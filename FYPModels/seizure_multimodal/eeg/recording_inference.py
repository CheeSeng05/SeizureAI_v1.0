from __future__ import annotations

import csv
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..progress import report_progress
from .eeg_adapter import EEGAdapter, EEGPreprocessor
from .loaders import EEGMetadata, EEGRecording, discover_eeg_files, load_eeg
from .montage import (
    BIPOLAR_CHANNELS,
    STANDARD_ELECTRODE_ALIASES,
    BipolarMontageError,
    BipolarMontagePlan,
    build_bipolar_montage_plan,
)


class EEGCompatibilityError(RuntimeError):
    """Raised rather than forcing scientifically unsupported model input."""


@dataclass(frozen=True)
class ModelEEGRequirements:
    sampling_rate: float
    channel_count: int
    required_channels: tuple[str, ...]
    expected_units: str | None
    expected_montage: str | None
    window_duration_sec: float
    window_stride_sec: float
    input_samples: int
    channel_aliases: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class CompatibilityItem:
    property: str
    model_expects: Any
    recording_value: Any
    status: str
    action: str


@dataclass(frozen=True)
class PreparedEEGWindows:
    windows: np.ndarray
    start_seconds: np.ndarray
    end_seconds: np.ndarray
    selected_channels: tuple[str, ...]
    aliases_used: Mapping[str, str]
    source_sampling_rate: float
    target_sampling_rate: float
    resampling_performed: bool


EEGAnnotationFields = (
    "recording_id", "event_id", "time_sec", "duration_sec", "raw_name",
    "normalized_type", "annotation_source", "clinical_status", "event_type",
    "event_guid", "source_user_available",
)


def requirements_from_config(config: Mapping[str, Any]) -> ModelEEGRequirements:
    section = config.get("eeg_encoder", {})
    if not isinstance(section, Mapping):
        raise ValueError("eeg_encoder configuration must be a mapping.")
    aliases_raw = section.get("channel_aliases") or STANDARD_ELECTRODE_ALIASES
    if not isinstance(aliases_raw, Mapping):
        raise ValueError("eeg_encoder.channel_aliases must be a mapping.")
    aliases = {
        str(key).upper(): tuple(str(value) for value in values)
        for key, values in aliases_raw.items()
        if isinstance(values, (list, tuple))
    }
    required = section.get("required_channels", ())
    if not isinstance(required, (list, tuple)):
        raise ValueError("eeg_encoder.required_channels must be a list.")
    sampling_rate = float(section.get("sampling_rate", 256.0))
    window_duration = float(section.get("window_duration_sec", 4.0))
    window_stride = float(section.get("window_stride_sec", 2.0))
    return ModelEEGRequirements(
        sampling_rate=sampling_rate,
        channel_count=int(section.get("channels", 18)),
        required_channels=tuple(str(value).upper() for value in required),
        expected_units=str(section.get("signal_units") or "uV"),
        expected_montage=str(section.get("montage_reference") or "CHB-MIT bipolar"),
        window_duration_sec=window_duration,
        window_stride_sec=window_stride,
        input_samples=int(round(sampling_rate * window_duration)),
        channel_aliases=aliases,
    )


def _channel_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def resolve_channel_indices(
    available: Sequence[str], required: Sequence[str],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[int, ...]:
    """Resolve an explicitly ordered set of channels (legacy public helper)."""
    if not required:
        raise EEGCompatibilityError("No required EEG channels were configured.")
    by_key: dict[str, list[int]] = {}
    for index, name in enumerate(available):
        by_key.setdefault(_channel_key(name), []).append(index)
    selected: list[int] = []
    missing: list[str] = []
    for required_name in required:
        match = None
        for candidate in (required_name, *(aliases or {}).get(required_name, ())):
            matches = by_key.get(_channel_key(candidate), ())
            if len(matches) == 1:
                match = matches[0]
                break
            if len(matches) > 1:
                break
        if match is None:
            missing.append(required_name)
        elif match in selected:
            raise EEGCompatibilityError(
                "Configured channel aliases map more than one required channel to the same source channel."
            )
        else:
            selected.append(match)
    if missing:
        raise EEGCompatibilityError("Missing or ambiguous required EEG channels: " + ", ".join(missing))
    return tuple(selected)


def _units_key(value: str) -> str:
    return value.strip().casefold().replace("μ", "u").replace("µ", "u")


def convert_signal_units(data: np.ndarray, source_units: Sequence[str], target_unit: str) -> np.ndarray:
    if data.ndim != 2 or len(source_units) != data.shape[0]:
        raise ValueError("Unit conversion expects [channels, samples] and one unit per channel.")
    target = _units_key(target_unit)
    to_volts = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
    if target not in to_volts:
        raise EEGCompatibilityError(f"Unsupported target EEG unit: {target_unit}")
    result = np.asarray(data, dtype=np.float64).copy()
    for index, unit in enumerate(source_units):
        source = _units_key(unit)
        if source not in to_volts:
            raise EEGCompatibilityError(
                f"Channel {index} has unknown/digital units '{unit}'; physical scaling cannot be guessed."
            )
        result[index] *= to_volts[source] / to_volts[target]
    return result.astype(np.float32)


def _plan(metadata: EEGMetadata, requirements: ModelEEGRequirements) -> BipolarMontagePlan:
    try:
        return build_bipolar_montage_plan(metadata.channel_names, aliases=requirements.channel_aliases)
    except BipolarMontageError as exc:
        raise EEGCompatibilityError(str(exc)) from exc


def _source_sampling_rate(metadata: EEGMetadata, plan: BipolarMontagePlan) -> float:
    rates = {round(float(metadata.sampling_rates[index]), 9) for index in plan.source_indices}
    if len(rates) != 1:
        raise EEGCompatibilityError("Required EEG electrodes do not share one sampling rate.")
    rate = next(iter(rates))
    if rate <= 0:
        raise EEGCompatibilityError("Required EEG electrodes have an invalid sampling rate.")
    return rate


def assess_compatibility(metadata: EEGMetadata, requirements: ModelEEGRequirements) -> list[CompatibilityItem]:
    items: list[CompatibilityItem] = []
    required_ok = (
        requirements.channel_count == len(BIPOLAR_CHANNELS)
        and requirements.required_channels == BIPOLAR_CHANNELS
        and requirements.input_samples == 1024
    )
    try:
        plan = _plan(metadata, requirements)
        source_rate = _source_sampling_rate(metadata, plan)
        plan_error = None
    except EEGCompatibilityError as exc:
        plan, source_rate, plan_error = None, None, str(exc)
    channel_status = "compatible" if required_ok and plan is not None else "incompatible"
    channel_action = (
        "Construct the exact explicit 18-channel order."
        if channel_status == "compatible" else
        plan_error or "Configuration must contain the exact CHB-MIT 18-channel order and 1024 samples."
    )
    items.append(CompatibilityItem(
        "channels", list(BIPOLAR_CHANNELS), list(metadata.channel_names), channel_status, channel_action,
    ))
    items.append(CompatibilityItem(
        "channel order", list(BIPOLAR_CHANNELS),
        list(plan.channel_order) if plan else "unavailable", channel_status, channel_action,
    ))
    rate_status = "incompatible" if source_rate is None else (
        "compatible" if np.isclose(source_rate, requirements.sampling_rate) else "adaptable"
    )
    items.append(CompatibilityItem(
        "sampling rate", requirements.sampling_rate, source_rate, rate_status,
        "Use scipy.signal.resample_poly and preserve time alignment." if rate_status == "adaptable"
        else "No resampling required." if rate_status == "compatible"
        else "Required electrodes must share one valid sampling rate.",
    ))
    units = [metadata.signal_units[index] for index in plan.source_indices] if plan else []
    units_status = "compatible" if units and all(
        _units_key(value) == _units_key(requirements.expected_units or "") for value in units
    ) else "adaptable" if units and all(
        _units_key(value) in {"v", "mv", "uv"} for value in units
    ) else "incompatible"
    items.append(CompatibilityItem(
        "units", requirements.expected_units, sorted(set(units)), units_status,
        "Convert metadata-backed physical values to microvolts before subtraction."
        if units_status == "adaptable" else
        "Use physical microvolts directly." if units_status == "compatible"
        else "Digital/unknown units cannot be guessed.",
    ))
    montage_status = "incompatible" if plan is None else (
        "compatible" if plan.uses_existing_bipolar_channels else "adaptable"
    )
    items.append(CompatibilityItem(
        "montage/reference", requirements.expected_montage, metadata.montage_reference or "unknown",
        montage_status,
        "Use the existing exact bipolar channels." if montage_status == "compatible"
        else "Construct A-B bipolar channels from physical common-reference electrodes."
        if montage_status == "adaptable" else channel_action,
    ))
    items.extend((
        CompatibilityItem("window size", 4.0, requirements.window_duration_sec,
                          "compatible" if np.isclose(requirements.window_duration_sec, 4.0) else "incompatible",
                          "Use complete four-second windows."),
        CompatibilityItem("window stride", 2.0, requirements.window_stride_sec,
                          "compatible" if np.isclose(requirements.window_stride_sec, 2.0) else "incompatible",
                          "Use a two-second stride (50% overlap)."),
        CompatibilityItem("input tensor shape", [None, 1024, 18],
                          [None, requirements.input_samples, requirements.channel_count],
                          "compatible" if required_ok else "incompatible",
                          "Fail unless the time-major model tensor is [B, 1024, 18]."),
        CompatibilityItem("labels", "not used for recording inference",
                          "embedded events retained separately", "compatible",
                          "Never pass annotations into the model."),
    ))
    return items


def require_compatible(items: Sequence[CompatibilityItem]) -> None:
    blockers = [item for item in items if item.status == "incompatible"]
    if blockers:
        raise EEGCompatibilityError(
            "Clinical EEG inference blocked: "
            + "; ".join(f"{item.property}: {item.action}" for item in blockers)
        )


def _source_sample_count(recording: EEGRecording, source_rate: float) -> int:
    duration_count = int(round(recording.metadata.duration_sec * source_rate))
    return min(recording.metadata.sample_count, duration_count) if duration_count else recording.metadata.sample_count


def _read_output_channel(
    recording: EEGRecording, plan: BipolarMontagePlan, output_index: int,
    sample_count: int, target_unit: str, *, start_sample: int = 0,
) -> np.ndarray:
    direct = plan.direct_indices[output_index]
    pair = plan.source_pairs[output_index]
    indices = (direct,) if direct is not None else pair
    if not indices:
        raise RuntimeError("Invalid bipolar montage plan.")
    physical = recording.read_samples(
        channel_indices=indices, start_sample=start_sample, stop_sample=start_sample + sample_count,
    )
    physical = convert_signal_units(
        physical, [recording.metadata.signal_units[index] for index in indices], target_unit,
    )
    return physical[0] if direct is not None else physical[0] - physical[1]


def _timeline_segments(
    metadata: EEGMetadata,
    *,
    source_count: int,
    source_rate: float,
    target_rate: float,
) -> list[dict[str, Any]]:
    configured = metadata.extra.get("timeline_segments", ()) if isinstance(metadata.extra, Mapping) else ()
    source_rows = list(configured) if isinstance(configured, (list, tuple)) else []
    if not source_rows:
        source_rows = [{
            "segment_index": 0,
            "source_sample_start": 0,
            "source_sample_stop": source_count,
            "global_start_sec": 0.0,
            "global_end_sec": source_count / source_rate,
            "duration_sec": source_count / source_rate,
        }]
    result: list[dict[str, Any]] = []
    target_cursor = 0
    for position, source in enumerate(source_rows):
        source_start = int(source.get("source_sample_start", 0))
        source_stop = min(int(source.get("source_sample_stop", source_count)), source_count)
        if source_start < 0 or source_stop <= source_start:
            raise EEGCompatibilityError("EEG timeline segment has invalid sample bounds.")
        segment_source_count = source_stop - source_start
        target_length = int(round(segment_source_count * target_rate / source_rate))
        global_start = float(source.get("global_start_sec", source_start / source_rate))
        result.append({
            "segment_index": int(source.get("segment_index", position)),
            "source_sample_start": source_start,
            "source_sample_stop": source_stop,
            "target_sample_start": target_cursor,
            "target_sample_stop": target_cursor + target_length,
            "global_start_sec": global_start,
            "global_end_sec": global_start + target_length / target_rate,
        })
        target_cursor += target_length
    if result[-1]["source_sample_stop"] != source_count:
        raise EEGCompatibilityError("EEG timeline segments do not cover the complete recording.")
    return result


def _resample(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    if np.isclose(source_rate, target_rate):
        return np.asarray(signal, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("SciPy is required for EEG resampling.") from exc
    source_int, target_int = int(round(source_rate)), int(round(target_rate))
    if not np.isclose(source_rate, source_int) or not np.isclose(target_rate, target_int):
        raise EEGCompatibilityError("Polyphase recording resampling requires integral source and target rates.")
    divisor = gcd(source_int, target_int)
    return resample_poly(
        signal, target_int // divisor, source_int // divisor, padtype="reflect",
    ).astype(np.float32)


def prepare_model_windows(
    recording: EEGRecording, requirements: ModelEEGRequirements,
) -> PreparedEEGWindows:
    """Prepare a small/ordinary recording as `[B, 1024, 18]` for tests and tools."""
    require_compatible(assess_compatibility(recording.metadata, requirements))
    plan = _plan(recording.metadata, requirements)
    source_rate = _source_sampling_rate(recording.metadata, plan)
    sample_count = _source_sample_count(recording, source_rate)
    preprocessor = EEGPreprocessor(sampling_rate=requirements.sampling_rate)
    channels = []
    for index in range(len(BIPOLAR_CHANNELS)):
        signal = _read_output_channel(
            recording, plan, index, sample_count, requirements.expected_units or "uV"
        )
        channels.append(preprocessor.transform(
            _resample(signal, source_rate, requirements.sampling_rate)[None, :]
        )[0])
    data = np.asarray(channels, dtype=np.float32)
    window_samples = requirements.input_samples
    stride_samples = int(round(requirements.window_stride_sec * requirements.sampling_rate))
    starts = np.arange(0, max(data.shape[1] - window_samples + 1, 0), stride_samples, dtype=int)
    windows = (
        np.stack([data[:, start:start + window_samples].T for start in starts]).astype(np.float32)
        if len(starts) else np.empty((0, window_samples, len(BIPOLAR_CHANNELS)), dtype=np.float32)
    )
    if windows.shape[1:] != (1024, 18):
        raise EEGCompatibilityError(f"Prepared EEG tensor has invalid shape {windows.shape}; expected [B, 1024, 18].")
    return PreparedEEGWindows(
        windows=windows,
        start_seconds=starts.astype(float) / requirements.sampling_rate,
        end_seconds=(starts + window_samples).astype(float) / requirements.sampling_rate,
        selected_channels=BIPOLAR_CHANNELS,
        aliases_used=dict(plan.aliases_used),
        source_sampling_rate=source_rate,
        target_sampling_rate=requirements.sampling_rate,
        resampling_performed=not np.isclose(source_rate, requirements.sampling_rate),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _resolve_output(project_root: Path, configured: str | Path) -> Path:
    value = Path(configured)
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def run_recording_inference(
    *, input_path: str | Path, project_root: str | Path, config: Mapping[str, Any],
    output_root: str | Path, batch_size: int | None = None,
) -> dict[str, Any]:
    """Run one EDF/Nicolet recording without materializing all windows in RAM/GPU."""
    if config.get("training", {}).get("enabled") is not False:
        raise RuntimeError("Clinical recording inference requires training.enabled=false.")
    threshold = float(config.get("testing", {}).get("threshold", 0.5))
    if not np.isclose(threshold, 0.5):
        raise EEGCompatibilityError("Clinical recording inference uses the frozen threshold 0.5.")
    project = Path(project_root).resolve()
    output = _resolve_output(project, output_root)
    output.mkdir(parents=True, exist_ok=True)
    requirements = requirements_from_config(config)
    configured_batch = int(config.get("eeg_encoder", {}).get("inference_batch_size", 64))
    inference_batch_size = int(batch_size or configured_batch)
    if inference_batch_size <= 0:
        raise ValueError("EEG inference batch size must be positive.")

    report_progress(config, stage="reading_eeg", label="Reading EEG recording", completed=0, total=1)
    recording = load_eeg(
        input_path, annotation_overrides=config.get("eeg_annotations", {}).get("overrides", {}),
    )
    try:
        report_progress(config, stage="reading_eeg", label="Reading EEG recording", completed=1, total=1)
        compatibility = assess_compatibility(recording.metadata, requirements)
        _write_csv(output / "annotations.csv", [asdict(event) for event in recording.annotations], EEGAnnotationFields)
        _write_json(output / "compatibility.json", [asdict(item) for item in compatibility])
        try:
            require_compatible(compatibility)
        except EEGCompatibilityError as exc:
            payload = {
                "status": "blocked", "prediction_only": True, "error": str(exc),
                "training_performed": False, "annotations_passed_to_model": False,
            }
            _write_json(output / "status.json", payload)
            _write_json(output / "metrics.json", payload)
            return payload

        plan = _plan(recording.metadata, requirements)
        source_rate = _source_sampling_rate(recording.metadata, plan)
        source_count = _source_sample_count(recording, source_rate)
        timeline_segments = _timeline_segments(
            recording.metadata,
            source_count=source_count,
            source_rate=source_rate,
            target_rate=requirements.sampling_rate,
        )
        target_count = int(timeline_segments[-1]["target_sample_stop"])
        window_samples = requirements.input_samples
        stride_samples = int(round(requirements.window_stride_sec * requirements.sampling_rate))
        window_positions: list[tuple[int, float, float, int]] = []
        for segment in timeline_segments:
            local_starts = np.arange(
                int(segment["target_sample_start"]),
                max(int(segment["target_sample_stop"]) - window_samples + 1, int(segment["target_sample_start"])),
                stride_samples,
                dtype=np.int64,
            )
            for start in local_starts:
                local_seconds = (int(start) - int(segment["target_sample_start"])) / requirements.sampling_rate
                global_start = float(segment["global_start_sec"]) + local_seconds
                window_positions.append(
                    (int(start), global_start, global_start + requirements.window_duration_sec, int(segment["segment_index"]))
                )
        starts = np.asarray([item[0] for item in window_positions], dtype=np.int64)
        if not len(starts):
            raise EEGCompatibilityError("The EEG recording is shorter than one four-second model window.")

        preprocessing_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=".eeg-working-", dir=output) as working_directory:
            processed = np.memmap(
                Path(working_directory) / "preprocessed.float32", dtype=np.float32, mode="w+",
                shape=(len(BIPOLAR_CHANNELS), target_count),
            )
            preprocessor = EEGPreprocessor(sampling_rate=requirements.sampling_rate)
            for index in range(len(BIPOLAR_CHANNELS)):
                for segment in timeline_segments:
                    segment_source_count = int(segment["source_sample_stop"]) - int(segment["source_sample_start"])
                    signal = _read_output_channel(
                        recording,
                        plan,
                        index,
                        segment_source_count,
                        requirements.expected_units or "uV",
                        start_sample=int(segment["source_sample_start"]),
                    )
                    resampled = _resample(signal, source_rate, requirements.sampling_rate)
                    expected_length = int(segment["target_sample_stop"]) - int(segment["target_sample_start"])
                    if len(resampled) != expected_length:
                        raise EEGCompatibilityError("Resampled EEG segments do not have consistent lengths.")
                    processed[
                        index,
                        int(segment["target_sample_start"]):int(segment["target_sample_stop"]),
                    ] = preprocessor.transform(resampled[None, :])[0]
                report_progress(
                    config, stage="preprocessing", label="Preparing EEG channels",
                    completed=index + 1, total=len(BIPOLAR_CHANNELS),
                    elapsed_sec=time.perf_counter() - preprocessing_started,
                )
            processed.flush()

            checkpoint_value = Path(str(config["eeg_encoder"]["checkpoint"]))
            checkpoint = checkpoint_value if checkpoint_value.is_absolute() else project / checkpoint_value
            adapter = EEGAdapter(checkpoint_path=checkpoint, channel_order=BIPOLAR_CHANNELS)
            if not adapter.is_inference_only:
                raise RuntimeError("EEG checkpoint is not frozen for inference-only execution.")

            prediction_path = output / "predictions.csv"
            fields = (
                "recording_id", "window_id", "start_sec", "end_sec",
                "eeg_segment_index", "probability", "logit", "predicted_label",
            )
            minimum, maximum = float("inf"), float("-inf")
            total_probability, positive_count = 0.0, 0
            inference_started = time.perf_counter()
            with prediction_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for batch_start in range(0, len(starts), inference_batch_size):
                    batch_starts = starts[batch_start:batch_start + inference_batch_size]
                    batch_positions = window_positions[batch_start:batch_start + inference_batch_size]
                    model_input = np.stack([
                        np.asarray(processed[:, start:start + window_samples]).T for start in batch_starts
                    ]).astype(np.float32)
                    if model_input.ndim != 3 or model_input.shape[1:] != (1024, 18):
                        raise EEGCompatibilityError(
                            f"Model EEG tensor has invalid shape {model_input.shape}; expected [B, 1024, 18]."
                        )
                    prediction = adapter.predict_probabilities(model_input, preprocessed=True)
                    probabilities = np.asarray(prediction["probability"], dtype=float).reshape(-1)
                    logits = np.asarray(prediction["logits"], dtype=float).reshape(-1)
                    for offset, (position, score, logit) in enumerate(zip(batch_positions, probabilities, logits)):
                        start, global_start, global_end, segment_index = position
                        window_index = batch_start + offset
                        writer.writerow({
                            "recording_id": recording.metadata.recording_id,
                            "window_id": f"window-{window_index + 1:06d}",
                            "start_sec": global_start,
                            "end_sec": global_end,
                            "eeg_segment_index": segment_index,
                            "probability": float(score), "logit": float(logit),
                            "predicted_label": int(score >= threshold),
                        })
                    minimum = min(minimum, float(probabilities.min()))
                    maximum = max(maximum, float(probabilities.max()))
                    total_probability += float(probabilities.sum())
                    positive_count += int(np.count_nonzero(probabilities >= threshold))
                    report_progress(
                        config, stage="running_model", label="Running EEG model",
                        completed=min(batch_start + len(batch_starts), len(starts)), total=len(starts),
                        elapsed_sec=time.perf_counter() - inference_started,
                    )
            del processed

        metadata = {
            "source_format": recording.metadata.source_format,
            "duration_sec": recording.metadata.duration_sec,
            "source_sampling_rate": source_rate, "target_sampling_rate": requirements.sampling_rate,
            "resampling_method": "scipy.signal.resample_poly",
            "resampling_performed": not np.isclose(source_rate, requirements.sampling_rate),
            "channel_order": list(BIPOLAR_CHANNELS), "aliases_used": dict(plan.aliases_used),
            "window_duration_sec": requirements.window_duration_sec,
            "window_stride_sec": requirements.window_stride_sec, "window_overlap": 0.5,
            "samples_per_window": window_samples,
            "model_input_shape": [len(starts), 1024, 18],
            "prediction_output_shape": [len(starts)],
            "inference_batch_size": inference_batch_size,
            "annotation_count": len(recording.annotations), "annotations_passed_to_model": False,
            "checkpoint_frozen": adapter.is_inference_only, "training_performed": False,
            "timeline": "recording_wall_clock_elapsed",
            "timeline_segments": timeline_segments,
            "wall_duration_sec": max(float(item["global_end_sec"]) for item in timeline_segments),
        }
        metrics = {
            "prediction_only": True, "threshold": threshold, "window_count": len(starts),
            "seizure_positive_windows": positive_count, "probability_min": minimum,
            "probability_max": maximum, "probability_mean": total_probability / len(starts),
            "duration_sec": recording.metadata.duration_sec,
        }
        payload = {"status": "completed", "metadata": metadata, "metrics": metrics}
        _write_json(output / "status.json", payload)
        _write_json(output / "metrics.json", payload)
        report_progress(config, stage="completed", label="EEG processing completed", completed=1, total=1)
        return payload
    finally:
        recording.close()


def run_first_recording_inference(
    *, dataset_root: str | Path | None, project_root: str | Path,
    config: Mapping[str, Any], output_root: str | Path = "results/private", batch_size: int = 64,
) -> dict[str, Any]:
    files = [path for path in discover_eeg_files(dataset_root) if path.suffix.casefold() == ".e"]
    if not files:
        raise FileNotFoundError("No .e recording was discovered recursively.")
    project = Path(project_root).resolve()
    private_root = _resolve_output(project, output_root)
    with load_eeg(files[0]) as recording:
        recording_id = recording.metadata.recording_id
    return run_recording_inference(
        input_path=files[0], project_root=project, config=config,
        output_root=private_root / recording_id, batch_size=batch_size,
    )
