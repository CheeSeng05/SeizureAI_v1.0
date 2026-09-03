"""Evaluate a frozen facial checkpoint on an untouched temporal test cohort."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ..evaluation.agreement import cohen_kappa
from ..evaluation.bootstrap import patient_bootstrap_confidence_intervals
from ..evaluation.classification import binary_classification_metrics
from ..evaluation.events import TemporalEvent, TemporalWindow
from ..evaluation.facial_biomarkers import biomarker_metadata, extract_facial_biomarkers
from ..evaluation.pipeline import evaluate_binary_timeline


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotation_events(
    annotations: Iterable[Mapping[str, Any]],
) -> tuple[list[TemporalEvent], list[TemporalEvent]]:
    """Parse one row per recording/event; blank onset denotes no seizure."""
    references: list[TemporalEvent] = []
    monitoring_by_recording: dict[tuple[str, str], TemporalEvent] = {}
    for row in annotations:
        patient = str(row["patient_id"])
        recording = str(row["recording_id"])
        monitor = TemporalEvent(
            patient,
            recording,
            float(row.get("monitor_start_sec") or 0.0),
            float(row["monitor_end_sec"]),
        )
        key = (patient, recording)
        previous = monitoring_by_recording.get(key)
        if previous is not None and previous != monitor:
            raise ValueError(f"Conflicting monitoring intervals for {key}")
        monitoring_by_recording[key] = monitor
        onset = row.get("seizure_onset_sec")
        offset = row.get("seizure_offset_sec")
        if onset not in (None, ""):
            if offset in (None, ""):
                raise ValueError(f"Seizure offset is required for temporal event {key}")
            references.append(TemporalEvent(patient, recording, float(onset), float(offset)))
    return references, list(monitoring_by_recording.values())


def _overlaps(window: TemporalWindow, event: TemporalEvent) -> bool:
    return window.patient_id == event.patient_id and window.recording_id == event.recording_id and (
        min(window.end_sec, event.end_sec) > max(window.start_sec, event.start_sec)
    )


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate aligned probability records using clinician event timestamps."""
    if not records:
        raise ValueError("Facial evaluation requires prediction records")
    windows = [
        TemporalWindow(
            str(row["patient_id"]), str(row["recording_id"]),
            float(row["start_sec"]), float(row["end_sec"]),
        )
        for row in records
    ]
    probabilities = [float(row["seizure_probability"]) for row in records]
    references, monitoring = annotation_events(annotations)
    record_keys = {(window.patient_id, window.recording_id) for window in windows}
    monitored_keys = {event.key for event in monitoring}
    missing = sorted(record_keys - monitored_keys)
    if missing:
        raise ValueError(
            "Every evaluated recording requires a monitoring annotation; missing "
            + ", ".join(f"{patient}/{recording}" for patient, recording in missing)
        )
    labels = [int(any(_overlaps(window, event) for event in references)) for window in windows]
    event_config = config["event_generation"]
    result = evaluate_binary_timeline(
        windows=windows,
        labels=labels,
        probabilities=probabilities,
        reference_events=references,
        monitoring_intervals=monitoring,
        threshold=float(event_config["probability_threshold"]),
        minimum_event_duration_sec=float(event_config["minimum_event_duration_sec"]),
        gap_tolerance_sec=float(event_config["gap_tolerance_sec"]),
        refractory_period_sec=float(event_config["refractory_period_sec"]),
        onset_reference=str(config["onset_reference"]),
        offset_annotations_available=bool(config["offset_annotations_available"]),
    )
    result["predicted_events"] = [asdict(event) for event in result["predicted_events"]]
    return result


def evaluate_clip_records(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate independent curated clips without inventing event timestamps."""
    if not records:
        raise ValueError("Facial clip evaluation requires prediction records")
    labels = [int(row["reference_label"]) for row in records]
    probabilities = [float(row["seizure_probability"]) for row in records]
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    unavailable_reason = (
        "Isolated curated clips have no continuous monitoring or clinician-verified "
        "onset/offset annotations."
    )
    return {
        "classification": binary_classification_metrics(
            labels, probabilities, threshold=threshold
        ),
        "cohen_kappa": cohen_kappa(labels, predictions),
        "predicted_events": [],
        "event_level_sensitivity": None,
        "false_positives_per_hour": {
            "false_alarm_events": None,
            "nonseizure_monitoring_hours": None,
            "false_positives_per_hour": None,
        },
        "localization": {
            "matched_events": 0,
            "onset_reference": None,
            "onset": None,
            "offset": None,
        },
        "temporal_metrics_available": False,
        "temporal_metrics_reason": unavailable_reason,
    }


def metric_summary(result: Mapping[str, Any]) -> dict[str, float | None]:
    classification = result["classification"]
    fp = result["false_positives_per_hour"]
    localization = result["localization"]
    onset = localization["onset"] or {}
    offset = localization["offset"] or {}
    return {
        "sensitivity": classification["sensitivity"],
        "specificity": classification["specificity"],
        "auroc": classification["auroc"],
        "f1": classification["f1"],
        "event_sensitivity": result["event_level_sensitivity"],
        "false_positives_per_hour": fp["false_positives_per_hour"],
        "onset_mae_sec": onset.get("mean_absolute_error_sec"),
        "offset_mae_sec": offset.get("mean_absolute_error_sec"),
    }


def patient_bootstrap_timeline_ci(
    records: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Recompute all metrics while resampling whole patients.

    Duplicate sampled patients are renamed so event matching cannot merge the
    same recording across bootstrap copies.
    """
    records_by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        records_by_patient.setdefault(str(row["patient_id"]), []).append(row)
    annotation_by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in annotations:
        annotation_by_patient.setdefault(str(row["patient_id"]), []).append(row)

    patients = sorted(records_by_patient)
    if not patients:
        raise ValueError("Patient bootstrap requires prediction records")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        renamed_records: list[dict[str, Any]] = []
        renamed_annotations: list[dict[str, Any]] = []
        for copy_index, selected in enumerate(
            rng.choice(patients, size=len(patients), replace=True)
        ):
            patient = str(selected)
            renamed = f"{patient}__bootstrap_{copy_index}"
            renamed_records.extend(
                {**row, "patient_id": renamed} for row in records_by_patient[patient]
            )
            renamed_annotations.extend(
                {**row, "patient_id": renamed}
                for row in annotation_by_patient.get(patient, [])
            )
        values = metric_summary(evaluate_records(renamed_records, renamed_annotations, config))
        for name, value in values.items():
            if value is not None and np.isfinite(value):
                samples.setdefault(name, []).append(float(value))
    tail = (1.0 - confidence) / 2.0
    return {
        name: {
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1.0 - tail)),
            "confidence": confidence,
            "method": "patient-level percentile bootstrap",
            "valid_resamples": len(values),
        }
        for name, values in samples.items()
    }


def patient_bootstrap_clip_ci(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Classification CIs that keep every clip from a sampled patient together."""
    return patient_bootstrap_confidence_intervals(
        records,
        patient_id=lambda row: str(row["patient_id"]),
        metric=lambda sample: metric_summary(
            evaluate_clip_records(sample, threshold=threshold)
        ),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def infer_checkpoint(
    *, manifest_path: Path, checkpoint_path: Path, split: str,
    minimum_face_detection: float, batch_size: int, device_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the frozen model and return prediction and biomarker rows."""
    import torch
    from torch.utils.data import DataLoader

    from .inference import load_model
    from .train import FaceWindowDataset, resolve_device

    device = resolve_device(device_name)
    dataset = FaceWindowDataset(manifest_path, split, minimum_face_detection)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model, classes = load_model(checkpoint_path, device)
    if "seizure" not in classes:
        raise ValueError("Checkpoint classes must include 'seizure'")
    seizure_index = classes.index("seizure")
    scores: list[float] = []
    with torch.inference_mode():
        for videos, _ in loader:
            scores.extend(model.predict_proba(videos.to(device))[:, seizure_index].cpu().tolist())

    root = manifest_path.resolve().parent
    predictions: list[dict[str, Any]] = []
    biomarkers: list[dict[str, Any]] = []
    for row, score in zip(dataset.rows, scores):
        base = {
            "window_id": row["window_id"], "patient_id": row["patient_id"],
            "recording_id": row["recording_id"], "start_sec": row["start_sec"],
            "end_sec": row["end_sec"], "seizure_probability": score,
            "reference_label": int(row["label"]),
        }
        predictions.append(base)
        face = np.load(root / row["face_path"], allow_pickle=False)
        pose = np.load(root / row["pose_path"], allow_pickle=False) if row.get("pose_path") else None
        geometry = np.load(root / row["face_geometry_path"], allow_pickle=False) if row.get("face_geometry_path") else None
        biomarkers.append({
            **base,
            **extract_facial_biomarkers(
                face_window=face, pose_window=pose, face_geometry_window=geometry,
                sampled_fps=float(row["sampled_fps"]),
            ),
        })
    return predictions, biomarkers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("clip", "event"), default="clip")
    parser.add_argument(
        "--annotations",
        help="Clinician event CSV; required only for --mode event",
    )
    parser.add_argument("--config", default="configs/facial_evaluation.yaml")
    parser.add_argument("--output", default="results/facial_evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = str(config.get("split", "test"))
    if split != "test":
        raise ValueError("Final facial evaluation is restricted to split='test'")
    if config.get("use_checkpoint_threshold", True):
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "probability_threshold" not in checkpoint:
            raise ValueError(
                "Checkpoint has no validation-selected probability_threshold; "
                "retrain it or set use_checkpoint_threshold: false with a pre-registered threshold"
            )
        config["event_generation"]["probability_threshold"] = float(
            checkpoint["probability_threshold"]
        )
    predictions, biomarkers = infer_checkpoint(
        manifest_path=Path(args.manifest), checkpoint_path=Path(args.checkpoint), split=split,
        minimum_face_detection=float(config.get("minimum_face_detection", 0.5)),
        batch_size=int(config.get("batch_size", 4)), device_name=args.device,
    )
    threshold = float(config["event_generation"]["probability_threshold"])
    for row in predictions:
        row["predicted_label"] = int(float(row["seizure_probability"]) >= threshold)
    bootstrap = config["bootstrap"]
    bootstrap_args = {
        "confidence": float(bootstrap["confidence"]),
        "n_resamples": int(bootstrap["resamples"]),
        "seed": int(bootstrap["seed"]),
    }
    if args.mode == "clip":
        result = evaluate_clip_records(predictions, threshold=threshold)
        result["confidence_intervals"] = patient_bootstrap_clip_ci(
            predictions, threshold=threshold, **bootstrap_args
        )
    else:
        if not args.annotations:
            parser.error("--annotations is required when --mode event")
        annotations = read_csv(args.annotations)
        references, _ = annotation_events(annotations)
        for row in predictions:
            window = TemporalWindow(
                str(row["patient_id"]), str(row["recording_id"]),
                float(row["start_sec"]), float(row["end_sec"]),
            )
            row["reference_label"] = int(
                any(_overlaps(window, event) for event in references)
            )
        result = evaluate_records(predictions, annotations, config)
        result["confidence_intervals"] = patient_bootstrap_timeline_ci(
            predictions, annotations, config, **bootstrap_args
        )
    result["biomarker_definitions"] = biomarker_metadata()
    result["evaluation_split"] = split
    result["evaluation_mode"] = args.mode
    result["probability_threshold_source"] = (
        "validation checkpoint" if config.get("use_checkpoint_threshold", True)
        else "pre-registered configuration"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "biomarkers.csv", biomarkers)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metric_summary(result), indent=2))


if __name__ == "__main__":
    main()
