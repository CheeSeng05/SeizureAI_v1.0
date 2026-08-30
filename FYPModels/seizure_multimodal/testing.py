from __future__ import annotations

import csv
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .checkpoints.provenance import sha256_file
from .discovery import ModalitySource, discover_sources, inspection_bundle, validate_verified_pair
from .eeg.csv_inference import normalized_label_mapping, predict_eeg_csv
from .evaluation.agreement import cohen_kappa
from .evaluation.classification import binary_classification_metrics
from .evaluation.events import TemporalEvent, TemporalWindow, probabilities_to_events
from .fusion.feature_fusion import FeatureFusionSpec
from .fusion.late_fusion import LateFusion
from .video.raw_pipeline import predict_video_file


SIMULATED_WARNING = """============================================================

SIMULATED MULTIMODAL PAIRING

The video and EEG used in this experiment are NOT from the
same patient or synchronized clinical recording.

They are being paired artificially only to test the multimodal
software pipeline.

All fusion metrics generated in this mode are SOFTWARE /
INTEGRATION TEST RESULTS and MUST NOT be interpreted as
clinical model performance.

============================================================
"""

STRICT_REJECTION = """Real multimodal testing was not performed.

The available video and EEG data are not verified as belonging
to the same patient and synchronized recording.
"""

RESULT_METADATA_FIELDS = (
    "test_mode",
    "video_model_used",
    "EEG_model_used",
    "fusion_used",
    "pairing_type",
    "training_performed",
    "video_checkpoint",
    "EEG_checkpoint",
)


@dataclass
class WorkflowResult:
    modality: str
    status: str
    rows: list[dict[str, Any]]
    probabilities: np.ndarray
    labels: np.ndarray | None
    embeddings: np.ndarray | None
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    error: str | None = None


def result_metadata(
    *,
    test_mode: str,
    video_model_used: bool,
    eeg_model_used: bool,
    fusion_used: bool,
    pairing_type: str,
    video_checkpoint: str,
    eeg_checkpoint: str,
) -> dict[str, Any]:
    return {
        "test_mode": test_mode,
        "video_model_used": video_model_used,
        "EEG_model_used": eeg_model_used,
        "fusion_used": fusion_used,
        "pairing_type": pairing_type,
        "training_performed": False,
        "video_checkpoint": video_checkpoint,
        "EEG_checkpoint": eeg_checkpoint,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_value) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_fieldnames = list(dict.fromkeys(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=unique_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _augment_rows(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{**row, **metadata} for row in rows]


def _na_metrics(reason: str) -> dict[str, Any]:
    return {
        "sensitivity": "N/A",
        "specificity": "N/A",
        "auroc": "N/A",
        "f1": "N/A",
        "false_positives_per_hour": "N/A",
        "onset_mean_signed_error_sec": "N/A",
        "onset_mean_absolute_error_sec": "N/A",
        "offset_mean_signed_error_sec": "N/A",
        "offset_mean_absolute_error_sec": "N/A",
        "cohen_kappa": "N/A",
        "reason": reason,
    }


def _classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return _na_metrics("Both binary reference classes are required for classification metrics.")
    classification = binary_classification_metrics(labels, probability)
    prediction = (probability >= 0.5).astype(int)
    return {
        "sensitivity": classification["sensitivity"],
        "specificity": classification["specificity"],
        "auroc": classification["auroc"],
        "f1": classification["f1"],
        "precision": classification["precision"],
        "cohen_kappa": cohen_kappa(labels, prediction),
        "confusion": {key: classification[key] for key in ("tn", "fp", "fn", "tp")},
        "false_positives_per_hour": "N/A",
        "onset_mean_signed_error_sec": "N/A",
        "onset_mean_absolute_error_sec": "N/A",
        "offset_mean_signed_error_sec": "N/A",
        "offset_mean_absolute_error_sec": "N/A",
    }


def _video_metrics(rows: Sequence[Mapping[str, Any]], run_details: Mapping[str, Any]) -> dict[str, Any]:
    annotation = run_details["annotation"]
    if annotation["alignment_status"] == "annotation_file_missing":
        metrics = _na_metrics(
            "No label workbook was supplied. This is a prediction-only video run, so accuracy "
            "and onset/offset errors cannot be calculated."
        )
        metrics["annotation_alignment_status"] = annotation["alignment_status"]
        metrics["prediction_only"] = True
        metrics["classification_label_policy"] = "No ground-truth labels were available."
        if rows:
            patient_id = annotation.get("patient_id") or "UNKNOWN"
            recording_id = str(rows[0]["recording_id"])
            windows = [
                TemporalWindow(patient_id, recording_id, float(row["start_sec"]), float(row["end_sec"]))
                for row in rows
            ]
            probabilities = [float(row["video_probability"]) for row in rows]
            events = probabilities_to_events(windows, probabilities, threshold=0.5)
            metrics["predicted_event_count"] = len(events)
            metrics["predicted_events"] = [asdict(event) for event in events]
        return metrics
    if annotation["alignment_status"] == "out_of_video_range_requires_real_clip_offset":
        return {
            **_na_metrics("Workbook timestamps are outside video-local time; a verified clip offset is required."),
            "annotation_alignment_status": annotation["alignment_status"],
        }
    valid_rows = [row for row in rows if row.get("reference_label") in (0, 1)]
    labels = np.asarray([row["reference_label"] for row in valid_rows], dtype=int)
    probabilities = np.asarray([row["video_probability"] for row in valid_rows], dtype=float)
    metrics = _classification_metrics(labels, probabilities)
    metrics["evaluated_windows"] = len(valid_rows)
    metrics["excluded_windows"] = len(rows) - len(valid_rows)
    metrics["annotation_alignment_status"] = annotation["alignment_status"]
    metrics["classification_label_policy"] = (
        "VSViG source policy: 0 through EEG onset, transition interval excluded, "
        "1 from clinical onset onward; no seizure offset was invented."
    )
    if not rows:
        return metrics
    patient_id = annotation.get("patient_id") or "UNKNOWN"
    recording_id = str(rows[0]["recording_id"])
    windows = [
        TemporalWindow(patient_id, recording_id, float(row["start_sec"]), float(row["end_sec"]))
        for row in rows
    ]
    all_probabilities = [float(row["video_probability"]) for row in rows]
    events = probabilities_to_events(windows, all_probabilities, threshold=0.5)
    # Model detections exist independently of whether ground truth was supplied.
    # Persist them for labelled runs as well as prediction-only runs so the UI
    # never mistakes "evaluated against labels" for "no model events".
    metrics["predicted_event_count"] = len(events)
    metrics["predicted_events"] = [asdict(event) for event in events]
    eeg_onset = annotation.get("eeg_onset_sec")
    clinical_onset = annotation.get("clinical_onset_sec")
    if eeg_onset is not None and eeg_onset > 0:
        false_events = sum(event.end_sec <= float(eeg_onset) for event in events)
        monitoring_hours = float(eeg_onset) / 3600.0
        metrics["false_positive_alarm_events"] = false_events
        metrics["nonseizure_monitoring_hours"] = monitoring_hours
        metrics["false_positives_per_hour"] = false_events / monitoring_hours
    else:
        metrics["false_positives_per_hour"] = "N/A"
    if clinical_onset is not None and eeg_onset is not None:
        candidates = [event for event in events if event.end_sec > float(eeg_onset)]
        if candidates:
            signed = float(candidates[0].start_sec - float(clinical_onset))
            metrics["predicted_onset_sec"] = candidates[0].start_sec
            metrics["onset_reference"] = "clinical_onset"
            metrics["onset_mean_signed_error_sec"] = signed
            metrics["onset_mean_absolute_error_sec"] = abs(signed)
        else:
            metrics["onset_mean_signed_error_sec"] = "N/A"
            metrics["onset_mean_absolute_error_sec"] = "N/A"
    metrics["offset_mean_signed_error_sec"] = "N/A"
    metrics["offset_mean_absolute_error_sec"] = "N/A"
    metrics["offset_reason"] = "Label.xlsx has no seizure offset annotation."
    return metrics


def _plot_timeline(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.plot([row["start_sec"] for row in rows], [row["video_probability"] for row in rows])
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="threshold")
    status = rows[0].get("annotation_alignment_status")
    if status != "out_of_video_range_requires_real_clip_offset":
        if rows[0].get("eeg_onset_sec") is not None:
            axis.axvline(float(rows[0]["eeg_onset_sec"]), color="tab:red", label="EEG onset")
        if rows[0].get("clinical_onset_sec") is not None:
            axis.axvline(
                float(rows[0]["clinical_onset_sec"]), color="tab:orange", label="clinical onset"
            )
    axis.set(xlabel="Video-local time (s)", ylabel="VSViG seizure probability", ylim=(0, 1))
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_eeg(labels: np.ndarray, probabilities: np.ndarray, output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = binary_classification_metrics(labels, probabilities)
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["non-ictal", "ictal"],
        yticklabels=["non-ictal", "ictal"],
        xlabel="Predicted",
        ylabel="Reference",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "eeg_confusion_matrix.png", dpi=150)
    plt.close(figure)

    order = np.argsort(-probabilities, kind="mergesort")
    truth = labels[order]
    tp = np.cumsum(truth == 1)
    fp = np.cumsum(truth == 0)
    tpr = np.r_[0.0, tp / max(1, int(np.sum(labels == 1)))]
    fpr = np.r_[0.0, fp / max(1, int(np.sum(labels == 0)))]
    figure, axis = plt.subplots()
    axis.plot(fpr, tpr, label=f"AUROC={metrics['auroc']:.4f}")
    axis.plot([0, 1], [0, 1], "--")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="EEG ROC")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "eeg_roc.png", dpi=150)
    plt.close(figure)


def _checkpoint_label(root: Path, configured: str | None) -> tuple[str, Path | None]:
    if not configured:
        return "not configured", None
    path = (root / configured).resolve()
    return (f"{path} (sha256={sha256_file(path)})" if path.is_file() else f"{path} (MISSING)"), path


def run_video_workflow(
    root: Path,
    config: Mapping[str, Any],
    source: ModalitySource,
    *,
    predictor: Callable[..., tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]] = predict_video_file,
) -> WorkflowResult:
    video_cfg = config.get("video_encoder", {})
    test_cfg = config.get("testing", {})
    video_label, video_checkpoint = _checkpoint_label(root, video_cfg.get("checkpoint"))
    eeg_label, _ = _checkpoint_label(root, config.get("eeg_encoder", {}).get("checkpoint"))
    metadata = result_metadata(
        test_mode="video_only",
        video_model_used=True,
        eeg_model_used=False,
        fusion_used=False,
        pairing_type="none",
        video_checkpoint=video_label,
        eeg_checkpoint=eeg_label,
    )
    output = root / str(test_cfg.get("output_directory", "results")) / "video_only"
    output.mkdir(parents=True, exist_ok=True)
    try:
        if video_checkpoint is None or not video_checkpoint.is_file():
            raise FileNotFoundError("VSViG checkpoint is missing")
        requested_device = str(test_cfg.get("device", "auto"))
        if requested_device == "auto":
            try:
                import torch

                requested_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                requested_device = "cpu"
        provenance = config.get("data", {}).get("video_provenance", {})
        provenance = provenance if isinstance(provenance, Mapping) else {}
        expected_hash = str(provenance.get("expected_sha256", "")).lower()
        actual_hash = sha256_file(source.video_path)
        hash_verified = bool(expected_hash and actual_hash == expected_hash)
        if source.label_path and expected_hash and not hash_verified:
            print(
                "video provenance hash did not match the configured sample hash; "
                "running with the supplied label workbook without sample-specific row filters.",
                flush=True,
            )
        annotation_patient_id = provenance.get("patient_id") if hash_verified else None
        annotation_seizure_id = provenance.get("seizure_id") if hash_verified else None
        annotation_seizure_type = provenance.get("seizure_type") if hash_verified else None
        rows, embeddings, run_details = predictor(
            video_path=source.video_path,
            label_path=source.label_path,
            vsvig_source=root / str(video_cfg.get("source")),
            vsvig_checkpoint=video_checkpoint,
            dynamic_partition=root / str(video_cfg.get("dynamic_partition")),
            pose_source=root / str(video_cfg.get("pose_source")),
            pose_checkpoint=root / str(video_cfg.get("pose_checkpoint")),
            annotation_patient_id=str(annotation_patient_id) if annotation_patient_id else None,
            annotation_seizure_id=str(annotation_seizure_id) if annotation_seizure_id else None,
            annotation_seizure_type=str(annotation_seizure_type) if annotation_seizure_type else None,
            recording_id=source.recording_id,
            device=requested_device,
            threshold=float(test_cfg.get("threshold", 0.5)),
            clip_frames=int(video_cfg.get("clip_frames", 30)),
            clip_stride_frames=int(video_cfg.get("clip_stride_frames", 30)),
            batch_size=int(video_cfg.get("inference_batch_size", 8)),
            pose_batch_size=int(video_cfg.get("pose_batch_size", 16)),
            pose_input_height=int(video_cfg.get("pose_input_height", 256)),
            progress_config=dict(config),
        )
        metrics = _video_metrics(rows, run_details)
        augmented = _augment_rows(rows, metadata)
        fields = list(rows[0]) if rows else [
            "recording_id",
            "start_sec",
            "end_sec",
            "video_probability",
            "video_prediction",
            "reference_label",
        ]
        fields += [field for field in RESULT_METADATA_FIELDS if field not in fields]
        _write_csv(output / "predictions.csv", augmented, fields)
        _write_csv(output / f"{source.source_id}_predictions.csv", augmented, fields)
        timeline_fields = [
            "recording_id",
            "start_sec",
            "end_sec",
            "video_probability",
            "video_prediction",
            "reference_label",
            "eeg_onset_sec",
            "clinical_onset_sec",
            "annotation_alignment_status",
            *RESULT_METADATA_FIELDS,
        ]
        _write_csv(output / "probability_timeline.csv", augmented, timeline_fields)
        _write_csv(output / f"{source.source_id}_probability_timeline.csv", augmented, timeline_fields)
        np.save(output / "embeddings.npy", embeddings)
        plot_error = None
        try:
            _plot_timeline(rows, output / "timeline.png")
            _plot_timeline(rows, output / f"{source.source_id}_timeline.png")
        except Exception as exc:
            plot_error = str(exc)
        payload = {
            "status": "completed",
            "metadata": metadata,
            "metrics": metrics,
            "run": run_details,
            "plot_error": plot_error,
        }
        _write_json(output / "metrics.json", payload)
        _write_json(output / f"{source.source_id}_metrics.json", payload)
        all_probabilities = np.asarray([row["video_probability"] for row in rows], dtype=float)
        all_labels = np.asarray(
            [row["reference_label"] if row["reference_label"] in (0, 1) else -1 for row in rows],
            dtype=int,
        )
        return WorkflowResult(
            "video", "completed", rows, all_probabilities, all_labels, embeddings, metrics, metadata
        )
    except Exception as exc:
        metadata = {**metadata, "video_model_used": False}
        metrics = _na_metrics(f"Video inference failed: {exc}")
        payload = {
            "status": "blocked",
            "metadata": metadata,
            "metrics": metrics,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output / "metrics.json", payload)
        _write_json(output / "Sz2PG_metrics.json", payload)
        _write_csv(output / "predictions.csv", [], [
            "recording_id", "start_sec", "end_sec", "video_probability",
            "video_prediction", "reference_label", *RESULT_METADATA_FIELDS,
        ])
        return WorkflowResult(
            "video", "blocked", [], np.array([]), None, None, metrics, metadata, str(exc)
        )


def run_eeg_workflow(
    root: Path,
    config: Mapping[str, Any],
    source: ModalitySource,
    *,
    predictor: Callable[..., Any] = predict_eeg_csv,
) -> WorkflowResult:
    eeg_cfg = config.get("eeg_encoder", {})
    test_cfg = config.get("testing", {})
    video_label, _ = _checkpoint_label(root, config.get("video_encoder", {}).get("checkpoint"))
    eeg_label, eeg_checkpoint = _checkpoint_label(root, eeg_cfg.get("checkpoint"))
    metadata = result_metadata(
        test_mode="eeg_only",
        video_model_used=False,
        eeg_model_used=True,
        fusion_used=False,
        pairing_type="none",
        video_checkpoint=video_label,
        eeg_checkpoint=eeg_label,
    )
    output = root / str(test_cfg.get("output_directory", "results")) / "eeg_only"
    output.mkdir(parents=True, exist_ok=True)
    mapping, positive = normalized_label_mapping(config)
    _write_json(
        output / "label_mapping.json",
        {
            "metadata": metadata,
            "mapping": mapping,
            "positive_value": positive,
            "binary_policy": "label 1 = ictal; labels 2,3,4,5 = non-ictal",
        },
    )
    try:
        if eeg_checkpoint is None or not eeg_checkpoint.is_file():
            raise FileNotFoundError(
                "The notebook-generated pretrained EEG checkpoint is absent. Expected "
                "best_seizure_model.h5, seizure_detector_model.h5, or seizure_detector.weights.h5; "
                "random/new weights are prohibited."
            )
        data, prediction = predictor(
            csv_path=source.eeg_path,
            checkpoint_path=eeg_checkpoint,
            config=config,
            batch_size=int(eeg_cfg.get("inference_batch_size", 64)),
            progress_config=config,
        )
        probabilities = np.asarray(prediction["probability"], dtype=float)
        labels = np.asarray(data.binary_labels, dtype=int)
        metrics = _classification_metrics(labels, probabilities)
        metrics["false_positives_per_hour"] = "N/A"
        metrics["false_positives_per_hour_reason"] = "CSV rows have no monitoring duration."
        metrics["onset_mean_absolute_error_sec"] = "N/A"
        metrics["offset_mean_absolute_error_sec"] = "N/A"
        rows = [
            {
                "sample_id": sample_id,
                "original_label": int(original),
                "true_label": int(label),
                "eeg_probability": float(probability),
                "eeg_logit": float(logit),
                "eeg_prediction": int(probability >= 0.5),
            }
            for sample_id, original, label, probability, logit in zip(
                data.sample_ids,
                data.original_labels,
                labels,
                probabilities,
                prediction["logits"],
            )
        ]
        augmented = _augment_rows(rows, metadata)
        fields = [*rows[0].keys(), *RESULT_METADATA_FIELDS]
        _write_csv(output / "eeg_predictions.csv", augmented, fields)
        _write_csv(output / "predictions.csv", augmented, fields)
        plot_error = None
        try:
            _plot_eeg(labels, probabilities, output)
        except Exception as exc:
            plot_error = str(exc)
        payload = {
            "status": "completed",
            "metadata": metadata,
            "metrics": metrics,
            "input_shape": [len(rows), 178],
            "model_input_shape": [len(rows), 1024, 18],
            "embedding_shape": list(np.asarray(prediction["embedding"]).shape),
            "plot_error": plot_error,
        }
        _write_json(output / "eeg_metrics.json", payload)
        _write_json(output / "metrics.json", payload)
        return WorkflowResult(
            "eeg",
            "completed",
            rows,
            probabilities,
            labels,
            np.asarray(prediction["embedding"]),
            metrics,
            metadata,
        )
    except Exception as exc:
        metadata = {**metadata, "EEG_model_used": False}
        metrics = _na_metrics(f"EEG inference failed: {exc}")
        payload = {
            "status": "blocked",
            "metadata": metadata,
            "metrics": metrics,
            "label_mapping": mapping,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output / "eeg_metrics.json", payload)
        _write_json(output / "metrics.json", payload)
        _write_csv(output / "eeg_predictions.csv", [], [
            "sample_id", "true_label", "eeg_probability", "eeg_prediction", *RESULT_METADATA_FIELDS,
        ])
        return WorkflowResult("eeg", "blocked", [], np.array([]), None, None, metrics, metadata, str(exc))


def run_strict_workflow(
    root: Path,
    config: Mapping[str, Any],
    video_source: ModalitySource | None,
    eeg_source: ModalitySource | None,
) -> WorkflowResult:
    output = root / str(config.get("testing", {}).get("output_directory", "results")) / "strict_multimodal"
    output.mkdir(parents=True, exist_ok=True)
    video_label, _ = _checkpoint_label(root, config.get("video_encoder", {}).get("checkpoint"))
    eeg_label, _ = _checkpoint_label(root, config.get("eeg_encoder", {}).get("checkpoint"))
    metadata = result_metadata(
        test_mode="strict_multimodal",
        video_model_used=False,
        eeg_model_used=False,
        fusion_used=False,
        pairing_type="none",
        video_checkpoint=video_label,
        eeg_checkpoint=eeg_label,
    )
    if video_source is None or eeg_source is None:
        reasons = ["both a video source and an EEG source are required"]
    else:
        _, reasons = validate_verified_pair(video_source, eeg_source)
    metrics = _na_metrics("; ".join(reasons))
    payload = {
        "status": "skipped",
        "message": STRICT_REJECTION.strip(),
        "reasons": reasons,
        "metadata": metadata,
        "metrics": metrics,
    }
    _write_json(output / "metrics.json", payload)
    (output / "SKIPPED.txt").write_text(STRICT_REJECTION, encoding="utf-8")
    return WorkflowResult(
        "strict_multimodal", "skipped", [], np.array([]), None, None, metrics, metadata
    )


def run_simulated_workflow(
    root: Path,
    config: Mapping[str, Any],
    video: WorkflowResult,
    eeg: WorkflowResult,
    *,
    allow_simulated_pairing: bool,
    calculate_artificial_metrics: bool | None = None,
    reference_labels: str | None = None,
) -> WorkflowResult:
    if not allow_simulated_pairing:
        raise PermissionError(
            "Simulated pairing is disabled. Re-run with --allow-simulated-pairing to acknowledge "
            "that the data are unrelated and clinically invalid for fusion."
        )
    print(SIMULATED_WARNING)
    test_cfg = config.get("testing", {})
    simulated_cfg = config.get("simulated_multimodal", {})
    output = root / str(test_cfg.get("output_directory", "results")) / "simulated_multimodal"
    output.mkdir(parents=True, exist_ok=True)
    (output / "WARNING.txt").write_text(SIMULATED_WARNING, encoding="utf-8")
    metadata = result_metadata(
        test_mode="simulated_multimodal",
        video_model_used=video.status == "completed",
        eeg_model_used=eeg.status == "completed",
        fusion_used=video.status == "completed" and eeg.status == "completed",
        pairing_type="simulated",
        video_checkpoint=str(video.metadata.get("video_checkpoint", "not available")),
        eeg_checkpoint=str(eeg.metadata.get("EEG_checkpoint", "not available")),
    )
    if video.status != "completed" or eeg.status != "completed":
        reason = (
            f"Simulated fusion could not run because video status={video.status} and EEG status={eeg.status}. "
            "No substitute probabilities were generated."
        )
        metrics = _na_metrics(reason)
        payload = {"status": "blocked", "metadata": metadata, "metrics": metrics, "reason": reason}
        _write_json(output / "metrics.json", payload)
        _write_csv(output / "pairing_manifest.csv", [], [
            "sim_window_id", "video_index", "eeg_index", "pairing_type", *RESULT_METADATA_FIELDS,
        ])
        _write_csv(output / "predictions.csv", [], [
            "sim_window_id", "video_probability", "eeg_probability", "fusion_probability",
            "fusion_prediction", "pairing_type", *RESULT_METADATA_FIELDS,
        ])
        return WorkflowResult(
            "simulated_multimodal", "blocked", [], np.array([]), None, None, metrics, metadata, reason
        )
    count = min(len(video.probabilities), len(eeg.probabilities))
    video_probability = video.probabilities[:count]
    eeg_probability = eeg.probabilities[:count]
    fused = LateFusion(
        video_weight=float(config.get("fusion", {}).get("late_video_weight", 0.5)),
        eeg_weight=float(config.get("fusion", {}).get("late_eeg_weight", 0.5)),
    ).predict(video_probability=video_probability, eeg_probability=eeg_probability)
    rows = []
    manifests = []
    for index in range(count):
        window_id = f"sim_window_{index + 1:04d}"
        manifests.append(
            {
                "sim_window_id": window_id,
                "video_index": index,
                "eeg_index": index,
                "pairing_policy": "first N in original order; N=min(video windows, EEG rows)",
                "pairing_type": "simulated",
            }
        )
        rows.append(
            {
                "sim_window_id": window_id,
                "video_probability": float(video_probability[index]),
                "eeg_probability": float(eeg_probability[index]),
                "fusion_probability": float(fused[index]),
                "fusion_prediction": int(fused[index] >= 0.5),
                "pairing_type": "simulated",
                "validity": "simulated",
                "clinical_interpretation": "prohibited",
            }
        )
    augmented_manifest = _augment_rows(manifests, metadata)
    augmented_rows = _augment_rows(rows, metadata)
    _write_csv(output / "pairing_manifest.csv", augmented_manifest, [
        *manifests[0].keys(), *RESULT_METADATA_FIELDS,
    ])
    _write_csv(output / "predictions.csv", augmented_rows, [*rows[0].keys(), *RESULT_METADATA_FIELDS])
    calculate = (
        bool(simulated_cfg.get("calculate_artificial_metrics", False))
        if calculate_artificial_metrics is None
        else calculate_artificial_metrics
    )
    reference = reference_labels or str(simulated_cfg.get("reference_labels", "video"))
    metrics = _na_metrics("Pipeline-only simulated test: no shared clinical ground truth exists.")
    if calculate:
        if reference == "video":
            labels = video.labels[:count] if video.labels is not None else None
        elif reference == "eeg":
            labels = eeg.labels[:count] if eeg.labels is not None else None
        else:
            raise ValueError("Simulated artificial reference_labels must be video or eeg")
        if labels is None or np.any(np.asarray(labels) < 0) or len(np.unique(labels)) < 2:
            metrics = _na_metrics(
                f"SIMULATED TEST METRIC unavailable: {reference} labels are absent/incomplete."
            )
        else:
            metrics = _classification_metrics(np.asarray(labels), fused)
        metrics.update(validity="simulated", clinical_interpretation="prohibited", label="SIMULATED TEST METRIC")
    payload = {
        "status": "completed",
        "warning": SIMULATED_WARNING.strip(),
        "metadata": metadata,
        "pairing_policy": "first N in original order",
        "N": count,
        "artificial_metrics_enabled": calculate,
        "reference_labels": reference if calculate else None,
        "metrics": metrics,
        "feature_fusion": {
            "trained": False,
            "shape_contract": asdict(FeatureFusionSpec(384, 32)),
            "status": "dimensions verified only; no fusion head was trained",
        },
    }
    _write_json(output / "metrics.json", payload)
    return WorkflowResult(
        "simulated_multimodal", "completed", rows, fused, None, None, metrics, metadata
    )


def _metric(payload: Mapping[str, Any], key: str) -> Any:
    metrics = payload.get("metrics", payload)
    return metrics.get(key, "N/A") if isinstance(metrics, Mapping) else "N/A"


def write_summary(root: Path, config: Mapping[str, Any]) -> None:
    results_root = root / str(config.get("testing", {}).get("output_directory", "results"))
    rows = []
    definitions = (
        ("Sz2PG video-only", "video_only"),
        ("EEG CSV only", "eeg_only"),
        ("Strict multimodal", "strict_multimodal"),
        ("Simulated multimodal", "simulated_multimodal"),
    )
    for name, directory in definitions:
        path = results_root / directory / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
        pairing_value = metadata.get("pairing_type", "not run")
        pairing = {
            "none": "N/A" if directory in {"video_only", "eeg_only"} else "Not verified",
            "simulated": "Artificial",
            "verified": "Verified",
        }.get(pairing_value, pairing_value)
        rows.append(
            {
                "Test": name,
                "Video": "Yes" if metadata.get("video_model_used") else "No",
                "EEG": "Yes" if metadata.get("EEG_model_used") else "No",
                "Fusion": "Yes" if metadata.get("fusion_used") else "No",
                "Pairing": pairing,
                "Status": payload.get("status", "not run"),
                "Sensitivity": _metric(payload, "sensitivity"),
                "Specificity": _metric(payload, "specificity"),
                "AUROC": _metric(payload, "auroc"),
                "F1": _metric(payload, "f1"),
                "FP/h": _metric(payload, "false_positives_per_hour"),
                "Onset MAE": _metric(payload, "onset_mean_absolute_error_sec"),
                "Offset MAE": _metric(payload, "offset_mean_absolute_error_sec"),
                "Cohen kappa": _metric(payload, "cohen_kappa"),
            }
        )
    fields = list(rows[0])
    _write_csv(results_root / "summary.csv", rows, fields)
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows]
    (results_root / "summary.md").write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")


def run_test_mode(
    *,
    root: str | Path,
    config: Mapping[str, Any],
    mode: str,
    allow_simulated_pairing: bool = False,
    calculate_artificial_metrics: bool | None = None,
    reference_labels: str | None = None,
) -> dict[str, WorkflowResult]:
    project_root = Path(root).resolve()
    if config.get("training", {}).get("enabled") is not False:
        raise RuntimeError("Test modes require training.enabled=false")
    if mode == "simulated_multimodal" and not allow_simulated_pairing:
        raise PermissionError(
            "Simulated pairing is disabled. Re-run with --allow-simulated-pairing to acknowledge "
            "that the data are unrelated and clinically invalid for fusion."
        )
    sources = discover_sources(project_root, config)
    video_source = next((source for source in sources if source.has_video), None)
    eeg_source = next((source for source in sources if source.has_eeg), None)
    results_root = project_root / str(config.get("testing", {}).get("output_directory", "results"))
    _write_json(results_root / "data_inspection.json", inspection_bundle(project_root, config))
    results: dict[str, WorkflowResult] = {}
    if mode in {"unimodal", "video", "simulated_multimodal"} and video_source:
        results["video"] = run_video_workflow(project_root, config, video_source)
    if mode in {"unimodal", "eeg", "simulated_multimodal"} and eeg_source:
        mapping, positive = normalized_label_mapping(config)
        print(f"EEG label mapping: {mapping}; positive/seizure value={positive}")
        results["eeg"] = run_eeg_workflow(project_root, config, eeg_source)
    if mode == "strict_multimodal":
        results["strict_multimodal"] = run_strict_workflow(
            project_root, config, video_source, eeg_source
        )
    if mode == "simulated_multimodal":
        if "video" not in results:
            results["video"] = WorkflowResult(
                "video", "missing", [], np.array([]), None, None, _na_metrics("Video missing"), {}, "Video missing"
            )
        if "eeg" not in results:
            results["eeg"] = WorkflowResult(
                "eeg", "missing", [], np.array([]), None, None, _na_metrics("EEG missing"), {}, "EEG missing"
            )
        results["simulated_multimodal"] = run_simulated_workflow(
            project_root,
            config,
            results["video"],
            results["eeg"],
            allow_simulated_pairing=allow_simulated_pairing,
            calculate_artificial_metrics=calculate_artificial_metrics,
            reference_labels=reference_labels,
        )
    if mode not in {"unimodal", "video", "eeg", "strict_multimodal", "simulated_multimodal"}:
        raise ValueError(f"Unsupported test mode: {mode}")
    write_summary(project_root, config)
    return results
