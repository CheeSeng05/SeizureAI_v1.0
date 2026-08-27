from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .evaluation.agreement import cohen_kappa
from .evaluation.classification import binary_classification_metrics
from .evaluation.events import (
    TemporalEvent,
    TemporalWindow,
    event_level_sensitivity,
    false_positives_per_hour,
    probabilities_to_events,
)
from .evaluation.localization import localization_metrics
from .evaluation.report import write_report_bundle
from .fusion.feature_fusion import FeatureFusionSpec, build_feature_fusion_model
from .fusion.late_fusion import LateFusion
from .reproducibility import build_run_record, save_run_record
from .synchronization.manifest import validate_manifest, write_manifest
from .synchronization.windows import make_window_id
from .types import WindowRecord


BANNER = "SYNTHETIC SOFTWARE TEST — NO CLINICAL MEANING"


def run_synthetic_smoke_test(output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0], dtype=int)
    video_probability = np.array([0.10, 0.18, 0.25, 0.58, 0.74, 0.69, 0.30, 0.22, 0.35, 0.61, 0.72, 0.29])
    eeg_probability = np.array([0.08, 0.15, 0.20, 0.78, 0.83, 0.76, 0.24, 0.18, 0.27, 0.80, 0.84, 0.21])
    video_embedding = rng.normal(size=(labels.size, 384)).astype(np.float32)
    eeg_embedding = rng.normal(size=(labels.size, 32)).astype(np.float32)
    spec = FeatureFusionSpec(384, 32, projection_dim=128, hidden_dim=128)
    spec.validate_inputs(video_embedding, eeg_embedding)

    manifest: list[WindowRecord] = []
    event_windows: list[TemporalWindow] = []
    for index in range(labels.size):
        start, end = index * 5.0, (index + 1) * 5.0
        manifest.append(
            WindowRecord(
                window_id=make_window_id("SYN_REC001", start, end),
                dataset="SYNTHETIC_SOFTWARE_ONLY",
                patient_id="SYN_P001",
                recording_id="SYN_REC001",
                start_sec=start,
                end_sec=end,
                label="ictal" if labels[index] else "normal",
                split="test",
                has_video=True,
                has_eeg=True,
                video_fps=30.0,
                eeg_sampling_rate=256.0,
            )
        )
        event_windows.append(TemporalWindow("SYN_P001", "SYN_REC001", start, end))
    validate_manifest(manifest, require_paired=True)
    write_manifest(manifest, output / "synthetic_manifest.csv")

    fused_probability = LateFusion().predict(
        video_probability=video_probability, eeg_probability=eeg_probability
    )
    classification = binary_classification_metrics(labels, fused_probability, threshold=0.5)
    predictions = (fused_probability >= 0.5).astype(int)
    events = probabilities_to_events(
        event_windows,
        fused_probability,
        threshold=0.5,
        minimum_event_duration_sec=5.0,
        gap_tolerance_sec=0.0,
        refractory_period_sec=0.0,
    )
    references = [
        TemporalEvent("SYN_P001", "SYN_REC001", 15.0, 30.0),
        TemporalEvent("SYN_P001", "SYN_REC001", 45.0, 55.0),
    ]
    monitoring = [TemporalEvent("SYN_P001", "SYN_REC001", 0.0, 60.0)]
    event_metrics = {
        "event_sensitivity": event_level_sensitivity(events, references),
        **false_positives_per_hour(events, references, monitoring),
        **localization_metrics(
            events,
            references,
            onset_reference="clinical_onset",
            offset_annotations_available=True,
        ),
    }
    feature_runtime = "shape contract verified; PyTorch runtime unavailable"
    feature_shape: list[int] | None = None
    try:
        import torch

        torch.manual_seed(42)
        model = build_feature_fusion_model(spec)
        model.eval()
        with torch.no_grad():
            feature_logits = model(torch.from_numpy(video_embedding), torch.from_numpy(eeg_embedding))
        feature_shape = list(feature_logits.shape)
        feature_runtime = "untrained PyTorch architecture forward pass completed"
    except (ImportError, RuntimeError):
        pass

    payload: dict[str, object] = {
        "warning": BANNER,
        "seed": 42,
        "windows": labels.size,
        "aligned_patient_id": "SYN_P001",
        "aligned_recording_id": "SYN_REC001",
        "video_embedding_shape": list(video_embedding.shape),
        "eeg_embedding_shape": list(eeg_embedding.shape),
        "feature_fusion_concatenated_dim": spec.concatenated_dim,
        "feature_fusion_output_shape": feature_shape,
        "feature_fusion_runtime": feature_runtime,
        "late_fusion_policy": "equal weights, fixed and unvalidated",
        "classification": classification,
        "cohen_kappa": cohen_kappa(labels, predictions),
        "events": [asdict(event) for event in events],
        "event_metrics": event_metrics,
    }
    (output / "synthetic_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_run_record(
        build_run_record(
            {
                "experiment_id": "synthetic_smoke_seed_42",
                "dataset": "SYNTHETIC_SOFTWARE_ONLY",
                "dataset_version": "1",
                "patient_split": {"test": ["SYN_P001"]},
                "random_seed": 42,
                "video_checkpoint": "not loaded (synthetic embedding)",
                "eeg_checkpoint": "not loaded (synthetic embedding)",
                "fusion_configuration": {
                    "late_weights": {"video": 0.5, "eeg": 0.5},
                    "weights_validated": False,
                    "feature_projection_dim": 128,
                },
                "window_duration_sec": 5.0,
                "window_stride_sec": 5.0,
                "prediction_threshold": 0.5,
                "synchronization_configuration": {
                    "same_patient": True,
                    "same_recording": True,
                    "video_offset_sec": 0.0,
                    "eeg_offset_sec": 0.0,
                    "video_drift_ppm": 0.0,
                    "eeg_drift_ppm": 0.0,
                },
            }
        ),
        output / "run_metadata.json",
    )
    write_report_bundle(
        output_directory=output,
        results={},
        metadata={"warning": BANNER, "software_smoke_test_completed": True},
        paired_real_veeg_verified=False,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=BANNER)
    parser.add_argument("--output", default="results/synthetic_smoke")
    args = parser.parse_args()
    payload = run_synthetic_smoke_test(args.output)
    print(BANNER)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
