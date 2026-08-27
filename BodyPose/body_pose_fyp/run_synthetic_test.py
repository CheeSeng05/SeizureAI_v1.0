from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def generate_pose_like_features(output_dir: Path, patient_count: int = 9) -> pd.DataFrame:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    annotation_rows = []
    fps = 10.0
    frame_count = 120
    time = np.arange(frame_count) / fps

    for patient_index in range(patient_count):
        patient_id = f"SYN{patient_index + 1:03d}"
        media_id = f"synthetic_{patient_id}"
        rng = np.random.default_rng(1000 + patient_index)
        ictal = (time >= 6.0).astype(float)
        phase = time * (0.7 + patient_index * 0.015)
        jitter = ictal * 0.10 * np.sin(2 * np.pi * 3.0 * time)
        noise = rng.normal(0.0, 0.006, size=frame_count)
        nose_x = 0.50 + 0.03 * np.sin(phase) + jitter + noise
        nose_y = 0.25 + 0.02 * np.cos(phase) + 0.5 * jitter + noise
        left_x = 0.42 + 0.02 * np.sin(phase + 0.3) + jitter
        right_x = 0.58 + 0.02 * np.sin(phase - 0.3) - jitter
        speed = np.abs(np.gradient(nose_x, 1 / fps))
        acceleration = np.abs(np.gradient(speed, 1 / fps))

        frame = pd.DataFrame({
            "processed_frame": np.arange(frame_count),
            "source_frame": np.arange(frame_count),
            "timestamp_sec": time,
            "source_fps": fps,
            "pose_detected": 1,
            "mean_visibility": np.clip(0.94 + rng.normal(0, 0.01, frame_count), 0, 1),
            "nose_x": nose_x,
            "nose_y": nose_y,
            "nose_z": 0.01 * np.sin(phase),
            "nose_visibility": 0.95,
            "left_shoulder_x": left_x,
            "left_shoulder_y": 0.40 + 0.01 * np.cos(phase),
            "right_shoulder_x": right_x,
            "right_shoulder_y": 0.40 - 0.01 * np.cos(phase),
            "nose_speed": speed,
            "nose_acceleration_magnitude": acceleration,
            "left_elbow_angle": 120 + 8 * np.sin(phase) + 30 * jitter,
            "right_elbow_angle": 120 - 8 * np.sin(phase) - 30 * jitter,
        })
        frame.to_csv(feature_dir / f"{media_id}.csv", index=False)
        annotation_rows.extend([
            {
                "Patient": patient_id,
                "VideoName": media_id,
                "BeginFrame": 0,
                "FinishFrame": 60,
                "ClinicalState": "Normal",
            },
            {
                "Patient": patient_id,
                "VideoName": media_id,
                "BeginFrame": 60,
                "FinishFrame": 120,
                "ClinicalState": "Seizure",
            },
        ])
    return pd.DataFrame(annotation_rows)


def synthetic_config(output_dir: Path) -> dict:
    absolute = output_dir.resolve()
    return {
        "paths": {
            "raw_annotations": str(absolute / "annotations" / "synthetic_raw.csv"),
            "canonical_annotations": str(absolute / "annotations" / "canonical.csv"),
            "features": str(absolute / "features"),
            "dataset": str(absolute / "datasets" / "windows.npz"),
            "window_manifest": str(absolute / "datasets" / "window_manifest.csv"),
            "checkpoint": str(absolute / "checkpoints" / "best_lstm.pt"),
            "scaler": str(absolute / "checkpoints" / "standard_scaler.joblib"),
            "results": str(absolute / "results"),
            "body_embeddings_csv": str(absolute / "results" / "body_embeddings.csv"),
            "body_embeddings_npz": str(absolute / "results" / "body_embeddings.npz"),
        },
        "annotation_adapter": {
            "type": "generic_tabular",
            "dataset": "synthetic_non_clinical",
            "source": str(absolute / "annotations" / "synthetic_raw.csv"),
            "column_map": {
                "media_id": "VideoName",
                "patient_id": "Patient",
                "start_sec": "BeginFrame",
                "end_sec": "FinishFrame",
                "label": "ClinicalState",
            },
            "label_map": {"Normal": "normal", "Seizure": "ictal"},
            "time_unit": "frames",
            "fps": 10,
        },
        "labels": {"active": ["normal", "ictal"]},
        "dataset": {
            "window_seconds": 2.0,
            "stride_seconds": 1.0,
            "target_fps": 10.0,
            "minimum_label_coverage": 0.8,
            "split_seed": 42,
            "train_fraction": 0.67,
            "validation_fraction": 0.11,
            "test_fraction": 0.22,
        },
        "training": {
            "batch_size": 16,
            "hidden_size": 8,
            "num_layers": 1,
            "bidirectional": True,
            "dropout": 0.1,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "max_epochs": 3,
            "patience": 3,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "seed": 42,
        },
    }


def run_module(project_root: Path, module: str, config_path: Path, *args: str) -> None:
    print(f"\n--- {module} ---", flush=True)
    interpreter = [sys.executable]
    if sys.flags.no_site:
        # Preserve an isolated verification runtime when this script itself uses -S.
        interpreter.append("-S")
    subprocess.run(
        [*interpreter, "-m", module, "--config", str(config_path), *args],
        cwd=project_root,
        check=True,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "synthetic_test_output"
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    raw_annotations = generate_pose_like_features(output_dir)
    raw_annotations.to_csv(output_dir / "annotations" / "synthetic_raw.csv", index=False)
    config = synthetic_config(output_dir)
    config_path = output_dir / "synthetic_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(
        "SYNTHETIC SOFTWARE TEST ONLY - outputs have no clinical or diagnostic meaning.",
        flush=True,
    )
    run_module(project_root, "body_pose.prepare_annotations", config_path)
    run_module(project_root, "body_pose.build_dataset", config_path)
    run_module(project_root, "body_pose.train_lstm", config_path)
    run_module(project_root, "body_pose.evaluate", config_path, "--split", "test")
    run_module(project_root, "body_pose.export_embeddings", config_path, "--split", "all")

    manifest = pd.read_csv(output_dir / "datasets" / "window_manifest.csv")
    leakage = manifest.groupby("patient_id")["split"].nunique()
    if (leakage > 1).any():
        raise AssertionError("Synthetic test detected patient leakage.")
    expected = [
        output_dir / "checkpoints" / "best_lstm.pt",
        output_dir / "results" / "test_metrics.json",
        output_dir / "results" / "body_embeddings.csv",
        output_dir / "results" / "body_embeddings.npz",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise AssertionError(f"Synthetic test outputs are missing: {missing}")
    print(
        f"\nPASS: {len(manifest)} windows, {manifest['patient_id'].nunique()} patients, "
        f"splits={sorted(manifest['split'].unique())}."
    )
    print("These synthetic results test software mechanics only; they have no clinical meaning.")


if __name__ == "__main__":
    main()
