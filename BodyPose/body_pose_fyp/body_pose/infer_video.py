from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .build_dataset import deterministic_window_id
from .models.lstm import PoseLSTMClassifier
from .utils import get_device, load_config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained body-pose stream on one unseen video.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/inference_predictions.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = Path(args.video)
    video_id = video.stem
    keypoint_path = Path(cfg["paths"]["keypoints"]) / f"{video_id}.csv"
    feature_path = Path(cfg["paths"]["features"]) / f"{video_id}.csv"

    subprocess.run([
        sys.executable, "-m", "body_pose.extract_pose",
        "--config", args.config, "--input", str(video),
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "body_pose.extract_features",
        "--config", args.config,
    ], check=True)

    df = pd.read_csv(feature_path)
    checkpoint = torch.load(cfg["paths"]["checkpoint"], map_location="cpu", weights_only=False)
    label_names = [str(name) for name in checkpoint.get("label_names", [])]
    if not label_names:
        raise ValueError("Checkpoint has no label_names; retrain with the current pipeline.")
    feature_names = checkpoint["feature_names"]
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Inference features missing: {sorted(missing)[:10]}")

    if len(df) < 2:
        raise ValueError("At least two pose-feature frames are required for inference.")
    source_times = df["timestamp_sec"].to_numpy(float)
    frame_dt = float(np.median(np.diff(source_times)))
    window_seconds = float(cfg["dataset"]["window_seconds"])
    stride_seconds = float(cfg["dataset"]["stride_seconds"])
    target_fps = float(cfg["dataset"].get("target_fps", 10.0))
    samples_per_window = max(2, round(window_seconds * target_fps))
    values = df[feature_names].to_numpy(np.float32)

    windows, rows = [], []
    available_end = float(source_times[-1] + frame_dt)
    first_start = float(source_times[0])
    window_count = int(
        np.floor((available_end - first_start - window_seconds) / stride_seconds)
    ) + 1
    visibility = df["mean_visibility"].to_numpy(float)
    for window_offset in range(max(window_count, 0)):
        start_sec = round(first_start + window_offset * stride_seconds, 9)
        end_sec = round(start_sec + window_seconds, 9)
        sample_times = start_sec + np.arange(samples_per_window) / target_fps
        windows.append(
            np.column_stack(
                [
                    np.interp(sample_times, source_times, values[:, feature_index])
                    for feature_index in range(values.shape[1])
                ]
            ).astype(np.float32)
        )
        rows.append({
            "window_id": deterministic_window_id(video_id, start_sec, end_sec),
            "media_id": video_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "mean_pose_visibility": float(
                np.interp(sample_times, source_times, visibility).mean()
            ),
        })
    if not windows:
        raise ValueError("Video is shorter than the configured temporal window.")

    x = np.stack(windows)
    scaler = joblib.load(cfg["paths"]["scaler"])
    x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    device = get_device()
    model = PoseLSTMClassifier(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probabilities = torch.softmax(
        model(torch.as_tensor(x, dtype=torch.float32, device=device)), dim=1
    ).cpu().numpy()
    output = pd.DataFrame(rows)
    output["predicted_label"] = [label_names[int(i)] for i in probabilities.argmax(axis=1)]
    for class_id, class_name in enumerate(label_names):
        output[f"probability_{class_name}"] = probabilities[:, class_id]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
