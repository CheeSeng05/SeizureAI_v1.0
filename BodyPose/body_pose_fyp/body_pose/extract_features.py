from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from .constants import ANGLE_TRIPLETS, POSE_LANDMARK_NAMES
from .utils import ensure_dir, load_config

LOGGER = logging.getLogger(__name__)


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosine = np.divide(
        np.sum(ba * bc, axis=1),
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=denom > 1e-8,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def interpolate_and_smooth(series: pd.Series, limit: int, window: int) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = result.interpolate(method="linear", limit=limit, limit_direction="both")
    result = result.ffill().bfill().fillna(0.0)
    if window >= 3 and len(result) >= window:
        if window % 2 == 0:
            window += 1
        window = min(window, len(result) if len(result) % 2 == 1 else len(result) - 1)
        if window >= 3:
            result = pd.Series(
                savgol_filter(result.to_numpy(float), window_length=window, polyorder=min(2, window - 1)),
                index=result.index,
            )
    return result


def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    left_hip = out[["left_hip_x", "left_hip_y"]].to_numpy(float)
    right_hip = out[["right_hip_x", "right_hip_y"]].to_numpy(float)
    left_shoulder = out[["left_shoulder_x", "left_shoulder_y"]].to_numpy(float)
    right_shoulder = out[["right_shoulder_x", "right_shoulder_y"]].to_numpy(float)

    center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    torso_scale = np.linalg.norm(shoulder_center - center, axis=1)
    torso_scale[~np.isfinite(torso_scale) | (torso_scale < 1e-6)] = 1.0

    for landmark in POSE_LANDMARK_NAMES:
        xy = out[[f"{landmark}_x", f"{landmark}_y"]].to_numpy(float)
        xy = (xy - center) / torso_scale[:, None]
        out[f"{landmark}_x"] = xy[:, 0]
        out[f"{landmark}_y"] = xy[:, 1]
        out[f"{landmark}_z"] = out[f"{landmark}_z"].to_numpy(float) / torso_scale
    return out


def create_features(input_csv: Path, output_csv: Path, feature_cfg: dict) -> None:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"No frames in {input_csv}")

    limit = int(feature_cfg.get("interpolate_limit", 15))
    smooth_window = int(feature_cfg.get("smoothing_window", 5))

    coordinate_columns = [
        f"{landmark}_{axis}"
        for landmark in POSE_LANDMARK_NAMES
        for axis in ("x", "y", "z")
    ]
    visibility_columns = [f"{landmark}_visibility" for landmark in POSE_LANDMARK_NAMES]

    visibility_threshold = float(feature_cfg.get("visibility_threshold", 0.30))
    for landmark in POSE_LANDMARK_NAMES:
        low_conf = df[f"{landmark}_visibility"] < visibility_threshold
        for axis in ("x", "y", "z"):
            df.loc[low_conf, f"{landmark}_{axis}"] = np.nan

    for column in coordinate_columns:
        df[column] = interpolate_and_smooth(df[column], limit, smooth_window)

    if bool(feature_cfg.get("normalize_body", True)):
        df = normalize_coordinates(df)

    dt = np.gradient(df["timestamp_sec"].to_numpy(float))
    dt[~np.isfinite(dt) | (dt <= 0)] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1 / 25

    metadata = df[[
        "processed_frame", "source_frame", "timestamp_sec", "source_fps",
        "pose_detected", "mean_visibility",
    ]].copy()
    feature_data: dict[str, np.ndarray] = {}

    if bool(feature_cfg.get("include_coordinates", True)):
        for column in coordinate_columns:
            feature_data[column] = df[column].to_numpy(float)
    if bool(feature_cfg.get("include_visibility", True)):
        for column in visibility_columns:
            feature_data[column] = df[column].to_numpy(float)

    for landmark in POSE_LANDMARK_NAMES:
        xyz = df[[f"{landmark}_x", f"{landmark}_y", f"{landmark}_z"]].to_numpy(float)
        velocity = np.gradient(xyz, axis=0) / dt[:, None]
        acceleration = np.gradient(velocity, axis=0) / dt[:, None]

        if bool(feature_cfg.get("include_velocity", True)):
            for i, axis in enumerate(("x", "y", "z")):
                feature_data[f"{landmark}_v{axis}"] = velocity[:, i]
        if bool(feature_cfg.get("include_acceleration", True)):
            for i, axis in enumerate(("x", "y", "z")):
                feature_data[f"{landmark}_a{axis}"] = acceleration[:, i]
        if bool(feature_cfg.get("include_speed", True)):
            feature_data[f"{landmark}_speed"] = np.linalg.norm(velocity, axis=1)
            feature_data[f"{landmark}_acceleration_magnitude"] = np.linalg.norm(
                acceleration, axis=1
            )
        if bool(feature_cfg.get("include_trajectory", True)):
            step_distance = np.linalg.norm(
                np.diff(xyz, axis=0, prepend=xyz[[0]]), axis=1
            )
            trajectory_window = max(1, int(feature_cfg.get("trajectory_window_frames", 15)))
            feature_data[f"{landmark}_trajectory_step"] = step_distance
            feature_data[f"{landmark}_trajectory_path"] = (
                pd.Series(step_distance)
                .rolling(trajectory_window, min_periods=1)
                .sum()
                .to_numpy()
            )

    if bool(feature_cfg.get("include_joint_angles", True)):
        for angle_name, (a_name, b_name, c_name) in ANGLE_TRIPLETS.items():
            a = df[[f"{a_name}_x", f"{a_name}_y", f"{a_name}_z"]].to_numpy(float)
            b = df[[f"{b_name}_x", f"{b_name}_y", f"{b_name}_z"]].to_numpy(float)
            c = df[[f"{c_name}_x", f"{c_name}_y", f"{c_name}_z"]].to_numpy(float)
            feature_data[angle_name] = angle_degrees(a, b, c)

    output = pd.concat(
        [metadata, pd.DataFrame(feature_data, index=df.index)], axis=1
    )

    numeric = output.select_dtypes(include=[np.number]).columns
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan)
    output[numeric] = output[numeric].interpolate(limit_direction="both").fillna(0.0)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    LOGGER.info("Saved %s with shape %s", output_csv, output.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create kinematic features from pose landmark CSV files.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    input_dir = Path(args.input_dir or cfg["paths"]["keypoints"])
    output_dir = ensure_dir(args.output_dir or cfg["paths"]["features"])
    csv_files = sorted(p for p in input_dir.glob("*.csv") if not p.name.endswith("_overlay.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No keypoint CSV files found in {input_dir}")

    for path in tqdm(csv_files, desc="Feature files"):
        create_features(path, output_dir / path.name, cfg["features"])


if __name__ == "__main__":
    main()
