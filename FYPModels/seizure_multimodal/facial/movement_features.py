"""Label-free movement features from the existing RGB/flow/geometry manifest."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SIGNALS = ("right_eye_aperture", "left_eye_aperture", "mouth_aperture", "eye_asymmetry")
FEATURE_NAMES = tuple(
    f"{signal}_{stat}" for signal in SIGNALS
    for stat in ("mean", "std", "speed_mean", "speed_p95")
) + ("geometry_speed_mean", "geometry_speed_p95", "flow_mean", "flow_p95", "flow_temporal_std")
DEFAULT_QUALITY = {"minimum_face_region": 0.8, "minimum_valid_geometry": 0.8}


class UnusableWindow(ValueError):
    """A structurally valid window with insufficient trustworthy observations."""


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject ambiguous identities and patient leakage without reading labels."""
    patients: dict[str, str] = {}
    windows: set[str] = set()
    for row in rows:
        for name in ("window_id", "patient_id", "recording_id", "split"):
            if not str(row.get(name, "")).strip():
                raise ValueError(f"Manifest requires nonempty {name}")
        patient, split = str(row["patient_id"]), str(row["split"])
        if split not in {"train", "validation", "test", "unassigned"}:
            raise ValueError(f"Unknown split: {split}")
        if patient in patients and patients[patient] != split:
            raise ValueError(f"Patient {patient} appears in multiple splits")
        patients[patient] = split
        window = str(row["window_id"])
        if window in windows:
            raise ValueError(f"Duplicate window_id: {window}")
        windows.add(window)
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f"Invalid timestamps for {window}")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest contains no windows")
    validate_manifest(rows)
    return rows


def validate_quality(quality: Mapping[str, float]) -> None:
    if set(quality) != set(DEFAULT_QUALITY):
        raise ValueError("Unknown or missing quality settings")
    if any(not np.isfinite(value) or not 0 < value <= 1 for value in quality.values()):
        raise ValueError("Quality fractions must be finite and in (0, 1]")


def extract_movement_features(
    face: np.ndarray, geometry: np.ndarray, sampled_fps: float,
    *, minimum_valid_geometry: float = 0.8,
) -> np.ndarray:
    """Summarize motion; differences never cross missing landmark observations.

    Geometry follows face_preprocess.FACE_LANDMARK_PAIRS (10 x/y vectors).
    Apertures divide vertical pair length by the corresponding region width.
    All-zero geometry denotes a failed detection in existing exports.
    """
    face, geometry = np.asarray(face), np.asarray(geometry, dtype=float)
    if face.ndim != 4 or face.shape[1] != 4 or min(face.shape[0], *face.shape[2:]) < 1:
        raise ValueError("face must have nonempty shape [T,4,H,W]")
    if geometry.shape != (face.shape[0], 20):
        raise ValueError("geometry must have shape [T,20] aligned with face")
    if not np.isfinite(sampled_fps) or sampled_fps <= 0:
        raise ValueError("sampled_fps must be positive and finite")
    if not np.isfinite(minimum_valid_geometry) or not 0 < minimum_valid_geometry <= 1:
        raise ValueError("minimum_valid_geometry must be in (0, 1]")

    lengths = np.linalg.norm(geometry.reshape(-1, 10, 2), axis=2)
    valid = np.isfinite(geometry).all(axis=1) & (lengths[:, [0, 1, 4]] > 1e-6).all(axis=1)
    if valid.mean() < minimum_valid_geometry:
        raise UnusableWindow("insufficient_valid_geometry")
    adjacent = valid[:-1] & valid[1:]
    if not adjacent.any():
        raise UnusableWindow("no_adjacent_valid_geometry")

    apertures = np.full((len(geometry), 3), np.nan)
    apertures[valid] = lengths[valid][:, [2, 3, 5]] / lengths[valid][:, [0, 1, 4]]
    signals = np.column_stack((apertures, np.abs(apertures[:, 0] - apertures[:, 1])))
    speeds = np.abs(np.diff(signals, axis=0)[adjacent]) * sampled_fps
    values: list[float] = []
    for index in range(signals.shape[1]):
        observed = signals[valid, index]
        values.extend((observed.mean(), observed.std(), speeds[:, index].mean(),
                       np.quantile(speeds[:, index], 0.95)))
    geometry_speed = np.linalg.norm(np.diff(geometry, axis=0)[adjacent], axis=1) * sampled_fps
    # Flow at t depends on t-1. Exclude the first frame and invalid transitions.
    flow = np.asarray(face[1:, 3][adjacent], dtype=float)
    if not np.isfinite(flow).all():
        raise UnusableWindow("nonfinite_flow")
    values.extend((geometry_speed.mean(), np.quantile(geometry_speed, 0.95),
                   flow.mean(), np.quantile(flow, 0.95), flow.mean(axis=(1, 2)).std()))
    features = np.asarray(values, dtype=float)
    if not np.isfinite(features).all():
        raise UnusableWindow("nonfinite_features")
    return features


def load_features(
    manifest_path: str | Path, rows: Sequence[Mapping[str, Any]], *,
    quality: Mapping[str, float] = DEFAULT_QUALITY,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Return every row's quality status, and features for scored rows in order.

    Missing files and malformed arrays are errors, rather than silent exclusions.
    Neither labels nor class-folder names enter the model's features.
    """
    validate_quality(quality)
    root = Path(manifest_path).resolve().parent
    records, features = [], []
    for row in rows:
        record = {name: str(row[name]) for name in ("window_id", "patient_id", "recording_id", "split")}
        record.update(start_sec=float(row["start_sec"]), end_sec=float(row["end_sec"]))
        record.update(status="scored", exclusion_reason="", anomaly_score=None, candidate_window=None)
        try:
            fraction = float(row.get("face_region_fraction") or row.get("face_detection_fraction") or "nan")
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError(f"Invalid face region fraction for {row['window_id']}")
            if fraction < quality["minimum_face_region"]:
                raise UnusableWindow("insufficient_face_region")
            if not row.get("face_geometry_path"):
                raise UnusableWindow("missing_geometry_path")
            face = np.load(root / row["face_path"], mmap_mode="r", allow_pickle=False)
            geometry = np.load(root / row["face_geometry_path"], allow_pickle=False)
            features.append(extract_movement_features(
                face, geometry, float(row["sampled_fps"]),
                minimum_valid_geometry=quality["minimum_valid_geometry"],
            ))
        except UnusableWindow as error:
            record.update(status="excluded", exclusion_reason=str(error))
        records.append(record)
    return records, np.asarray(features, dtype=float).reshape(-1, len(FEATURE_NAMES))
