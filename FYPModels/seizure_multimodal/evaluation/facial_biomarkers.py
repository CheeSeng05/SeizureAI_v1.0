"""Transparent motor and facial-expression biomarkers from preprocessed windows.

These quantities are measurement proxies, not diagnoses or emotion labels.  They
must be validated against independent clinician annotations before clinical use.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def extract_facial_biomarkers(
    *,
    face_window: np.ndarray,
    sampled_fps: float,
    pose_window: np.ndarray | None = None,
    face_geometry_window: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Calculate auditable per-window motion and facial-expression proxies.

    ``face_window`` is ``[T,4,H,W]`` (RGB plus normalized optical-flow
    magnitude), pose is ``[T,33,4]``, and geometry is ``[T,20]``.  Pose points
    with visibility below 0.2 are excluded.
    """
    face = np.asarray(face_window, dtype=float)
    if face.ndim != 4 or face.shape[1] != 4 or face.shape[0] == 0:
        raise ValueError("face_window must have non-empty shape [T,4,H,W]")
    if not np.isfinite(sampled_fps) or sampled_fps <= 0:
        raise ValueError("sampled_fps must be positive")

    flow = face[:, 3]
    result: dict[str, float | None] = {
        "optical_flow_mean": _finite_mean(flow),
        "optical_flow_p95": float(np.nanpercentile(flow, 95)),
        "head_speed_mean_per_sec": None,
        "wrist_speed_mean_per_sec": None,
        "left_right_wrist_speed_asymmetry": None,
        "immobility_fraction": None,
        "facial_geometry_change_per_sec": None,
        "eye_aperture_proxy": None,
        "mouth_opening_proxy": None,
    }

    if pose_window is not None:
        pose = np.asarray(pose_window, dtype=float)
        if pose.ndim != 3 or pose.shape[0] != face.shape[0] or pose.shape[1:] != (33, 4):
            raise ValueError("pose_window must have shape [T,33,4] aligned to face_window")
        visible = pose[:, :, 3] >= 0.2

        def speeds(indices: list[int]) -> np.ndarray:
            points = np.nanmean(pose[:, indices, :3], axis=1)
            valid = visible[:, indices].any(axis=1)
            values = np.linalg.norm(np.diff(points, axis=0), axis=1) * sampled_fps
            values[~(valid[1:] & valid[:-1])] = np.nan
            return values

        head_speed = speeds(list(range(11)))
        left_wrist_speed = speeds([15])
        right_wrist_speed = speeds([16])
        wrist_speed = np.nanmean(np.vstack((left_wrist_speed, right_wrist_speed)), axis=0)
        result["head_speed_mean_per_sec"] = _finite_mean(head_speed)
        result["wrist_speed_mean_per_sec"] = _finite_mean(wrist_speed)
        left = _finite_mean(left_wrist_speed)
        right = _finite_mean(right_wrist_speed)
        if left is not None and right is not None:
            result["left_right_wrist_speed_asymmetry"] = float(
                abs(left - right) / (left + right + 1e-8)
            )
        combined = np.nanmean(np.vstack((head_speed, wrist_speed)), axis=0)
        valid_combined = combined[np.isfinite(combined)]
        if valid_combined.size:
            # Relative, normalized-pose speed threshold; configurable studies
            # should pre-register a replacement threshold.
            result["immobility_fraction"] = float(np.mean(valid_combined < 0.02))

    if face_geometry_window is not None:
        geometry = np.asarray(face_geometry_window, dtype=float)
        if geometry.ndim != 2 or geometry.shape != (face.shape[0], 20):
            raise ValueError("face_geometry_window must have shape [T,20] aligned to face_window")
        changes = np.linalg.norm(np.diff(geometry, axis=0), axis=1) * sampled_fps
        result["facial_geometry_change_per_sec"] = _finite_mean(changes)
        pairs = geometry.reshape(geometry.shape[0], 10, 2)
        # Pair definitions are documented in face_preprocess.FACE_LANDMARK_PAIRS.
        result["eye_aperture_proxy"] = _finite_mean(
            np.mean(np.linalg.norm(pairs[:, [2, 3]], axis=2), axis=1)
        )
        result["mouth_opening_proxy"] = _finite_mean(np.linalg.norm(pairs[:, 5], axis=1))

    return result


def biomarker_metadata() -> dict[str, dict[str, Any]]:
    """Definitions shipped with result files for clinical auditability."""
    return {
        "optical_flow_mean": {"unit": "normalized flow", "construct": "facial motion"},
        "optical_flow_p95": {"unit": "normalized flow", "construct": "peak facial motion"},
        "head_speed_mean_per_sec": {"unit": "normalized pose units/s", "construct": "head movement"},
        "wrist_speed_mean_per_sec": {"unit": "normalized pose units/s", "construct": "upper-limb movement"},
        "left_right_wrist_speed_asymmetry": {"unit": "ratio [0,1]", "construct": "motor asymmetry"},
        "immobility_fraction": {"unit": "fraction [0,1]", "construct": "motor arrest proxy"},
        "facial_geometry_change_per_sec": {"unit": "inter-ocular-normalized units/s", "construct": "facial-expression change"},
        "eye_aperture_proxy": {"unit": "inter-ocular-normalized ratio", "construct": "eye aperture; not emotion"},
        "mouth_opening_proxy": {"unit": "inter-ocular-normalized ratio", "construct": "mouth opening; not emotion"},
    }
