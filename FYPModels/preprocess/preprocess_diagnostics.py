"""Frame-level coverage and crop-motion measurements, without accuracy claims."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FLAG_FIELDS = (
    "pose_detected", "face_landmarks_detected", "face_region_available",
    "pose_head_fallback", "box_reused",
)
FRAME_FIELDS = (
    "time_sec", *FLAG_FIELDS, "face_detector_detected",
    "raw_x1", "raw_y1", "raw_x2", "raw_y2",
    "crop_x1", "crop_y1", "crop_x2", "crop_y2",
    "box_center_step_fraction", "box_center_offset_fraction",
)


def frame_diagnostics(*, time_sec: float, raw_box: Any, crop_box: Any,
                      previous_box: Any, frame_shape: tuple[int, ...],
                      flags: tuple[bool, ...], face_detector_detected: bool = False) -> dict[str, Any]:
    row = dict.fromkeys(FRAME_FIELDS, None)
    row["time_sec"] = time_sec
    row.update(zip(FLAG_FIELDS, map(int, flags)))
    row["face_detector_detected"] = int(face_detector_detected)
    for prefix, box in (("raw", raw_box), ("crop", crop_box)):
        if box is not None:
            row.update({f"{prefix}_{name}": float(value)
                        for name, value in zip(("x1", "y1", "x2", "y2"), box)})
    diagonal = float(np.hypot(*frame_shape[:2]))
    if crop_box is not None:
        center = (crop_box[:2] + crop_box[2:]) / 2
        for name, reference in (("box_center_step_fraction", previous_box),
                                ("box_center_offset_fraction", raw_box)):
            if reference is not None:
                reference_center = (reference[:2] + reference[2:]) / 2
                row[name] = float(np.linalg.norm(center - reference_center) / diagonal)
    return row


def summarize_frames(rows: list[dict[str, Any]], processing_seconds: float) -> dict[str, Any]:
    def mean(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] is not None]
        return float(np.mean(values)) if values else None

    return {
        "sampled_frames": len(rows),
        "processing_seconds": processing_seconds,
        "processing_fps": len(rows) / processing_seconds if processing_seconds > 0 else None,
        "pose_detection_fraction": mean("pose_detected"),
        "face_landmark_fraction": mean("face_landmarks_detected"),
        "face_detector_fraction": mean("face_detector_detected"),
        "face_region_fraction": mean("face_region_available"),
        "pose_head_fallback_fraction": mean("pose_head_fallback"),
        "face_box_reused_fraction": mean("box_reused"),
        "mean_box_center_step_fraction": mean("box_center_step_fraction"),
        "mean_box_center_offset_fraction": mean("box_center_offset_fraction"),
    }


def write_diagnostics(directory: Path, sample_id: str, rows: list[dict[str, Any]],
                      settings: dict[str, Any], processing_seconds: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frames_path = directory / f"{sample_id}.frames.csv"
    with frames_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    path = directory / f"{sample_id}.json"
    path.write_text(json.dumps({
        "settings": settings,
        "metrics": summarize_frames(rows, processing_seconds),
        "frames_path": str(frames_path.resolve()),
        "interpretation": "Coverage and motion proxies only; patient identity and landmark accuracy require annotation.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
