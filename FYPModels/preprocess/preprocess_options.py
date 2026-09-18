"""Shared inference settings for facial preprocessing and comparisons."""
from __future__ import annotations

import argparse
import math
from typing import Any


CONFIDENCE_OPTIONS = (
    "min_pose_detection_confidence",
    "min_pose_presence_confidence",
    "min_pose_tracking_confidence",
    "min_face_detection_confidence",
    "min_face_presence_confidence",
    "min_face_tracking_confidence",
)

BASELINE_TUNING = {
    **dict.fromkeys(CONFIDENCE_OPTIONS, 0.5),
    "box_smoothing": 0.7,
    "max_box_reuse_frames": 15,
    "max_box_reuse_seconds": None,
}


def add_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    for name in CONFIDENCE_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=0.5)
    parser.add_argument("--box-smoothing", type=float, default=0.7,
                        help="Weight of the previous crop box; lower follows movement faster.")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--max-box-reuse-frames", type=int, default=15)
    reuse.add_argument("--max-box-reuse-seconds", type=float,
                       help="Maximum elapsed video time since the last detected face/head box.")


def tuning_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in BASELINE_TUNING}


def validate_tuning(settings: dict[str, Any]) -> None:
    for name in CONFIDENCE_OPTIONS:
        if not math.isfinite(settings[name]) or not 0 <= settings[name] <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    smoothing = settings["box_smoothing"]
    if not math.isfinite(smoothing) or not 0 <= smoothing < 1:
        raise ValueError("box_smoothing must be finite and in [0, 1)")
    frames = settings["max_box_reuse_frames"]
    if not isinstance(frames, int) or frames < 0:
        raise ValueError("max_box_reuse_frames must be a nonnegative integer")
    seconds = settings["max_box_reuse_seconds"]
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
        raise ValueError("max_box_reuse_seconds must be finite and nonnegative")


def can_reuse_box(*, missed_frames: int, elapsed_seconds: float,
                  max_frames: int, max_seconds: float | None) -> bool:
    """Seconds take precedence for API callers; CLI arguments are exclusive."""
    if max_seconds is not None:
        return max_seconds > 0 and elapsed_seconds <= max_seconds + 1e-9
    return missed_frames <= max_frames
