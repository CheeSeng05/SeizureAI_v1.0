"""Stream SPEINet-enhanced video frames into downstream preprocessing."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .speinet import DEFAULT_SPEINET_CHECKPOINT, SPEINetConfig, SPEINetEnhancer


def iter_enhanced_frames(
    capture: Any, *, source_fps: float, fps: float, enhancer: SPEINetEnhancer,
    max_duration_seconds: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Enhance full frames before face detection, preserving source timestamps.

    Decode/classify at native FPS, then restore requested center frames with
    native-rate neighbours. This gives the same temporal inputs as enhancement
    followed by sampling, without running the network on discarded outputs.
    At most ``2 * reference_radius + 1`` decoded frames are retained. Future
    context is bounded by the clip/duration limit, so this is offline processing.
    """
    if any(not math.isfinite(value) or value <= 0 for value in (source_fps, fps)):
        raise ValueError("source_fps and fps must be finite and positive")
    if max_duration_seconds is not None and (
        not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0
    ):
        raise ValueError("max_duration_seconds must be finite and positive")
    radius = enhancer.config.reference_radius
    frames: dict[int, np.ndarray] = {}
    sharp: dict[int, bool] = {}
    read_index, center, next_sample = 0, 0, 0.0
    finished = False
    shape = None
    while True:
        while not finished and read_index <= center + radius:
            if max_duration_seconds is not None and read_index / source_fps >= max_duration_seconds:
                finished = True
                break
            ok, frame = capture.read()
            if not ok:
                finished = True
                break
            if shape is not None and frame.shape != shape:
                raise ValueError("Video frame dimensions changed during SPEINet enhancement")
            shape = frame.shape
            frames[read_index] = frame
            sharp[read_index] = enhancer.is_sharp(frame)
            read_index += 1
        if center >= read_index:
            break
        time_sec = center / source_fps
        if time_sec + 1e-9 >= next_sample:
            # Reflect the temporal endpoints; a single-frame clip repeats itself.
            previous = center - 1 if center > 0 else min(1, read_index - 1)
            following = center + 1 if center + 1 < read_index else max(0, center - 1)
            candidates = [index for index in frames if abs(index - center) <= radius and sharp[index]]
            before = max((index for index in candidates if index <= center), default=None)
            after = min((index for index in candidates if index >= center), default=None)
            # SPEINet gates on slot 3 and reads slot 4: duplicate a sole reference.
            before = after if before is None else before
            after = before if after is None else after
            restored = enhancer.enhance([
                frames[previous], frames[center], frames[following],
                frames.get(before), frames.get(after),
            ])
            if restored.shape != frames[center].shape or restored.dtype != np.uint8:
                raise RuntimeError("Video enhancement must preserve frame shape and uint8 BGR format")
            yield time_sec, restored
            next_sample += 1 / fps
        center += 1
        for index in list(frames):
            if index < center - radius:
                del frames[index]
                del sharp[index]


def add_video_enhancement_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Video enhancement before face preprocessing")
    group.add_argument("--video-enhancement", choices=("none", "speinet"), default="none")
    group.add_argument("--speinet-checkpoint", type=Path, default=DEFAULT_SPEINET_CHECKPOINT)
    group.add_argument("--speinet-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    group.add_argument("--speinet-tile-size", type=int, default=200)
    group.add_argument("--speinet-tile-overlap", type=int, default=40)


def video_enhancement_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {"video_enhancement": None if args.video_enhancement == "none" else SPEINetConfig(
        checkpoint=args.speinet_checkpoint, device=args.speinet_device,
        tile_size=args.speinet_tile_size, tile_overlap=args.speinet_tile_overlap,
    )}
