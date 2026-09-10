"""Video decoding and sampling, independent of any detector or model."""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from ..data_inspection import MP4VideoMetadata, inspect_mp4


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int | None
    width: int | None
    height: int | None
    mp4: MP4VideoMetadata | None = None
    inspection_error: str | None = None

    @property
    def duration_sec(self) -> float | None:
        return self.frame_count / self.fps if self.frame_count is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "duration_sec": self.duration_sec,
                "fps_source": "opencv", "timestamp_policy": "frame_index / source_fps"}


def _positive_integer_property(capture: Any, key: int) -> int | None:
    value = float(capture.get(key))
    return int(value) if math.isfinite(value) and value >= 1 else None


@contextmanager
def open_video(path: str | Path, *, cv2: Any = None) -> Iterator[tuple[Any, VideoMetadata]]:
    """Open a decoder and always release it, including on validation failures.

    OpenCV FPS remains the sampling clock. MP4 track metadata is supplementary;
    unsupported containers or MP4 layouts can still be decoded by OpenCV.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if cv2 is None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to decode video") from exc
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise OSError(f"Video cannot be opened: {source}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"Video reports an invalid FPS: {source_fps!r}")
        mp4 = None
        inspection_error = None
        if source.suffix.lower() in {".mp4", ".m4v", ".mov"}:
            try:
                mp4 = inspect_mp4(source)
            except (OSError, ValueError) as exc:
                inspection_error = str(exc)
        frame_count = _positive_integer_property(capture, cv2.CAP_PROP_FRAME_COUNT)
        metadata = VideoMetadata(
            path=str(source), fps=source_fps,
            frame_count=frame_count if frame_count is not None else mp4.frame_count if mp4 else None,
            width=_positive_integer_property(capture, cv2.CAP_PROP_FRAME_WIDTH),
            height=_positive_integer_property(capture, cv2.CAP_PROP_FRAME_HEIGHT),
            mp4=mp4, inspection_error=inspection_error,
        )
        yield capture, metadata
    finally:
        capture.release()


def iter_sampled_frames(
    capture: Any, *, source_fps: float, fps: float,
    max_duration_seconds: float | None = None,
) -> Iterator[tuple[float, Any]]:
    """Yield (seconds, BGR frame) without resizing or duplicating frames.

    Sampling preserves the facial pipeline's constant-FPS timing policy; it
    does not claim presentation-timestamp support for variable-frame-rate video.
    The caller owns the capture, normally through ``open_video``.
    """
    if any(not math.isfinite(value) or value <= 0 for value in (source_fps, fps)):
        raise ValueError("source_fps and fps must be finite and positive")
    if max_duration_seconds is not None and (
        not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0
    ):
        raise ValueError("max_duration_seconds must be finite and positive")
    next_sample, frame_index = 0.0, 0
    while True:
        time_sec = frame_index / source_fps
        if max_duration_seconds is not None and time_sec >= max_duration_seconds:
            break
        ok, bgr = capture.read()
        if not ok:
            break
        frame_index += 1
        if time_sec + 1e-9 < next_sample:
            continue
        yield time_sec, bgr
        next_sample += 1 / fps
