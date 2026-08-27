from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Iterable, Sequence

import numpy as np

from ..types import WindowRecord
from .windows import make_window_id


@dataclass(frozen=True)
class ClockMapping:
    """Map the shared recording timeline to a modality's local timeline.

    ``offset_sec`` is the modality-local timestamp corresponding to shared time
    zero. ``drift_ppm`` models a linear clock-rate difference. Explicit frame or
    sample timestamps should be preferred when dropped frames/nonlinear drift are
    known.
    """

    offset_sec: float = 0.0
    drift_ppm: float = 0.0

    def to_local(self, shared_sec: float) -> float:
        if not isfinite(shared_sec):
            raise ValueError("Timestamp must be finite")
        return self.offset_sec + shared_sec * (1.0 + self.drift_ppm / 1_000_000.0)


def _validate_interval(start_sec: float, end_sec: float) -> None:
    if not all(isfinite(value) for value in (start_sec, end_sec)):
        raise ValueError("Timestamps must be finite")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("Interval must satisfy 0 <= start_sec < end_sec")


def video_frame_indices(
    start_sec: float,
    end_sec: float,
    *,
    fps: float | None = None,
    clock: ClockMapping = ClockMapping(),
    frame_timestamps_sec: Sequence[float] | None = None,
) -> np.ndarray:
    """Return frames whose local timestamps fall in ``[start, end)``.

    Supplying timestamps handles variable FPS and dropped frames without relying
    on raw array-index alignment.
    """

    _validate_interval(start_sec, end_sec)
    local_start, local_end = clock.to_local(start_sec), clock.to_local(end_sec)
    if frame_timestamps_sec is not None:
        timestamps = np.asarray(frame_timestamps_sec, dtype=float)
        if timestamps.ndim != 1 or np.any(~np.isfinite(timestamps)):
            raise ValueError("Frame timestamps must be a finite 1-D sequence")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("Frame timestamps must be strictly increasing")
        return np.flatnonzero((timestamps >= local_start) & (timestamps < local_end))
    if fps is None or not isfinite(fps) or fps <= 0:
        raise ValueError("A positive FPS is required when frame timestamps are absent")
    first = max(0, ceil(local_start * fps - 1e-9))
    stop = max(first, ceil(local_end * fps - 1e-9))
    return np.arange(first, stop, dtype=int)


def eeg_sample_bounds(
    start_sec: float,
    end_sec: float,
    *,
    sampling_rate: float | None = None,
    clock: ClockMapping = ClockMapping(),
    sample_timestamps_sec: Sequence[float] | None = None,
) -> tuple[int, int]:
    """Return half-open EEG sample bounds for the shared interval."""

    _validate_interval(start_sec, end_sec)
    local_start, local_end = clock.to_local(start_sec), clock.to_local(end_sec)
    if sample_timestamps_sec is not None:
        timestamps = np.asarray(sample_timestamps_sec, dtype=float)
        if timestamps.ndim != 1 or np.any(~np.isfinite(timestamps)):
            raise ValueError("EEG sample timestamps must be a finite 1-D sequence")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("EEG sample timestamps must be strictly increasing")
        return (
            int(np.searchsorted(timestamps, local_start, side="left")),
            int(np.searchsorted(timestamps, local_end, side="left")),
        )
    if sampling_rate is None or not isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("A positive sampling rate is required when timestamps are absent")
    return max(0, ceil(local_start * sampling_rate - 1e-9)), max(
        0, ceil(local_end * sampling_rate - 1e-9)
    )


def validate_extracted_duration(
    *,
    expected_duration_sec: float,
    video_timestamps_sec: Sequence[float],
    eeg_sample_count: int,
    eeg_sampling_rate: float,
    tolerance_sec: float,
) -> None:
    if expected_duration_sec <= 0 or tolerance_sec < 0:
        raise ValueError("Invalid expected duration or tolerance")
    video_ts = np.asarray(video_timestamps_sec, dtype=float)
    if video_ts.size < 2:
        raise ValueError("At least two video timestamps are needed for duration validation")
    video_duration = float(video_ts[-1] - video_ts[0])
    eeg_duration = eeg_sample_count / eeg_sampling_rate
    if abs(video_duration - expected_duration_sec) > tolerance_sec:
        raise ValueError(
            f"Video duration mismatch: {video_duration:.6f}s vs {expected_duration_sec:.6f}s"
        )
    if abs(eeg_duration - expected_duration_sec) > tolerance_sec:
        raise ValueError(
            f"EEG duration mismatch: {eeg_duration:.6f}s vs {expected_duration_sec:.6f}s"
        )
    if abs(video_duration - eeg_duration) > tolerance_sec:
        raise ValueError(
            f"Synchronization mismatch: video={video_duration:.6f}s EEG={eeg_duration:.6f}s"
        )


def assert_paired_windows(windows: Iterable[WindowRecord], *, require_files: bool = False) -> None:
    seen: set[str] = set()
    for window in windows:
        if window.window_id in seen:
            raise ValueError(f"Duplicate window_id: {window.window_id}")
        seen.add(window.window_id)
        expected_id = make_window_id(window.recording_id, window.start_sec, window.end_sec)
        if window.window_id != expected_id:
            raise ValueError(
                f"{window.window_id}: ID does not match recording/timestamps ({expected_id})"
            )
        if not (window.patient_id and window.recording_id and window.dataset):
            raise ValueError(f"{window.window_id}: missing patient, recording, or dataset identity")
        _validate_interval(window.start_sec, window.end_sec)
        if not (window.has_video and window.has_eeg):
            raise ValueError(f"{window.window_id}: both modalities are required for real fusion")
        if require_files:
            window.require_paired_files()


def overlapping_interval(
    video_start_sec: float,
    video_end_sec: float,
    eeg_start_sec: float,
    eeg_end_sec: float,
) -> tuple[float, float]:
    for start, end in ((video_start_sec, video_end_sec), (eeg_start_sec, eeg_end_sec)):
        _validate_interval(start, end)
    start, end = max(video_start_sec, eeg_start_sec), min(video_end_sec, eeg_end_sec)
    if end <= start:
        raise ValueError("Video and EEG intervals do not overlap")
    return start, end
