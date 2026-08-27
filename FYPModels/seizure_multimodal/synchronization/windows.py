from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Iterator


def _milliseconds(seconds: float) -> int:
    if seconds < 0:
        raise ValueError("Window timestamps cannot be negative")
    return int((Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def make_window_id(recording_id: str, start_sec: float, end_sec: float) -> str:
    if not recording_id or "__" in recording_id:
        raise ValueError("recording_id must be non-empty and cannot contain '__'")
    start_ms = _milliseconds(start_sec)
    end_ms = _milliseconds(end_sec)
    if end_ms <= start_ms:
        raise ValueError("Window end must be after start")
    return f"{recording_id}__{start_ms:010d}__{end_ms:010d}"


def generate_windows(
    recording_id: str,
    *,
    overlap_start_sec: float,
    overlap_end_sec: float,
    window_duration_sec: float,
    stride_sec: float,
    include_partial: bool = False,
) -> Iterator[tuple[str, float, float]]:
    """Yield deterministic windows only within the cross-modal overlap."""

    if overlap_start_sec < 0 or overlap_end_sec <= overlap_start_sec:
        raise ValueError("Invalid recording overlap interval")
    if window_duration_sec <= 0 or stride_sec <= 0:
        raise ValueError("Window duration and stride must be positive")
    start = float(overlap_start_sec)
    epsilon = 1e-9
    while start < overlap_end_sec - epsilon:
        end = min(start + window_duration_sec, overlap_end_sec)
        if end - start < window_duration_sec - epsilon and not include_partial:
            break
        yield make_window_id(recording_id, start, end), start, end
        start += stride_sec
