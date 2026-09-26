from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class EEGLoadError(RuntimeError):
    """Raised when an EEG container cannot be decoded safely."""


class UnsupportedEEGFormatError(EEGLoadError):
    """Raised when no verified loader supports the input container."""


@dataclass(frozen=True)
class EEGAnnotation:
    recording_id: str
    event_id: str
    time_sec: float
    duration_sec: float
    raw_name: str
    normalized_type: str
    annotation_source: str
    clinical_status: str = "unconfirmed"
    event_type: str | None = None
    event_guid: str | None = None
    source_user_available: bool = False

    @property
    def is_point_event(self) -> bool:
        return self.duration_sec == 0.0


@dataclass(frozen=True)
class EEGMetadata:
    recording_id: str
    source_format: str
    channel_names: tuple[str, ...]
    sampling_rates: tuple[float, ...]
    signal_units: tuple[str, ...]
    duration_sec: float
    sample_count: int
    montage_reference: str | None
    recording_start_time: datetime | None = None
    vendor: str | None = None
    reader: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sampling_rate(self) -> float | None:
        unique = {round(float(value), 9) for value in self.sampling_rates}
        return float(self.sampling_rates[0]) if len(unique) == 1 and self.sampling_rates else None


def private_recording_id(path: str | Path) -> str:
    """Return a non-identifying stable ID without exposing the source filename."""

    resolved = str(Path(path).resolve()).casefold().encode("utf-8")
    return f"recording-{hashlib.sha256(resolved).hexdigest()[:12]}"


def provisional_annotation_type(
    raw_name: str,
    event_type: str | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Apply conservative, configurable categories without asserting a diagnosis."""

    raw = raw_name.strip()
    kind = (event_type or "").strip()
    configured = {key.casefold(): value for key, value in (overrides or {}).items()}
    if raw.casefold() in configured:
        return configured[raw.casefold()]

    lowered = raw.casefold()
    kind_lower = kind.casefold()
    if lowered.startswith("fit ") or lowered == "fit":
        return "candidate_seizure_event"
    if lowered == "spike" or kind_lower == "spike":
        return "eeg_event"
    if kind_lower in {
        "detections active",
        "detections inactive",
        "review progress",
        "video start",
        "video stop",
        "video review progress",
    }:
        return "system_event"
    if kind_lower in {"exam start", "us. start"}:
        return "recording_metadata"
    if lowered in {"annotation", "anmerkning", ""}:
        return "generic_annotation"
    if kind_lower in {"annotation", "anmerkning"}:
        return "clinical_observation"
    return "unknown"


class EEGRecording(ABC):
    """Format-independent, read-only EEG recording interface."""

    path: Path
    metadata: EEGMetadata
    annotations: Sequence[EEGAnnotation]

    @abstractmethod
    def read_samples(
        self,
        *,
        channel_indices: Sequence[int],
        start_sample: int,
        stop_sample: int,
    ) -> np.ndarray:
        """Return physical samples as ``[channels, samples]`` for ``[start, stop)``."""

    def close(self) -> None:
        """Release reader resources when the implementation keeps any open."""

    def __enter__(self) -> "EEGRecording":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

