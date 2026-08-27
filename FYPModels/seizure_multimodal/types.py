from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EncoderOutput:
    """Common output contract for a frozen unimodal encoder."""

    embedding: Any
    logits: Any
    probability: Any

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowRecord:
    """One interval on the recording time axis shared by video and EEG."""

    window_id: str
    dataset: str
    patient_id: str
    recording_id: str
    start_sec: float
    end_sec: float
    label: str
    split: str
    has_video: bool
    has_eeg: bool
    video_path: Optional[str] = None
    eeg_path: Optional[str] = None
    video_fps: Optional[float] = None
    eeg_sampling_rate: Optional[float] = None
    video_offset_sec: float = 0.0
    eeg_offset_sec: float = 0.0
    video_drift_ppm: float = 0.0
    eeg_drift_ppm: float = 0.0
    eeg_onset_sec: Optional[float] = None
    clinical_onset_sec: Optional[float] = None
    seizure_offset_sec: Optional[float] = None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def require_paired_files(self) -> None:
        if not (self.has_video and self.has_eeg):
            raise ValueError(
                f"{self.window_id}: real multimodal evaluation requires "
                "has_video=true and has_eeg=true"
            )
        for modality, raw_path in (("video", self.video_path), ("EEG", self.eeg_path)):
            if not raw_path:
                raise ValueError(f"{self.window_id}: {modality} path is missing")
            if not Path(raw_path).exists():
                raise FileNotFoundError(f"{self.window_id}: {modality} file not found: {raw_path}")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "WindowRecord":
        values = dict(row)
        for key in ("video_path", "eeg_path"):
            if values.get(key) == "":
                values[key] = None
        for key in ("has_video", "has_eeg"):
            value = values.get(key, False)
            values[key] = value if isinstance(value, bool) else str(value).strip().lower() in {
                "1",
                "true",
                "yes",
            }
        float_fields = (
            "start_sec",
            "end_sec",
            "video_fps",
            "eeg_sampling_rate",
            "video_offset_sec",
            "eeg_offset_sec",
            "video_drift_ppm",
            "eeg_drift_ppm",
            "eeg_onset_sec",
            "clinical_onset_sec",
            "seizure_offset_sec",
        )
        for key in float_fields:
            value = values.get(key)
            if value in (None, ""):
                values[key] = None if key not in {
                    "video_offset_sec",
                    "eeg_offset_sec",
                    "video_drift_ppm",
                    "eeg_drift_ppm",
                } else 0.0
            else:
                values[key] = float(value)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})
