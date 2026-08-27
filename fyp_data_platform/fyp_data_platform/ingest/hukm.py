"""Future HUKM ingestion boundary; intentionally contains no clinical assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HukmSourceLayout:
    root: Path
    video_directory: str
    eeg_directory: str
    annotation_export: str


def validate_hukm_layout(layout: HukmSourceLayout) -> None:
    """Fail clearly until governance and the real export contract are supplied."""
    raise NotImplementedError(
        "HUKM ingestion is intentionally not implemented. First agree de-identification, "
        "consent/governance, ID mapping, video/EEG clock synchronization, annotation columns, "
        "time units, and canonical label definitions with the clinical team."
    )
