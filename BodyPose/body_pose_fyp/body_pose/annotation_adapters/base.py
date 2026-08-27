from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "dataset",
    "media_id",
    "recording_id",
    "patient_id",
    "start_sec",
    "end_sec",
    "label",
    "source_event_type",
    "lateralization",
    "localization",
    "confidence",
    "annotation_source",
    "derived_label",
]
REQUIRED_CANONICAL_COLUMNS = {
    "media_id",
    "patient_id",
    "start_sec",
    "end_sec",
    "label",
}


class AnnotationAdapter(ABC):
    """Convert one raw dataset annotation format to the shared internal schema."""

    def __init__(self, source: str | Path, config: dict[str, Any] | None = None) -> None:
        self.source = Path(source)
        self.config = config or {}

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Read raw annotations and return canonical rows."""

    def canonical(self, active_labels: list[str] | None = None) -> pd.DataFrame:
        return validate_canonical(self.read(), active_labels=active_labels)


def validate_canonical(
    frame: pd.DataFrame,
    active_labels: list[str] | None = None,
) -> pd.DataFrame:
    missing = REQUIRED_CANONICAL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical annotations are missing columns: {sorted(missing)}")

    result = frame.copy()
    for column in CANONICAL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    for column in ("media_id", "patient_id", "label"):
        result[column] = result[column].astype("string").str.strip()
        invalid = result[column].isna() | result[column].eq("")
        if invalid.any():
            raise ValueError(f"Canonical column '{column}' contains empty values.")

    result["label"] = result["label"].str.lower()
    result["start_sec"] = pd.to_numeric(result["start_sec"], errors="raise")
    result["end_sec"] = pd.to_numeric(result["end_sec"], errors="raise")
    if not np.isfinite(result["start_sec"].to_numpy(float)).all() or not np.isfinite(
        result["end_sec"].to_numpy(float)
    ).all():
        raise ValueError("Canonical start_sec and end_sec must be finite numbers.")
    if (result["start_sec"] < 0).any():
        raise ValueError("Canonical start_sec values cannot be negative.")
    if (result["end_sec"] <= result["start_sec"]).any():
        raise ValueError("Every canonical interval must have end_sec > start_sec.")

    if active_labels is not None:
        allowed = {str(label).strip().lower() for label in active_labels}
        unknown = sorted(set(result["label"].dropna()) - allowed)
        if unknown:
            raise ValueError(
                f"Canonical annotations contain inactive labels {unknown}; "
                f"configured labels are {sorted(allowed)}."
            )

    extra_columns = [column for column in result.columns if column not in CANONICAL_COLUMNS]
    return result[CANONICAL_COLUMNS + extra_columns].reset_index(drop=True)
