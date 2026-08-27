"""Stable, dataframe-free annotation boundary for sibling FYP components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .annotation_adapters.factory import create_adapter


@dataclass(frozen=True)
class AnnotationBatch:
    rows: list[dict[str, Any]]
    excluded_event_counts: dict[str, int]


def load_annotations(
    adapter_type: str,
    source: str | Path,
    *,
    config: dict[str, Any] | None = None,
    active_labels: list[str] | None = None,
) -> AnnotationBatch:
    """Return JSON-compatible canonical rows without exposing pandas to callers."""
    adapter = create_adapter(adapter_type, source, config=config)
    frame = adapter.canonical(active_labels=active_labels)
    clean = frame.astype(object).where(pd.notna(frame), None)
    return AnnotationBatch(
        rows=clean.to_dict(orient="records"),
        excluded_event_counts=dict(getattr(adapter, "excluded_counts", {})),
    )
