from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..types import WindowRecord
from .splits import assert_patient_disjoint
from .windows import make_window_id


REQUIRED_COLUMNS = (
    "window_id",
    "dataset",
    "patient_id",
    "recording_id",
    "start_sec",
    "end_sec",
    "label",
    "split",
    "has_video",
    "has_eeg",
)


def read_manifest(path: str | Path) -> list[WindowRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        return [WindowRecord.from_mapping(row) for row in reader]


def write_manifest(windows: Iterable[WindowRecord], path: str | Path) -> None:
    rows = [asdict(window) for window in windows]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(WindowRecord.__dataclass_fields__)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(
    windows: Iterable[WindowRecord],
    *,
    require_paired: bool = False,
    require_files: bool = False,
) -> list[WindowRecord]:
    rows = list(windows)
    ids: set[str] = set()
    allowed_splits = {"train", "validation", "test", "unassigned"}
    for row in rows:
        if row.window_id in ids:
            raise ValueError(f"Duplicate window_id: {row.window_id}")
        ids.add(row.window_id)
        expected = make_window_id(row.recording_id, row.start_sec, row.end_sec)
        if row.window_id != expected:
            raise ValueError(f"{row.window_id}: expected deterministic ID {expected}")
        if not row.patient_id or not row.recording_id or not row.dataset:
            raise ValueError(f"{row.window_id}: identity fields cannot be empty")
        if row.label == "pre_ictal":
            raise ValueError(
                f"{row.window_id}: pre_ictal requires an explicit clinical label policy; "
                "it is not inferred automatically"
            )
        if row.split not in allowed_splits:
            raise ValueError(f"{row.window_id}: invalid split {row.split!r}")
        if require_paired and not (row.has_video and row.has_eeg):
            raise ValueError(f"{row.window_id}: both video and EEG are required")
        if require_files and (row.has_video or row.has_eeg):
            if row.has_video and not row.video_path:
                raise ValueError(f"{row.window_id}: has_video=true but video_path is empty")
            if row.has_eeg and not row.eeg_path:
                raise ValueError(f"{row.window_id}: has_eeg=true but eeg_path is empty")
    assert_patient_disjoint(rows)
    return rows
