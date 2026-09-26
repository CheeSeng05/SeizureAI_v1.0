from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = (".e", ".edf")


def configured_eeg_root(root: str | Path | None = None) -> Path:
    value = root if root is not None else os.getenv("EEG_DATA_ROOT")
    if value in {None, ""}:
        raise ValueError("Set EEG_DATA_ROOT or pass an explicit dataset root.")
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError("The configured EEG dataset root is not a directory.")
    return resolved


def discover_eeg_files(
    root: str | Path | None = None,
    *,
    suffixes: Iterable[str] = SUPPORTED_SUFFIXES,
) -> list[Path]:
    """Recursively discover supported recordings without naming assumptions."""

    dataset_root = configured_eeg_root(root)
    normalized = {suffix.casefold() for suffix in suffixes}
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in normalized
    )

