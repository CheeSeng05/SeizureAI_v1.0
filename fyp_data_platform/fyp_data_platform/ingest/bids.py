from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SENSOR_SUFFIXES = {".edf", ".bdf", ".eeg", ".vhdr", ".vmrk", ".set", ".fdt"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}
RECORDING_SUFFIXES = (
    "_events",
    "_channels",
    "_electrodes",
    "_coordsystem",
    "_eeg",
)


@dataclass(frozen=True)
class BidsFile:
    path: Path
    relative_path: str
    subject_id: str | None
    session_id: str | None
    recording_id: str | None
    role: str
    modality: str | None
    media_type: str | None
    payload_state: str
    materializable: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sensor_payload(self) -> bool:
        return self.path.suffix.lower() in SENSOR_SUFFIXES

    @property
    def is_video_payload(self) -> bool:
        return self.path.suffix.lower() in VIDEO_SUFFIXES


def _entity(text: str, name: str) -> str | None:
    match = re.search(rf"(?:^|[/_])({re.escape(name)}-[^/_]+)", text)
    return match.group(1) if match else None


def _recording_id(path: Path) -> str | None:
    stem = path.name
    for suffix in (".tsv.gz", ".nii.gz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = path.stem
    for suffix in RECORDING_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem if stem.startswith("sub-") else None


def _payload_state(path: Path) -> tuple[str, bool]:
    if path.is_symlink() and not path.exists():
        return "annex-pointer", False
    if not path.is_file():
        return "unavailable", False
    try:
        if path.stat().st_size <= 2048:
            head = path.read_bytes()[:2048]
            if b"git-annex/objects" in head or b".git/annex/objects" in head:
                return "annex-pointer", False
            if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                return "lfs-pointer", False
    except OSError:
        return "unavailable", False
    return "materialized", True


def _role(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith("_events.tsv") or lower.endswith("_events.tsv.gz"):
        return "annotation"
    if path.suffix.lower() in SENSOR_SUFFIXES | VIDEO_SUFFIXES:
        return "source-media"
    if _recording_id(path):
        return "sidecar"
    return "dataset-metadata"


def _json_metadata(path: Path, materializable: bool) -> dict[str, Any]:
    if path.suffix.lower() != ".json" or not materializable:
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def discover_bids_files(
    root: Path,
    *,
    subjects: set[str] | None = None,
    sessions: set[str] | None = None,
    data_types: set[str] | None = None,
) -> list[BidsFile]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS dataset root not found: {root}")
    normalized_subjects = {
        item if item.startswith("sub-") else f"sub-{item}" for item in (subjects or set())
    }
    normalized_sessions = {
        item if item.startswith("ses-") else f"ses-{item}" for item in (sessions or set())
    }
    found: list[BidsFile] = []
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts or path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        subject = _entity(relative, "sub")
        session = _entity(relative, "ses")
        datatype = next(
            (
                part.lower()
                for part in Path(relative).parts
                if part.lower() in {"eeg", "ieeg", "meg", "ecg", "emg", "motion", "mov", "beh"}
            ),
            None,
        )
        if data_types is not None and datatype is not None and datatype not in data_types:
            # Deliberately skip non-target signal directories without opening their files.
            continue
        if normalized_subjects and subject and subject not in normalized_subjects:
            continue
        if normalized_sessions and session and session not in normalized_sessions:
            continue
        # Dataset-level metadata remains useful regardless of a subject filter.
        if normalized_subjects and not subject and relative not in {
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            "README",
            "CHANGES",
            "LICENSE",
        }:
            continue
        state, materializable = _payload_state(path)
        suffix = path.suffix.lower()
        modality = "video" if suffix in VIDEO_SUFFIXES else "eeg" if suffix in SENSOR_SUFFIXES or datatype == "eeg" else None
        found.append(
            BidsFile(
                path=path,
                relative_path=relative,
                subject_id=subject,
                session_id=session,
                recording_id=_recording_id(path),
                role=_role(path),
                modality=modality,
                media_type=mimetypes.guess_type(path.name)[0],
                payload_state=state,
                materializable=materializable,
                metadata=_json_metadata(path, materializable),
            )
        )
    return found


def dataset_description(files: list[BidsFile]) -> dict[str, Any]:
    for item in files:
        if item.relative_path == "dataset_description.json":
            return item.metadata
    return {}
