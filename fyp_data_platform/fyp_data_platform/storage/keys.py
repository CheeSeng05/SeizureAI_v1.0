from __future__ import annotations

import re
from pathlib import PurePosixPath


_UNSAFE_ENTITY = re.compile(r"[^A-Za-z0-9._=-]+")


def safe_entity(value: str) -> str:
    cleaned = _UNSAFE_ENTITY.sub("-", str(value).strip()).strip(".-")
    if not cleaned:
        raise ValueError("Storage entity cannot be empty.")
    return cleaned


def normalize_key(key: str) -> str:
    candidate = str(key).replace("\\", "/").strip("/")
    path = PurePosixPath(candidate)
    if not candidate or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe storage key: {key!r}")
    return path.as_posix()


def source_key(dataset_id: str, relative_path: str) -> str:
    relative = normalize_key(relative_path)
    return normalize_key(f"datasets/{safe_entity(dataset_id)}/source/{relative}")


def derived_key(dataset_id: str, recording_id: str, artifact_type: str, filename: str) -> str:
    return normalize_key(
        f"datasets/{safe_entity(dataset_id)}/derived/{safe_entity(recording_id)}/"
        f"{safe_entity(artifact_type)}/{safe_entity(filename)}"
    )
