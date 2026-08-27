from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_record(
    path: str | Path,
    *,
    model_name: str,
    source: str,
    notes: str = "",
) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return {
        "model_name": model_name,
        "source": source,
        "original_path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "immutable_original": True,
    }


def write_checkpoint_record(record: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
