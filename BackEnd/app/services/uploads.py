"""Receiving recordings.

Files here are hours of 1080p ward footage — the sample recording alone is
450 MB and the dataset's are several GB. Nothing is ever read into memory
whole; the request body is streamed to disk in chunks.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Optional

CHUNK = 4 * 1024 * 1024          # 4 MiB
VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg", ".wmv"}
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class UploadError(ValueError):
    """Rejected before anything touched disk."""


@dataclass
class Upload:
    upload_id: str
    filename: str
    stored_path: str
    size_bytes: int
    sha256: str
    received_at: str
    patient_id: Optional[str] = None
    seizure_id: Optional[str] = None
    seizure_type: Optional[str] = None
    eeg_onset_sec: Optional[float] = None
    clinical_onset_sec: Optional[float] = None


def safe_filename(name: str) -> str:
    """Strip any path components a client sent, then whitelist characters.

    A filename arrives from the browser and must never be trusted to stay
    inside the staging directory.
    """
    base = Path(name).name
    cleaned = SAFE_NAME.sub("_", base).lstrip(".")
    if not cleaned:
        raise UploadError("filename is empty after sanitising")
    if Path(cleaned).suffix.lower() not in VIDEO_SUFFIXES:
        raise UploadError(
            f"unsupported file type '{Path(cleaned).suffix}'. "
            f"Expected one of: {', '.join(sorted(VIDEO_SUFFIXES))}"
        )
    return cleaned


class UploadStore:
    def __init__(self, staging_dir: Path, max_bytes: int):
        self.dir = Path(staging_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _meta_path(self, upload_id: str) -> Path:
        return self.dir / f"{upload_id}.json"

    def save_stream(self, source: BinaryIO, filename: str, **annotation) -> Upload:
        """Stream to disk, hashing as we go.

        The hash is computed during the write rather than by re-reading the
        file, and it lets a duplicate upload be recognised later.
        """
        name = safe_filename(filename)
        upload_id = uuid.uuid4().hex[:12]
        target = self.dir / f"{upload_id}__{name}"

        digest = hashlib.sha256()
        written = 0
        try:
            with target.open("wb") as out:
                while True:
                    chunk = source.read(CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > self.max_bytes:
                        raise UploadError(
                            f"file exceeds the {self.max_bytes / 1e9:.1f} GB limit"
                        )
                    digest.update(chunk)
                    out.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)      # never leave a partial file
            raise

        if written == 0:
            target.unlink(missing_ok=True)
            raise UploadError("uploaded file is empty")

        record = Upload(
            upload_id=upload_id,
            filename=name,
            stored_path=str(target),
            size_bytes=written,
            sha256=digest.hexdigest(),
            received_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **annotation,
        )
        self._meta_path(upload_id).write_text(json.dumps(asdict(record), indent=2))
        return record

    def get(self, upload_id: str) -> Optional[Upload]:
        path = self._meta_path(upload_id)
        if not path.exists():
            return None
        return Upload(**json.loads(path.read_text()))

    def list(self) -> list[Upload]:
        return sorted(
            (Upload(**json.loads(p.read_text())) for p in self.dir.glob("*.json")),
            key=lambda u: u.received_at,
            reverse=True,
        )

    def delete(self, upload_id: str) -> bool:
        record = self.get(upload_id)
        if not record:
            return False
        Path(record.stored_path).unlink(missing_ok=True)
        self._meta_path(upload_id).unlink(missing_ok=True)
        return True

    def free_space_bytes(self) -> int:
        return shutil.disk_usage(self.dir).free