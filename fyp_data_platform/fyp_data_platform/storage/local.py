from __future__ import annotations

import shutil
from pathlib import Path

from .base import ObjectMetadata, StorageBackend
from .integrity import sha256_file
from .keys import normalize_key


class LocalStorage(StorageBackend):
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        target = (self.root / Path(normalize_key(key))).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"Storage key escapes the configured root: {key!r}")
        return target

    def put_file(self, source: Path, key: str, *, sha256: str | None = None) -> ObjectMetadata:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        checksum = sha256 or sha256_file(source)
        destination = self._path(key)
        existing = self.stat(key)
        if existing and existing.size_bytes == source.stat().st_size and existing.sha256 == checksum:
            return existing
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != checksum:
            temporary.unlink(missing_ok=True)
            raise IOError(f"Checksum verification failed while storing {source}")
        temporary.replace(destination)
        return ObjectMetadata(key=normalize_key(key), size_bytes=destination.stat().st_size, sha256=checksum)

    def download_file(self, key: str, destination: Path) -> ObjectMetadata:
        metadata = self.stat(key)
        if metadata is None:
            raise FileNotFoundError(key)
        source = self._path(key)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        if metadata.sha256 and sha256_file(destination) != metadata.sha256:
            destination.unlink(missing_ok=True)
            raise IOError(f"Checksum verification failed while reading {key}")
        return metadata

    def stat(self, key: str) -> ObjectMetadata | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return ObjectMetadata(
            key=normalize_key(key), size_bytes=path.stat().st_size, sha256=sha256_file(path)
        )
