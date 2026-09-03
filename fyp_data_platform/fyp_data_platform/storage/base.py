from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator


@dataclass(frozen=True)
class ObjectMetadata:
    key: str
    size_bytes: int
    sha256: str | None = None
    etag: str | None = None


class StorageBackend(ABC):
    """Minimal object-store contract shared by local disk and S3."""

    @abstractmethod
    def put_file(self, source: Path, key: str, *, sha256: str | None = None) -> ObjectMetadata:
        """Persist a local file at key, returning its stored metadata."""

    @abstractmethod
    def download_file(self, key: str, destination: Path) -> ObjectMetadata:
        """Materialize key at destination."""

    @abstractmethod
    def stat(self, key: str) -> ObjectMetadata | None:
        """Return metadata, or None when key does not exist."""

    @abstractmethod
    def iter_bytes(
        self, key: str, *, start: int = 0, end: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        """Stream an inclusive byte range without materializing the complete object."""

    def exists(self, key: str) -> bool:
        return self.stat(key) is not None
