from __future__ import annotations

from pathlib import Path

import pytest

from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.local import LocalStorage


def test_local_storage_roundtrip_and_idempotence(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("research-data", encoding="utf-8")
    storage = LocalStorage(tmp_path / "objects")
    first = storage.put_file(source, "datasets/test/source/source.txt")
    second = storage.put_file(source, "datasets/test/source/source.txt")
    assert first == second
    destination = tmp_path / "download" / "source.txt"
    storage.download_file(first.key, destination)
    assert sha256_file(destination) == first.sha256


def test_local_storage_rejects_traversal(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "objects")
    with pytest.raises(ValueError):
        storage.stat("../outside.txt")
