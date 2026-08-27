"""Safe structural inspection helpers that do not deserialize checkpoint code."""

from __future__ import annotations

import collections
import io
import pickle
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .provenance import sha256_file


def inspect_checkpoint_archive(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    result: dict[str, Any] = {
        "path": str(checkpoint.resolve()),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
        "is_zip_archive": zipfile.is_zipfile(checkpoint),
    }
    if not result["is_zip_archive"]:
        result["archive_entries"] = None
        return result
    with zipfile.ZipFile(checkpoint) as archive:
        entries = archive.infolist()
        result.update(
            archive_entries=len(entries),
            data_entries=sum("/data/" in entry.filename for entry in entries),
            pickle_entries=[entry.filename for entry in entries if entry.filename.endswith("data.pkl")],
            uncompressed_bytes=sum(entry.file_size for entry in entries),
        )
    return result


@dataclass(frozen=True)
class TensorMetadata:
    storage_key: str
    dtype: str
    storage_size: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool


@dataclass(frozen=True)
class _StorageMetadata:
    key: str
    dtype: str
    size: int


@dataclass(frozen=True)
class _StorageType:
    dtype: str


def _rebuild_tensor_v2(
    storage: _StorageMetadata,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    _backward_hooks: Any,
) -> TensorMetadata:
    return TensorMetadata(
        storage_key=storage.key,
        dtype=storage.dtype,
        storage_size=storage.size,
        storage_offset=storage_offset,
        shape=tuple(size),
        stride=tuple(stride),
        requires_grad=requires_grad,
    )


class _TorchMetadataUnpickler(pickle.Unpickler):
    """Restricted unpickler supporting only state-dict tensor metadata."""

    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("collections", "Counter"): collections.Counter,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
            ("torch", "FloatStorage"): _StorageType("float32"),
            ("torch", "LongStorage"): _StorageType("int64"),
        }
        try:
            return allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}") from exc

    def persistent_load(self, persistent_id: Any) -> _StorageMetadata:
        if not isinstance(persistent_id, tuple) or len(persistent_id) < 5:
            raise pickle.UnpicklingError("Unsupported persistent checkpoint identifier")
        kind, storage_type, key, _location, size = persistent_id[:5]
        if kind != "storage" or not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError("Unsupported checkpoint storage")
        return _StorageMetadata(str(key), storage_type.dtype, int(size))


def read_state_dict_metadata(path: str | Path) -> dict[str, TensorMetadata]:
    """Read PyTorch tensor names/shapes without importing torch or tensor bytes."""

    checkpoint = Path(path)
    if not zipfile.is_zipfile(checkpoint):
        raise ValueError("Only modern zip-based PyTorch checkpoints are supported")
    with zipfile.ZipFile(checkpoint) as archive:
        pickle_names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if len(pickle_names) != 1:
            raise ValueError(f"Expected one data.pkl entry, found {len(pickle_names)}")
        root = _TorchMetadataUnpickler(io.BytesIO(archive.read(pickle_names[0]))).load()

    candidates: list[dict[str, TensorMetadata]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            tensor_items = {key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, TensorMetadata)}
            if tensor_items:
                candidates.append(tensor_items)
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(root)
    if not candidates:
        raise ValueError("No tensor state dictionary found in checkpoint")
    return max(candidates, key=len)


def serializable_state_dict_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    return {key: asdict(value) for key, value in read_state_dict_metadata(path).items()}
