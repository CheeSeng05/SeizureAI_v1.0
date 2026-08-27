from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AnnotationAdapter
from .generic_tabular import GenericTabularAdapter
from .seizeit2 import SeizeIT2Adapter


def create_adapter(
    adapter_type: str,
    source: str | Path,
    config: dict[str, Any] | None = None,
) -> AnnotationAdapter:
    name = str(adapter_type).strip().lower().replace("-", "_")
    adapters: dict[str, type[AnnotationAdapter]] = {
        "generic": GenericTabularAdapter,
        "generic_tabular": GenericTabularAdapter,
        "seizeit2": SeizeIT2Adapter,
        "ds005873": SeizeIT2Adapter,
    }
    if name not in adapters:
        raise ValueError(
            f"Unknown annotation adapter '{adapter_type}'. Expected one of {sorted(adapters)}."
        )
    return adapters[name](source=source, config=config)
