from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def video_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def active_labels(config: dict[str, Any]) -> list[str]:
    """Return the ordered, normalized class names configured for this dataset."""
    labels = config.get("labels", {}).get("active")
    if not labels:
        raise ValueError("config.labels.active must contain at least two class names.")
    normalized = [str(label).strip().lower() for label in labels]
    if len(normalized) < 2:
        raise ValueError("At least two active labels are required for classification.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("config.labels.active contains duplicate labels.")
    return normalized


def label_maps(config: dict[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
    names = active_labels(config)
    label_to_id = {label: index for index, label in enumerate(names)}
    return label_to_id, {index: label for label, index in label_to_id.items()}


def window_manifest_path(config: dict[str, Any]) -> Path:
    paths = config["paths"]
    value = paths.get("window_manifest", paths.get("metadata"))
    if value is None:
        raise ValueError("config.paths.window_manifest is required.")
    return Path(value)


def seizeit2_video_message() -> str:
    return (
        "The SeizeIT2 public release can be used to validate seizure annotations, "
        "but it cannot train the MediaPipe body-pose stream because the required "
        "patient video is not included."
    )
