from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .eeg_adapter import EEGAdapter


@dataclass(frozen=True)
class EEGCSVData:
    sample_ids: list[str]
    features: np.ndarray
    original_labels: np.ndarray
    binary_labels: np.ndarray
    feature_columns: list[str]


def normalized_label_mapping(config: Mapping[str, Any]) -> tuple[dict[int, str], int]:
    section = config.get("eeg_labels", {})
    if not isinstance(section, Mapping):
        raise ValueError("eeg_labels configuration must be a mapping")
    raw_mapping = section.get("mapping", {})
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise ValueError("eeg_labels.mapping is required")
    mapping = {int(key): str(value) for key, value in raw_mapping.items()}
    positive = int(section.get("positive_value", 1))
    if positive not in mapping:
        raise ValueError("eeg_labels.positive_value is absent from the mapping")
    return mapping, positive


def load_eeg_csv(path: str | Path, config: Mapping[str, Any]) -> EEGCSVData:
    mapping, positive = normalized_label_mapping(config)
    sample_ids: list[str] = []
    labels: list[int] = []
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        feature_columns = sorted(
            (column for column in columns if column.startswith("X") and column[1:].isdigit()),
            key=lambda column: int(column[1:]),
        )
        if feature_columns != [f"X{index}" for index in range(1, 179)]:
            raise ValueError("EEG CSV must contain the exact ordered feature set X1..X178")
        if "y" not in columns:
            raise ValueError("EEG CSV target column 'y' is missing")
        for index, row in enumerate(reader):
            label = int(row["y"])
            if label not in mapping:
                raise ValueError(f"EEG label {label} has no configured mapping")
            sample_ids.append(row.get("Unnamed") or f"row_{index:05d}")
            labels.append(label)
            rows.append([float(row[column]) for column in feature_columns])
    features = np.asarray(rows, dtype=np.float32)
    original_labels = np.asarray(labels, dtype=int)
    return EEGCSVData(
        sample_ids=sample_ids,
        features=features,
        original_labels=original_labels,
        binary_labels=(original_labels == positive).astype(int),
        feature_columns=feature_columns,
    )


def adapt_csv_for_notebook_model(features: Any) -> np.ndarray:
    """Reproduce notebook cell 17's cross-dataset transformation exactly.

    Each 178-value single-channel row is linearly expanded to 1024 time steps,
    repeated across 18 channels, and globally z-scored over the complete table.
    This is an adaptation for the CHB-MIT-trained model, not native CSV training.
    """

    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the notebook's EEG CSV adaptation") from exc
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 178:
        raise ValueError("EEG CSV features must have shape [samples, 178]")
    expanded = zoom(values, (1, 1024 / 178), order=1)
    if expanded.shape[1] != 1024:
        raise RuntimeError(f"Notebook interpolation produced unexpected shape {expanded.shape}")
    expanded = np.repeat(expanded.reshape(-1, 1024, 1), 18, axis=2).astype(np.float32)
    return ((expanded - np.mean(expanded)) / (np.std(expanded) + 1e-8)).astype(np.float32)


def _expanded_single_channel(features: np.ndarray) -> tuple[np.ndarray, float, float]:
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the notebook's EEG CSV adaptation") from exc
    expanded = zoom(features, (1, 1024 / 178), order=1).astype(np.float32)
    if expanded.shape[1] != 1024:
        raise RuntimeError(f"Notebook interpolation produced unexpected shape {expanded.shape}")
    return expanded, float(np.mean(expanded)), float(np.std(expanded))


def predict_eeg_csv(
    *,
    csv_path: str | Path,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    batch_size: int = 64,
) -> tuple[EEGCSVData, dict[str, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("EEG inference batch size must be positive")
    data = load_eeg_csv(csv_path, config)
    expanded, global_mean, global_std = _expanded_single_channel(data.features)
    adapter = EEGAdapter(checkpoint_path=checkpoint_path)
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for start in range(0, len(expanded), batch_size):
        batch = ((expanded[start : start + batch_size] - global_mean) / (global_std + 1e-8))
        model_input = np.repeat(batch[:, :, None], 18, axis=2).astype(np.float32)
        output = adapter.predict(model_input, preprocessed=True)
        embeddings.append(np.asarray(output["embedding"]))
        logits.append(np.asarray(output["logits"]).reshape(-1))
        probabilities.append(np.asarray(output["probability"]).reshape(-1))
    return data, {
        "embedding": np.concatenate(embeddings),
        "logits": np.concatenate(logits),
        "probability": np.concatenate(probabilities),
        "prediction": (np.concatenate(probabilities) >= 0.5).astype(int),
    }
