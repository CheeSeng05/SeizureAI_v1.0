"""Validation statistics for biomarkers and independent clinician ratings."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .agreement import cohen_kappa


def validate_continuous_biomarker(
    predicted: Sequence[float], reference: Sequence[float]
) -> dict[str, float | int | None]:
    """Return paired-sample error and Pearson/Spearman construct validity."""
    x = np.asarray(predicted, dtype=float).reshape(-1)
    y = np.asarray(reference, dtype=float).reshape(-1)
    if x.size == 0 or x.shape != y.shape:
        raise ValueError("Predicted and reference scores must be non-empty and aligned")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return {"n": int(x.size), "mae": None, "pearson_r": None, "spearman_rho": None}

    def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
        if np.std(first) == 0 or np.std(second) == 0:
            return None
        return float(np.corrcoef(first, second)[0, 1])

    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = (start + 1 + stop) / 2.0
            start = stop
        return ranks

    return {
        "n": int(x.size),
        "mae": float(np.mean(np.abs(x - y))),
        "pearson_r": correlation(x, y),
        "spearman_rho": correlation(average_ranks(x), average_ranks(y)),
    }


def validate_binary_biomarker(
    predicted: Sequence[int], reference: Sequence[int]
) -> dict[str, float | int | None]:
    """Agreement for a pre-registered binary biomarker definition."""
    predicted_array = np.asarray(predicted).reshape(-1)
    reference_array = np.asarray(reference).reshape(-1)
    if predicted_array.size == 0 or predicted_array.shape != reference_array.shape:
        raise ValueError("Predicted and reference labels must be non-empty and aligned")
    if not np.all(np.isin(predicted_array, [0, 1])) or not np.all(np.isin(reference_array, [0, 1])):
        raise ValueError("Binary biomarker labels must contain only 0 and 1")
    return {"n": int(predicted_array.size), "cohen_kappa": cohen_kappa(reference_array, predicted_array)}
