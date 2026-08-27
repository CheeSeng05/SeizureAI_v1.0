from __future__ import annotations

from typing import Any

import numpy as np


def cohen_kappa(reference: Any, prediction: Any) -> float | None:
    """Model-vs-reference Cohen's kappa on the same temporal units."""

    truth = np.asarray(reference).reshape(-1)
    predicted = np.asarray(prediction).reshape(-1)
    if truth.size == 0 or truth.shape != predicted.shape:
        raise ValueError("Reference and predictions must be non-empty and aligned")
    classes = sorted(set(truth.tolist()) | set(predicted.tolist()), key=str)
    observed = float(np.mean(truth == predicted))
    expected = sum(float(np.mean(truth == c) * np.mean(predicted == c)) for c in classes)
    if np.isclose(expected, 1.0):
        return 1.0 if np.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)
