from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .classification import binary_classification_metrics


def select_threshold(
    y_true: Any,
    y_probability: Any,
    *,
    split: str,
    objective: str = "f1",
    candidates: Iterable[float] | None = None,
) -> dict[str, float]:
    """Select on validation data only; final test labels are explicitly rejected."""

    if split != "validation":
        raise ValueError("Threshold selection is permitted only on the validation split")
    if objective != "f1":
        raise ValueError("Currently supported threshold objective: f1")
    values = np.asarray(list(candidates) if candidates is not None else np.linspace(0.05, 0.95, 91))
    if values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("Threshold candidates must be in [0, 1]")
    best_threshold = float(values[0])
    best_score = -1.0
    for threshold in values:
        score = binary_classification_metrics(y_true, y_probability, threshold=float(threshold))["f1"]
        numeric = -1.0 if score is None else float(score)
        if numeric > best_score:
            best_threshold, best_score = float(threshold), numeric
    return {"threshold": best_threshold, "validation_f1": best_score}
