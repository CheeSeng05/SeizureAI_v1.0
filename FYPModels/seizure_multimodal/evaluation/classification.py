from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _arrays(y_true: Any, y_probability: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(y_probability, dtype=float).reshape(-1)
    if labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must be non-empty and have equal length")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Binary labels must contain only 0 and 1")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return labels.astype(int), probabilities


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def binary_confusion(
    y_true: Any, y_probability: Any, *, threshold: float = 0.5
) -> dict[str, int]:
    labels, probabilities = _arrays(y_true, y_probability)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
    }


def binary_auroc(y_true: Any, y_probability: Any) -> float | None:
    """AUROC from continuous probabilities, using average ranks for ties."""

    labels, probabilities = _arrays(y_true, y_probability)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    ranks = np.empty(labels.size, dtype=float)
    index = 0
    while index < labels.size:
        stop = index + 1
        while stop < labels.size and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        average_rank = (index + 1 + stop) / 2.0
        ranks[order[index:stop]] = average_rank
        index = stop
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def binary_classification_metrics(
    y_true: Any,
    y_probability: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int | None | str]:
    """Window-level binary metrics; event sensitivity is calculated separately."""

    confusion = binary_confusion(y_true, y_probability, threshold=threshold)
    tn, fp, fn, tp = (confusion[key] for key in ("tn", "fp", "fn", "tp"))
    precision = _safe_ratio(tp, tp + fp)
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    accuracy = _safe_ratio(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (
        None if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2
    )
    f1 = (
        None
        if precision is None or sensitivity is None or precision + sensitivity == 0
        else 2 * precision * sensitivity / (precision + sensitivity)
    )
    return {
        **confusion,
        "evaluation_unit": "window",
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": sensitivity,
        "f1": f1,
        "auroc": binary_auroc(y_true, y_probability),
        "evaluated_samples": int(tp + tn + fp + fn),
    }


def multiclass_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, Any]:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if labels.size == 0 or labels.shape != predictions.shape:
        raise ValueError("Multiclass arrays must be non-empty and have equal shape")
    classes = sorted(set(labels.tolist()) | set(predictions.tolist()), key=str)
    per_class: dict[str, float | None] = {}
    supports: list[int] = []
    scores: list[float] = []
    for label in classes:
        tp = int(np.sum((labels == label) & (predictions == label)))
        fp = int(np.sum((labels != label) & (predictions == label)))
        fn = int(np.sum((labels == label) & (predictions != label)))
        support = int(np.sum(labels == label))
        denominator = 2 * tp + fp + fn
        score = 2 * tp / denominator if denominator else 0.0
        per_class[str(label)] = score
        supports.append(support)
        scores.append(score)
    total = sum(supports)
    return {
        "per_class_f1": per_class,
        "macro_f1": float(np.mean(scores)),
        "weighted_f1": float(np.average(scores, weights=supports)) if total else None,
    }
