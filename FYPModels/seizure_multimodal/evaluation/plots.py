from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .classification import binary_confusion


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required for analysis plots") from exc
    return plt


def roc_curve_points(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    tpr, fpr = [], []
    positives, negatives = int(truth.sum()), int(truth.size - truth.sum())
    for threshold in thresholds:
        prediction = scores >= threshold
        tpr.append(float(np.sum(prediction & (truth == 1)) / positives) if positives else np.nan)
        fpr.append(float(np.sum(prediction & (truth == 0)) / negatives) if negatives else np.nan)
    return np.asarray(fpr), np.asarray(tpr)


def precision_recall_points(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    precision, recall = [], []
    positives = int(truth.sum())
    for threshold in thresholds:
        prediction = scores >= threshold
        tp = int(np.sum(prediction & (truth == 1)))
        fp = int(np.sum(prediction & (truth == 0)))
        precision.append(tp / (tp + fp) if tp + fp else 1.0)
        recall.append(tp / positives if positives else np.nan)
    return np.asarray(recall), np.asarray(precision)


def write_analysis_plots(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float],
    window_start_sec: Sequence[float],
    output_directory: str | Path,
    threshold: float,
    eeg_onset_sec: float | None = None,
    clinical_onset_sec: float | None = None,
) -> None:
    """Write confusion, ROC, PR, and probability-over-time plots when applicable."""

    plt = _matplotlib()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    confusion = binary_confusion(labels, probabilities, threshold=threshold)
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "ictal"], yticklabels=["normal", "ictal"], xlabel="Predicted", ylabel="Reference")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    fpr, tpr = roc_curve_points(labels, probabilities)
    recall, precision = precision_recall_points(labels, probabilities)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].plot(recall, precision)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    figure.tight_layout()
    figure.savefig(output / "roc_precision_recall.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 3))
    axis.plot(window_start_sec, probabilities, label="model probability")
    axis.axhline(threshold, linestyle="--", label="threshold")
    if eeg_onset_sec is not None:
        axis.axvline(eeg_onset_sec, color="tab:red", label="EEG onset")
    if clinical_onset_sec is not None:
        axis.axvline(clinical_onset_sec, color="tab:orange", label="clinical onset")
    axis.set(xlabel="Recording time (s)", ylabel="Seizure probability", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "probability_timeline.png", dpi=150)
    plt.close(figure)
