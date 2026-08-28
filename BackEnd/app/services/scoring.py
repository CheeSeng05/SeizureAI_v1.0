"""Threshold and persistence scoring.

Single source of truth for turning per-window probabilities into predictions
and metrics. The frontend never recomputes any of this, so the number on screen
and the number in a report always come from the same code.

Label policy follows the source model exactly:
  windows before EEG onset            -> 0
  windows from clinical onset onward  -> 1
  the EEG-to-clinical transition      -> excluded from hard-label metrics
No seizure offset is invented. When the annotation has none, every window after
clinical onset counts as ictal and offset error stays None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class Annotation:
    eeg_onset_sec: Optional[float]
    clinical_onset_sec: Optional[float]
    seizure_offset_sec: Optional[float] = None


def apply_persistence(above: np.ndarray, min_consecutive: int) -> np.ndarray:
    """Raise an alarm only after `min_consecutive` windows stay above threshold.

    Isolated spikes are the dominant false-alarm mode in ward footage - a nurse
    passing, a bed adjustment - while a seizure is sustained. Requiring a run
    suppresses the former without touching the latter.
    """
    if min_consecutive <= 1:
        return above.astype(np.int8)
    out = np.zeros_like(above, dtype=np.int8)
    run = 0
    for i, v in enumerate(above):
        run = run + 1 if v else 0
        if run >= min_consecutive:
            out[i - min_consecutive + 1: i + 1] = 1
    return out


def reference_labels(start_sec: np.ndarray, ann: Annotation) -> np.ndarray:
    """-1 marks windows excluded from scoring."""
    y = np.full(start_sec.shape, -1, dtype=np.int8)
    if ann.eeg_onset_sec is None:
        return y
    y[start_sec < ann.eeg_onset_sec] = 0
    if ann.clinical_onset_sec is not None:
        ictal = start_sec >= ann.clinical_onset_sec
        if ann.seizure_offset_sec is not None:
            ictal &= start_sec < ann.seizure_offset_sec
        y[ictal] = 1
    return y


def count_alarm_events(pred: np.ndarray) -> int:
    """Contiguous runs of 1, not individual windows.

    FP/hour is a clinical burden measure: one 30-second alarm interrupts a
    nurse once, not thirty times.
    """
    if pred.size == 0:
        return 0
    return int(np.sum(np.diff(np.concatenate(([0], pred))) == 1))


def score(
    start_sec: Sequence[float],
    probability: Sequence[float],
    ann: Annotation,
    threshold: float = 0.5,
    min_consecutive: int = 1,
    window_sec: float = 1.001,
) -> dict:
    start = np.asarray(start_sec, dtype=float)
    prob = np.asarray(probability, dtype=float)

    pred = apply_persistence(prob >= threshold, min_consecutive)
    y = reference_labels(start, ann)
    scored = y >= 0

    yt, yp, pt = y[scored], pred[scored], prob[scored]
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    # FP/hour is measured only over known non-seizure monitoring time.
    interictal = pred[y == 0]
    events = count_alarm_events(interictal)
    hours = float(interictal.size * window_sec / 3600.0)

    # First alarm at or after EEG onset - when a monitoring system would
    # actually have notified someone.
    first_alarm = None
    if ann.eeg_onset_sec is not None:
        after = np.flatnonzero((start >= ann.eeg_onset_sec) & (pred == 1))
        if after.size:
            first_alarm = float(start[after[0]])

    onset_err = None
    if first_alarm is not None and ann.clinical_onset_sec is not None:
        onset_err = first_alarm - ann.clinical_onset_sec

    both = len(np.unique(yt)) > 1
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "auroc": float(roc_auc_score(yt, pt)) if both else None,
        "f1": float(f1_score(yt, yp, zero_division=0)) if both else None,
        "cohen_kappa": float(cohen_kappa_score(yt, yp)) if both else None,
        "false_positives_per_hour": events / hours if hours > 0 else None,
        "alarm_events": events,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "onset_error_sec": onset_err,
        "offset_error_sec": None,
        "first_alarm_sec": first_alarm,
        "prediction": pred.tolist(),
    }
