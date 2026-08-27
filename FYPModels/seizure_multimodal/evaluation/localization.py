from __future__ import annotations

from typing import Iterable

import numpy as np

from .events import TemporalEvent, match_events


def _summary(errors: list[float]) -> dict[str, float] | None:
    if not errors:
        return None
    values = np.asarray(errors, dtype=float)
    return {
        "mean_signed_error_sec": float(values.mean()),
        "median_signed_error_sec": float(np.median(values)),
        "mean_absolute_error_sec": float(np.abs(values).mean()),
        "median_absolute_error_sec": float(np.median(np.abs(values))),
    }


def localization_metrics(
    predictions: Iterable[TemporalEvent],
    references: Iterable[TemporalEvent],
    *,
    onset_reference: str,
    offset_annotations_available: bool,
) -> dict[str, object]:
    if onset_reference not in {"eeg_onset", "clinical_onset"}:
        raise ValueError("Onset reference must be eeg_onset or clinical_onset; they cannot be mixed")
    matches = match_events(predictions, references)
    onset_errors = [prediction.start_sec - reference.start_sec for prediction, reference in matches]
    offset_errors = [prediction.end_sec - reference.end_sec for prediction, reference in matches]
    return {
        "matched_events": len(matches),
        "onset_reference": onset_reference,
        "onset": _summary(onset_errors),
        "offset": _summary(offset_errors) if offset_annotations_available else None,
    }
