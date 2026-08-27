from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .agreement import cohen_kappa
from .classification import binary_classification_metrics
from .events import (
    TemporalEvent,
    TemporalWindow,
    event_level_sensitivity,
    false_positives_per_hour,
    probabilities_to_events,
)
from .localization import localization_metrics


def evaluate_binary_timeline(
    *,
    windows: Sequence[TemporalWindow],
    labels: Sequence[int],
    probabilities: Sequence[float],
    reference_events: Iterable[TemporalEvent],
    monitoring_intervals: Iterable[TemporalEvent],
    threshold: float,
    minimum_event_duration_sec: float,
    gap_tolerance_sec: float,
    refractory_period_sec: float,
    onset_reference: str,
    offset_annotations_available: bool,
) -> dict[str, object]:
    """Evaluate aligned windows and merged events with one shared policy."""

    if len(windows) != len(labels) or len(windows) != len(probabilities):
        raise ValueError("Windows, labels, and probabilities must be aligned")
    classification = binary_classification_metrics(labels, probabilities, threshold=threshold)
    predictions = probabilities_to_events(
        windows,
        probabilities,
        threshold=threshold,
        minimum_event_duration_sec=minimum_event_duration_sec,
        gap_tolerance_sec=gap_tolerance_sec,
        refractory_period_sec=refractory_period_sec,
    )
    references = list(reference_events)
    prediction_labels = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "classification": classification,
        "cohen_kappa": cohen_kappa(labels, prediction_labels),
        "predicted_events": predictions,
        "event_level_sensitivity": event_level_sensitivity(predictions, references),
        "false_positives_per_hour": false_positives_per_hour(
            predictions, references, monitoring_intervals
        ),
        "localization": localization_metrics(
            predictions,
            references,
            onset_reference=onset_reference,
            offset_annotations_available=offset_annotations_available,
        ),
    }
