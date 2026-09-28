"""Summarize anomalies, with optional independent seizure reference evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluation.classification import binary_classification_metrics
from ..evaluation.events import (
    TemporalEvent, TemporalWindow, event_level_sensitivity,
    false_positives_per_hour, probabilities_to_events,
)
from ..evaluation.localization import localization_metrics
from .evaluation import annotation_events
from .movement_features import validate_manifest


DEFAULT_EVENTS = {
    "minimum_event_duration_sec": 2.0,
    "gap_tolerance_sec": 0.0,
    "refractory_period_sec": 0.0,
}


def validate_events(settings: Mapping[str, float]) -> None:
    if set(settings) != set(DEFAULT_EVENTS):
        raise ValueError("Unknown or missing event settings")
    if any(not np.isfinite(value) or value < 0 for value in settings.values()):
        raise ValueError("Event settings must be finite and nonnegative")


def coverage_intervals(records: Sequence[Mapping[str, Any]]) -> list[TemporalEvent]:
    """Union windows per recording, without counting overlaps or unknown gaps."""
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["patient_id"]), str(row["recording_id"]))].append(
            (float(row["start_sec"]), float(row["end_sec"]))
        )
    result = []
    for key, intervals in sorted(grouped.items()):
        start, end = sorted(intervals)[0]
        for next_start, next_end in sorted(intervals)[1:]:
            if next_start <= end + 1e-9:
                end = max(end, next_end)
            else:
                result.append(TemporalEvent(*key, start, end))
                start, end = next_start, next_end
        result.append(TemporalEvent(*key, start, end))
    return result


def candidate_events(
    records: Sequence[Mapping[str, Any]], threshold: float, settings: Mapping[str, float],
) -> list[TemporalEvent]:
    """Merge bounded anomaly scores, never bridging an unobserved time gap."""
    validate_events(settings)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["patient_id"]), str(row["recording_id"]))].append(row)
    events = []
    for interval in coverage_intervals(records):
        rows = [row for row in grouped[interval.key]
                if interval.start_sec <= float(row["start_sec"]) < interval.end_sec]
        # This utility also accepts bounded ranking scores; no probabilities
        # are estimated or calibrated by this anomaly workflow.
        events.extend(probabilities_to_events(
            [TemporalWindow(str(row["patient_id"]), str(row["recording_id"]),
                            float(row["start_sec"]), float(row["end_sec"])) for row in rows],
            [float(row["anomaly_score"]) for row in rows], threshold=threshold,
            **{**settings, "refractory_period_sec": 0.0},
        ))
    accepted: list[TemporalEvent] = []
    for event in sorted(events):
        previous = accepted[-1] if accepted and accepted[-1].key == event.key else None
        if previous and event.start_sec < previous.end_sec + settings["refractory_period_sec"]:
            continue
        accepted.append(event)
    return accepted


def summarize_anomalies(
    records: Sequence[Mapping[str, Any]], *, threshold: float,
    event_settings: Mapping[str, float] = DEFAULT_EVENTS,
) -> dict[str, Any]:
    validate_manifest(records)
    validate_events(event_settings)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Anomaly threshold must be in [0, 1]")
    if any(row.get("status") not in {"scored", "excluded"} for row in records):
        raise ValueError("Every prediction requires a scored/excluded status")
    scored = [row for row in records if row["status"] == "scored"]
    scores = np.asarray([float(row["anomaly_score"]) for row in scored])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Anomaly scores must be finite and in [0, 1]")
    events = candidate_events(scored, threshold, event_settings)
    available = sum(interval.duration_sec for interval in coverage_intervals(records))
    scored_seconds = sum(interval.duration_sec for interval in coverage_intervals(scored))
    return {
        "score_type": "isolation_forest_anomaly_score",
        "score_interpretation": "Higher means more unusual; not a seizure probability.",
        "anomaly_threshold": threshold,
        "event_settings": dict(event_settings),
        "total_windows": len(records), "scored_windows": len(scored),
        "excluded_windows": len(records) - len(scored),
        "exclusion_reasons": dict(Counter(str(row["exclusion_reason"]) for row in records
                                          if row["status"] == "excluded")),
        "available_window_hours": available / 3600,
        "scored_window_hours": scored_seconds / 3600,
        "time_coverage_fraction": scored_seconds / available if available else None,
        "score_quantiles": dict(zip(("min", "median", "p95", "max"),
                                    np.quantile(scores, [0, .5, .95, 1]).tolist())) if len(scores) else None,
        "candidate_windows": int(np.sum(scores >= threshold)),
        "candidate_events": [asdict(event) for event in events],
        "candidate_events_per_scored_hour": len(events) / (scored_seconds / 3600) if scored_seconds else None,
        "reference_evaluation": None,
    }


def evaluate_anomalies(
    records: Sequence[Mapping[str, Any]], *, threshold: float,
    event_settings: Mapping[str, float] = DEFAULT_EVENTS,
    labels: Sequence[Mapping[str, Any]] | None = None,
    annotations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen detector. Unknown or excluded windows are never negatives.

    ``labels`` contains window_id,label (0/1); missing labels remain unknown.
    Event annotations assert exhaustively reviewed monitoring intervals using
    the same schema as facial.evaluation. Temporal metrics require full scored
    coverage of those intervals; otherwise only window metrics are reported.
    """
    result = summarize_anomalies(records, threshold=threshold, event_settings=event_settings)
    if any(row["split"] != "test" for row in records):
        raise ValueError("Reference evaluation requires split='test'")
    if (labels is None) == (annotations is None):
        raise ValueError("Provide exactly one of window labels or event annotations")
    scored = [row for row in records if row["status"] == "scored"]
    selected, targets = [], []
    references: list[TemporalEvent] = []
    monitoring: list[TemporalEvent] = []
    if labels is not None:
        by_id = {}
        known_ids = {row["window_id"] for row in records}
        for row in labels:
            identifier = str(row["window_id"])
            if identifier in by_id or identifier not in known_ids:
                raise ValueError(f"Duplicate or unknown annotated window_id: {identifier}")
            value = str(row.get("label", "")).strip()
            if value not in {"", "0", "1"}:
                raise ValueError("Reference labels must be blank, 0, or 1")
            by_id[identifier] = value
        for row in scored:
            value = by_id.get(row["window_id"], "")
            if value != "":
                selected.append(row)
                targets.append(int(value))
    else:
        for row in annotations or []:
            onset_missing = row.get("seizure_onset_sec") in (None, "")
            offset_missing = row.get("seizure_offset_sec") in (None, "")
            if onset_missing != offset_missing:
                raise ValueError("Seizure onset and offset must both be supplied or both be blank")
        references, monitoring = annotation_events(annotations or [])
        if len(set(references)) != len(references):
            raise ValueError("Duplicate reference seizure event")
        monitors = {event.key: event for event in monitoring}
        keys = {(row["patient_id"], row["recording_id"]) for row in records}
        if set(monitors) != keys:
            raise ValueError("Annotations must cover exactly the predicted recordings")
        for event in [*references, *monitoring]:
            monitor = monitors[event.key]
            if not np.isfinite([event.start_sec, event.end_sec]).all() or not (
                monitor.start_sec <= event.start_sec < event.end_sec <= monitor.end_sec
            ):
                raise ValueError("Invalid annotation or event outside monitoring interval")
        for row in records:
            monitor = monitors[(row["patient_id"], row["recording_id"])]
            if not monitor.start_sec <= float(row["start_sec"]) < float(row["end_sec"]) <= monitor.end_sec:
                raise ValueError("Prediction window lies outside reviewed monitoring interval")
        selected = scored
        targets = [int(any(
            (row["patient_id"], row["recording_id"]) == event.key
            and min(float(row["end_sec"]), event.end_sec) > max(float(row["start_sec"]), event.start_sec)
            for event in references
        )) for row in selected]

    evaluation: dict[str, Any] = {
        "labelled_scored_windows": len(selected),
        "unlabelled_scored_windows": len(scored) - len(selected),
        "classification": binary_classification_metrics(
            targets, [float(row["anomaly_score"]) for row in selected], threshold=threshold,
        ) if selected else None,
        "event_level_sensitivity": None, "false_positives_per_hour": None, "localization": None,
        "temporal_metrics_available": False,
        "temporal_metrics_reason": "Window labels do not establish continuous event annotations.",
    }
    if annotations is not None:
        # Complete monitoring is essential for honest missed-event and FP/hour metrics.
        covered = {(event.key, round(event.start_sec, 8), round(event.end_sec, 8))
                   for event in coverage_intervals(scored)}
        required = {(event.key, round(event.start_sec, 8), round(event.end_sec, 8)) for event in monitoring}
        if required and covered == required:
            events = candidate_events(scored, threshold, event_settings)
            evaluation.update(
                event_level_sensitivity=event_level_sensitivity(events, references),
                false_positives_per_hour=false_positives_per_hour(events, references, monitoring),
                localization=localization_metrics(events, references, onset_reference="clinical_onset",
                                                   offset_annotations_available=True),
                temporal_metrics_available=True, temporal_metrics_reason=None,
            )
        else:
            evaluation["temporal_metrics_reason"] = (
                "Scored windows do not fully cover reviewed monitoring; gaps/exclusions remain unknown."
            )
    result["reference_evaluation"] = evaluation
    return result
