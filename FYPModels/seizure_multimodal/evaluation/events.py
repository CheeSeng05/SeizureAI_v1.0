from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class TemporalWindow:
    patient_id: str
    recording_id: str
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if not self.patient_id or not self.recording_id or self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise ValueError("Invalid temporal window")


@dataclass(frozen=True, order=True)
class TemporalEvent:
    patient_id: str
    recording_id: str
    start_sec: float
    end_sec: float
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.patient_id or not self.recording_id or self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise ValueError("Invalid temporal event")

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def key(self) -> tuple[str, str]:
        return self.patient_id, self.recording_id


def probabilities_to_events(
    windows: Sequence[TemporalWindow],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
    minimum_event_duration_sec: float = 0.0,
    gap_tolerance_sec: float = 0.0,
    refractory_period_sec: float = 0.0,
) -> list[TemporalEvent]:
    """Threshold and merge positive windows into alarm events.

    The same function is intended for EEG, video, and fusion comparisons.
    """

    if len(windows) != len(probabilities):
        raise ValueError("Windows and probabilities must be aligned")
    if not 0 <= threshold <= 1 or min(
        minimum_event_duration_sec, gap_tolerance_sec, refractory_period_sec
    ) < 0:
        raise ValueError("Invalid event-generation configuration")
    grouped: dict[tuple[str, str], list[tuple[TemporalWindow, float]]] = defaultdict(list)
    for window, probability in zip(windows, probabilities):
        value = float(probability)
        if not 0 <= value <= 1:
            raise ValueError("Event probabilities must be in [0, 1]")
        grouped[(window.patient_id, window.recording_id)].append((window, value))
    candidates: list[TemporalEvent] = []
    for key, entries in grouped.items():
        entries.sort(key=lambda item: item[0].start_sec)
        previous_start = -1.0
        current_start: float | None = None
        current_end = 0.0
        current_score = 0.0
        for window, score in entries:
            if window.start_sec < previous_start:
                raise ValueError(f"Unsorted input windows for {key}")
            previous_start = window.start_sec
            if score < threshold:
                continue
            if current_start is None:
                current_start, current_end, current_score = window.start_sec, window.end_sec, score
            elif window.start_sec <= current_end + gap_tolerance_sec:
                current_end = max(current_end, window.end_sec)
                current_score = max(current_score, score)
            else:
                candidates.append(TemporalEvent(*key, current_start, current_end, current_score))
                current_start, current_end, current_score = window.start_sec, window.end_sec, score
        if current_start is not None:
            candidates.append(TemporalEvent(*key, current_start, current_end, current_score))
    candidates = [event for event in candidates if event.duration_sec >= minimum_event_duration_sec]
    accepted: list[TemporalEvent] = []
    for event in sorted(candidates):
        previous = accepted[-1] if accepted and accepted[-1].key == event.key else None
        if previous is not None and event.start_sec < previous.end_sec + refractory_period_sec:
            continue
        accepted.append(event)
    return accepted


def event_overlap(first: TemporalEvent, second: TemporalEvent) -> float:
    if first.key != second.key:
        return 0.0
    return max(0.0, min(first.end_sec, second.end_sec) - max(first.start_sec, second.start_sec))


def match_events(
    predictions: Iterable[TemporalEvent], references: Iterable[TemporalEvent]
) -> list[tuple[TemporalEvent, TemporalEvent]]:
    predicted = sorted(predictions)
    reference = sorted(references)
    used: set[int] = set()
    matches: list[tuple[TemporalEvent, TemporalEvent]] = []
    for prediction in predicted:
        choices = [
            (event_overlap(prediction, target), index, target)
            for index, target in enumerate(reference)
            if index not in used and event_overlap(prediction, target) > 0
        ]
        if choices:
            _, index, target = max(choices, key=lambda item: item[0])
            used.add(index)
            matches.append((prediction, target))
    return matches


def event_level_sensitivity(
    predictions: Iterable[TemporalEvent], references: Iterable[TemporalEvent]
) -> float | None:
    predicted, reference = list(predictions), list(references)
    if not reference:
        return None
    return len(match_events(predicted, reference)) / len(reference)


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    sorted_intervals = sorted(intervals)
    if not sorted_intervals:
        return 0.0
    total = 0.0
    start, end = sorted_intervals[0]
    for next_start, next_end in sorted_intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def false_positives_per_hour(
    predictions: Iterable[TemporalEvent],
    references: Iterable[TemporalEvent],
    monitoring_intervals: Iterable[TemporalEvent],
) -> dict[str, float | int | None]:
    """Count unmatched merged alarm events per non-seizure monitoring hour."""

    predicted, reference, monitoring = list(predictions), list(references), list(monitoring_intervals)
    matched_prediction_ids = {id(prediction) for prediction, _ in match_events(predicted, reference)}
    false_alarms = sum(id(event) not in matched_prediction_ids for event in predicted)
    by_key_monitor: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    by_key_seizure: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for event in monitoring:
        by_key_monitor[event.key].append((event.start_sec, event.end_sec))
    for event in reference:
        by_key_seizure[event.key].append((event.start_sec, event.end_sec))
    monitoring_seconds = sum(_union_duration(values) for values in by_key_monitor.values())
    seizure_seconds = 0.0
    for key, seizures in by_key_seizure.items():
        intersections: list[tuple[float, float]] = []
        for seizure_start, seizure_end in seizures:
            intersections.extend(
                [
                (max(seizure_start, start), min(seizure_end, end))
                for start, end in by_key_monitor.get(key, [])
                if min(seizure_end, end) > max(seizure_start, start)
                ]
            )
        seizure_seconds += _union_duration(intersections)
    nonseizure_hours = max(0.0, monitoring_seconds - seizure_seconds) / 3600.0
    return {
        "false_alarm_events": false_alarms,
        "nonseizure_monitoring_hours": nonseizure_hours,
        "false_positives_per_hour": false_alarms / nonseizure_hours if nonseizure_hours > 0 else None,
    }
