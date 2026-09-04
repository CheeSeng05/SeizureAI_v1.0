"""Analysis for TLE facial-expression *recognition* rating experiments.

This is deliberately separate from :mod:`seizure_fusion.face_emotion`.  The
image model estimates the expression visible in an image; this module measures
how a participant attributes emotion to standardized faces, following the
independent 0--5 ratings used by Yamada et al. (2005), DOI:
10.1016/j.neuropsychologia.2004.06.019.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


RATED_EMOTIONS = ("happiness", "sadness", "fear", "anger", "disgust", "surprise")
STIMULUS_EMOTIONS = RATED_EMOTIONS + ("neutral",)
NEGATIVE_EMOTIONS = frozenset(("sadness", "fear", "anger"))
_ALIASES = {"happy": "happiness", "sad": "sadness", "angry": "anger",
            **{name: name for name in STIMULUS_EMOTIONS}}
REQUIRED_COLUMNS = frozenset(
    ("participant_id", "cohort", "session", "stimulus_emotion", "rated_emotion", "rating")
)


def _emotion(value: str, allowed: tuple[str, ...], field: str) -> str:
    emotion = _ALIASES.get(value.strip().lower())
    if emotion not in allowed:
        raise ValueError(f"Unknown {field}={value!r}; expected one of {', '.join(allowed)}")
    return emotion


def read_ratings(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Rating CSV is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def analyze_ratings(rows: Iterable[dict[str, str]]) -> dict:
    """Return participant/session bias summaries and control-referenced z-scores.

    Repeated trials of the same participant/session/stimulus/rated-emotion cell
    are averaged first so that participants, rather than trials, are the unit of
    normative comparison.
    """
    cells: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        participant = row.get("participant_id", "").strip()
        cohort = row.get("cohort", "").strip().lower()
        session = row.get("session", "").strip().lower()
        if not participant or not cohort or not session:
            raise ValueError(f"Row {row_number}: participant_id, cohort, and session are required")
        stimulus = _emotion(row.get("stimulus_emotion", ""), STIMULUS_EMOTIONS, "stimulus_emotion")
        rated = _emotion(row.get("rated_emotion", ""), RATED_EMOTIONS, "rated_emotion")
        try:
            rating = float(row.get("rating", ""))
        except ValueError as error:
            raise ValueError(f"Row {row_number}: rating must be numeric") from error
        if not np.isfinite(rating) or not 0 <= rating <= 5:
            raise ValueError(f"Row {row_number}: rating must be between 0 and 5")
        cells[(participant, cohort, session, stimulus, rated)].append(rating)

    means = {key: float(np.mean(values)) for key, values in cells.items()}
    controls: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, cohort, _, stimulus, rated), value in means.items():
        if cohort == "control":
            controls[(stimulus, rated)].append(value)

    normative = {}
    for (stimulus, rated), values in controls.items():
        array = np.asarray(values, dtype=float)
        normative[f"{stimulus}:{rated}"] = {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
            "n": len(array),
        }

    sessions: dict[tuple[str, str, str], list[tuple[str, str, float]]] = defaultdict(list)
    for (participant, cohort, session, stimulus, rated), value in means.items():
        sessions[(participant, cohort, session)].append((stimulus, rated, value))

    summaries = []
    for (participant, cohort, session), values in sorted(sessions.items()):
        matched = [value for stimulus, rated, value in values if stimulus == rated]
        cross = [value for stimulus, rated, value in values if stimulus != rated]
        negative_cross = [value for stimulus, rated, value in values
                          if rated in NEGATIVE_EMOTIONS and stimulus != rated]
        neutral_negative = [value for stimulus, rated, value in values
                            if stimulus == "neutral" and rated in NEGATIVE_EMOTIONS]
        z_scores = {}
        if cohort != "control":
            for stimulus, rated, value in values:
                reference = normative.get(f"{stimulus}:{rated}")
                if reference and reference["sd"] not in (None, 0.0):
                    z_scores[f"{stimulus}:{rated}"] = (value - reference["mean"]) / reference["sd"]
        summaries.append({
            "participant_id": participant,
            "cohort": cohort,
            "session": session,
            "matched_mean": float(np.mean(matched)) if matched else None,
            "cross_emotion_mean": float(np.mean(cross)) if cross else None,
            "negative_cross_emotion_mean": float(np.mean(negative_cross)) if negative_cross else None,
            "negative_neutral_mean": float(np.mean(neutral_negative)) if neutral_negative else None,
            "control_referenced_z_scores": z_scores,
        })
    return {"scale": [0, 5], "normative_cells": normative, "session_summaries": summaries}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze TLE facial-expression intensity ratings")
    parser.add_argument("--ratings", required=True, help="Long-format rating CSV")
    parser.add_argument("--output", help="Optional JSON output path (stdout when omitted)")
    args = parser.parse_args()
    result = analyze_ratings(read_ratings(args.ratings))
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
