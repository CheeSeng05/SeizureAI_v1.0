from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from ..types import WindowRecord


def assert_patient_disjoint(windows: Iterable[WindowRecord]) -> None:
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for window in windows:
        if window.split != "unassigned":
            patient_splits[window.patient_id].add(window.split)
    leaks = {patient: sorted(splits) for patient, splits in patient_splits.items() if len(splits) > 1}
    if leaks:
        raise ValueError(f"Patient leakage across splits: {leaks}")


def assign_patient_splits(
    windows: Sequence[WindowRecord],
    *,
    fractions: Mapping[str, float] | None = None,
    seed: int = 42,
) -> list[WindowRecord]:
    fractions = fractions or {"train": 0.7, "validation": 0.15, "test": 0.15}
    if set(fractions) != {"train", "validation", "test"}:
        raise ValueError("Fractions must define train, validation, and test")
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or any(v < 0 for v in fractions.values()):
        raise ValueError("Split fractions must be non-negative and sum to one")
    boundaries = (
        fractions["train"],
        fractions["train"] + fractions["validation"],
    )
    assignments: dict[str, str] = {}
    for patient in sorted({window.patient_id for window in windows}):
        digest = hashlib.sha256(f"{seed}:{patient}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        assignments[patient] = (
            "train" if value < boundaries[0] else "validation" if value < boundaries[1] else "test"
        )
    result = [replace(window, split=assignments[window.patient_id]) for window in windows]
    assert_patient_disjoint(result)
    return result
