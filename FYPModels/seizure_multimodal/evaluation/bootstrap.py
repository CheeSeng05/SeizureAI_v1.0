from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping

import numpy as np


def patient_bootstrap_confidence_intervals(
    records: Iterable[Any],
    *,
    patient_id: Callable[[Any], str],
    metric: Callable[[list[Any]], Mapping[str, float | None]],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float] | None]:
    """Percentile CIs resampling patients, not correlated temporal windows."""

    if not 0 < confidence < 1 or n_resamples <= 0:
        raise ValueError("Invalid bootstrap configuration")
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[patient_id(record)].append(record)
    patients = sorted(grouped)
    if not patients:
        raise ValueError("Patient bootstrap requires records")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_resamples):
        selected = rng.choice(patients, size=len(patients), replace=True)
        resample = [record for patient in selected for record in grouped[str(patient)]]
        for name, value in metric(resample).items():
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    tail = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float] | None] = {}
    for name, values in samples.items():
        result[name] = (
            {
                "lower": float(np.quantile(values, tail)),
                "upper": float(np.quantile(values, 1.0 - tail)),
                "confidence": confidence,
                "method": "patient-level percentile bootstrap",
            }
            if values
            else None
        )
    return result
