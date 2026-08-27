from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping


def metrics_by_group(
    records: Iterable[Any],
    *,
    group_key: Callable[[Any], str],
    metric: Callable[[list[Any]], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Generate per-patient or per-seizure result dictionaries."""

    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    return {key: dict(metric(values)) for key, values in sorted(grouped.items())}
