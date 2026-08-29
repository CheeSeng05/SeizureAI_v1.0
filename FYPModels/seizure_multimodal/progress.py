from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def report_progress(
    config: Mapping[str, Any], *, stage: str, label: str,
    completed: int | float, total: int | float,
    elapsed_sec: float | None = None,
) -> None:
    """Publish small cross-process progress snapshots for the web orchestrator."""
    testing = config.get("testing", {})
    path_value = testing.get("progress_file") if isinstance(testing, Mapping) else None
    if not path_value:
        return
    total_value = max(float(total), 0.0)
    completed_value = min(max(float(completed), 0.0), total_value) if total_value else 0.0
    percent = completed_value / total_value * 100.0 if total_value else 0.0
    remaining = None
    if elapsed_sec is not None and completed_value > 0 and completed_value < total_value:
        remaining = elapsed_sec / completed_value * (total_value - completed_value)
    payload = {
        "stage": stage, "stage_label": label,
        "completed_units": completed_value, "total_units": total_value,
        "percent": percent, "stage_elapsed_sec": elapsed_sec,
        "estimated_remaining_sec": remaining,
    }
    path = Path(str(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)
