from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_RUN_FIELDS = (
    "experiment_id",
    "dataset",
    "dataset_version",
    "patient_split",
    "random_seed",
    "video_checkpoint",
    "eeg_checkpoint",
    "fusion_configuration",
    "window_duration_sec",
    "window_stride_sec",
    "prediction_threshold",
    "synchronization_configuration",
)


def build_run_record(values: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_RUN_FIELDS if field not in values]
    if missing:
        raise ValueError(f"Run metadata is missing fields: {missing}")
    return {**dict(values), "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def save_run_record(record: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), indent=2) + "\n", encoding="utf-8")
