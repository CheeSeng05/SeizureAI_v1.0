from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .checkpoints.provenance import sha256_file


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


def build_preprocessing_record(
    *, pipeline: str, sample_id: str, inputs: Mapping[str, str | Path],
    settings: Mapping[str, Any], source_metadata: Mapping[str, Any],
    software_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Record preprocessing inputs without inventing training/fusion fields.

    Hashes identify the input bytes; they do not verify patient identity or
    guarantee deterministic inference across hardware and library versions.
    """
    return {
        "schema_version": 1,
        "record_type": "preprocessing",
        "pipeline": pipeline,
        "sample_id": sample_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "settings": dict(settings),
        "source_metadata": dict(source_metadata),
        "software_versions": dict(software_versions),
    }


def save_run_record(record: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), indent=2) + "\n", encoding="utf-8")
