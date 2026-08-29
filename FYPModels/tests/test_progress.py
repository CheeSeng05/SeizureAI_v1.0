from __future__ import annotations

import json

import pytest

from seizure_multimodal.progress import report_progress


def test_progress_snapshot_contains_percentage_and_eta(tmp_path) -> None:
    path = tmp_path / "progress.json"
    report_progress(
        {"testing": {"progress_file": str(path)}},
        stage="video_inference", label="Processing video clips",
        completed=25, total=100, elapsed_sec=10,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["percent"] == 25
    assert payload["estimated_remaining_sec"] == pytest.approx(30)
    assert payload["stage_label"] == "Processing video clips"
