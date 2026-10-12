from __future__ import annotations

import csv
from contextlib import nullcontext
import struct
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seizure_multimodal.synchronization.video_segments import (
    NICOLET_VIDEO_SYNC_HEADER_BYTES,
    NICOLET_VIDEO_SYNC_RECORD_BYTES,
    NicoletVideoSyncRecord,
    OLE_EPOCH,
    VideoSegment,
    build_video_segment_manifest,
    fuse_prediction_timelines,
    parse_nicolet_video_sync_payload,
    video_rows_on_global_timeline,
)
from seizure_multimodal.testing import (
    WorkflowResult,
    run_fusion_workflow,
    run_video_segments_workflow,
)


def _sync_payload(start: datetime, rows: list[tuple[float, str]]) -> bytes:
    payload = bytearray(NICOLET_VIDEO_SYNC_HEADER_BYTES)
    for offset_sec, filename in rows:
        record = bytearray(NICOLET_VIDEO_SYNC_RECORD_BYTES)
        value = start + timedelta(seconds=offset_sec)
        ole_days = (value - OLE_EPOCH).total_seconds() / 86400.0
        whole_days = float(int(ole_days))
        fractional_seconds = (ole_days - int(ole_days)) * 86400.0
        struct.pack_into("<d", record, 8, whole_days)
        struct.pack_into("<d", record, 16, fractional_seconds)
        encoded = filename.encode("utf-16-le")
        record[24:24 + len(encoded)] = encoded
        payload.extend(record)
    return bytes(payload)


def _config(**fusion):
    return {"fusion": {
        "eeg_weight": 0.8,
        "video_weight": 0.2,
        "threshold": 0.5,
        "minimum_video_coverage": 1.0,
        **fusion,
    }}


def test_nicolet_video_sync_payload_preserves_eeg_clock_and_filename() -> None:
    start = datetime(2026, 1, 2, 3, 4, 5)
    records = parse_nicolet_video_sync_payload(
        _sync_payload(start, [(4.416, "segment-a.avi"), (64.420, "segment-a.avi"), (905.752, "segment-b.avi")]),
        recording_start=start,
    )
    assert [row.filename for row in records] == ["segment-a.avi", "segment-a.avi", "segment-b.avi"]
    np.testing.assert_allclose([row.global_time_sec for row in records], [4.416, 64.420, 905.752], atol=1e-5)


def test_manifest_uses_embedded_names_and_times_not_upload_order(tmp_path: Path, monkeypatch) -> None:
    eeg = tmp_path / "recording.e"
    eeg.write_bytes(b"synthetic")
    first = tmp_path / "z-segment.avi"
    second = tmp_path / "a-segment.avi"
    first.write_bytes(b"video")
    second.write_bytes(b"video")
    header = SimpleNamespace(
        startDateTime=datetime(2026, 1, 1),
        Events=[
            SimpleNamespace(IDStr="Video Start", date=datetime(2026, 1, 1) + timedelta(seconds=10)),
            SimpleNamespace(IDStr="Video Start", date=datetime(2026, 1, 1) + timedelta(seconds=50)),
        ],
    )
    monkeypatch.setattr(
        "seizure_multimodal.synchronization.video_segments.read_nicolet_video_sync",
        lambda _path: ([
            NicoletVideoSyncRecord("z-segment.avi", 10.0),
            NicoletVideoSyncRecord("a-segment.avi", 50.0),
        ], header),
    )

    manifest = build_video_segment_manifest(
        eeg_path=eeg,
        video_paths=[second, first],
        recording_id="recording",
        video_inspector=lambda path: SimpleNamespace(
            duration_sec=20.0 if Path(path).name == "z-segment.avi" else 30.0,
            fps=25.0,
        ),
    )

    assert [item.filename for item in manifest] == ["z-segment.avi", "a-segment.avi"]
    assert [(item.global_start_sec, item.global_end_sec) for item in manifest] == [
        (10.0, 30.0), (50.0, 80.0),
    ]


def test_video_local_time_maps_to_global_eeg_time() -> None:
    segment = VideoSegment(
        "segment-0001", "segment.avi", "segment.avi", 320.5, 420.5, 100.0, 30.0,
    )
    rows = video_rows_on_global_timeline(
        [{"start_sec": 15.0, "end_sec": 16.0, "video_probability": 0.75}],
        segment,
        recording_id="recording",
    )
    assert rows[0]["local_start_sec"] == 15.0
    assert rows[0]["start_sec"] == 335.5
    assert rows[0]["end_sec"] == 336.5


def test_overlap_weighted_alignment_and_80_20_fusion() -> None:
    eeg = [{"window_id": "eeg-1", "start_sec": 0, "end_sec": 4, "probability": 0.9}]
    video = [
        {"start_sec": 0, "end_sec": 1, "video_probability": 0.2, "video_segment_id": "a"},
        {"start_sec": 1, "end_sec": 2, "video_probability": 0.4, "video_segment_id": "a"},
        {"start_sec": 2, "end_sec": 3, "video_probability": 0.6, "video_segment_id": "b"},
        {"start_sec": 3, "end_sec": 4, "video_probability": 0.8, "video_segment_id": "b"},
    ]
    row = fuse_prediction_timelines(eeg_rows=eeg, video_rows=video, recording_id="r", config=_config())[0]
    assert row["video_probability"] == pytest.approx(0.5)
    assert row["video_coverage"] == pytest.approx(1.0)
    assert row["fused_probability"] == pytest.approx(0.82)
    assert row["fused_prediction"] == 1
    assert row["video_segment_ids"] == "a|b"


def test_alignment_uses_timestamps_not_array_index() -> None:
    eeg = [{"window_id": "late", "start_sec": 100, "end_sec": 104, "probability": 0.5}]
    video = [
        {"start_sec": 0, "end_sec": 4, "video_probability": 0.99, "video_segment_id": "wrong"},
        {"start_sec": 102, "end_sec": 104, "video_probability": 0.8, "video_segment_id": "right-b"},
        {"start_sec": 100, "end_sec": 102, "video_probability": 0.2, "video_segment_id": "right-a"},
    ]
    row = fuse_prediction_timelines(eeg_rows=eeg, video_rows=video, recording_id="r", config=_config())[0]
    assert row["video_probability"] == pytest.approx(0.5)
    assert "wrong" not in row["video_segment_ids"]


def test_real_gap_is_not_counted_as_video_coverage() -> None:
    eeg = [{"window_id": "gap", "start_sec": 0, "end_sec": 4, "probability": 0.8}]
    video = [
        {"start_sec": 0, "end_sec": 1, "video_probability": 0.2, "video_segment_id": "a"},
        {"start_sec": 3, "end_sec": 4, "video_probability": 0.8, "video_segment_id": "b"},
    ]
    row = fuse_prediction_timelines(eeg_rows=eeg, video_rows=video, recording_id="r", config=_config())[0]
    assert row["video_coverage"] == pytest.approx(0.5)
    assert row["video_probability"] == pytest.approx(0.5)
    assert row["fusion_available"] is False
    assert row["fused_probability"] is None


def test_partial_coverage_can_be_explicitly_enabled() -> None:
    eeg = [{"window_id": "partial", "start_sec": 0, "end_sec": 4, "probability": 0.4}]
    video = [{"start_sec": 0, "end_sec": 2, "video_probability": 0.9, "video_segment_id": "a"}]
    row = fuse_prediction_timelines(
        eeg_rows=eeg,
        video_rows=video,
        recording_id="r",
        config=_config(minimum_video_coverage=0.5),
    )[0]
    assert row["fusion_available"] is True
    assert row["fused_probability"] == pytest.approx(0.5)
    assert row["fused_prediction"] == 1


@pytest.mark.parametrize("eeg_weight,video_weight", [(0.8, 0.3), (-0.1, 1.1), (1.1, -0.1)])
def test_invalid_fusion_weights_are_rejected(eeg_weight: float, video_weight: float) -> None:
    with pytest.raises(ValueError):
        fuse_prediction_timelines(
            eeg_rows=[{"start_sec": 0, "end_sec": 4, "probability": 0.5}],
            video_rows=[{"start_sec": 0, "end_sec": 4, "video_probability": 0.5}],
            recording_id="r",
            config=_config(eeg_weight=eeg_weight, video_weight=video_weight),
        )


def test_missing_video_never_becomes_zero_probability() -> None:
    row = fuse_prediction_timelines(
        eeg_rows=[{"start_sec": 0, "end_sec": 4, "probability": 0.7}],
        video_rows=[],
        recording_id="r",
        config=_config(),
    )[0]
    assert row["video_probability"] is None
    assert row["fused_probability"] is None
    assert row["fusion_available"] is False


@pytest.mark.parametrize("labelled", [False, True])
def test_real_fusion_workflow_runs_both_frozen_models_and_saves_traceable_rows(
    tmp_path: Path, monkeypatch, labelled,
) -> None:
    eeg_path = tmp_path / "recording.e"
    video_a = tmp_path / "a.avi"
    video_b = tmp_path / "b.avi"
    for path in (eeg_path, video_a, video_b):
        path.write_bytes(b"synthetic")
    manifest = [
        VideoSegment("segment-0001", str(video_a), video_a.name, 0.0, 2.0, 2.0, 30.0),
        VideoSegment("segment-0002", str(video_b), video_b.name, 2.0, 4.0, 2.0, 30.0),
    ]
    monkeypatch.setattr(
        "seizure_multimodal.testing.build_video_segment_manifest",
        lambda **_kwargs: manifest,
    )
    eeg_result = WorkflowResult(
        "eeg", "completed",
        [{"window_id": "eeg-1", "start_sec": 0.0, "end_sec": 4.0, "probability": 0.9}],
        np.asarray([0.9]), None, None, {}, {},
    )
    monkeypatch.setattr("seizure_multimodal.testing.run_eeg_workflow", lambda *_args, **_kwargs: eeg_result)

    def fake_video(_root, _config, source):
        probability = 0.2 if source.source_id == "segment-0001" else 0.8
        return WorkflowResult(
            "video", "completed",
            [{"start_sec": 0.0, "end_sec": 2.0, "video_probability": probability,
              "video_window_id": f"{source.source_id}-window"}],
            np.asarray([probability]), None, None, {}, {},
        )

    monkeypatch.setattr("seizure_multimodal.testing.run_video_workflow", fake_video)
    output = tmp_path / "results"
    config = {
        "training": {"enabled": False},
        "testing": {"output_directory": str(output)},
        "data": {
            "recording_id": "recording",
            "eeg_file": str(eeg_path),
            "video_segments": [str(video_b), str(video_a)],
        },
        "video_encoder": {"checkpoint": "missing-video.pth"},
        "eeg_encoder": {"checkpoint": "missing-eeg.keras"},
        "fusion": {"eeg_weight": 0.8, "video_weight": 0.2, "threshold": 0.5,
                   "minimum_video_coverage": 1.0},
    }

    if labelled:
        labels = tmp_path / "labels.csv"
        labels.write_text("start_sec,end_sec,label\n0,4,1\n", encoding="utf-8")
        config["data"]["video_labels"] = str(labels)
        monkeypatch.setattr("seizure_multimodal.eeg.loaders.load_eeg", lambda _: nullcontext(
            SimpleNamespace(metadata=SimpleNamespace(duration_sec=8.0))
        ))
    result = run_fusion_workflow(tmp_path, config)

    assert result.status == "completed"
    assert result.rows[0]["video_segment_ids"] == "segment-0001|segment-0002"
    assert result.rows[0]["video_probability"] == pytest.approx(0.5)
    assert result.rows[0]["fused_probability"] == pytest.approx(0.82)
    assert result.metrics["feature_fusion_used"] is False
    assert result.metrics["prediction_only"] is (not labelled)
    if labelled:
        assert result.rows[0]["reference_label"] == 1
        assert result.metrics["true_positives"] == 1
        assert result.metrics["accuracy"] == 1
    with (output / "fusion" / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        saved = next(csv.DictReader(stream))
    assert saved["recording_id"] == "recording"
    assert saved["video_window_ids"] == "segment-0001-window|segment-0002-window"


@pytest.mark.parametrize("offsets,match", [
    ({}, "requires a start time"),
    ({"video.avi": -1}, "Invalid EDF"),
    ({"video.avi": float('nan')}, "Invalid EDF"),
    ({"video.avi": 9}, "extends beyond"),
])
def test_edf_manifest_rejects_invalid_timing(tmp_path, offsets, match):
    with pytest.raises(ValueError, match=match):
        build_video_segment_manifest(
            eeg_path=tmp_path / "recording.edf", video_paths=[tmp_path / "video.avi"],
            recording_id="edf", video_start_offsets=offsets, eeg_duration_sec=10,
            video_inspector=lambda _: SimpleNamespace(duration_sec=4, fps=25),
        )


def test_edf_manifest_preserves_human_declared_offset(tmp_path):
    segments = build_video_segment_manifest(
        eeg_path=tmp_path / "recording.edf", video_paths=[tmp_path / "video.avi"],
        recording_id="edf", video_start_offsets={"video.avi": 2.5}, eeg_duration_sec=10,
        video_inspector=lambda _: SimpleNamespace(duration_sec=4, fps=25),
    )
    assert segments[0].global_start_sec == 2.5
    assert segments[0].global_end_sec == 6.5
    assert segments[0].sync_source == "human_declared_offsets+video_container_duration"


def test_video_only_workflow_processes_every_multimodal_segment(tmp_path: Path, monkeypatch) -> None:
    eeg_path = tmp_path / "recording.e"
    video_a = tmp_path / "a.avi"
    video_b = tmp_path / "b.avi"
    for path in (eeg_path, video_a, video_b):
        path.write_bytes(b"synthetic")
    manifest = [
        VideoSegment("segment-0001", str(video_a), video_a.name, 5.0, 7.0, 2.0, 30.0),
        VideoSegment("segment-0002", str(video_b), video_b.name, 7.0, 9.0, 2.0, 30.0),
    ]
    monkeypatch.setattr(
        "seizure_multimodal.testing.build_video_segment_manifest", lambda **_kwargs: manifest,
    )
    calls: list[str] = []

    def fake_video(_root, _config, source):
        calls.append(source.source_id)
        return WorkflowResult(
            "video", "completed",
            [{"start_sec": 0.0, "end_sec": 2.0, "video_probability": 0.6,
              "video_prediction": 1}],
            np.asarray([0.6]), None, None, {}, {},
        )

    monkeypatch.setattr("seizure_multimodal.testing.run_video_workflow", fake_video)
    output = tmp_path / "results"
    result = run_video_segments_workflow(tmp_path, {
        "testing": {"output_directory": str(output)},
        "data": {"recording_id": "recording", "eeg_file": str(eeg_path),
                 "video_segments": [str(video_a), str(video_b)]},
        "video_encoder": {"checkpoint": "video.pth"},
        "eeg_encoder": {"checkpoint": "eeg.keras"},
    })

    assert result.status == "completed"
    assert calls == ["segment-0001", "segment-0002"]
    assert [row["start_sec"] for row in result.rows] == [5.0, 7.0]
    assert result.metrics["video_segment_count"] == 2
    assert (output / "video_only" / "video_segment_manifest.json").is_file()
