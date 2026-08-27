from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from fyp_data_platform.db.models import ModelRun, StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.integrations.seizure_model import SeizureModelIntegration
from fyp_data_platform.storage.local import LocalStorage


def test_model_run_stages_inputs_uploads_results_and_updates_catalog(
    tmp_path: Path, settings, migrated_db, monkeypatch
) -> None:
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    video = tmp_path / "small.mp4"
    labels = tmp_path / "labels.xlsx"
    video.write_bytes(b"video")
    labels.write_bytes(b"labels")
    video_meta = storage.put_file(video, "datasets/test/source/video.mp4")
    label_meta = storage.put_file(labels, "datasets/test/source/labels.xlsx")
    with factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset("test", name="Test", version=None, description=None, source_url=None, license=None, metadata_json={})
        patient, _ = repo.patient("test", "p1", metadata_json={})
        recording, _ = repo.recording(
            "test", "r1", patient_id=patient.id, session_id=None, task=None,
            modality="video", duration_sec=None, video_fps=None, start_datetime=None, metadata_json={},
        )
        for modality, relative, key, meta in (
            ("video", "manual/p1/r1/video.mp4", "datasets/test/source/video.mp4", video_meta),
            ("labels", "manual/p1/r1/labels.xlsx", "datasets/test/source/labels.xlsx", label_meta),
        ):
            repo.stored_file(
                "test", relative, recording_id=recording.id, role="source-media", modality=modality,
                storage_backend="local", storage_key=key, payload_state="stored",
                size_bytes=meta.size_bytes, sha256=meta.sha256, media_type=None, metadata_json={},
            )

    def fake_run(command, cwd, check):
        config_path = Path(command[command.index("--config") + 1])
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        result_dir = Path(config["testing"]["output_directory"]) / "video_only"
        result_dir.mkdir(parents=True)
        (result_dir / "metrics.json").write_text(json.dumps({"sensitivity": 0.8}), encoding="utf-8")

    monkeypatch.setattr("fyp_data_platform.integrations.seizure_model.subprocess.run", fake_run)
    result = SeizureModelIntegration(settings, storage, factory).run("test", "r1", mode="video")

    assert result.status == "completed"
    assert result.artifacts[0]["relative_path"] == "video_only/metrics.json"
    with factory() as session:
        run = session.get(ModelRun, result.run_id)
        outputs = session.scalars(select(StoredFile).where(StoredFile.role == "model-output")).all()
    assert run is not None and run.status == "completed"
    assert len(outputs) == 1


def test_video_model_run_allows_prediction_without_labels(
    tmp_path: Path, settings, migrated_db, monkeypatch
) -> None:
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    video = tmp_path / "small.mp4"
    video.write_bytes(b"video")
    video_meta = storage.put_file(video, "datasets/test/source/video-only.mp4")
    with factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset("test", name="Test", version=None, description=None, source_url=None, license=None, metadata_json={})
        patient, _ = repo.patient("test", "p1", metadata_json={})
        recording, _ = repo.recording(
            "test", "r2", patient_id=patient.id, session_id=None, task=None,
            modality="video", duration_sec=None, video_fps=None, start_datetime=None, metadata_json={},
        )
        repo.stored_file(
            "test", "manual/p1/r2/video-only.mp4", recording_id=recording.id,
            role="source-media", modality="video", storage_backend="local",
            storage_key="datasets/test/source/video-only.mp4", payload_state="stored",
            size_bytes=video_meta.size_bytes, sha256=video_meta.sha256,
            media_type=None, metadata_json={},
        )

    def fake_run(command, cwd, check):
        config_path = Path(command[command.index("--config") + 1])
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["data"]["video_labels"] is None
        result_dir = Path(config["testing"]["output_directory"]) / "video_only"
        result_dir.mkdir(parents=True)
        (result_dir / "metrics.json").write_text(
            json.dumps({"status": "completed", "prediction_only": True}), encoding="utf-8"
        )

    monkeypatch.setattr("fyp_data_platform.integrations.seizure_model.subprocess.run", fake_run)
    result = SeizureModelIntegration(settings, storage, factory).run("test", "r2", mode="video")

    assert result.status == "completed"
    assert result.artifacts[0]["relative_path"] == "video_only/metrics.json"


def test_video_model_run_can_require_labels_for_development_mode(
    tmp_path: Path, settings, migrated_db
) -> None:
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    video = tmp_path / "small.mp4"
    video.write_bytes(b"video")
    video_meta = storage.put_file(video, "datasets/test/source/video-missing-label.mp4")
    with factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset("test", name="Test", version=None, description=None, source_url=None, license=None, metadata_json={})
        patient, _ = repo.patient("test", "p1", metadata_json={})
        recording, _ = repo.recording(
            "test", "r3", patient_id=patient.id, session_id=None, task=None,
            modality="video", duration_sec=None, video_fps=None, start_datetime=None, metadata_json={},
        )
        repo.stored_file(
            "test", "manual/p1/r3/video-missing-label.mp4", recording_id=recording.id,
            role="source-media", modality="video", storage_backend="local",
            storage_key="datasets/test/source/video-missing-label.mp4", payload_state="stored",
            size_bytes=video_meta.size_bytes, sha256=video_meta.sha256,
            media_type=None, metadata_json={},
        )

    try:
        SeizureModelIntegration(settings, storage, factory).run(
            "test", "r3", mode="video", require_labels=True
        )
    except LookupError as error:
        assert "labels" in str(error)
    else:
        raise AssertionError("Development video run should require registered labels")
