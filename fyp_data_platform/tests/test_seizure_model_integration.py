from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

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
                size_bytes=meta.size_bytes, sha256=meta.sha256, media_type=None,
                metadata_json=(
                    {"label_identity": {"patient_id": "p1", "seizure_id": "Sz1", "seizure_type": "PG"}}
                    if modality == "labels" else {}
                ),
            )

    def fake_run(command, cwd, check, stdout, stderr):
        config_path = Path(command[command.index("--config") + 1])
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["data"]["video_provenance"] == {
            "expected_sha256": video_meta.sha256,
            "source": "fyp_data_platform catalog",
            "patient_id": "p1",
            "seizure_id": "Sz1",
            "seizure_type": "PG",
        }
        result_dir = Path(config["testing"]["output_directory"]) / "video_only"
        result_dir.mkdir(parents=True)
        (result_dir / "metrics.json").write_text(json.dumps({"sensitivity": 0.8}), encoding="utf-8")

    monkeypatch.setattr("fyp_data_platform.integrations.seizure_model.subprocess.run", fake_run)
    progress = []
    result = SeizureModelIntegration(settings, storage, factory).run(
        "test", "r1", mode="video", require_labels=True, progress_callback=progress.append
    )

    assert result.status == "completed"
    assert result.runtime["timings"]["total_sec"] >= 0
    assert result.runtime["timings"]["model_pipeline_sec"] >= 0
    assert {item["relative_path"] for item in result.artifacts} == {"pipeline.log", "video_only/metrics.json"}
    assert [item["stage"] for item in progress] == [
        "input_staging", "input_staging", "configuration", "configuration",
        "model_loading", "artifact_persistence", "artifact_persistence",
    ]
    with factory() as session:
        run = session.get(ModelRun, result.run_id)
        outputs = session.scalars(select(StoredFile).where(StoredFile.role == "model-output")).all()
    assert run is not None and run.status == "completed"
    assert len(outputs) == 2


def test_video_model_run_allows_prediction_without_labels(
    tmp_path: Path, settings, migrated_db, monkeypatch
) -> None:
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    video = tmp_path / "small.mp4"
    video.write_bytes(b"video")
    labels = tmp_path / "attached.csv"
    labels.write_text(
        "recording_id,patient_id,start_sec,end_sec,label\nr2,p1,0,999,1\n",
        encoding="utf-8",
    )
    video_meta = storage.put_file(video, "datasets/test/source/video-only.mp4")
    label_meta = storage.put_file(labels, "datasets/test/source/attached.csv")
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
        repo.stored_file(
            "test", "manual/p1/r2/attached.csv", recording_id=recording.id,
            role="ground-truth", modality="labels", storage_backend="local",
            storage_key="datasets/test/source/attached.csv", payload_state="stored",
            size_bytes=label_meta.size_bytes, sha256=label_meta.sha256,
            media_type=None, metadata_json={"label_identity": {"patient_id": "p1"}},
        )

    def fake_run(command, cwd, check, stdout, stderr):
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
    assert "video_only/metrics.json" in {item["relative_path"] for item in result.artifacts}


def test_legacy_csv_recording_selector_and_subprocess_diagnostics(tmp_path, settings, migrated_db, monkeypatch):
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'video')
    labels = tmp_path / 'labels.csv'
    labels.write_text('recording_id,patient_id,start_sec,end_sec,label\nstitched-test-001,p1,0,5,0\nstitched-test-001,p1,5,10,1\n', encoding='utf-8')
    with factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset('test', name='Test', version=None, description=None, source_url=None, license=None, metadata_json={})
        patient, _ = repo.patient('test', 'p1', metadata_json={})
        recording, _ = repo.recording('test', 'Stitched_Video', patient_id=patient.id, session_id=None, task=None, modality='video', duration_sec=None, video_fps=None, start_datetime=None, metadata_json={})
        for modality, path in [('video', video), ('labels', labels)]:
            key = f'datasets/test/source/{path.name}'
            meta = storage.put_file(path, key)
            repo.stored_file('test', path.name, recording_id=recording.id, role='source-media', modality=modality, storage_backend='local', storage_key=key, payload_state='stored', size_bytes=meta.size_bytes, sha256=meta.sha256, media_type=None, metadata_json={'label_identity': {'patient_id': 'P1'}} if modality == 'labels' else {})

    def fail(command, cwd, check, stdout, stderr):
        import yaml
        config = yaml.safe_load(Path(command[command.index('--config') + 1]).read_text())
        assert config['data']['video_provenance']['recording_id'] == 'stitched-test-001'
        stdout.write(b'Traceback: specific model failure\nValueError: bad test input\n')
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr('fyp_data_platform.integrations.seizure_model.subprocess.run', fail)
    with pytest.raises(RuntimeError, match='bad test input'):
        SeizureModelIntegration(settings, storage, factory).run('test', 'Stitched_Video', require_labels=True)
    with factory() as session:
        run = session.scalar(select(ModelRun))
        assert run.status == 'failed'
        assert 'bad test input' in run.metrics_json['error']
        assert run.metrics_json['artifacts'][0]['relative_path'] == 'pipeline.log'


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


@pytest.mark.parametrize(("filename", "expected_format"), [("recording.edf", "edf"), ("recording.e", "nicolet_e")])
def test_raw_eeg_uses_general_recording_config(
    tmp_path, filename, expected_format, settings, migrated_db
) -> None:
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    staged = tmp_path / filename
    staged.write_bytes(b"synthetic")
    integration = SeizureModelIntegration(settings, storage, factory)

    config = integration._build_config(
        settings.model_project_path.resolve(), tmp_path / "results", {"eeg": staged.resolve()}
    )

    assert config["data"]["eeg_file"] == str(staged.resolve())
    assert config["data"]["eeg_csv"] is None
    assert config["data"]["eeg_format"] == expected_format


def test_csv_eeg_keeps_backward_compatible_config(settings, migrated_db, tmp_path) -> None:
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    staged = tmp_path / "benchmark.csv"
    staged.write_text("X1,y\n0,1\n", encoding="utf-8")
    config = SeizureModelIntegration(settings, storage, factory)._build_config(
        settings.model_project_path.resolve(), tmp_path / "results", {"eeg": staged.resolve()}
    )

    assert config["data"]["eeg_csv"] == str(staged.resolve())
    assert config["data"]["eeg_file"] is None
    assert config["data"]["eeg_format"] == "csv"


def test_managed_raw_eeg_run_routes_to_recording_pipeline(
    tmp_path, settings, migrated_db, monkeypatch
) -> None:
    _, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    source = tmp_path / "recording.e"
    source.write_bytes(b"synthetic nicolet")
    source_meta = storage.put_file(source, "datasets/test/source/recording.e")
    with factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset("test", name="Test", version=None, description=None, source_url=None, license=None, metadata_json={})
        patient, _ = repo.patient("test", "p1", metadata_json={})
        recording, _ = repo.recording(
            "test", "eeg-raw", patient_id=patient.id, session_id=None, task=None,
            modality="eeg", duration_sec=None, video_fps=None, start_datetime=None, metadata_json={},
        )
        repo.stored_file(
            "test", "manual/p1/eeg-raw/recording.e", recording_id=recording.id,
            role="source-data", modality="eeg", storage_backend="local",
            storage_key="datasets/test/source/recording.e", payload_state="stored",
            size_bytes=source_meta.size_bytes, sha256=source_meta.sha256,
            media_type="application/octet-stream", metadata_json={"format": "nicolet-e"},
        )

    def fake_run(command, cwd, check, stdout, stderr):
        assert command[command.index("--mode") + 1] == "eeg"
        config_path = Path(command[command.index("--config") + 1])
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["data"]["eeg_format"] == "nicolet_e"
        assert config["data"]["eeg_file"].endswith("eeg.e")
        output = Path(config["testing"]["output_directory"]) / "eeg_only"
        output.mkdir(parents=True)
        (output / "predictions.csv").write_text(
            "window_id,start_sec,end_sec,probability,predicted_label\nwindow-000001,0,4,0.75,1\n",
            encoding="utf-8",
        )
        (output / "metrics.json").write_text(
            json.dumps({"status": "completed", "metrics": {"prediction_only": True, "window_count": 1}}),
            encoding="utf-8",
        )

    monkeypatch.setattr("fyp_data_platform.integrations.seizure_model.subprocess.run", fake_run)
    result = SeizureModelIntegration(settings, storage, factory).run(
        "test", "eeg-raw", mode="eeg"
    )

    assert result.status == "completed"
    assert "eeg_only/predictions.csv" in {item["relative_path"] for item in result.artifacts}
