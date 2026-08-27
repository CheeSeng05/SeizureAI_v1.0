from __future__ import annotations

from sqlalchemy import func, select

from fyp_data_platform.db.models import Prediction, TemporalWindow
from fyp_data_platform.db.repository import CatalogRepository


def test_window_id_is_the_prediction_fusion_key(migrated_db) -> None:
    _, factory = migrated_db
    window_id = "sub-001__0000000000__0000003000"
    with factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset(
            "test",
            name="Test",
            version="1",
            description=None,
            source_url=None,
            license=None,
            metadata_json={},
        )
        patient, _ = repo.patient("test", "sub-001", metadata_json={})
        recording, _ = repo.recording(
            "test",
            "rec-001",
            patient_id=patient.id,
            session_id="ses-01",
            task=None,
            modality="video",
            duration_sec=10,
            video_fps=25,
            start_datetime=None,
            metadata_json={},
        )
        repo.temporal_window(
            window_id,
            dataset_id="test",
            recording_id=recording.id,
            patient_id=patient.id,
            start_sec=0,
            end_sec=3,
            label="normal",
            label_id=0,
            split="train",
            feature_file_id=None,
            metadata_json={},
        )
        repo.model_run(
            "run-001",
            name="Body baseline",
            model_type="lstm",
            model_version="1",
            modality="body",
            status="completed",
            git_commit=None,
            config_json={},
            metrics_json={},
            artifact_file_id=None,
            started_at=None,
            completed_at=None,
        )
        first, created = repo.prediction(
            "run-001",
            window_id,
            predicted_label="normal",
            confidence=0.8,
            probabilities_json={"normal": 0.8, "ictal": 0.2},
            is_correct=True,
        )
        second, created_again = repo.prediction(
            "run-001",
            window_id,
            predicted_label="normal",
            confidence=0.9,
            probabilities_json={"normal": 0.9, "ictal": 0.1},
            is_correct=True,
        )
        assert created
        assert not created_again
        assert first.id == second.id

    with factory() as session:
        assert session.scalar(select(func.count()).select_from(TemporalWindow)) == 1
        assert session.scalar(select(func.count()).select_from(Prediction)) == 1
        assert session.scalar(select(Prediction.window_id)) == window_id
