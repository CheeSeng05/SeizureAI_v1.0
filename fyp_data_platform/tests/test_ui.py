from __future__ import annotations

from io import BytesIO

from openpyxl import Workbook
import pytest

from fyp_data_platform import ui
from fyp_data_platform.storage.local import LocalStorage


def test_health_and_empty_catalog_use_platform_database(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))

    client = ui.create_app().test_client()
    health = client.get("/api/health")
    catalog = client.get("/api/catalog")

    assert health.status_code == 200
    assert health.get_json()["counts"] == {
        "datasets": 0, "patients": 0, "recordings": 0, "files": 0, "runs": 0
    }
    assert health.get_json()["compute"]["mode"] in {"cpu", "cuda"}
    assert "pytorch_version" in health.get_json()["compute"]
    assert catalog.status_code == 200
    assert catalog.get_json() == {"recordings": [], "runs": []}


def test_upload_requires_a_video(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))

    response = ui.create_app().test_client().post("/api/uploads", data={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Choose at least one video file."


def test_upload_registers_exactly_matched_video_labels(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["PatID", "Seizure Type", "#Seizure", "EEG onset", "Clinical Onset"])
    sheet.append(["Pat01", "PG", "Sz2", 10 / 86400, 15 / 86400])
    label_bytes = BytesIO()
    workbook.save(label_bytes)
    label_bytes.seek(0)

    client = ui.create_app().test_client()
    response = client.post(
        "/api/uploads",
        data={
            "videos": (BytesIO(b"video"), "Sz2PG.mp4"),
            "labels": (label_bytes, "Label.xlsx"),
            "dataset_id": "ui-videos", "dataset_name": "UI videos",
            "patient_id": "Pat01", "recording_id": "Sz2PG",
            "seizure_id": "Sz2", "seizure_type": "PG",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    assert response.get_json()["labels"][0]["identity"]["clinical_onset_sec"] == pytest.approx(15)
    recording = client.get("/api/catalog").get_json()["recordings"][0]
    assert recording["video"]["filename"] == "Sz2PG.mp4"
    assert recording["label"]["identity"]["seizure_id"] == "Sz2"


def _eeg_csv(*, include_y: bool = True) -> bytes:
    columns = ["Unnamed", *[f"X{index}" for index in range(1, 179)]]
    values = ["sample-1", *[str(index / 10) for index in range(1, 179)]]
    if include_y:
        columns.append("y")
        values.append("1")
    return (",".join(columns) + "\n" + ",".join(values) + "\n").encode()


def test_upload_registers_eeg_csv(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))
    client = ui.create_app().test_client()

    response = client.post(
        "/api/uploads",
        data={
            "mode": "eeg", "eeg": (BytesIO(_eeg_csv()), "patient.csv"),
            "dataset_id": "ui-recordings", "patient_id": "Pat01",
            "recording_id": "eeg-001",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    assert response.get_json()["eeg"]["validation"]["row_count"] == 1
    recording = client.get("/api/catalog").get_json()["recordings"][0]
    assert recording["video"] is None
    assert recording["eeg"]["filename"] == "patient.csv"


def test_upload_rejects_eeg_without_ground_truth(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))

    response = ui.create_app().test_client().post(
        "/api/uploads",
        data={"mode": "eeg", "eeg": (BytesIO(_eeg_csv(include_y=False)), "patient.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "EEG CSV must contain the ground-truth column y."


def test_both_mode_requires_shared_recording_id(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))

    response = ui.create_app().test_client().post(
        "/api/uploads",
        data={
            "mode": "unimodal", "videos": (BytesIO(b"video"), "video.mp4"),
            "eeg": (BytesIO(_eeg_csv()), "eeg.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Recording ID is required when uploading video and EEG together."


def test_run_rejects_unknown_mode(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))

    response = ui.create_app().test_client().post(
        "/api/runs", json={"dataset_id": "data", "recording_id": "recording", "mode": "fusion"}
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Run mode must be video, eeg, or unimodal."
