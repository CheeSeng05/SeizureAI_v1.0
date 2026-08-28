from __future__ import annotations

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
    assert catalog.get_json() == {"videos": [], "runs": []}


def test_upload_requires_a_video(settings, migrated_db, monkeypatch) -> None:
    engine, factory = migrated_db
    monkeypatch.setattr(ui, "load_settings", lambda _: settings)
    monkeypatch.setattr(ui, "create_engine_and_session", lambda _: (engine, factory))
    monkeypatch.setattr(ui, "create_storage", lambda _: LocalStorage(settings.local_storage_root))

    response = ui.create_app().test_client().post("/api/uploads", data={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Choose at least one video file."
