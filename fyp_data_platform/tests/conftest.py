from __future__ import annotations

import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from fyp_data_platform.config.settings import PROJECT_ROOT, Settings
from fyp_data_platform.db.session import create_engine_and_session


@pytest.fixture()
def bids_root(tmp_path: Path) -> Path:
    root = tmp_path / "ds005873"
    eeg = root / "sub-001" / "ses-01" / "eeg"
    eeg.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "SeizeIT2", "BIDSVersion": "1.9.0", "License": "CC0"}),
        encoding="utf-8",
    )
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\nsub-001\t42\tF\nsub-002\t50\tM\n", encoding="utf-8"
    )
    prefix = "sub-001_ses-01_task-szMonitoring_run-01"
    (eeg / f"{prefix}_events.tsv").write_text(
        "onset\tduration\teventType\tlateralization\tlocalization\tvigilance\tconfidence\tchannels\tdateTime\trecordingDuration\n"
        "0\t10\tbckg\tn/a\tn/a\tawake\t1\tn/a\t2024-01-01T00:00:00\t60\n"
        "10\t5\tsz-foc\tleft\ttemporal\tawake\t0.9\tT3\t2024-01-01T00:00:10\t60\n"
        "15\t2\tartifact\tn/a\tn/a\tawake\t0.5\tn/a\t2024-01-01T00:00:15\t60\n",
        encoding="utf-8",
    )
    (eeg / f"{prefix}_eeg.json").write_text(
        json.dumps({"RecordingDuration": 60, "TaskName": "szMonitoring"}), encoding="utf-8"
    )
    (eeg / f"{prefix}_eeg.edf").write_text(
        ".git/annex/objects/AA/BB/fake-edf/fake-edf", encoding="utf-8"
    )
    other = root / "sub-002" / "ses-01" / "eeg"
    other.mkdir(parents=True)
    (other / "sub-002_ses-01_task-szMonitoring_run-01_eeg.json").write_text(
        json.dumps({"RecordingDuration": 30}), encoding="utf-8"
    )
    return root


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        storage_backend="local",
        local_storage_root=tmp_path / "objects",
        database_url=f"sqlite:///{(tmp_path / 'catalog.sqlite').as_posix()}",
        s3_bucket=None,
        s3_endpoint_url=None,
        aws_region=None,
        s3_prefix="",
        aws_profile=None,
        s3_server_side_encryption="AES256",
        s3_kms_key_id=None,
        model_project_path=PROJECT_ROOT.parent / "FYPModels",
        model_python=None,
        work_root=tmp_path / "work",
        log_level="INFO",
    )


@pytest.fixture()
def migrated_db(settings: Settings):
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = settings.database_url
    command.upgrade(config, "head")
    engine, factory = create_engine_and_session(settings.database_url)
    try:
        yield engine, factory
    finally:
        engine.dispose()
