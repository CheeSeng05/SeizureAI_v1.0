from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from fyp_data_platform.config.settings import Settings
from fyp_data_platform.db.models import Annotation, Dataset, Patient, Recording, StoredFile
from fyp_data_platform.ingest.service import DatasetSpec, ingest_seizeit2
from fyp_data_platform.storage.local import LocalStorage


def test_metadata_ingest_is_idempotent(
    bids_root: Path, settings: Settings, migrated_db
) -> None:
    _, factory = migrated_db
    arguments = dict(
        settings=settings,
        storage=LocalStorage(settings.local_storage_root),
        session_factory=factory,
        source=bids_root,
        spec=DatasetSpec(),
        subjects={"sub-001"},
    )
    first = ingest_seizeit2(**arguments)
    second = ingest_seizeit2(**arguments)

    assert first.created_patients == 1
    assert first.created_recordings == 1
    assert first.created_annotations == 2
    assert first.excluded_event_counts == {"artifact": 1}
    assert second.created_patients == 0
    assert second.created_recordings == 0
    assert second.created_file_rows == 0
    assert second.created_annotations == 0

    with factory() as session:
        counts = {
            model.__tablename__: session.scalar(select(func.count()).select_from(model))
            for model in (Dataset, Patient, Recording, StoredFile, Annotation)
        }
    assert counts == {
        "datasets": 1,
        "patients": 1,
        "recordings": 1,
        "files": 5,
        "annotations": 2,
    }
    edf = next(settings.local_storage_root.rglob("*.edf"), None)
    assert edf is None


def test_dry_run_needs_no_database_or_storage(
    bids_root: Path, settings: Settings, tmp_path: Path
) -> None:
    from fyp_data_platform.db.session import create_engine_and_session

    engine, factory = create_engine_and_session(f"sqlite:///{(tmp_path / 'not-migrated.sqlite').as_posix()}")
    try:
        report = ingest_seizeit2(
            settings=settings,
            storage=None,
            session_factory=factory,
            source=bids_root,
            spec=DatasetSpec(),
            subjects={"sub-001"},
            dry_run=True,
        )
    finally:
        engine.dispose()
    assert report.annotation_rows == 2
    assert report.dry_run
    assert not settings.local_storage_root.exists()
