from __future__ import annotations

from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    Annotation,
    Dataset,
    ModelRun,
    Patient,
    Prediction,
    Recording,
    StoredFile,
    TemporalWindow,
)


ModelT = TypeVar("ModelT")


def _upsert(session: Session, model: type[ModelT], identity: dict[str, Any], values: dict[str, Any]) -> tuple[ModelT, bool]:
    instance = session.scalar(select(model).filter_by(**identity))
    created = instance is None
    if instance is None:
        instance = model(**identity, **values)
        session.add(instance)
    else:
        for key, value in values.items():
            setattr(instance, key, value)
    session.flush()
    return instance, created


class CatalogRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def dataset(self, dataset_id: str, **values: Any) -> tuple[Dataset, bool]:
        return _upsert(self.session, Dataset, {"id": dataset_id}, values)

    def patient(self, dataset_id: str, source_patient_id: str, **values: Any) -> tuple[Patient, bool]:
        patient_id = f"{dataset_id}::{source_patient_id}"
        return _upsert(
            self.session,
            Patient,
            {"id": patient_id},
            {"dataset_id": dataset_id, "source_patient_id": source_patient_id, **values},
        )

    def recording(self, dataset_id: str, source_recording_id: str, **values: Any) -> tuple[Recording, bool]:
        recording_id = f"{dataset_id}::{source_recording_id}"
        return _upsert(
            self.session,
            Recording,
            {"id": recording_id},
            {"dataset_id": dataset_id, "source_recording_id": source_recording_id, **values},
        )

    def stored_file(self, dataset_id: str, source_relative_path: str, **values: Any) -> tuple[StoredFile, bool]:
        return _upsert(
            self.session,
            StoredFile,
            {"dataset_id": dataset_id, "source_relative_path": source_relative_path},
            values,
        )

    def annotation(self, annotation_key: str, **values: Any) -> tuple[Annotation, bool]:
        return _upsert(
            self.session,
            Annotation,
            {"annotation_key": annotation_key},
            values,
        )

    def temporal_window(self, window_id: str, **values: Any) -> tuple[TemporalWindow, bool]:
        return _upsert(self.session, TemporalWindow, {"window_id": window_id}, values)

    def model_run(self, model_run_id: str, **values: Any) -> tuple[ModelRun, bool]:
        return _upsert(self.session, ModelRun, {"id": model_run_id}, values)

    def prediction(
        self, model_run_id: str, window_id: str, **values: Any
    ) -> tuple[Prediction, bool]:
        return _upsert(
            self.session,
            Prediction,
            {"model_run_id": model_run_id, "window_id": window_id},
            values,
        )
