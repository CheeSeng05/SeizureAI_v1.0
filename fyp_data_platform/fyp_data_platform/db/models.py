from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(255))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    patients: Mapped[list["Patient"]] = relationship(back_populates="dataset")
    recordings: Mapped[list["Recording"]] = relationship(back_populates="dataset")


class Patient(TimestampMixin, Base):
    __tablename__ = "patients"
    __table_args__ = (UniqueConstraint("dataset_id", "source_patient_id"),)

    id: Mapped[str] = mapped_column(String(220), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    source_patient_id: Mapped[str] = mapped_column(String(100), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="patients")
    recordings: Mapped[list["Recording"]] = relationship(back_populates="patient")


class Recording(TimestampMixin, Base):
    __tablename__ = "recordings"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_recording_id"),
        Index("ix_recordings_patient_session", "patient_id", "session_id"),
    )

    id: Mapped[str] = mapped_column(String(300), primary_key=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    deleting: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    source_recording_id: Mapped[str] = mapped_column(String(200), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(100))
    task: Mapped[str | None] = mapped_column(String(100))
    modality: Mapped[str | None] = mapped_column(String(50))
    duration_sec: Mapped[float | None] = mapped_column(Float)
    video_fps: Mapped[float | None] = mapped_column(Float)
    start_datetime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="recordings")
    patient: Mapped[Patient] = relationship(back_populates="recordings")
    files: Mapped[list["StoredFile"]] = relationship(back_populates="recording")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="recording")


class StoredFile(TimestampMixin, Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_relative_path"),
        UniqueConstraint("storage_backend", "storage_key"),
        Index("ix_files_recording_role", "recording_id", "role"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    recording_id: Mapped[str | None] = mapped_column(ForeignKey("recordings.id"), index=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    modality: Mapped[str | None] = mapped_column(String(50))
    source_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(20), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    payload_state: Mapped[str] = mapped_column(String(30), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64))
    media_type: Mapped[str | None] = mapped_column(String(150))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    recording: Mapped[Recording | None] = relationship(back_populates="files")


class Annotation(TimestampMixin, Base):
    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint("annotation_key"),
        Index("ix_annotations_recording_label_time", "recording_id", "label", "start_sec"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    annotation_key: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id"), nullable=False)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    source_event_type: Mapped[str | None] = mapped_column(String(100))
    lateralization: Mapped[str | None] = mapped_column(String(100))
    localization: Mapped[str | None] = mapped_column(String(255))
    confidence: Mapped[float | None] = mapped_column(Float)
    derived_label: Mapped[str | None] = mapped_column(String(100))
    source_file_id: Mapped[str | None] = mapped_column(ForeignKey("files.id"))
    source_relative_path: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    recording: Mapped[Recording] = relationship(back_populates="annotations")


class TemporalWindow(TimestampMixin, Base):
    __tablename__ = "temporal_windows"
    __table_args__ = (
        UniqueConstraint("window_id"),
        Index("ix_windows_recording_time", "recording_id", "start_sec"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    # Exact deterministic fusion key exported by the synchronized model pipeline.
    window_id: Mapped[str] = mapped_column(String(300), nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id"), nullable=False)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str | None] = mapped_column(String(100), index=True)
    label_id: Mapped[int | None] = mapped_column(Integer)
    split: Mapped[str | None] = mapped_column(String(30), index=True)
    feature_file_id: Mapped[str | None] = mapped_column(ForeignKey("files.id"))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ModelRun(TimestampMixin, Base):
    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    deleting: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str | None] = mapped_column(String(100))
    model_version: Mapped[str | None] = mapped_column(String(100))
    modality: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30), default="created", nullable=False)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifact_file_id: Mapped[str | None] = mapped_column(ForeignKey("files.id"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("model_run_id", "window_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.id"), nullable=False, index=True)
    window_id: Mapped[str] = mapped_column(
        String(300), ForeignKey("temporal_windows.window_id"), nullable=False, index=True
    )
    predicted_label: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    probabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    is_correct: Mapped[bool | None] = mapped_column(Boolean)
