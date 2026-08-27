"""Create the research data catalog.

Revision ID: 20260815_0001
Revises:
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260815_0001"
down_revision = None
branch_labels = None
depends_on = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(100)),
        sa.Column("description", sa.Text()),
        sa.Column("source_url", sa.Text()),
        sa.Column("license", sa.String(255)),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "patients",
        sa.Column("id", sa.String(220), primary_key=True),
        sa.Column("dataset_id", sa.String(100), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("source_patient_id", sa.String(100), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("dataset_id", "source_patient_id"),
    )
    op.create_index("ix_patients_dataset_id", "patients", ["dataset_id"])
    op.create_table(
        "recordings",
        sa.Column("id", sa.String(300), primary_key=True),
        sa.Column("dataset_id", sa.String(100), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("patient_id", sa.String(220), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("source_recording_id", sa.String(200), nullable=False),
        sa.Column("session_id", sa.String(100)),
        sa.Column("task", sa.String(100)),
        sa.Column("modality", sa.String(50)),
        sa.Column("duration_sec", sa.Float()),
        sa.Column("video_fps", sa.Float()),
        sa.Column("start_datetime", sa.DateTime(timezone=True)),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("dataset_id", "source_recording_id"),
    )
    op.create_index("ix_recordings_dataset_id", "recordings", ["dataset_id"])
    op.create_index("ix_recordings_patient_id", "recordings", ["patient_id"])
    op.create_index("ix_recordings_patient_session", "recordings", ["patient_id", "session_id"])
    op.create_table(
        "files",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_id", sa.String(100), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("recording_id", sa.String(300), sa.ForeignKey("recordings.id")),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("modality", sa.String(50)),
        sa.Column("source_relative_path", sa.Text(), nullable=False),
        sa.Column("storage_backend", sa.String(20), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("payload_state", sa.String(30), nullable=False),
        sa.Column("size_bytes", sa.Integer()),
        sa.Column("sha256", sa.String(64)),
        sa.Column("media_type", sa.String(150)),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("dataset_id", "source_relative_path"),
        sa.UniqueConstraint("storage_backend", "storage_key"),
    )
    op.create_index("ix_files_dataset_id", "files", ["dataset_id"])
    op.create_index("ix_files_recording_id", "files", ["recording_id"])
    op.create_index("ix_files_recording_role", "files", ["recording_id", "role"])
    op.create_table(
        "annotations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("annotation_key", sa.String(64), nullable=False, unique=True),
        sa.Column("dataset_id", sa.String(100), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("recording_id", sa.String(300), sa.ForeignKey("recordings.id"), nullable=False),
        sa.Column("patient_id", sa.String(220), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("source_event_type", sa.String(100)),
        sa.Column("lateralization", sa.String(100)),
        sa.Column("localization", sa.String(255)),
        sa.Column("confidence", sa.Float()),
        sa.Column("derived_label", sa.String(100)),
        sa.Column("source_file_id", sa.String(36), sa.ForeignKey("files.id")),
        sa.Column("source_relative_path", sa.Text()),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
    )
    op.create_index("ix_annotations_dataset_id", "annotations", ["dataset_id"])
    op.create_index("ix_annotations_patient_id", "annotations", ["patient_id"])
    op.create_index(
        "ix_annotations_recording_label_time", "annotations", ["recording_id", "label", "start_sec"]
    )
    op.create_table(
        "temporal_windows",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("window_id", sa.String(300), nullable=False, unique=True),
        sa.Column("dataset_id", sa.String(100), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("recording_id", sa.String(300), sa.ForeignKey("recordings.id"), nullable=False),
        sa.Column("patient_id", sa.String(220), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("label", sa.String(100)),
        sa.Column("label_id", sa.Integer()),
        sa.Column("split", sa.String(30)),
        sa.Column("feature_file_id", sa.String(36), sa.ForeignKey("files.id")),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
    )
    op.create_index("ix_temporal_windows_dataset_id", "temporal_windows", ["dataset_id"])
    op.create_index("ix_temporal_windows_patient_id", "temporal_windows", ["patient_id"])
    op.create_index("ix_temporal_windows_label", "temporal_windows", ["label"])
    op.create_index("ix_temporal_windows_split", "temporal_windows", ["split"])
    op.create_index("ix_windows_recording_time", "temporal_windows", ["recording_id", "start_sec"])
    op.create_table(
        "model_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("model_type", sa.String(100)),
        sa.Column("model_version", sa.String(100)),
        sa.Column("modality", sa.String(50)),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("git_commit", sa.String(64)),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("artifact_file_id", sa.String(36), sa.ForeignKey("files.id")),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        *_timestamps(),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_run_id", sa.String(36), sa.ForeignKey("model_runs.id"), nullable=False),
        sa.Column(
            "window_id", sa.String(300), sa.ForeignKey("temporal_windows.window_id"), nullable=False
        ),
        sa.Column("predicted_label", sa.String(100), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column("probabilities_json", sa.JSON(), nullable=False),
        sa.Column("is_correct", sa.Boolean()),
        *_timestamps(),
        sa.UniqueConstraint("model_run_id", "window_id"),
    )
    op.create_index("ix_predictions_model_run_id", "predictions", ["model_run_id"])
    op.create_index("ix_predictions_window_id", "predictions", ["window_id"])


def downgrade() -> None:
    for table in (
        "predictions",
        "model_runs",
        "temporal_windows",
        "annotations",
        "files",
        "recordings",
        "patients",
        "datasets",
    ):
        op.drop_table(table)
