from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
import mimetypes
import tempfile
import threading
import time
import uuid
from functools import wraps
from fyp_data_platform import lifecycle
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from seizure_multimodal.data_inspection import resolved_interval_label_records, resolved_label_records
from seizure_multimodal.synchronization.video_segments import build_video_segment_manifest

from fyp_data_platform.config.settings import PROJECT_ROOT, Settings, load_settings
from fyp_data_platform.compute import compute_info
from fyp_data_platform.db.models import Dataset, ModelRun, Patient, Recording, StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.db.session import create_engine_and_session
from fyp_data_platform.integrations.seizure_model import SeizureModelIntegration
from fyp_data_platform.integrations.cancellation import ModelRunStopped
from fyp_data_platform.storage.factory import create_storage
from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.keys import safe_entity, source_key
from fyp_data_platform.video_preview import VideoPreviews


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
ALLOWED_EEG_EXTENSIONS = {".csv", ".edf", ".e"}
ALLOWED_LABEL_EXTENSIONS = {".csv", ".xlsx"}
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _recording_source_id(recording_id: str, file_stem: str, count: int) -> str:
    stem = safe_entity(file_stem)
    if recording_id and count == 1:
        return safe_entity(recording_id)
    return f"{safe_entity(recording_id)}-{stem}" if recording_id else stem


def _register_video(
    *, settings: Settings, storage: Any, session_factory: Any, upload: FileStorage,
    dataset_id: str, dataset_name: str, patient_id: str, recording_id: str,
    file_count: int, shared_recording: bool = False,
    recording_metadata: dict[str, Any] | None = None,
    file_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filename = secure_filename(upload.filename or "")
    if not filename:
        raise ValueError("Uploaded file is missing a filename.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video file extension: {suffix}")
    dataset_id = safe_entity(dataset_id or "ui-recordings")
    dataset_name = dataset_name or "UI uploaded recordings"
    patient_id = safe_entity(patient_id or "local-patient-001")
    source_recording_id = (
        safe_entity(recording_id)
        if shared_recording else _recording_source_id(recording_id, Path(filename).stem, file_count)
    )
    relative = f"manual/{patient_id}/{source_recording_id}/{filename}"
    key = source_key(dataset_id, relative)
    with tempfile.TemporaryDirectory(prefix="fyp-ui-upload-") as directory:
        temporary = Path(directory) / filename
        upload.save(temporary)
        metadata = storage.put_file(temporary, key, sha256=sha256_file(temporary))
    with session_factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset(
            dataset_id, name=dataset_name, version=None,
            description="Locally uploaded research videos", source_url=None,
            license=None, metadata_json={"uploaded_via": "web"},
        )
        patient, _ = repo.patient(dataset_id, patient_id, metadata_json={"uploaded_via": "web"})
        recording, _ = repo.recording(
            dataset_id, source_recording_id, patient_id=patient.id,
            session_id=None, task=None, modality="multimodal" if shared_recording else "video", duration_sec=None,
            video_fps=None, start_datetime=None,
            metadata_json={"uploaded_via": "web", **(recording_metadata or {})},
        )
        file_row, _ = repo.stored_file(
            dataset_id, relative, recording_id=recording.id, role="source-media",
            modality="video", storage_backend=settings.storage_backend,
            storage_key=key, payload_state="stored", size_bytes=metadata.size_bytes,
            sha256=metadata.sha256, media_type=None,
            metadata_json={"uploaded_via": "web", "original_filename": filename, **(file_metadata or {})},
        )
    return {
        "file_id": file_row.id, "dataset_id": dataset_id,
        "patient_id": patient_id, "recording_id": source_recording_id,
        "filename": filename, "storage_key": key,
        "sha256": metadata.sha256, "size_bytes": metadata.size_bytes,
    }


def _validate_eeg_path(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EEG_EXTENSIONS:
        raise ValueError("EEG data must be a .csv, .edf, or .e file.")
    if suffix != ".csv":
        from seizure_multimodal.eeg.loaders import load_eeg

        with load_eeg(path) as recording:
            expected_format = "edf" if suffix == ".edf" else "nicolet-e"
            if recording.metadata.source_format != expected_format:
                raise ValueError(
                    f"EEG {suffix} content does not match the expected {expected_format} format."
                )
            return {
                "format": recording.metadata.source_format,
                "duration_sec": recording.metadata.duration_sec,
                "channel_count": len(recording.metadata.channel_names),
                "sampling_rates": sorted(set(recording.metadata.sampling_rates)),
                "annotation_count": len(recording.annotations),
            }
    rows = 0
    labels: set[int] = set()
    try:
        text = path.read_bytes().decode("utf-8-sig")
        reader = csv.DictReader(StringIO(text))
        expected = [f"X{index}" for index in range(1, 179)]
        columns = reader.fieldnames or []
        feature_columns = sorted(
            (column for column in columns if column.startswith("X") and column[1:].isdigit()),
            key=lambda column: int(column[1:]),
        )
        if feature_columns != expected:
            raise ValueError("EEG CSV must contain the exact feature columns X1 through X178.")
        if "y" not in columns:
            raise ValueError("EEG CSV must contain the ground-truth column y.")
        for row_number, row in enumerate(reader, start=2):
            for column in expected:
                float(row[column])
            label = int(row["y"])
            if label not in {1, 2, 3, 4, 5}:
                raise ValueError(f"EEG label on row {row_number} must be between 1 and 5.")
            labels.add(label)
            rows += 1
    except UnicodeDecodeError as error:
        raise ValueError("EEG CSV must use UTF-8 text encoding.") from error
    except (TypeError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith("EEG"):
            raise
        raise ValueError(f"EEG CSV contains invalid numeric data: {error}") from error
    if rows == 0:
        raise ValueError("EEG CSV does not contain any data rows.")
    return {"format": "csv", "row_count": rows, "labels": sorted(labels), "feature_count": 178}


def _validate_eeg_upload(upload: FileStorage) -> dict[str, Any]:
    """Compatibility wrapper used by callers that have not staged the upload yet."""
    filename = secure_filename(upload.filename or "")
    if Path(filename).suffix.lower() not in ALLOWED_EEG_EXTENSIONS:
        raise ValueError("EEG data must be a .csv, .edf, or .e file.")
    try:
        with tempfile.TemporaryDirectory(prefix="fyp-ui-eeg-validation-") as directory:
            temporary = Path(directory) / filename
            upload.save(temporary)
            return _validate_eeg_path(temporary)
    finally:
        upload.stream.seek(0)


def _validate_multimodal_uploads(
    *,
    eeg_upload: FileStorage,
    videos: list[FileStorage],
    recording_id: str,
    video_start_offsets: dict[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Stage one upload set and verify its Nicolet-to-video synchronization."""

    try:
        with tempfile.TemporaryDirectory(prefix="fyp-ui-fusion-validation-") as directory:
            root = Path(directory)
            eeg_name = secure_filename(eeg_upload.filename or "")
            eeg_path = root / eeg_name
            eeg_upload.save(eeg_path)
            validation = _validate_eeg_path(eeg_path)
            video_paths: list[Path] = []
            seen: set[str] = set()
            for upload in videos:
                filename = secure_filename(upload.filename or "")
                if not filename:
                    raise ValueError("Every video segment must have a filename.")
                if filename.casefold() in seen:
                    raise ValueError(f"Duplicate video segment filename: {filename}")
                seen.add(filename.casefold())
                path = root / filename
                upload.save(path)
                video_paths.append(path)
            segments = build_video_segment_manifest(
                eeg_path=eeg_path,
                video_paths=video_paths,
                recording_id=recording_id,
                **({"video_start_offsets": video_start_offsets, "eeg_duration_sec": validation["duration_sec"]} if eeg_path.suffix.lower() == ".edf" else {}),
            )
            public = [item.public_dict() for item in segments]
            by_filename = {
                item.filename.casefold(): {
                    "video_sync": item.public_dict(),
                }
                for item in segments
            }
            return validation, public, by_filename
    finally:
        eeg_upload.stream.seek(0)
        for upload in videos:
            upload.stream.seek(0)


def _register_eeg(
    *, settings: Settings, storage: Any, session_factory: Any, upload: FileStorage,
    dataset_id: str, dataset_name: str, patient_id: str, recording_id: str,
    validation: dict[str, Any] | None = None, shared_recording: bool = False,
    recording_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filename = secure_filename(upload.filename or "")
    dataset_id = safe_entity(dataset_id or "ui-recordings")
    dataset_name = dataset_name or "UI uploaded recordings"
    patient_id = safe_entity(patient_id or "local-patient-001")
    source_recording_id = safe_entity(recording_id or Path(filename).stem)
    relative = f"manual/{patient_id}/{source_recording_id}/{filename}"
    key = source_key(dataset_id, relative)
    with tempfile.TemporaryDirectory(prefix="fyp-ui-eeg-upload-") as directory:
        temporary = Path(directory) / filename
        upload.save(temporary)
        validation = validation or _validate_eeg_path(temporary)
        metadata = storage.put_file(temporary, key, sha256=sha256_file(temporary))
    with session_factory.begin() as session:
        repo = CatalogRepository(session)
        repo.dataset(
            dataset_id, name=dataset_name, version=None,
            description="Locally uploaded research recordings", source_url=None,
            license=None, metadata_json={"uploaded_via": "web"},
        )
        patient, _ = repo.patient(dataset_id, patient_id, metadata_json={"uploaded_via": "web"})
        recording, _ = repo.recording(
            dataset_id, source_recording_id, patient_id=patient.id,
            session_id=None, task=None, modality="multimodal" if shared_recording else "eeg", duration_sec=validation.get("duration_sec"),
            video_fps=None, start_datetime=None,
            metadata_json={"uploaded_via": "web", **(recording_metadata or {})},
        )
        file_row, _ = repo.stored_file(
            dataset_id, relative, recording_id=recording.id, role="source-data",
            modality="eeg", storage_backend=settings.storage_backend,
            storage_key=key, payload_state="stored", size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
            media_type="text/csv" if Path(filename).suffix.lower() == ".csv" else "application/octet-stream",
            metadata_json={
                "uploaded_via": "web", "original_filename": filename, **validation,
                **({"multimodal_sync": recording_metadata} if recording_metadata else {}),
            },
        )
    return {
        "file_id": file_row.id, "dataset_id": dataset_id, "patient_id": patient_id,
        "recording_id": source_recording_id, "filename": filename,
        "storage_key": key, "sha256": metadata.sha256,
        "size_bytes": metadata.size_bytes, "validation": validation,
    }


def _validate_label_upload(
    upload: FileStorage, *, patient_id: str, seizure_id: str, seizure_type: str,
    recording_id: str = "",
) -> dict[str, Any]:
    filename = secure_filename(upload.filename or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_LABEL_EXTENSIONS:
        raise ValueError("Ground-truth labels must be a .csv interval file or .xlsx workbook.")
    with tempfile.TemporaryDirectory(prefix="fyp-ui-label-check-") as directory:
        temporary = Path(directory) / filename
        upload.save(temporary)
        if suffix == ".csv":
            records = resolved_interval_label_records(temporary)
        else:
            if not all((patient_id, seizure_id, seizure_type)):
                raise ValueError(
                    "Patient ID, seizure ID, and seizure type are required with workbook labels."
                )
            records = resolved_label_records(temporary)
    upload.stream.seek(0)
    if suffix == ".csv":
        if any(record.get("resolved_PatID") for record in records):
            records = [
                record for record in records
                if str(record.get("resolved_PatID") or "").strip().casefold() == patient_id.casefold()
            ]
        if not records:
            raise ValueError(f"No interval label rows matched Patient ID {patient_id}.")
        recording_ids = {str(row.get('recording_id') or '').strip() for row in records}
        if len(recording_ids) > 1:
            records = [row for row in records if str(row.get('recording_id') or '').strip().casefold() == recording_id.casefold()]
            if not recording_id or not records:
                raise ValueError("Label CSV contains multiple recordings. Enter a Recording ID matching the CSV.")
        selected_recording = str(records[0].get('recording_id') or '').strip() or None
        ordered = sorted(records, key=lambda row: row["start_sec"])
        for index, row in enumerate(ordered):
            if not all(math.isfinite(row[key]) for key in ("start_sec", "end_sec")) or row["start_sec"] < 0:
                raise ValueError("Label intervals must have finite, non-negative times.")
            if index and row["start_sec"] < ordered[index - 1]["end_sec"]:
                raise ValueError("Ground-truth label intervals must not overlap.")
        seizure_rows = [record for record in records if int(record["label"]) == 1]
        return {
            "patient_id": patient_id,
            "recording_id": selected_recording,
            "seizure_id": seizure_id or None,
            "seizure_type": seizure_type or None,
            "label_format": "interval_csv",
            "interval_count": len(records),
            "seizure_interval_count": len(seizure_rows),
            "eeg_onset_sec": min((record["start_sec"] for record in seizure_rows), default=None),
            "clinical_onset_sec": min((record["start_sec"] for record in seizure_rows), default=None),
            "seizure_end_sec": max((record["end_sec"] for record in seizure_rows), default=None),
            "label_end_sec": max(record["end_sec"] for record in records),
        }
    matches = [
        record for record in records
        if str(record.get("resolved_PatID") or "").strip().casefold() == patient_id.casefold()
        and str(record.get("#Seizure") or "").strip().casefold() == seizure_id.casefold()
        and str(record.get("resolved_Seizure Type") or "").strip().casefold() == seizure_type.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one label row for {patient_id} / {seizure_id} / {seizure_type}; "
            f"found {len(matches)}."
        )
    match = matches[0]
    return {
        "patient_id": patient_id,
        "seizure_id": seizure_id,
        "seizure_type": seizure_type,
        "label_format": "workbook",
        "eeg_onset_sec": match.get("EEG onset seconds"),
        "clinical_onset_sec": match.get("Clinical Onset seconds"),
        "seizure_end_sec": next(
            (match.get(key) for key in ("Seizure End seconds", "Seizure end seconds", "Offset seconds") if match.get(key) is not None),
            None,
        ),
    }


def _register_label(
    *, settings: Settings, storage: Any, session_factory: Any, upload: FileStorage,
    dataset_id: str, patient_id: str, recording_id: str, identity: dict[str, Any],
) -> dict[str, Any]:
    filename = secure_filename(upload.filename or "")
    relative = f"manual/{patient_id}/{recording_id}/labels/{filename}"
    key = source_key(dataset_id, relative)
    with tempfile.TemporaryDirectory(prefix="fyp-ui-label-upload-") as directory:
        temporary = Path(directory) / filename
        upload.save(temporary)
        metadata = storage.put_file(temporary, key, sha256=sha256_file(temporary))
    with session_factory.begin() as session:
        recording = session.get(Recording, f"{dataset_id}::{recording_id}")
        if recording is None:
            raise LookupError("The video recording was not registered before its labels.")
        file_row, _ = CatalogRepository(session).stored_file(
            dataset_id, relative, recording_id=recording.id,
            role="ground-truth-labels", modality="labels",
            storage_backend=settings.storage_backend, storage_key=key,
            payload_state="stored", size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
            media_type=(
                "text/csv"
                if Path(filename).suffix.lower() == ".csv"
                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            metadata_json={"uploaded_via": "web", "original_filename": filename, "label_identity": identity},
        )
    return {
        "file_id": file_row.id, "filename": filename, "storage_key": key,
        "sha256": metadata.sha256, "size_bytes": metadata.size_bytes,
        "identity": identity,
    }


def _metric_documents(row: ModelRun) -> list[dict[str, Any]]:
    values = ((row.metrics_json or {}).get("metrics") or {}).values()
    return [value for value in values if isinstance(value, dict)]


def _metric_body(document: dict[str, Any]) -> dict[str, Any]:
    nested = document.get("metrics")
    return nested if isinstance(nested, dict) else document


def _best_metric_document(row: ModelRun) -> dict[str, Any]:
    metrics = (row.metrics_json or {}).get("metrics") or {}
    if row.modality == "fusion":
        for path, document in metrics.items():
            if str(path).replace("\\", "/").endswith("fusion/metrics.json") and isinstance(document, dict):
                return document
    preferred = f"{row.modality}_only/metrics.json"
    for path, document in metrics.items():
        if str(path).replace("\\", "/") == preferred and isinstance(document, dict):
            return document
    documents = _metric_documents(row)
    for document in documents:
        body = _metric_body(document)
        if "predicted_event_count" in body or "predicted_events" in body:
            return document
    return documents[0] if documents else {}


def _artifacts(row: ModelRun) -> list[dict[str, Any]]:
    return [item for item in (row.metrics_json or {}).get("artifacts", []) if isinstance(item, dict)]


def _find_artifact(row: ModelRun, *suffixes: str) -> dict[str, Any] | None:
    for suffix in suffixes:
        for artifact in _artifacts(row):
            relative = str(artifact.get("relative_path", ""))
            if relative == suffix or relative.endswith(f"/{suffix}"):
                return artifact
    return None


def _download_artifact_bytes(storage: Any, artifact: dict[str, Any]) -> bytes:
    key = str(artifact.get("key", ""))
    if not key:
        raise LookupError("Artifact does not have a storage key.")
    suffix = Path(str(artifact.get("relative_path", "artifact"))).suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="fyp-ui-artifact-") as directory:
        destination = Path(directory) / f"artifact{suffix}"
        storage.download_file(key, destination)
        return destination.read_bytes()


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prediction_windows(
    storage: Any, row: ModelRun, *preferred_suffixes: str,
) -> list[dict[str, Any]]:
    artifact = _find_artifact(
        row, *(preferred_suffixes or (
            "fusion/predictions.csv", "video_only/predictions.csv", "predictions.csv",
            "video_only/probability_timeline.csv", "probability_timeline.csv",
        )),
    )
    if artifact is None:
        return []
    text = _download_artifact_bytes(storage, artifact).decode("utf-8-sig")
    windows = []
    for csv_row in csv.DictReader(StringIO(text)):
        start = _float_or_none(csv_row.get("start_sec"))
        end = _float_or_none(csv_row.get("end_sec"))
        if start is None or end is None:
            continue
        probability = _float_or_none(
            csv_row.get("fused_probability") or csv_row.get("video_probability") or csv_row.get("probability")
            or csv_row.get("seizure_probability")
        )
        prediction_value = (
            csv_row.get("fused_prediction") or csv_row.get("video_prediction") or csv_row.get("prediction")
            or csv_row.get("predicted_label")
        )
        try:
            prediction = int(float(prediction_value))
        except (TypeError, ValueError):
            prediction = 0
        reference_value = _float_or_none(csv_row.get("reference_label"))
        windows.append({
            "start_sec": start, "end_sec": end, "duration_sec": max(end - start, 0.0),
            "probability": probability, "prediction": prediction,
            "reference_label": int(reference_value) if reference_value in {0.0, 1.0} else None,
            "eeg_onset_sec": _float_or_none(csv_row.get("eeg_onset_sec")),
            "clinical_onset_sec": _float_or_none(csv_row.get("clinical_onset_sec")),
            "seizure_end_sec": _float_or_none(csv_row.get("seizure_end_sec")),
            "window_id": csv_row.get("window_id"),
            "video_segment_id": csv_row.get("video_segment_id"),
            "eeg_probability": _float_or_none(csv_row.get("eeg_probability")),
            "video_probability": _float_or_none(csv_row.get("video_probability")),
            "fused_probability": _float_or_none(csv_row.get("fused_probability")),
            "eeg_prediction": _float_or_none(csv_row.get("eeg_prediction")),
            "video_prediction": _float_or_none(csv_row.get("video_prediction")),
            "fused_prediction": _float_or_none(csv_row.get("fused_prediction")),
            "fusion_available": str(csv_row.get("fusion_available", "")).strip().lower() in {"true", "1", "yes"},
            "video_coverage": _float_or_none(csv_row.get("video_coverage")),
            "video_segment_ids": [value for value in str(csv_row.get("video_segment_ids", "")).split("|") if value],
            "video_window_ids": [value for value in str(csv_row.get("video_window_ids", "")).split("|") if value],
        })
    return windows


def _events_from_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for window in windows:
        if not window["prediction"]:
            if current:
                events.append(current)
                current = None
            continue
        score = window["probability"] or 0.0
        if current is None:
            current = {"start_sec": window["start_sec"], "end_sec": window["end_sec"], "score": score}
        else:
            current["end_sec"] = window["end_sec"]
            current["score"] = max(float(current["score"]), float(score))
    if current:
        events.append(current)
    return events


def _outcome(row: ModelRun, windows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    if row.status != "completed":
        return None
    windows = windows or []
    body = _metric_body(_best_metric_document(row))
    metric_events = body.get("predicted_events")
    events = []
    if isinstance(metric_events, list):
        events = [
            {
                "start_sec": _float_or_none(event.get("start_sec")) or 0.0,
                "end_sec": _float_or_none(event.get("end_sec")) or 0.0,
                "score": _float_or_none(event.get("score")) or 0.0,
            }
            for event in metric_events if isinstance(event, dict)
        ]
    if not events:
        events = _events_from_windows(windows)
    try:
        event_count = int(float(body.get("predicted_event_count")))
    except (TypeError, ValueError):
        event_count = len(events)
    prediction_only = bool(body.get("prediction_only"))
    return {
        "detected": event_count > 0 or any(item["prediction"] for item in windows),
        "event_count": event_count, "events": events,
        "max_probability": max(
            (float(item["probability"]) for item in windows if item["probability"] is not None),
            default=None,
        ),
        "prediction_only": prediction_only,
        "note": (
            "Prediction-only run. Accuracy metrics require registered labels."
            if prediction_only else "Evaluation run using the registered ground-truth labels."
        ),
    }


def _run_summary(row: ModelRun) -> dict[str, Any]:
    config = row.config_json or {}
    payload = row.metrics_json or {}
    artifacts = _artifacts(row)
    started = row.started_at
    ended = row.completed_at or (datetime.now(timezone.utc) if row.status == 'running' else None)
    elapsed = None
    if started and ended:
        elapsed = max(0.0, (ended.replace(tzinfo=ended.tzinfo or timezone.utc) - started.replace(tzinfo=started.tzinfo or timezone.utc)).total_seconds())
    return {
        "id": row.id, "status": row.status, "modality": row.modality,
        "archived": row.archived, "deleting": row.deleting,
        "dataset_id": config.get("dataset_id"), "recording_id": config.get("recording_id"),
        "compute": config.get("compute"),
        "runtime": payload.get("runtime"),
        "elapsed_sec": elapsed,
        "require_labels": bool(config.get("require_labels")), "artifact_count": len(artifacts),
        "output_prefix": str(artifacts[0].get("key", "")).rsplit("/", 1)[0] if artifacts else None,
        "outcome": _outcome(row), "error": payload.get("error"),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


def _source_video(session: Any, row: ModelRun) -> StoredFile | None:
    config = row.config_json or {}
    dataset_id = config.get("dataset_id")
    recording_id = config.get("recording_id")
    if not dataset_id or not recording_id:
        return None
    return session.scalar(
        select(StoredFile).where(
            StoredFile.dataset_id == dataset_id,
            StoredFile.recording_id == f"{dataset_id}::{recording_id}",
            StoredFile.modality == "video", StoredFile.payload_state == "stored",
        ).order_by(StoredFile.created_at.desc())
    )


def _run_patient(session, row):
    config = row.config_json or {}
    recording = session.scalar(select(Recording).where(
        Recording.dataset_id == config.get('dataset_id'),
        Recording.source_recording_id == config.get('recording_id'),
    ))
    return session.get(Patient, recording.patient_id) if recording else None


def _patient_summary(patient):
    if patient is None:
        return None
    profile = (patient.metadata_json or {}).get('profile', {})
    return {'id': patient.source_patient_id, 'age': profile.get('age'),
            'notes': profile.get('notes', ''), 'updated_at': profile.get('updated_at')}


def _run_result(storage: Any, session: Any, row: ModelRun) -> dict[str, Any]:
    windows = _prediction_windows(storage, row)
    eeg_windows = _prediction_windows(storage, row, "eeg_only/predictions.csv")
    video_windows = _prediction_windows(storage, row, "fusion/video_timeline.csv")
    summary = _run_summary(row)
    document = _best_metric_document(row)
    body = _metric_body(document)
    run_info = document.get("run", {}) if isinstance(document.get("run"), dict) else {}
    video_info = run_info.get("video", {}) if isinstance(run_info.get("video"), dict) else {}
    timeline = _find_artifact(row, "video_only/timeline.png", "timeline.png")
    source_video = _source_video(session, row)
    ground_truth = {
        key: next((window.get(key) for window in windows if window.get(key) is not None), None)
        for key in ("eeg_onset_sec", "clinical_onset_sec", "seizure_end_sec")
    }
    workflow_metrics: dict[str, Any] = {}
    for path, value in ((row.metrics_json or {}).get("metrics") or {}).items():
        if not isinstance(value, dict):
            continue
        lowered_path = str(path).lower()
        name = "fusion" if "fusion/metrics" in lowered_path else "eeg" if "eeg" in lowered_path else "video" if "video" in lowered_path else str(path)
        workflow_metrics[name] = {
            "status": value.get("status"), "metrics": _metric_body(value),
            "metadata": value.get("metadata"), "error": value.get("error"),
        }
    return {
        **summary,
        "patient": _patient_summary(_run_patient(session, row)),
        # Older labelled runs did not persist predicted_events in metrics. Use
        # their saved prediction windows so they display correctly without a rerun.
        "outcome": _outcome(row, windows),
        "duration_sec": (
            video_info.get("duration_sec")
            or (document.get("metadata", {}).get("duration_sec") if isinstance(document.get("metadata"), dict) else None)
            or body.get("duration_sec")
            or max((item["end_sec"] for item in windows), default=None)
        ),
        "metrics": body, "workflow_metrics": workflow_metrics,
        "run": run_info, "windows": windows, "fusion_windows": windows if row.modality == "fusion" else [],
        "eeg_windows": eeg_windows, "video_windows": video_windows,
        "ground_truth": ground_truth,
        "source_video": (
            {
                "filename": Path(source_video.source_relative_path).name,
                "size_bytes": source_video.size_bytes,
                "media_type": mimetypes.guess_type(source_video.source_relative_path)[0] or "application/octet-stream",
                "browser_playable": Path(source_video.source_relative_path).suffix.lower() == ".mp4",
                "url": f"/api/runs/{row.id}/source-video",
                "preview_status_url": f"/api/runs/{row.id}/preview",
                "pose_url": next((f"/api/runs/{row.id}/artifacts/{item['relative_path']}" for item in _artifacts(row) if item.get("relative_path") == "video_only/pose_keypoints.json"), None),
            }
            if source_video else None
        ),
        "artifacts": [
            {
                "relative_path": item.get("relative_path"), "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
                "url": f"/api/runs/{row.id}/artifacts/{item.get('relative_path')}",
            } for item in _artifacts(row)
        ],
        "timeline_image": f"/api/runs/{row.id}/artifacts/{timeline.get('relative_path')}" if timeline else None,
    }


def create_app(config_path: str | None = None) -> Any:
    try:
        from flask import Flask, Response, jsonify, request, send_from_directory, send_file
    except ImportError as error:
        raise RuntimeError("Install the web dependencies with: pip install -e '.[ui]'") from error
    settings = load_settings(config_path)
    engine, session_factory = create_engine_and_session(settings.database_url)
    storage = create_storage(settings)
    previews = VideoPreviews(settings.work_root, storage)
    jobs: dict[str, dict[str, Any]] = {}
    cancellations: dict[str, threading.Event] = {}
    jobs_lock = threading.Lock()
    run_lock = threading.Lock()
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024

    def exclusive_change(handler):
        @wraps(handler)
        def guarded(*args, **kwargs):
            if not run_lock.acquire(blocking=False):
                return jsonify({"error": "Processing or another data change is active. Try again when it finishes."}), 409
            try:
                with previews.lock:
                    if previews.busy:
                        return jsonify({"error": "A video preview is being prepared. Try again when it finishes."}), 409
                    return handler(*args, **kwargs)
            finally:
                run_lock.release()
        return guarded

    @app.route("/api/manage/<kind>/<path:identity>", methods=["GET", "PATCH", "DELETE"])
    @exclusive_change
    def manage(kind, identity):
        try:
            data = request.get_json(silent=True) or {}
            if request.method == "DELETE":
                return jsonify(lifecycle.permanently_delete(session_factory, storage, settings, kind, identity, data.get("confirmation")))
            with session_factory.begin() as session:
                if request.method == "PATCH":
                    if not isinstance(data.get("archived"), bool):
                        raise ValueError("archived must be true or false")
                    lifecycle.archive(session, kind, identity, data["archived"])
                return jsonify(lifecycle.preview(session, kind, identity))
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 409
        except Exception:
            app.logger.exception("Managed cleanup failed")
            return jsonify({"error": "Cleanup did not finish. The item remains archived; retry permanent deletion to finish removing its files."}), 500

    @app.get("/api/health")
    def health() -> Any:
        with session_factory() as session:
            counts = {
                "datasets": session.scalar(select(func.count()).select_from(Dataset)) or 0,
                "patients": session.scalar(select(func.count()).select_from(Patient)) or 0,
                "recordings": session.scalar(select(func.count()).select_from(Recording)) or 0,
                "files": session.scalar(select(func.count()).select_from(StoredFile)) or 0,
                "runs": session.scalar(select(func.count()).select_from(ModelRun)) or 0,
            }
        return jsonify({
            "status": "ok", "environment": settings.environment,
            "storage_backend": settings.storage_backend, "counts": counts,
            "compute": compute_info(),
        })

    @app.get("/api/catalog")
    def catalog() -> Any:
        show_archived = request.args.get("archived") == "true"
        with session_factory() as session:
            records = {r.id: r for r in session.scalars(select(Recording)).all()}
            inputs = session.scalars(
                select(StoredFile).where(
                    StoredFile.modality.in_(("video", "eeg", "labels")),
                    StoredFile.payload_state == "stored",
                )
                .order_by(StoredFile.created_at.desc())
            ).all()
            runs = session.scalars(select(ModelRun).order_by(ModelRun.started_at.desc())).all()
            runs = [r for r in runs if bool(r.archived or (records.get(lifecycle.run_recording_id(r)) and records[lifecycle.run_recording_id(r)].archived)) == show_archived]
        grouped: dict[str, dict[str, Any]] = {}
        for row in inputs:
            if not row.recording_id:
                continue
            parent = records.get(row.recording_id)
            if not parent or bool(parent.archived) != show_archived:
                continue
            source_id = row.recording_id.split("::", 1)[1] if "::" in row.recording_id else row.recording_id
            recording = grouped.setdefault(row.recording_id, {
                "id": row.recording_id, "dataset_id": row.dataset_id,
                "recording_id": row.recording_id, "recording_source_id": source_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "video": None, "videos": [], "eeg": None, "label": None,
                "recording_type": (parent.metadata_json or {}).get("recording_type") or parent.modality,
                "sync_status": (parent.metadata_json or {}).get("sync_status", "unavailable"),
                "sync_verified": bool((parent.metadata_json or {}).get("sync_verified")),
                "sync_source": (parent.metadata_json or {}).get("sync_source"),
                "archived": parent.archived, "deleting": parent.deleting,
            })
            if row.modality != "video" and recording.get("label" if row.modality == "labels" else row.modality) is not None:
                continue
            file_payload = {
                "id": row.id, "filename": Path(row.source_relative_path).name,
                "storage_key": row.storage_key, "size_bytes": row.size_bytes,
                "sha256": row.sha256,
                "metadata": row.metadata_json or {},
            }
            if row.modality == "labels":
                recording["label"] = {
                    **file_payload,
                    "identity": (row.metadata_json or {}).get("label_identity"),
                }
            elif row.modality == "video":
                recording["videos"].append(file_payload)
                if recording["video"] is None:
                    recording["video"] = file_payload
            else:
                recording[row.modality] = file_payload
        for parent in records.values():
            if bool(parent.archived) == show_archived and parent.id not in grouped:
                grouped[parent.id] = {"id": parent.id, "dataset_id": parent.dataset_id, "recording_source_id": parent.source_recording_id, "created_at": parent.created_at.isoformat(), "archived": parent.archived, "deleting": parent.deleting, "video": None, "videos": [], "eeg": None, "label": None, "recording_type": (parent.metadata_json or {}).get("recording_type") or parent.modality, "sync_status": (parent.metadata_json or {}).get("sync_status", "unavailable"), "sync_verified": bool((parent.metadata_json or {}).get("sync_verified")), "sync_source": (parent.metadata_json or {}).get("sync_source")}
        for recording in grouped.values():
            recording["videos"].sort(key=lambda item: float(((item.get("metadata") or {}).get("video_sync") or {}).get("global_start_sec", 0.0)))
        recordings = sorted(grouped.values(), key=lambda item: item.get("created_at") or "", reverse=True)
        summaries = []
        for row in runs:
            parent = records.get(lifecycle.run_recording_id(row))
            summaries.append({**_run_summary(row), "parent_archived": bool(parent and parent.archived)})
        return jsonify({"recordings": recordings, "runs": summaries})

    @app.get("/api/runs/<run_id>")
    def run_result(run_id: str) -> Any:
        with session_factory() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                return jsonify({"error": "Unknown model run."}), 404
            try:
                result = _run_result(storage, session, row)
                with jobs_lock:
                    result['stop_requested'] = any(job.get('run_id') == row.id and job.get('stop_requested') for job in jobs.values())
                if row.status == 'running':
                    config = row.config_json or {}
                    progress_path = settings.work_root / safe_entity(config.get('dataset_id', '')) / safe_entity(config.get('recording_id', '')) / safe_entity(row.id) / 'progress.json'
                    try:
                        with progress_path.open(encoding='utf-8') as handle:
                            progress = json.loads(handle.read(65536))
                        if isinstance(progress, dict):
                            result['progress'] = progress
                    except (OSError, ValueError):
                        pass
                return jsonify(result)
            except Exception as error:
                return jsonify({"error": str(error)}), 500

    @app.get('/api/runs/<run_id>/evidence')
    @app.get('/api/runs/<run_id>/evidence/eeg')
    @app.get('/api/runs/<run_id>/evidence/video/<int:index>/preview')
    @app.get('/api/runs/<run_id>/evidence/video/<int:index>/media')
    def run_evidence(run_id, index=None):
        from .evidence import evidence_sources, input_path, waveform
        # Also prevents deletion of retained inputs while evidence is being read.
        with previews.lock, session_factory() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                return jsonify({'error': 'Unknown model run.'}), 404
            config = row.config_json or {}
            parent = session.get(Recording, f"{config.get('dataset_id')}::{config.get('recording_id')}")
            if row.deleting or (parent and parent.deleting):
                return jsonify({'error': 'Recording deletion is in progress.'}), 409
            data = config.get('config', {}).get('data', {})
            try:
                sources = evidence_sources(settings, row, storage)
                if request.path.endswith('/eeg'):
                    path = input_path(settings, row, data.get('eeg_file'))
                    payload = waveform(path, float(request.args.get('start', '0')), float(request.args.get('end', '12')), 'all' if request.args.get('channel') == 'all' else int(request.args.get('channel', '0')))
                    return jsonify(payload)
                if index is not None:
                    if not any(item['input_index'] == index for item in sources['videos']):
                        return jsonify({'error': 'Verified video evidence is unavailable.'}), 404
                    source = input_path(settings, row, data['video_segments'][index])
                    stamp = source.stat()
                    destination = previews.path(config['dataset_id'], config['recording_id'], str(source), f'{stamp.st_size}:{stamp.st_mtime_ns}')
                    if request.path.endswith('/media'):
                        if not destination.is_file():
                            return jsonify({'error': 'Preview is not ready.'}), 409
                        return send_file(destination, mimetype='video/mp4', conditional=True, max_age=3600)
                    state = previews.ensure(destination, str(source), source_path=source)
                    return jsonify({**state, 'url': f'/api/runs/{run_id}/evidence/video/{index}/media' if state['status'] == 'ready' else None})
                return jsonify(sources)
            except (ValueError, TypeError, IndexError) as error:
                return jsonify({'error': str(error)}), 400
            except Exception:
                app.logger.exception('Evidence could not be loaded')
                return jsonify({'error': 'Could not read the retained evidence for this run.'}), 422

    @app.get('/api/runs/<run_id>/preview')
    @app.get('/api/runs/<run_id>/preview-video')
    def run_preview(run_id):
        with previews.lock, session_factory() as session:
            row = session.get(ModelRun, run_id)
            source = _source_video(session, row) if row else None
            if not source:
                return jsonify({'error': 'Source video not found.'}), 404
            parent = session.get(Recording, source.recording_id)
            if row.deleting or (parent and parent.deleting):
                return jsonify({'error': 'Recording deletion is in progress.'}), 409
            config = row.config_json or {}
            destination = previews.path(config['dataset_id'], config['recording_id'], source.storage_key, source.sha256)
            if request.path.endswith('/preview-video'):
                if not destination.is_file():
                    return jsonify({'error': 'Preview is not ready.'}), 409
                response = send_file(destination, mimetype='video/mp4', conditional=True, max_age=3600)
                response.headers['Cache-Control'] = 'private, max-age=3600'
                return response
            state = previews.ensure(destination, source.storage_key)
            return jsonify({**state, 'url': f'/api/runs/{run_id}/preview-video' if state['status'] == 'ready' else None})

    @app.get("/api/runs/<run_id>/source-video")
    def run_source_video(run_id: str) -> Any:
        with session_factory() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                return jsonify({"error": "Unknown model run."}), 404
            source = _source_video(session, row)
            if source is None:
                return jsonify({"error": "This run does not have a registered source video."}), 404
            metadata = storage.stat(source.storage_key)
            if metadata is None or metadata.size_bytes <= 0:
                return jsonify({"error": "The registered source video is unavailable."}), 404
            size = metadata.size_bytes
            range_header = request.headers.get("Range")
            start, end, status = 0, size - 1, 200
            if range_header:
                if not range_header.startswith("bytes=") or "," in range_header:
                    return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
                bounds = range_header[6:].split("-", 1)
                try:
                    if not bounds[0]:
                        suffix_length = int(bounds[1])
                        if suffix_length <= 0:
                            raise ValueError
                        start = max(0, size - suffix_length)
                    else:
                        start = int(bounds[0])
                    end = min(int(bounds[1]), size - 1) if bounds[1] else size - 1
                    if start < 0 or start >= size or end < start:
                        raise ValueError
                except ValueError:
                    return Response(status=416, headers={"Content-Range": f"bytes */{size}"})
                status = 206
            headers = {
                "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1),
                "Content-Disposition": f'inline; filename="{Path(source.source_relative_path).name}"',
                "Cache-Control": "private, max-age=3600",
            }
            if status == 206:
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            return Response(
                storage.iter_bytes(source.storage_key, start=start, end=end),
                status=status,
                mimetype=mimetypes.guess_type(source.source_relative_path)[0] or "application/octet-stream",
                headers=headers,
                direct_passthrough=True,
            )

    @app.get("/api/runs/<run_id>/artifacts/<path:relative_path>")
    def run_artifact(run_id: str, relative_path: str) -> Any:
        with session_factory() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                return jsonify({"error": "Unknown model run."}), 404
            artifact = next((item for item in _artifacts(row) if str(item.get("relative_path")) == relative_path), None)
            if artifact is None:
                return jsonify({"error": "Unknown artifact."}), 404
            try:
                body = _download_artifact_bytes(storage, artifact)
            except Exception as error:
                return jsonify({"error": str(error)}), 500
        return Response(body, mimetype=mimetypes.guess_type(relative_path)[0] or "application/octet-stream")

    @app.post("/api/uploads")
    @exclusive_change
    def upload() -> Any:
        mode = str(request.form.get("mode", "video")).strip().lower()
        if mode not in {"video", "eeg", "unimodal", "fusion"}:
            return jsonify({"error": "Upload mode must be video, eeg, fusion, or unimodal."}), 400
        files = [item for item in request.files.getlist("videos") if item.filename]
        eeg_upload = request.files.get("eeg")
        label_upload = request.files.get("labels")
        if mode in {"video", "unimodal", "fusion"} and not files:
            return jsonify({"error": "Choose at least one video file."}), 400
        if mode in {"eeg", "unimodal", "fusion"} and not (eeg_upload and eeg_upload.filename):
            return jsonify({"error": "Choose one EEG .csv, .edf, or .e file."}), 400
        if mode == "unimodal" and len(files) != 1:
            return jsonify({"error": "Upload exactly one video with one EEG file when running both independently."}), 400
        try:
            patient_id = safe_entity(request.form.get("patient_id", "local-patient-001"))
            recording_id = str(request.form.get("recording_id", "")).strip()
            with session_factory() as session:
                dataset = safe_entity(request.form.get("dataset_id", "ui-recordings"))
                if mode == "fusion":
                    lifecycle.require_available(session, dataset, safe_entity(recording_id))
                else:
                    for item in files:
                        lifecycle.require_available(session, dataset, _recording_source_id(recording_id, Path(secure_filename(item.filename)).stem, len(files)))
                if eeg_upload and eeg_upload.filename:
                    lifecycle.require_available(session, dataset, safe_entity(recording_id or Path(secure_filename(eeg_upload.filename)).stem))
            if mode == "unimodal" and not recording_id:
                raise ValueError("Recording ID is required when uploading video and EEG together.")
            if mode == "fusion" and not recording_id:
                raise ValueError("Recording ID is required for a synchronized multimodal upload.")
            fusion_validation = None
            fusion_segments: list[dict[str, Any]] = []
            fusion_file_metadata: dict[str, dict[str, Any]] = {}
            recording_metadata: dict[str, Any] = {}
            if mode == "fusion" and eeg_upload:
                offsets = None
                if Path(eeg_upload.filename or "").suffix.lower() == ".edf":
                    offsets = json.loads(request.form.get("video_start_offsets", "{}"))
                    if not isinstance(offsets, dict):
                        raise ValueError("Video start offsets must map filenames to seconds.")
                    offsets = {secure_filename(name): float(value) for name, value in offsets.items()}
                fusion_validation, fusion_segments, fusion_file_metadata = _validate_multimodal_uploads(
                    eeg_upload=eeg_upload,
                    videos=files,
                    recording_id=safe_entity(recording_id),
                    **({"video_start_offsets": offsets} if offsets is not None else {}),
                )
                recording_metadata = {
                    "recording_type": "multimodal",
                    "sync_status": "verified",
                    "sync_verified": True,
                    "sync_source": fusion_segments[0]["sync_source"],
                    "video_segments": fusion_segments,
                }
            label_identity = None
            if label_upload and label_upload.filename:
                if mode == "eeg":
                    raise ValueError("The video label workbook cannot be attached to an EEG-only upload.")
                if mode != "fusion" and len(files) != 1:
                    raise ValueError("Upload exactly one video when attaching a label workbook.")
                label_identity = _validate_label_upload(
                    label_upload,
                    patient_id=patient_id,
                    recording_id=recording_id,
                    seizure_id=str(request.form.get("seizure_id", "")).strip(),
                    seizure_type=str(request.form.get("seizure_type", "")).strip(),
                )
                if mode == "fusion":
                    selector = label_identity.get("recording_id")
                    if selector and str(selector).casefold() != recording_id.casefold():
                        raise ValueError("Ground-truth Recording ID must match the multimodal recording.")
                    times = [label_identity.get(key) for key in ("label_end_sec", "eeg_onset_sec", "clinical_onset_sec", "seizure_end_sec")]
                    if any(value is not None and (not math.isfinite(value) or value < 0 or value > fusion_validation["duration_sec"]) for value in times):
                        raise ValueError("Ground-truth times must be within the EEG recording duration.")
            uploaded_videos = [
                _register_video(
                    settings=settings, storage=storage, session_factory=session_factory, upload=item,
                    dataset_id=request.form.get("dataset_id", "ui-recordings"),
                    dataset_name=request.form.get("dataset_name", "UI uploaded recordings"),
                    patient_id=patient_id,
                    recording_id=recording_id, file_count=len(files),
                    shared_recording=mode == "fusion",
                    recording_metadata=recording_metadata,
                    file_metadata=fusion_file_metadata.get(secure_filename(item.filename or "").casefold()),
                ) for item in files
            ]
            uploaded_eeg = None
            if eeg_upload and eeg_upload.filename:
                uploaded_eeg = _register_eeg(
                    settings=settings, storage=storage, session_factory=session_factory,
                    upload=eeg_upload, dataset_id=request.form.get("dataset_id", "ui-recordings"),
                    dataset_name=request.form.get("dataset_name", "UI uploaded recordings"),
                    patient_id=patient_id, recording_id=recording_id,
                    validation=fusion_validation,
                    shared_recording=mode == "fusion",
                    recording_metadata=recording_metadata,
                )
            registered_labels = []
            if label_identity and label_upload:
                registered_labels.append(_register_label(
                    settings=settings, storage=storage, session_factory=session_factory,
                    upload=label_upload, dataset_id=uploaded_videos[0]["dataset_id"],
                    patient_id=uploaded_videos[0]["patient_id"], recording_id=uploaded_videos[0]["recording_id"],
                    identity=label_identity,
                ))
        except Exception as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({
            "mode": mode, "videos": uploaded_videos, "eeg": uploaded_eeg,
            "synchronization": recording_metadata or None,
            "labels": registered_labels,
            "uploaded": [*uploaded_videos, *([uploaded_eeg] if uploaded_eeg else [])],
        }), 201

    @app.patch('/api/runs/<run_id>/patient')
    def update_patient(run_id):
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Patient information must be an object.'}), 400
        age = data.get('age')
        notes = data.get('notes', '')
        if age is not None and (type(age) is not int or not 0 <= age <= 130):
            return jsonify({'error': 'Age must be a whole number from 0 to 130, or blank.'}), 400
        if not isinstance(notes, str) or len(notes) > 4000:
            return jsonify({'error': 'Patient notes must be text of at most 4000 characters.'}), 400
        with session_factory.begin() as session:
            row = session.get(ModelRun, run_id)
            patient = _run_patient(session, row) if row else None
            if patient is None:
                return jsonify({'error': 'Patient not found for this run.'}), 404
            patient.metadata_json = {**(patient.metadata_json or {}), 'profile': {
                'age': age, 'notes': notes.strip(), 'updated_at': datetime.now(timezone.utc).isoformat(),
            }}
            return jsonify(_patient_summary(patient))

    @app.post("/api/runs")
    def start_run() -> Any:
        data = request.get_json(silent=True) or request.form
        dataset_id = str(data.get("dataset_id", "")).strip()
        recording_id = str(data.get("recording_id", "")).strip()
        mode = str(data.get("mode", "video")).strip().lower()
        require_labels = str(data.get("require_labels", "false")).lower() in {"true", "1", "yes"}
        if not dataset_id or not recording_id:
            return jsonify({"error": "dataset_id and recording_id are required."}), 400
        if mode not in {"video", "eeg", "fusion", "unimodal"}:
            return jsonify({"error": "Run mode must be video, eeg, fusion, or unimodal."}), 400
        if require_labels and mode == "eeg":
            return jsonify({"error": "require_labels applies to video evaluation, not EEG CSV labels."}), 400
        if mode == "fusion":
            with session_factory() as session:
                recording = session.get(Recording, f"{dataset_id}::{recording_id}")
                metadata = recording.metadata_json if recording else {}
                if not recording:
                    return jsonify({"error": "Recording not found."}), 404
                if not bool((metadata or {}).get("sync_verified")):
                    return jsonify({"error": "Multimodal fusion unavailable: Video and EEG synchronization could not be verified."}), 409
                modalities = set(session.scalars(select(StoredFile.modality).where(
                    StoredFile.recording_id == recording.id,
                    StoredFile.payload_state == "stored",
                )).all())
                missing = [name for name in ("video", "eeg") if name not in modalities]
                if missing:
                    return jsonify({"error": f"Multimodal fusion unavailable: missing {', '.join(missing)} input."}), 409
        if not run_lock.acquire(blocking=False):
            return jsonify({"error": "Another model run is already active."}), 409
        try:
            with session_factory() as session:
                lifecycle.require_available(session, dataset_id, recording_id)
        except ValueError as error:
            run_lock.release()
            return jsonify({"error": str(error)}), 409
        job_id = str(uuid.uuid4())
        job_started = time.monotonic()
        cancel_event = threading.Event()
        with jobs_lock:
            cancellations[job_id] = cancel_event
            jobs[job_id] = {
                "id": job_id, "status": "running", "dataset_id": dataset_id,
                "recording_id": recording_id, "mode": mode,
                "progress": {"stage": "queued", "stage_label": "Starting model run", "percent": 0.0, "elapsed_sec": 0.0},
            }

        def update_progress(payload: dict[str, Any]) -> None:
            with jobs_lock:
                current = jobs.get(job_id)
                if current and current.get("status") == "running":
                    if payload.get('run_id'):
                        current['run_id'] = payload['run_id']
                    current["progress"] = {**payload, "elapsed_sec": time.monotonic() - job_started}

        def worker() -> None:
            try:
                result = SeizureModelIntegration(settings, storage, session_factory).run(
                    dataset_id, recording_id, mode=mode, require_labels=require_labels,
                    progress_callback=update_progress,
                    cancel_event=cancel_event,
                )
                with jobs_lock:
                    jobs[job_id] = {
                        "id": job_id, "status": result.status, "result": asdict(result),
                        "progress": {"stage": result.status, "stage_label": "Run stopped" if result.status == 'stopped' else "Run completed", "percent": 0.0 if result.status == 'stopped' else 100.0, "elapsed_sec": time.monotonic() - job_started},
                    }
            except ModelRunStopped:
                with jobs_lock:
                    jobs[job_id] = {'id': job_id, 'status': 'stopped', 'progress': {'stage': 'stopped', 'stage_label': 'Run stopped', 'elapsed_sec': time.monotonic() - job_started}}
            except Exception as error:
                with jobs_lock:
                    jobs[job_id] = {
                        "id": job_id, "status": "failed", "error": str(error),
                        "progress": {"stage": "failed", "stage_label": "Run failed", "percent": 0.0, "elapsed_sec": time.monotonic() - job_started},
                    }
            finally:
                with jobs_lock:
                    cancellations.pop(job_id, None)
                run_lock.release()

        threading.Thread(target=worker, name=f"model-run-{job_id}", daemon=True).start()
        return jsonify(jobs[job_id]), 202

    @app.get("/api/jobs/<job_id>")
    def job(job_id: str) -> Any:
        with jobs_lock:
            payload = jobs.get(job_id)
        return jsonify(payload) if payload else (jsonify({"error": "Unknown job."}), 404)

    @app.post('/api/jobs/<identity>/stop')
    @app.post('/api/runs/<identity>/stop')
    def stop_model(identity):
        with jobs_lock:
            job_id = identity if request.path.startswith('/api/jobs/') else next((key for key, value in jobs.items() if value.get('run_id') == identity), None)
            current = jobs.get(job_id)
            event = cancellations.get(job_id)
            if not current or current.get('status') != 'running' or event is None:
                return jsonify({'error': 'This run is no longer active in this app.'}), 409
            if current.get('progress', {}).get('stage') == 'artifact_persistence':
                return jsonify({'error': 'The model has finished and is saving its results. Please wait for completion.'}), 409
            event.set()
            current['stop_requested'] = True
            return jsonify({'status': 'stopping'}), 202

    @app.get("/")
    @app.get("/<path:path>")
    def frontend(path: str = "") -> Any:
        if not FRONTEND_DIST.is_dir():
            return jsonify({"error": "React build not found.", "help": "Run npm install and npm run build in fyp_data_platform/frontend."}), 503
        requested = FRONTEND_DIST / path
        if path and requested.is_file():
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    app.extensions["fyp_engine"] = engine
    return app


def run_ui(*, config_path: str | None, host: str, port: int, debug: bool) -> None:
    app = create_app(config_path)
    print(f"SeizureAI running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
