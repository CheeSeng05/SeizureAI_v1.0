from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from fyp_data_platform.config.settings import Settings
from fyp_data_platform.db.models import StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.storage.base import StorageBackend
from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.keys import source_key

from .bids import BidsFile, dataset_description, discover_bids_files


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str = "ds005873"
    name: str = "SeizeIT2"
    version: str = "1.1.0"
    source_url: str = "https://openneuro.org/datasets/ds005873"
    license: str | None = None
    description: str | None = None
    subjects: set[str] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)


@dataclass
class IngestReport:
    dataset_id: str
    dry_run: bool
    discovered_files: int = 0
    discovered_patients: int = 0
    discovered_recordings: int = 0
    stored_files: int = 0
    referenced_payloads: int = 0
    unavailable_payloads: int = 0
    created_patients: int = 0
    created_recordings: int = 0
    created_file_rows: int = 0
    created_annotations: int = 0
    updated_rows: int = 0
    annotation_rows: int = 0
    annotation_label_counts: dict[str, int] = field(default_factory=dict)
    excluded_event_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset_spec(path: Path | None) -> DatasetSpec:
    if path is None:
        return DatasetSpec()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    dataset = data.get("dataset", data)
    subset = data.get("subset", {})
    return DatasetSpec(
        dataset_id=str(dataset.get("id", "ds005873")),
        name=str(dataset.get("name", "SeizeIT2")),
        version=str(dataset.get("version", "1.1.0")),
        source_url=str(dataset.get("source_url", "https://openneuro.org/datasets/ds005873")),
        license=dataset.get("license"),
        description=dataset.get("description"),
        subjects={str(value) for value in subset.get("subjects", [])},
        sessions={str(value) for value in subset.get("sessions", [])},
    )


def _participants(root: Path) -> dict[str, dict[str, str]]:
    path = root / "participants.tsv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["participant_id"]: {key: value for key, value in row.items() if key != "participant_id"}
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("participant_id")
        }


def _annotation_key(dataset_id: str, row: dict[str, Any]) -> str:
    identity = {
        "dataset": dataset_id,
        "recording": row["recording_id"],
        "patient": row["patient_id"],
        "start": float(row["start_sec"]),
        "end": float(row["end_sec"]),
        "source": row.get("source_tsv_path") or row.get("annotation_source"),
        "event": row.get("source_event_type"),
        "lateralization": row.get("lateralization"),
        "localization": row.get("localization"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _should_store(item: BidsFile, *, include_edf: bool, include_video: bool) -> bool:
    if not item.materializable:
        return False
    if item.is_sensor_payload:
        return include_edf and item.path.suffix.lower() == ".edf"
    if item.is_video_payload:
        return include_video
    return True


def _recording_metadata(files: list[BidsFile]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in files:
        if item.recording_id:
            result.setdefault(item.recording_id, {}).update(item.metadata)
    return result


def _seizeit2_annotations(files: list[BidsFile]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read the two canonical SeizeIT2 classes directly from BIDS events.tsv."""
    annotations: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    for item in files:
        if item.role != "annotation" or not item.materializable or not item.recording_id or not item.subject_id:
            continue
        with item.path.open("r", encoding="utf-8-sig", newline="") as handle:
            for source in csv.DictReader(handle, delimiter="\t"):
                event_type = str(source.get("eventType") or source.get("trial_type") or "").strip()
                lowered = event_type.lower()
                label = "normal" if lowered in {"bckg", "background", "normal"} else "ictal" if lowered.startswith(("sz", "seizure")) else None
                if label is None:
                    excluded[event_type or "unknown"] = excluded.get(event_type or "unknown", 0) + 1
                    continue
                try:
                    start = float(source["onset"])
                    duration = float(source["duration"])
                except (KeyError, TypeError, ValueError):
                    excluded["invalid_time"] = excluded.get("invalid_time", 0) + 1
                    continue
                annotations.append({
                    "dataset": "seizeit2", "media_id": item.recording_id,
                    "recording_id": item.recording_id, "patient_id": item.subject_id,
                    "session_id": item.session_id, "start_sec": start,
                    "end_sec": start + duration, "label": label,
                    "source_event_type": event_type,
                    "lateralization": _optional_string(source.get("lateralization")),
                    "localization": _optional_string(source.get("localization")),
                    "confidence": _float_or_none(source.get("confidence")),
                    "derived_label": label, "annotation_source": item.relative_path,
                    "source_tsv_path": item.relative_path,
                })
    return annotations, excluded


def ingest_seizeit2(
    *,
    settings: Settings,
    storage: StorageBackend | None,
    session_factory: sessionmaker,
    source: Path,
    spec: DatasetSpec,
    subjects: set[str] | None = None,
    sessions: set[str] | None = None,
    include_edf: bool = False,
    include_video: bool = False,
    dry_run: bool = False,
) -> IngestReport:
    source = Path(source).resolve()
    selected_subjects = subjects if subjects is not None else spec.subjects
    selected_sessions = sessions if sessions is not None else spec.sessions
    files = discover_bids_files(
        source,
        subjects=selected_subjects,
        sessions=selected_sessions,
        data_types={"eeg"},
    )
    report = IngestReport(dataset_id=spec.dataset_id, dry_run=dry_run, discovered_files=len(files))
    if not files:
        report.warnings.append("No files matched the requested dataset subset.")
        return report

    annotations, excluded = _seizeit2_annotations(files)
    report.annotation_rows = len(annotations)
    report.excluded_event_counts = excluded
    for row in annotations:
        label = str(row["label"])
        report.annotation_label_counts[label] = report.annotation_label_counts.get(label, 0) + 1

    for item in files:
        if _should_store(item, include_edf=include_edf, include_video=include_video):
            report.stored_files += 1
        elif item.materializable:
            report.referenced_payloads += 1
        else:
            report.unavailable_payloads += 1
    report.discovered_patients = len(
        {item.subject_id for item in files if item.subject_id}
        | {str(row["patient_id"]) for row in annotations}
    )
    report.discovered_recordings = len(
        {item.recording_id for item in files if item.recording_id}
        | {str(row["recording_id"]) for row in annotations}
    )
    if dry_run:
        LOGGER.info("SeizeIT2 dry-run plan: %s", report.to_dict())
        return report
    if storage is None:
        raise ValueError("A storage backend is required for non-dry-run ingestion.")

    description = dataset_description(files)
    participants = _participants(source)
    recordings = _recording_metadata(files)
    stored_objects = {}
    for item in files:
        if not _should_store(item, include_edf=include_edf, include_video=include_video):
            continue
        checksum = sha256_file(item.path)
        stored_objects[item.relative_path] = storage.put_file(
            item.path, source_key(spec.dataset_id, item.relative_path), sha256=checksum
        )

    with session_factory.begin() as session:
        repo = CatalogRepository(session)
        _, created = repo.dataset(
            spec.dataset_id,
            name=spec.name,
            version=spec.version,
            description=spec.description or description.get("DatasetDescription"),
            source_url=spec.source_url,
            license=spec.license or description.get("License"),
            metadata_json=_json_safe(description),
        )
        report.updated_rows += int(not created)

        patient_ids = sorted(
            {item.subject_id for item in files if item.subject_id}
            | {str(row["patient_id"]) for row in annotations}
        )
        for source_patient_id in patient_ids:
            _, created = repo.patient(
                spec.dataset_id,
                source_patient_id,
                metadata_json=_json_safe(participants.get(source_patient_id, {})),
            )
            report.created_patients += int(created)
            report.updated_rows += int(not created)

        recording_sources = {
            item.recording_id: (item.subject_id, item.session_id, item.modality)
            for item in files
            if item.recording_id and item.subject_id
        }
        for row in annotations:
            source_recording_id = str(row["recording_id"])
            recording_sources.setdefault(
                source_recording_id,
                (str(row["patient_id"]), row.get("session_id"), "eeg"),
            )
        for source_recording_id, (patient_id, session_id, modality) in sorted(recording_sources.items()):
            meta = recordings.get(source_recording_id, {})
            duration = meta.get("RecordingDuration")
            _, created = repo.recording(
                spec.dataset_id,
                source_recording_id,
                patient_id=f"{spec.dataset_id}::{patient_id}",
                session_id=session_id,
                task=_entity_value(source_recording_id, "task"),
                modality=modality,
                duration_sec=float(duration) if duration not in (None, "n/a") else None,
                video_fps=None,
                start_datetime=None,
                metadata_json=_json_safe(meta),
            )
            report.created_recordings += int(created)
            report.updated_rows += int(not created)

        for item in files:
            stored = _should_store(item, include_edf=include_edf, include_video=include_video)
            stored_object = stored_objects.get(item.relative_path)
            checksum = stored_object.sha256 if stored_object else None
            file_size = (
                stored_object.size_bytes
                if stored_object
                else item.path.stat().st_size if item.materializable else None
            )
            key = source_key(spec.dataset_id, item.relative_path)
            _, created = repo.stored_file(
                spec.dataset_id,
                item.relative_path,
                recording_id=(f"{spec.dataset_id}::{item.recording_id}" if item.recording_id else None),
                role=item.role,
                modality=item.modality,
                storage_backend=settings.storage_backend if stored else "source",
                storage_key=key if stored else f"{spec.dataset_id}/{item.relative_path}",
                payload_state="stored" if stored else item.payload_state if not item.materializable else "source-reference",
                size_bytes=file_size,
                sha256=checksum,
                media_type=item.media_type,
                metadata_json=_json_safe(item.metadata),
            )
            report.created_file_rows += int(created)
            report.updated_rows += int(not created)

        canonical_fields = {
            "dataset",
            "media_id",
            "recording_id",
            "patient_id",
            "start_sec",
            "end_sec",
            "label",
            "source_event_type",
            "lateralization",
            "localization",
            "confidence",
            "derived_label",
            "annotation_source",
            "source_tsv_path",
        }
        source_files = {
            item.source_relative_path: item
            for item in session.scalars(
                select(StoredFile).where(StoredFile.dataset_id == spec.dataset_id)
            )
        }
        for row in annotations:
            key = _annotation_key(spec.dataset_id, row)
            source_path = _optional_string(
                row.get("source_tsv_path") or row.get("annotation_source")
            )
            _, created = repo.annotation(
                key,
                dataset_id=spec.dataset_id,
                recording_id=f"{spec.dataset_id}::{row['recording_id']}",
                patient_id=f"{spec.dataset_id}::{row['patient_id']}",
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                label=str(row["label"]),
                source_event_type=_optional_string(row.get("source_event_type")),
                lateralization=_optional_string(row.get("lateralization")),
                localization=_optional_string(row.get("localization")),
                confidence=(float(row["confidence"]) if row.get("confidence") is not None else None),
                derived_label=_optional_string(row.get("derived_label")),
                source_file_id=source_files[source_path].id if source_path in source_files else None,
                source_relative_path=source_path,
                metadata_json=_json_safe(
                    {key: value for key, value in row.items() if key not in canonical_fields}
                ),
            )
            report.created_annotations += int(created)
            report.updated_rows += int(not created)
    LOGGER.info("SeizeIT2 ingestion completed: %s", report.to_dict())
    return report


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "", "n/a") else str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value in (None, "", "n/a") else float(value)
    except (TypeError, ValueError):
        return None


def _entity_value(recording_id: str, name: str) -> str | None:
    for part in recording_id.split("_"):
        if part.startswith(f"{name}-"):
            return part.split("-", 1)[1]
    return None
