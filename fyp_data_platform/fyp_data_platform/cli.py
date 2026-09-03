from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from botocore.exceptions import BotoCoreError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from fyp_data_platform.config.settings import PROJECT_ROOT, load_settings
from fyp_data_platform.db.models import Annotation, Dataset, Patient, Recording, StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.db.session import create_engine_and_session
from fyp_data_platform.ingest.bids import discover_bids_files
from fyp_data_platform.ingest.service import ingest_seizeit2, load_dataset_spec
from fyp_data_platform.integrations.seizure_model import SeizureModelIntegration
from fyp_data_platform.storage.factory import create_storage
from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.keys import source_key


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _settings_and_db(config_path: str | None):
    settings = load_settings(config_path)
    engine, session_factory = create_engine_and_session(settings.database_url)
    return settings, engine, session_factory


def cmd_db_upgrade(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    alembic = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic.attributes["database_url"] = settings.database_url
    command.upgrade(alembic, args.revision)
    _print({"database": settings.database_url.rsplit("@", 1)[-1], "revision": args.revision})


def cmd_inspect(args: argparse.Namespace) -> None:
    subjects = set((args.subject or []) + (args.subjects or []))
    sessions = set((args.session or []) + (args.sessions or []))
    files = discover_bids_files(
        Path(args.source), subjects=subjects, sessions=sessions
    )
    summary: dict[str, Any] = {
        "root": str(Path(args.source).resolve()),
        "files": len(files),
        "subjects": sorted({item.subject_id for item in files if item.subject_id}),
        "sessions": sorted({item.session_id for item in files if item.session_id}),
        "recordings": len({item.recording_id for item in files if item.recording_id}),
        "roles": {},
        "payload_states": {},
    }
    for item in files:
        summary["roles"][item.role] = summary["roles"].get(item.role, 0) + 1
        summary["payload_states"][item.payload_state] = (
            summary["payload_states"].get(item.payload_state, 0) + 1
        )
    _print(summary)


def cmd_ingest(args: argparse.Namespace) -> None:
    if args.metadata_only and (args.include_edf or args.include_video):
        raise ValueError("--metadata-only cannot be combined with --include-edf or --include-video")
    subjects = (args.subject or []) + (args.subjects or [])
    sessions = (args.session or []) + (args.sessions or [])
    settings, engine, session_factory = _settings_and_db(args.config)
    try:
        report = ingest_seizeit2(
            settings=settings,
            storage=None if args.dry_run else create_storage(settings),
            session_factory=session_factory,
            source=Path(args.source),
            spec=load_dataset_spec(Path(args.dataset_config) if args.dataset_config else None),
            subjects=set(subjects) if subjects else None,
            sessions=set(sessions) if sessions else None,
            include_edf=args.include_edf,
            include_video=args.include_video,
            dry_run=args.dry_run,
        )
        _print(report.to_dict())
    finally:
        engine.dispose()


def cmd_catalog_summary(args: argparse.Namespace) -> None:
    _, engine, session_factory = _settings_and_db(args.config)
    try:
        with session_factory() as session:
            _print(
                {
                    "datasets": session.scalar(select(func.count()).select_from(Dataset)),
                    "patients": session.scalar(select(func.count()).select_from(Patient)),
                    "recordings": session.scalar(select(func.count()).select_from(Recording)),
                    "files": session.scalar(select(func.count()).select_from(StoredFile)),
                    "annotations": session.scalar(select(func.count()).select_from(Annotation)),
                }
            )
    finally:
        engine.dispose()


def cmd_storage_check(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    metadata = create_storage(settings).stat(args.key)
    _print({"exists": metadata is not None, "object": asdict(metadata) if metadata else None})


def cmd_register_media(args: argparse.Namespace) -> None:
    settings, engine, session_factory = _settings_and_db(args.config)
    source = Path(args.file).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    relative = f"manual/{args.patient_id}/{args.recording_id}/{source.name}"
    key = source_key(args.dataset_id, relative)
    checksum = sha256_file(source)
    storage = create_storage(settings)
    metadata = storage.put_file(source, key, sha256=checksum)
    try:
        with session_factory.begin() as session:
            repo = CatalogRepository(session)
            repo.dataset(
                args.dataset_id,
                name=args.dataset_name,
                version=None,
                description="Manually registered development media",
                source_url=None,
                license=None,
                metadata_json={},
            )
            patient, _ = repo.patient(args.dataset_id, args.patient_id, metadata_json={})
            recording, _ = repo.recording(
                args.dataset_id,
                args.recording_id,
                patient_id=patient.id,
                session_id=args.session_id,
                task=None,
                modality=args.modality,
                duration_sec=None,
                video_fps=None,
                start_datetime=None,
                metadata_json={"development_only": True},
            )
            file_row, _ = repo.stored_file(
                args.dataset_id,
                relative,
                recording_id=recording.id,
                role="source-media",
                modality=args.modality,
                storage_backend=settings.storage_backend,
                storage_key=key,
                payload_state="stored",
                size_bytes=metadata.size_bytes,
                sha256=metadata.sha256,
                media_type=None,
                metadata_json={"development_only": True},
            )
        _print({"file_id": file_row.id, "storage_key": key, "sha256": metadata.sha256})
    finally:
        engine.dispose()


def cmd_model_run(args: argparse.Namespace) -> None:
    settings, engine, session_factory = _settings_and_db(args.config)
    try:
        result = SeizureModelIntegration(
            settings, create_storage(settings), session_factory
        ).run(
            args.dataset_id,
            args.recording_id,
            mode=args.mode,
            video_key=args.video_key,
            labels_key=args.labels_key,
            eeg_key=args.eeg_key,
            require_labels=args.require_labels,
            cleanup=args.cleanup,
        )
        _print(asdict(result))
    finally:
        engine.dispose()


def cmd_ui(args: argparse.Namespace) -> None:
    try:
        from fyp_data_platform.ui import run_ui
    except ImportError as error:
        raise RuntimeError(
            "The local UI requires Flask. Install it with: python -m pip install -e \".[ui]\""
        ) from error
    run_ui(config_path=args.config, host=args.host, port=args.port, debug=args.debug)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FYP multimodal research data platform")
    parser.add_argument("--config", help="YAML platform config; environment variables override it")
    subparsers = parser.add_subparsers(dest="command", required=True)

    db = subparsers.add_parser("db-upgrade", help="Apply Alembic migrations")
    db.add_argument("--revision", default="head")
    db.set_defaults(handler=cmd_db_upgrade)

    inspect = subparsers.add_parser("inspect-bids", help="Inspect a local/sparse BIDS checkout")
    inspect.add_argument("--source", "--dataset-path", dest="source", required=True)
    inspect.add_argument("--subject", action="append")
    inspect.add_argument("--subjects", nargs="+", default=[])
    inspect.add_argument("--session", action="append")
    inspect.add_argument("--sessions", nargs="+", default=[])
    inspect.set_defaults(handler=cmd_inspect)

    ingest = subparsers.add_parser("ingest-seizeit2", help="Ingest ds005873 metadata and annotations")
    ingest.add_argument(
        "--source", "--dataset-path", dest="source", required=True,
        help="Local or DataLad ds005873 root",
    )
    ingest.add_argument("--dataset-config", help="Dataset/subset manifest YAML")
    ingest.add_argument("--subject", action="append", help="Repeatable BIDS subject filter")
    ingest.add_argument("--subjects", nargs="+", default=[], help="One or more BIDS subject filters")
    ingest.add_argument("--session", action="append", help="Repeatable BIDS session filter")
    ingest.add_argument("--sessions", nargs="+", default=[], help="One or more BIDS session filters")
    ingest.add_argument(
        "--metadata-only",
        action="store_true",
        help="Explicit safe mode (also the default when payload flags are absent)",
    )
    ingest.add_argument("--include-edf", action="store_true", help="Store materialized EDF payloads")
    ingest.add_argument("--include-video", action="store_true", help="Store materialized video payloads")
    ingest.add_argument("--dry-run", action="store_true", help="Plan only; write neither objects nor rows")
    ingest.set_defaults(handler=cmd_ingest)

    summary = subparsers.add_parser("catalog-summary", help="Count catalog records")
    summary.set_defaults(handler=cmd_catalog_summary)

    check = subparsers.add_parser("storage-check", help="Check one logical storage key")
    check.add_argument("--key", required=True)
    check.set_defaults(handler=cmd_storage_check)

    register = subparsers.add_parser(
        "register-media", help="Store and catalog one explicitly selected development media file"
    )
    register.add_argument("--file", required=True)
    register.add_argument("--dataset-id", required=True)
    register.add_argument("--dataset-name", required=True)
    register.add_argument("--patient-id", required=True)
    register.add_argument("--recording-id", required=True)
    register.add_argument("--session-id")
    register.add_argument("--modality", choices=["video", "eeg", "labels"], required=True)
    register.set_defaults(handler=cmd_register_media)

    model = subparsers.add_parser(
        "model-run", help="Run the authoritative seizure_multimodal pipeline on stored inputs"
    )
    model.add_argument("--dataset-id", required=True)
    model.add_argument("--recording-id", required=True)
    model.add_argument(
        "--mode", choices=["video", "eeg", "unimodal", "strict_multimodal"], default="video"
    )
    model.add_argument("--video-key")
    model.add_argument("--labels-key")
    model.add_argument("--eeg-key")
    model.add_argument(
        "--require-labels",
        action="store_true",
        help="Require a registered label workbook and calculate video evaluation metrics.",
    )
    model.add_argument("--cleanup", action="store_true")
    model.set_defaults(handler=cmd_model_run)

    ui = subparsers.add_parser("ui", help="Run the local browser UI for uploads and model jobs")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8080)
    ui.add_argument("--debug", action="store_true")
    ui.set_defaults(handler=cmd_ui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        args.handler(args)
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        LookupError,
        OSError,
        subprocess.SubprocessError,
        SQLAlchemyError,
        BotoCoreError,
    ) as error:
        logging.getLogger(__name__).error("%s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
