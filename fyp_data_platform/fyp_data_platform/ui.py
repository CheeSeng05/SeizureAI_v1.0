from __future__ import annotations

import csv
import mimetypes
import tempfile
import threading
import uuid
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from fyp_data_platform.config.settings import PROJECT_ROOT, Settings, load_settings
from fyp_data_platform.compute import compute_info
from fyp_data_platform.db.models import Dataset, ModelRun, Patient, Recording, StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.db.session import create_engine_and_session
from fyp_data_platform.integrations.seizure_model import SeizureModelIntegration
from fyp_data_platform.storage.factory import create_storage
from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.keys import safe_entity, source_key


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _recording_source_id(recording_id: str, file_stem: str, count: int) -> str:
    stem = safe_entity(file_stem)
    if recording_id and count == 1:
        return safe_entity(recording_id)
    return f"{safe_entity(recording_id)}-{stem}" if recording_id else stem


def _register_video(
    *, settings: Settings, storage: Any, session_factory: Any, upload: FileStorage,
    dataset_id: str, dataset_name: str, patient_id: str, recording_id: str,
    file_count: int,
) -> dict[str, Any]:
    filename = secure_filename(upload.filename or "")
    if not filename:
        raise ValueError("Uploaded file is missing a filename.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video file extension: {suffix}")
    dataset_id = safe_entity(dataset_id or "ui-videos")
    dataset_name = dataset_name or "UI uploaded videos"
    patient_id = safe_entity(patient_id or "local-patient-001")
    source_recording_id = _recording_source_id(recording_id, Path(filename).stem, file_count)
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
            session_id=None, task=None, modality="video", duration_sec=None,
            video_fps=None, start_datetime=None, metadata_json={"uploaded_via": "web"},
        )
        file_row, _ = repo.stored_file(
            dataset_id, relative, recording_id=recording.id, role="source-media",
            modality="video", storage_backend=settings.storage_backend,
            storage_key=key, payload_state="stored", size_bytes=metadata.size_bytes,
            sha256=metadata.sha256, media_type=None,
            metadata_json={"uploaded_via": "web", "original_filename": filename},
        )
    return {
        "file_id": file_row.id, "dataset_id": dataset_id,
        "patient_id": patient_id, "recording_id": source_recording_id,
        "filename": filename, "storage_key": key,
        "sha256": metadata.sha256, "size_bytes": metadata.size_bytes,
    }


def _metric_documents(row: ModelRun) -> list[dict[str, Any]]:
    values = ((row.metrics_json or {}).get("metrics") or {}).values()
    return [value for value in values if isinstance(value, dict)]


def _metric_body(document: dict[str, Any]) -> dict[str, Any]:
    nested = document.get("metrics")
    return nested if isinstance(nested, dict) else document


def _best_metric_document(row: ModelRun) -> dict[str, Any]:
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


def _prediction_windows(storage: Any, row: ModelRun) -> list[dict[str, Any]]:
    artifact = _find_artifact(
        row, "video_only/predictions.csv", "predictions.csv",
        "video_only/probability_timeline.csv", "probability_timeline.csv",
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
            csv_row.get("video_probability") or csv_row.get("probability")
            or csv_row.get("seizure_probability")
        )
        prediction_value = (
            csv_row.get("video_prediction") or csv_row.get("prediction")
            or csv_row.get("predicted_label")
        )
        try:
            prediction = int(float(prediction_value))
        except (TypeError, ValueError):
            prediction = 0
        windows.append({
            "start_sec": start, "end_sec": end, "duration_sec": max(end - start, 0.0),
            "probability": probability, "prediction": prediction,
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
    return {
        "id": row.id, "status": row.status, "modality": row.modality,
        "dataset_id": config.get("dataset_id"), "recording_id": config.get("recording_id"),
        "compute": config.get("compute"),
        "require_labels": bool(config.get("require_labels")), "artifact_count": len(artifacts),
        "output_prefix": str(artifacts[0].get("key", "")).rsplit("/", 1)[0] if artifacts else None,
        "outcome": _outcome(row), "error": payload.get("error"),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


def _run_result(storage: Any, row: ModelRun) -> dict[str, Any]:
    windows = _prediction_windows(storage, row)
    document = _best_metric_document(row)
    body = _metric_body(document)
    run_info = document.get("run", {}) if isinstance(document.get("run"), dict) else {}
    video_info = run_info.get("video", {}) if isinstance(run_info.get("video"), dict) else {}
    timeline = _find_artifact(row, "video_only/timeline.png", "timeline.png")
    return {
        **_run_summary(row),
        "duration_sec": video_info.get("duration_sec") or max((item["end_sec"] for item in windows), default=None),
        "metrics": body, "run": run_info, "windows": windows,
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
        from flask import Flask, Response, jsonify, request, send_from_directory
    except ImportError as error:
        raise RuntimeError("Install the web dependencies with: pip install -e '.[ui]'") from error
    settings = load_settings(config_path)
    engine, session_factory = create_engine_and_session(settings.database_url)
    storage = create_storage(settings)
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()
    run_lock = threading.Lock()
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024

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
        with session_factory() as session:
            videos = session.scalars(
                select(StoredFile).where(StoredFile.modality == "video", StoredFile.payload_state == "stored")
                .order_by(StoredFile.created_at.desc()).limit(100)
            ).all()
            runs = session.scalars(select(ModelRun).order_by(ModelRun.started_at.desc()).limit(100)).all()
        video_rows = []
        for row in videos:
            source_id = row.recording_id.split("::", 1)[1] if row.recording_id and "::" in row.recording_id else ""
            video_rows.append({
                "id": row.id, "dataset_id": row.dataset_id, "recording_id": row.recording_id,
                "recording_source_id": source_id, "filename": Path(row.source_relative_path).name,
                "storage_key": row.storage_key, "size_bytes": row.size_bytes, "sha256": row.sha256,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })
        return jsonify({"videos": video_rows, "runs": [_run_summary(row) for row in runs]})

    @app.get("/api/runs/<run_id>")
    def run_result(run_id: str) -> Any:
        with session_factory() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                return jsonify({"error": "Unknown model run."}), 404
            try:
                return jsonify(_run_result(storage, row))
            except Exception as error:
                return jsonify({"error": str(error)}), 500

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
    def upload() -> Any:
        files = [item for item in request.files.getlist("videos") if item.filename]
        if not files:
            return jsonify({"error": "Choose at least one video file."}), 400
        try:
            uploaded = [
                _register_video(
                    settings=settings, storage=storage, session_factory=session_factory, upload=item,
                    dataset_id=request.form.get("dataset_id", "ui-videos"),
                    dataset_name=request.form.get("dataset_name", "UI uploaded videos"),
                    patient_id=request.form.get("patient_id", "local-patient-001"),
                    recording_id=request.form.get("recording_id", ""), file_count=len(files),
                ) for item in files
            ]
        except Exception as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({"uploaded": uploaded}), 201

    @app.post("/api/runs")
    def start_run() -> Any:
        data = request.get_json(silent=True) or request.form
        dataset_id = str(data.get("dataset_id", "")).strip()
        recording_id = str(data.get("recording_id", "")).strip()
        require_labels = str(data.get("require_labels", "false")).lower() in {"true", "1", "yes"}
        if not dataset_id or not recording_id:
            return jsonify({"error": "dataset_id and recording_id are required."}), 400
        if not run_lock.acquire(blocking=False):
            return jsonify({"error": "Another model run is already active."}), 409
        job_id = str(uuid.uuid4())
        with jobs_lock:
            jobs[job_id] = {"id": job_id, "status": "running", "dataset_id": dataset_id, "recording_id": recording_id}

        def worker() -> None:
            try:
                result = SeizureModelIntegration(settings, storage, session_factory).run(
                    dataset_id, recording_id, mode="video", require_labels=require_labels
                )
                with jobs_lock:
                    jobs[job_id] = {"id": job_id, "status": result.status, "result": asdict(result)}
            except Exception as error:
                with jobs_lock:
                    jobs[job_id] = {"id": job_id, "status": "failed", "error": str(error)}
            finally:
                run_lock.release()

        threading.Thread(target=worker, name=f"model-run-{job_id}", daemon=True).start()
        return jsonify(jobs[job_id]), 202

    @app.get("/api/jobs/<job_id>")
    def job(job_id: str) -> Any:
        with jobs_lock:
            payload = jobs.get(job_id)
        return jsonify(payload) if payload else (jsonify({"error": "Unknown job."}), 404)

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
