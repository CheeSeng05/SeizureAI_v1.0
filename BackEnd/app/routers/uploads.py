from __future__ import annotations

import shlex
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..config import get_settings
from ..dependencies import get_jobs, get_uploads
from ..services import config_builder, progress
from ..services.config_builder import ConfigError
from ..services.jobs import JobRunner
from ..services.uploads import Upload, UploadError, UploadStore

router = APIRouter(prefix="/api", tags=["uploads"])


def build_command(template: str, video_path: str) -> list[str]:
    """Split a command string into argv, safely on Windows.

    shlex's default POSIX mode treats backslash as an escape character, so a
    path like C:\\Users\\me\\python.exe loses every separator and becomes
    C:Usersmepython.exe. posix=False preserves them but leaves quotes attached
    to the token, so those are stripped afterwards.
    """
    parts = []
    for raw in shlex.split(template, posix=False):
        token = raw
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        parts.append(token.replace("{video}", video_path))
    return parts




@router.post("/uploads", status_code=201)
async def create_upload(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    seizure_id: Optional[str] = Form(None),
    seizure_type: Optional[str] = Form(None),
    eeg_onset_sec: Optional[float] = Form(None),
    clinical_onset_sec: Optional[float] = Form(None),
    uploads: UploadStore = Depends(get_uploads),
):
    """Stage a recording.

    Onsets are optional here but required for metrics: without them there is no
    reference to score predictions against, and the run yields probabilities
    only. The UI says so rather than failing silently later.
    """
    try:
        record = uploads.save_stream(
            file.file, file.filename or "recording.mp4",
            patient_id=patient_id, seizure_id=seizure_id,
            seizure_type=seizure_type,
            eeg_onset_sec=eeg_onset_sec,
            clinical_onset_sec=clinical_onset_sec,
        )
    except UploadError as e:
        raise HTTPException(400, str(e))
    except OSError as e:
        raise HTTPException(507, f"could not write to staging directory: {e}")

    return {
        **record.__dict__,
        "scorable": record.eeg_onset_sec is not None and record.clinical_onset_sec is not None,
    }


@router.get("/uploads")
def list_uploads(uploads: UploadStore = Depends(get_uploads)):
    settings = get_settings()
    return {
        "uploads": [u.__dict__ for u in uploads.list()],
        "free_bytes": uploads.free_space_bytes(),
        "max_upload_bytes": settings.max_upload_bytes,
    }


@router.delete("/uploads/{upload_id}", status_code=204)
def delete_upload(upload_id: str, uploads: UploadStore = Depends(get_uploads)):
    if not uploads.delete(upload_id):
        raise HTTPException(404, "no such upload")


@router.post("/uploads/{upload_id}/analyse", status_code=202)
def analyse(
    upload_id: str,
    uploads: UploadStore = Depends(get_uploads),
    jobs: JobRunner = Depends(get_jobs),
):
    """Kick off inference in the model repository.

    Serial by design: inference is GPU-bound, so a second request while one is
    running is refused rather than queued silently behind it.
    """
    record = uploads.get(upload_id)
    if not record:
        raise HTTPException(404, "no such upload")
    if not Path(record.stored_path).exists():
        raise HTTPException(410, "the staged file is gone; upload it again")
    if jobs.busy():
        raise HTTPException(409, "another analysis is already running")

    settings = get_settings()
    if not settings.model_repo.is_dir():
        raise HTTPException(
            500,
            f"model repository not found at {settings.model_repo.resolve()}. "
            "Set SEIZUREAI_MODEL_REPO.",
        )

    job_id = uuid.uuid4().hex[:12]

    # The model reads its input from a config file, so give it one naming this
    # upload rather than the recording pinned in the repository's base config.
    config_arg = None
    if settings.generate_config:
        try:
            cfg = config_builder.build(
                config_builder.load_base(settings.model_repo, settings.base_config),
                video_path=Path(record.stored_path),
                sha256=record.sha256,
                patient_id=record.patient_id,
                seizure_id=record.seizure_id,
                seizure_type=record.seizure_type,
                job_id=job_id,
            )
        except ConfigError as e:
            raise HTTPException(500, str(e))
        written = config_builder.write(cfg, settings.model_repo, job_id)
        config_arg = config_builder.relative_to_repo(written, settings.model_repo)

    command = build_command(settings.inference_command, record.stored_path)
    if config_arg and "--config" not in command:
        command += ["--config", config_arg]

    exe = Path(command[0])
    if exe.is_absolute() and not exe.exists():
        raise HTTPException(
            400,
            f"interpreter not found: {command[0]}. Check SEIZUREAI_INFERENCE_CMD.",
        )

    job = jobs.submit(upload_id, command, settings.model_repo, job_id=job_id)
    return _with_progress(job, jobs)


@router.get("/jobs")
def list_jobs(jobs: JobRunner = Depends(get_jobs)):
    return [_with_progress(j, jobs) for j in jobs.list()]


def _with_progress(job, jobs: JobRunner) -> dict:
    p = progress.compute(
        status=job.status, started_at=job.started_at, finished_at=job.finished_at,
        log=job.log, last_output_at=job.last_output_at, job_dir=jobs.state_dir,
    )
    return {**job.public(), "progress": p.__dict__}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, jobs: JobRunner = Depends(get_jobs)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _with_progress(job, jobs)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, jobs: JobRunner = Depends(get_jobs)):
    if not jobs.cancel(job_id):
        raise HTTPException(409, "job is not running")
    return {"cancelled": job_id}


@router.get("/inference-config")
def inference_config():
    """What the server would actually execute.

    Command strings pass through shell quoting, env vars, and argv splitting;
    when a launch fails it is far quicker to look at the resolved argv than to
    guess which layer mangled it.
    """
    settings = get_settings()
    command = build_command(settings.inference_command, "<VIDEO_PATH>")
    exe = Path(command[0])
    return {
        "raw": settings.inference_command,
        "argv": command,
        "interpreter": command[0] if command else None,
        "interpreter_exists": exe.exists() if exe.is_absolute() else "relative path",
        "model_repo": str(settings.model_repo.resolve()),
        "model_repo_exists": settings.model_repo.is_dir(),
        "staging_dir": str(settings.staging_dir.resolve()),
        "job_timeout_sec": settings.job_timeout_sec,
    }


@router.get("/uploads/{upload_id}/config-preview")
def config_preview(upload_id: str, uploads: UploadStore = Depends(get_uploads)):
    """The exact config a run would use.

    Inference takes hours on CPU, so being able to check what it will analyse
    before starting is worth an endpoint.
    """
    record = uploads.get(upload_id)
    if not record:
        raise HTTPException(404, "no such upload")
    settings = get_settings()
    try:
        cfg = config_builder.build(
            config_builder.load_base(settings.model_repo, settings.base_config),
            video_path=Path(record.stored_path), sha256=record.sha256,
            patient_id=record.patient_id, seizure_id=record.seizure_id,
            seizure_type=record.seizure_type, job_id="preview",
        )
    except ConfigError as e:
        raise HTTPException(500, str(e))

    prov = cfg.get("data", {}).get("video_provenance", {})
    return {
        "video": cfg["data"]["video"],
        "provenance": prov,
        "labels_file": cfg["data"].get("video_labels"),
        "identified": bool(record.patient_id and record.seizure_id),
        "note": (
            "Onset times are read from the labels workbook using the patient "
            "and seizure IDs. If this recording has no row there, the run "
            "produces probabilities but no metrics."
        ),
    }