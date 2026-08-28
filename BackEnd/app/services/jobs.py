"""Running inference as a background job.

Inference on a full recording takes minutes, so an HTTP request cannot wait for
it. The job is spawned as a subprocess, its output is captured, and the client
polls for status.

This service does NOT reimplement any part of the model. It invokes the model
repository's own CLI, so the dashboard can never disagree with what the
documented command would produce.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

def _kill_tree(proc: subprocess.Popen, grace_sec: float = 5.0) -> None:
    """Kill a process and everything it spawned.

    terminate() signals only the immediate child. The model CLI spawns its own
    workers, so terminating the launcher leaves those running - they keep
    consuming a full core each and compete with the next job for CPU. On
    Windows taskkill /T walks the tree; elsewhere a process group is used.
    """
    if proc.poll() is not None:
        return

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()

    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()          # last resort on the launcher itself


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

MAX_LOG_LINES = 400          # keep the tail; a long run can print a lot


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    job_id: str
    upload_id: str
    command: list[str]
    cwd: str
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    last_output_at: Optional[str] = None
    log: list[str] = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d["log"] = self.log[-MAX_LOG_LINES:]
        return d


class JobRunner:
    """Single-worker job registry.

    Deliberately serial: inference is GPU-bound, and running two at once on one
    machine makes both slower and can exhaust VRAM. A second request queues.
    """

    def __init__(self, state_dir: Path, timeout_sec: int = 60 * 60):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._active: Optional[subprocess.Popen] = None
        self._load()

    # -- persistence -------------------------------------------------------
    def _path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _save(self, job: Job) -> None:
        self._path(job.job_id).write_text(json.dumps(asdict(job), indent=2))

    def _load(self) -> None:
        for p in self.state_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                job = Job(**data)
                # A job marked running in a previous process is not running now.
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.error = "interrupted — the server restarted"
                self._jobs[job.job_id] = job
            except (json.JSONDecodeError, TypeError):
                continue

    # -- api ---------------------------------------------------------------
    def busy(self) -> bool:
        with self._lock:
            return any(j.status == "running" for j in self._jobs.values())

    def submit(self, upload_id: str, command: list[str], cwd: Path,
               job_id: str | None = None) -> Job:
        job = Job(job_id=job_id or uuid.uuid4().hex[:12], upload_id=upload_id,
                  command=command, cwd=str(cwd))
        with self._lock:
            self._jobs[job.job_id] = job
        self._save(job)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status != "running":
            return False
        with self._lock:
            proc = self._active
        if proc and proc.poll() is None:
            _kill_tree(proc)
            job.status = "cancelled"
            job.finished_at = _now()
            self._save(job)
            return True
        return False

    # -- worker ------------------------------------------------------------
    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = _now()
        self._save(job)

        try:
            proc = subprocess.Popen(
                job.command,
                cwd=job.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # interleave, so ordering is real
                text=True,
                bufsize=1,
                # Own process group / job object, so the whole tree can be
                # signalled together rather than just the launcher.
                start_new_session=(sys.platform != "win32"),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               if sys.platform == "win32" else 0),
            )
            with self._lock:
                self._active = proc

            # The stdout loop below blocks until the process exits, so a
            # wait(timeout=...) after it would never fire while the job is
            # stuck. A watchdog thread enforces the limit independently.
            timed_out = threading.Event()

            def _watchdog():
                if proc.poll() is None:
                    timed_out.set()
                    _kill_tree(proc)

            watchdog = threading.Timer(self.timeout_sec, _watchdog)
            watchdog.daemon = True
            watchdog.start()

            assert proc.stdout is not None
            for line in proc.stdout:
                job.log.append(line.rstrip())
                job.last_output_at = _now()
                if len(job.log) > MAX_LOG_LINES * 2:
                    del job.log[:MAX_LOG_LINES]     # trim, keep the tail
                self._save(job)

            proc.wait()
            watchdog.cancel()
            job.exit_code = proc.returncode

            if timed_out.is_set():
                job.status = "failed"
                job.error = (f"inference exceeded {self.timeout_sec}s and was "
                             f"stopped. Raise SEIZUREAI_JOB_TIMEOUT if the "
                             f"recording is long and this machine has no GPU.")
            elif job.status != "cancelled":
                job.status = "succeeded" if proc.returncode == 0 else "failed"
                if proc.returncode != 0:
                    job.error = f"inference exited with code {proc.returncode}"

        except FileNotFoundError as e:
            job.status = "failed"
            job.error = (f"could not start inference: {e}. Check the command and "
                         f"model_repo path in the backend settings.")
        except Exception as e:                     # noqa: BLE001
            job.status = "failed"
            job.error = str(e)
        finally:
            job.finished_at = _now()
            with self._lock:
                self._active = None
            self._save(job)