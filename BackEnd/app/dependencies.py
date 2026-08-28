from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .services.jobs import JobRunner
from .services.results_store import ResultsStore
from .services.uploads import UploadStore


@lru_cache
def get_store() -> ResultsStore:
    return ResultsStore(get_settings().results_dir)


@lru_cache
def get_uploads() -> UploadStore:
    s = get_settings()
    return UploadStore(s.staging_dir, s.max_upload_bytes)


@lru_cache
def get_jobs() -> JobRunner:
    s = get_settings()
    return JobRunner(s.jobs_dir, s.job_timeout_sec)