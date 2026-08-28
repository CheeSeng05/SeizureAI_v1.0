"""Configuration. Point RESULTS_DIR at the model repo's results/ folder."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.results_dir = Path(
            os.environ.get("SEIZUREAI_RESULTS_DIR", "../FYPModels/results")
        ).expanduser()
        # Vite's dev server; add the deployed origin for production.
        self.cors_origins = os.environ.get(
            "SEIZUREAI_CORS", "http://localhost:5173,http://localhost:3000"
        ).split(",")

        # Where uploaded recordings are staged before inference.
        self.staging_dir = Path(
            os.environ.get("SEIZUREAI_STAGING_DIR", "./staging")
        ).expanduser()

        # Job state, so status survives a server restart.
        self.jobs_dir = Path(os.environ.get("SEIZUREAI_JOBS_DIR", "./jobs")).expanduser()

        # The model repository — inference is invoked there, never reimplemented.
        self.model_repo = Path(
            os.environ.get("SEIZUREAI_MODEL_REPO", "../FYPModels")
        ).expanduser()

        # The command that runs inference. {video} is substituted with the
        # staged file path. Override if your CLI differs.
        self.inference_command = os.environ.get(
            "SEIZUREAI_INFERENCE_CMD",
            "python -m seizure_multimodal test --mode video",
        )

        self.max_upload_bytes = int(
            os.environ.get("SEIZUREAI_MAX_UPLOAD_GB", "8") ) * 1024**3
        self.job_timeout_sec = int(os.environ.get("SEIZUREAI_JOB_TIMEOUT", str(60 * 60)))

        # Write a per-upload config so the model analyses the uploaded file
        # rather than the recording pinned in its base config.
        self.generate_config = os.environ.get(
            "SEIZUREAI_GENERATE_CONFIG", "true").lower() not in {"0", "false", "no"}
        self.base_config = os.environ.get("SEIZUREAI_BASE_CONFIG", "inference.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()