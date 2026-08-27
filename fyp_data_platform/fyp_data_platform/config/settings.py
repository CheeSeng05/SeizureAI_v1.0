from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _env(name: str, fallback: Any = None) -> Any:
    value = os.getenv(name)
    return fallback if value is None or value == "" else value


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _configured_path(env_name: str, fallback: str | Path, yaml_base: Path) -> Path:
    env_value = os.getenv(env_name)
    # Relative .env/environment paths are project-relative; YAML paths are file-relative.
    return _path(env_value, PROJECT_ROOT) if env_value else _path(fallback, yaml_base)


@dataclass(frozen=True)
class Settings:
    environment: str
    storage_backend: str
    local_storage_root: Path
    database_url: str
    s3_bucket: str | None
    s3_endpoint_url: str | None
    aws_region: str | None
    s3_prefix: str
    aws_profile: str | None
    s3_server_side_encryption: str | None
    s3_kms_key_id: str | None
    model_project_path: Path
    model_python: Path | None
    body_pose_project_path: Path
    body_pose_python: Path | None
    work_root: Path
    log_level: str

    def validate(self) -> None:
        if self.storage_backend not in {"local", "s3"}:
            raise ValueError("storage.backend must be 'local' or 's3'.")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("S3_BUCKET or storage.s3.bucket is required for S3 mode.")
        if not self.database_url:
            raise ValueError("DATABASE_URL or database.url is required.")


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    selected = config_path or os.getenv("FYP_CONFIG")
    data: dict[str, Any] = {}
    base = PROJECT_ROOT
    if selected:
        path = Path(selected).expanduser()
        path = path if path.is_absolute() else (Path.cwd() / path)
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        base = path.parent

    local_root = _env(
        "FYP_LOCAL_STORAGE_ROOT", _nested(data, "storage", "local_root", default="data")
    )
    body_project = _env(
        "BODY_POSE_PROJECT_PATH",
        _nested(data, "body_pose", "project_path", default="../BodyPose/body_pose_fyp"),
    )
    body_python_value = _env(
        "BODY_POSE_PYTHON", _nested(data, "body_pose", "python", default=None)
    )
    model_project = _env(
        "FYP_MODEL_PROJECT_PATH",
        _nested(data, "model", "project_path", default="../FYPModels"),
    )
    model_python_value = _env(
        "FYP_MODEL_PYTHON", _nested(data, "model", "python", default=None)
    )
    work_root = _env(
        "FYP_WORK_ROOT",
        _nested(data, "model", "work_root", default=_nested(data, "body_pose", "work_root", default="work")),
    )

    settings = Settings(
        environment=str(_env("FYP_ENVIRONMENT", data.get("environment", "local"))).lower(),
        storage_backend=str(
            _env("FYP_STORAGE_BACKEND", _nested(data, "storage", "backend", default="local"))
        ).lower(),
        local_storage_root=_configured_path("FYP_LOCAL_STORAGE_ROOT", local_root, base),
        database_url=str(
            _env(
                "DATABASE_URL",
                _nested(
                    data,
                    "database",
                    "url",
                    default="postgresql+psycopg://fyp_user:change_me@localhost:5432/fyp_data",
                ),
            )
            or ""
        ),
        s3_bucket=_env("S3_BUCKET", _nested(data, "storage", "s3", "bucket")),
        s3_endpoint_url=_env(
            "S3_ENDPOINT_URL", _nested(data, "storage", "s3", "endpoint_url")
        ),
        aws_region=_env(
            "AWS_REGION",
            _env("AWS_DEFAULT_REGION", _nested(data, "storage", "s3", "region")),
        ),
        s3_prefix=str(_env("S3_PREFIX", _nested(data, "storage", "s3", "prefix", default="")) or ""),
        aws_profile=_env("AWS_PROFILE", _nested(data, "storage", "s3", "profile")),
        s3_server_side_encryption=_env(
            "S3_SERVER_SIDE_ENCRYPTION",
            _nested(data, "storage", "s3", "server_side_encryption", default="AES256"),
        ),
        s3_kms_key_id=_env(
            "S3_KMS_KEY_ID", _nested(data, "storage", "s3", "kms_key_id")
        ),
        model_project_path=_configured_path("FYP_MODEL_PROJECT_PATH", model_project, base),
        model_python=(
            _configured_path("FYP_MODEL_PYTHON", model_python_value, base)
            if model_python_value
            else None
        ),
        body_pose_project_path=_configured_path("BODY_POSE_PROJECT_PATH", body_project, base),
        body_pose_python=(
            _configured_path("BODY_POSE_PYTHON", body_python_value, base)
            if body_python_value
            else None
        ),
        work_root=_configured_path("FYP_WORK_ROOT", work_root, base),
        log_level=str(_env("LOG_LEVEL", _nested(data, "logging", "level", default="INFO"))).upper(),
    )
    settings.validate()
    return settings
