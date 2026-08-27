from __future__ import annotations

from fyp_data_platform.config import settings as settings_module


def test_docker_config_loads_local_s3_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *args, **kwargs: None)
    for name in (
        "FYP_STORAGE_BACKEND",
        "S3_BUCKET",
        "S3_ENDPOINT_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "S3_PREFIX",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = settings_module.load_settings(
        settings_module.PROJECT_ROOT / "configs" / "docker.example.yaml"
    )

    assert settings.environment == "docker"
    assert settings.storage_backend == "s3"
    assert settings.s3_bucket == "fyp-local"
    assert settings.s3_endpoint_url == "http://localhost:9000"
    assert settings.s3_prefix == "fyp-dev"
