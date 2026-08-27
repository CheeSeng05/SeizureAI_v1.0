from __future__ import annotations

from fyp_data_platform.config.settings import Settings

from .base import StorageBackend
from .local import LocalStorage
from .s3 import S3Storage


def create_storage(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorage(settings.local_storage_root)
    return S3Storage(
        settings.s3_bucket or "",
        prefix=settings.s3_prefix,
        region=settings.aws_region,
        profile=settings.aws_profile,
        endpoint_url=settings.s3_endpoint_url,
        server_side_encryption=settings.s3_server_side_encryption,
        kms_key_id=settings.s3_kms_key_id,
    )
