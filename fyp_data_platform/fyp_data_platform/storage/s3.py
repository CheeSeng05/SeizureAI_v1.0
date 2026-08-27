from __future__ import annotations

from pathlib import Path
from typing import Any

from botocore.config import Config
from botocore.exceptions import ClientError

from .base import ObjectMetadata, StorageBackend
from .integrity import sha256_file
from .keys import normalize_key


class S3Storage(StorageBackend):
    """S3 adapter that assumes the bucket already exists and never changes its ACL."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str | None = None,
        profile: str | None = None,
        endpoint_url: str | None = None,
        server_side_encryption: str | None = "AES256",
        kms_key_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.server_side_encryption = server_side_encryption
        self.kms_key_id = kms_key_id
        if client is None:
            import boto3

            session = boto3.Session(profile_name=profile, region_name=region)
            client_options: dict[str, Any] = {"endpoint_url": endpoint_url}
            if endpoint_url:
                client_options["config"] = Config(s3={"addressing_style": "path"})
            client = session.client("s3", **client_options)
        self.client = client

    def _key(self, key: str) -> str:
        key = normalize_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, source: Path, key: str, *, sha256: str | None = None) -> ObjectMetadata:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        checksum = sha256 or sha256_file(source)
        size = source.stat().st_size
        existing = self.stat(key)
        if existing and existing.size_bytes == size and existing.sha256 == checksum:
            return existing

        extra: dict[str, Any] = {"Metadata": {"sha256": checksum}}
        if self.server_side_encryption:
            extra["ServerSideEncryption"] = self.server_side_encryption
        if self.kms_key_id:
            extra["SSEKMSKeyId"] = self.kms_key_id
        self.client.upload_file(str(source), self.bucket, self._key(key), ExtraArgs=extra)
        stored = self.stat(key)
        if stored is None or stored.size_bytes != size or stored.sha256 != checksum:
            raise IOError(f"S3 integrity verification failed for s3://{self.bucket}/{self._key(key)}")
        return stored

    def download_file(self, key: str, destination: Path) -> ObjectMetadata:
        metadata = self.stat(key)
        if metadata is None:
            raise FileNotFoundError(key)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial")
        self.client.download_file(self.bucket, self._key(key), str(temporary))
        if temporary.stat().st_size != metadata.size_bytes:
            temporary.unlink(missing_ok=True)
            raise IOError(f"Downloaded size mismatch for {key}")
        if metadata.sha256 and sha256_file(temporary) != metadata.sha256:
            temporary.unlink(missing_ok=True)
            raise IOError(f"Downloaded checksum mismatch for {key}")
        temporary.replace(destination)
        return metadata

    def stat(self, key: str) -> ObjectMetadata | None:
        try:
            result = self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return ObjectMetadata(
            key=normalize_key(key),
            size_bytes=int(result["ContentLength"]),
            sha256=(result.get("Metadata") or {}).get("sha256"),
            etag=str(result.get("ETag", "")).strip('"') or None,
        )
