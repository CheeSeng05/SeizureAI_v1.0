from __future__ import annotations

from pathlib import Path
from io import BytesIO

import boto3
from botocore.exceptions import ClientError

from fyp_data_platform.storage.s3 import S3Storage


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str], dict]] = {}
        self.upload_calls = 0

    def upload_file(self, source, bucket, key, ExtraArgs):
        self.upload_calls += 1
        self.objects[(bucket, key)] = (Path(source).read_bytes(), ExtraArgs["Metadata"], ExtraArgs)

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "HeadObject")
        data, metadata, _ = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "Metadata": metadata, "ETag": '"fake"'}

    def download_file(self, bucket, key, destination):
        Path(destination).write_bytes(self.objects[(bucket, key)][0])

    def get_object(self, Bucket, Key, Range):
        start, end = (int(value) for value in Range.removeprefix("bytes=").split("-"))
        stream = BytesIO(self.objects[(Bucket, Key)][0][start : end + 1])
        stream.iter_chunks = lambda chunk_size: iter(lambda: stream.read(chunk_size), b"")
        return {"Body": stream}


def test_s3_uses_checksum_metadata_encryption_and_skips_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "object.bin"
    source.write_bytes(b"payload")
    client = FakeS3()
    storage = S3Storage("private-bucket", prefix="fyp", client=client)
    first = storage.put_file(source, "dataset/object.bin")
    second = storage.put_file(source, "dataset/object.bin")
    assert first == second
    assert client.upload_calls == 1
    _, metadata, extra = client.objects[("private-bucket", "fyp/dataset/object.bin")]
    assert metadata["sha256"] == first.sha256
    assert extra["ServerSideEncryption"] == "AES256"
    assert b"".join(storage.iter_bytes(first.key, start=1, end=4, chunk_size=2)) == b"aylo"


def test_s3_passes_local_endpoint_and_uses_path_addressing(monkeypatch) -> None:
    captured = {}

    class FakeSession:
        def client(self, service_name, **kwargs):
            captured["service_name"] = service_name
            captured.update(kwargs)
            return FakeS3()

    monkeypatch.setattr(boto3, "Session", lambda **kwargs: FakeSession())
    S3Storage(
        "fyp-local",
        region="ap-southeast-1",
        endpoint_url="http://localhost:5000",
    )

    assert captured["service_name"] == "s3"
    assert captured["endpoint_url"] == "http://localhost:5000"
    assert captured["config"].s3["addressing_style"] == "path"
