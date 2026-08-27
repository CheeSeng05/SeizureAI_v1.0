from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import select

from fyp_data_platform.config.settings import Settings
from fyp_data_platform.db.models import StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.storage.base import StorageBackend
from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.keys import derived_key, safe_entity


@contextmanager
def _body_project_import(project_path: Path) -> Iterator[None]:
    project = str(project_path.resolve())
    if not Path(project).is_dir():
        raise FileNotFoundError(f"BodyPose project not found: {project}")
    added = project not in sys.path
    if added:
        sys.path.insert(0, project)
    try:
        yield
    finally:
        if added:
            sys.path.remove(project)


def load_body_annotations(
    settings: Settings,
    *,
    adapter_type: str,
    source: Path,
    config: dict[str, Any] | None = None,
    active_labels: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Consume BodyPose's stable annotation interface; no adapter logic is copied here."""
    with _body_project_import(settings.body_pose_project_path):
        module = importlib.import_module("body_pose.annotation_interface")
        batch = module.load_annotations(
            adapter_type, source, config=config, active_labels=active_labels
        )
    return batch.rows, batch.excluded_event_counts


@dataclass(frozen=True)
class BodyRunResult:
    recording_id: str
    pose_outputs: list[dict[str, Any]]
    feature_outputs: list[dict[str, Any]]
    command_log: list[list[str]]


class BodyPoseIntegration:
    """Stage media, invoke existing BodyPose CLIs, then persist derived artifacts."""

    def __init__(self, settings: Settings, storage: StorageBackend, session_factory: Any) -> None:
        self.settings = settings
        self.storage = storage
        self.session_factory = session_factory

    def run(
        self,
        dataset_id: str,
        recording_id: str,
        source_key: str | None = None,
        *,
        cleanup: bool = False,
    ) -> BodyRunResult:
        source_key = source_key or self._resolve_video_key(dataset_id, recording_id)
        run_root = self.settings.work_root / safe_entity(dataset_id) / safe_entity(recording_id)
        input_path = run_root / "input" / Path(source_key).name
        pose_dir = run_root / "pose"
        feature_dir = run_root / "features"
        self.storage.download_file(source_key, input_path)
        python = str(self.settings.body_pose_python or Path(sys.executable))
        commands = [
            [python, "-m", "body_pose.extract_pose", "--input", str(input_path), "--output-dir", str(pose_dir)],
            [python, "-m", "body_pose.extract_features", "--input-dir", str(pose_dir), "--output-dir", str(feature_dir)],
        ]
        for command in commands:
            subprocess.run(command, cwd=self.settings.body_pose_project_path, check=True)
        pose_manifest = self._upload_tree(dataset_id, recording_id, "pose", pose_dir)
        feature_manifest = self._upload_tree(dataset_id, recording_id, "features", feature_dir)
        with self.session_factory.begin() as session:
            repo = CatalogRepository(session)
            for kind, manifest in (("pose", pose_manifest), ("features", feature_manifest)):
                for item in manifest:
                    repo.stored_file(
                        dataset_id,
                        f"derived/{recording_id}/{kind}/{item['relative_path']}",
                        recording_id=f"{dataset_id}::{recording_id}",
                        role="derived-artifact",
                        modality="body",
                        storage_backend=self.settings.storage_backend,
                        storage_key=item["key"],
                        payload_state="stored",
                        size_bytes=item["size_bytes"],
                        sha256=item["sha256"],
                        media_type=None,
                        metadata_json={"artifact_type": kind},
                    )
        result = BodyRunResult(
            recording_id=recording_id,
            pose_outputs=pose_manifest,
            feature_outputs=feature_manifest,
            command_log=commands,
        )
        if cleanup:
            resolved_work = self.settings.work_root.resolve()
            resolved_run = run_root.resolve()
            if resolved_work not in resolved_run.parents:
                raise ValueError(f"Refusing to clean a path outside the work root: {resolved_run}")
            shutil.rmtree(resolved_run)
        return result

    def _resolve_video_key(self, dataset_id: str, recording_id: str) -> str:
        with self.session_factory() as session:
            row = session.scalar(
                select(StoredFile).where(
                    StoredFile.dataset_id == dataset_id,
                    StoredFile.recording_id == f"{dataset_id}::{recording_id}",
                    StoredFile.role == "source-media",
                    StoredFile.modality == "video",
                    StoredFile.payload_state == "stored",
                )
            )
        if row is None:
            raise LookupError(
                f"No stored video is registered for {dataset_id}/{recording_id}; "
                "supply --source-key or register the video first."
            )
        return row.storage_key

    def _upload_tree(self, dataset_id: str, recording_id: str, kind: str, root: Path) -> list[dict[str, Any]]:
        if not root.is_dir():
            raise FileNotFoundError(f"BodyPose did not produce expected directory: {root}")
        outputs: list[dict[str, Any]] = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            key = derived_key(dataset_id, recording_id, kind, path.relative_to(root).as_posix())
            metadata = self.storage.put_file(path, key, sha256=sha256_file(path))
            outputs.append({**asdict(metadata), "relative_path": path.relative_to(root).as_posix()})
        return outputs
