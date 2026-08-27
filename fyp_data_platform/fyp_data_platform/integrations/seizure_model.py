from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from fyp_data_platform.config.settings import Settings
from fyp_data_platform.db.models import ModelRun, StoredFile
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.storage.base import StorageBackend
from fyp_data_platform.storage.integrity import sha256_file
from fyp_data_platform.storage.keys import normalize_key, safe_entity


@dataclass(frozen=True)
class ModelRunResult:
    run_id: str
    dataset_id: str
    recording_id: str
    mode: str
    status: str
    output_prefix: str
    artifacts: list[dict[str, Any]]
    metrics: dict[str, Any]
    command: list[str]


class SeizureModelIntegration:
    """Stage catalogued inputs, run seizure_multimodal, and persist its outputs."""

    def __init__(self, settings: Settings, storage: StorageBackend, session_factory: Any) -> None:
        self.settings = settings
        self.storage = storage
        self.session_factory = session_factory

    def run(
        self,
        dataset_id: str,
        recording_id: str,
        *,
        mode: str = "video",
        video_key: str | None = None,
        labels_key: str | None = None,
        eeg_key: str | None = None,
        require_labels: bool = False,
        cleanup: bool = False,
    ) -> ModelRunResult:
        if mode not in {"video", "eeg", "unimodal", "strict_multimodal"}:
            raise ValueError("Managed runs support video, eeg, unimodal, or strict_multimodal.")
        project = self.settings.model_project_path.resolve()
        if not (project / "seizure_multimodal" / "__main__.py").is_file():
            raise FileNotFoundError(f"Authoritative model project not found: {project}")

        run_id = str(uuid.uuid4())
        run_root = self.settings.work_root / safe_entity(dataset_id) / safe_entity(recording_id) / run_id
        input_root = run_root / "inputs"
        output_root = run_root / "results"
        input_root.mkdir(parents=True, exist_ok=True)

        resolved = {
            "video": video_key or self._resolve_key(dataset_id, recording_id, "video"),
            "labels": labels_key or self._resolve_key(dataset_id, recording_id, "labels"),
            "eeg": eeg_key or self._resolve_key(dataset_id, recording_id, "eeg"),
        }
        if mode == "video":
            required = {"video", "labels"} if require_labels else {"video"}
        elif mode == "eeg":
            required = {"eeg"}
        elif mode == "strict_multimodal":
            required = {"video", "labels", "eeg"}
        else:
            required = {"video", "labels", "eeg"} if require_labels else set()
        missing = sorted(name for name in required if not resolved[name])
        if missing:
            raise LookupError(f"Missing registered model input(s): {', '.join(missing)}")
        if mode == "unimodal" and not any(resolved[name] for name in ("video", "eeg")):
            raise LookupError("Missing registered model input(s): video or eeg")

        staged: dict[str, Path] = {}
        for name, key in resolved.items():
            if key:
                destination = input_root / f"{name}{Path(key).suffix}"
                self.storage.download_file(key, destination)
                staged[name] = destination.resolve()

        config = self._build_config(project, output_root.resolve(), staged)
        config_path = run_root / "inference.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        python = str(self.settings.model_python or Path(sys.executable))
        command = [
            python, "-m", "seizure_multimodal", "test", "--mode", mode,
            "--config", str(config_path.resolve()), "--root", str(project), "--device", "auto",
        ]
        self._create_run(
            run_id,
            dataset_id,
            recording_id,
            mode,
            config,
            command,
            require_labels=require_labels,
        )
        try:
            subprocess.run(command, cwd=project, check=True)
            artifacts = self._upload_outputs(dataset_id, recording_id, run_id, output_root)
            metrics = self._collect_metrics(output_root)
            workflow_statuses = {
                value.get("status")
                for value in metrics.values()
                if isinstance(value, dict) and value.get("status")
            }
            status = "blocked" if "blocked" in workflow_statuses else "completed"
            self._finish_run(run_id, status, metrics, artifacts, None)
            result = ModelRunResult(
                run_id, dataset_id, recording_id, mode, status,
                self._output_prefix(dataset_id, recording_id, run_id), artifacts, metrics, command,
            )
        except Exception as error:
            self._finish_run(run_id, "failed", {}, [], str(error))
            raise
        finally:
            if cleanup and run_root.exists():
                self._safe_cleanup(run_root)
        return result

    def _resolve_key(self, dataset_id: str, recording_id: str, modality: str) -> str | None:
        with self.session_factory() as session:
            row = session.scalar(
                select(StoredFile).where(
                    StoredFile.dataset_id == dataset_id,
                    StoredFile.recording_id == f"{dataset_id}::{recording_id}",
                    StoredFile.modality == modality,
                    StoredFile.payload_state == "stored",
                ).order_by(StoredFile.created_at.desc())
            )
        return row.storage_key if row else None

    def _build_config(self, project: Path, output: Path, staged: dict[str, Path]) -> dict[str, Any]:
        config = yaml.safe_load((project / "configs" / "inference.yaml").read_text(encoding="utf-8"))
        config["training"] = {"enabled": False}
        config.setdefault("testing", {})["output_directory"] = str(output)
        data = config.setdefault("data", {})
        data["video"] = str(staged.get("video", project / "__missing_video__"))
        data["video_labels"] = str(staged["labels"]) if "labels" in staged else None
        data["eeg_csv"] = str(staged.get("eeg", project / "__missing_eeg__"))
        for section, keys in {
            "video_encoder": ("source", "checkpoint", "dynamic_partition", "pose_source", "pose_checkpoint"),
            "eeg_encoder": ("checkpoint",),
        }.items():
            for key in keys:
                value = config.get(section, {}).get(key)
                if value:
                    config[section][key] = str((project / str(value)).resolve())
        return config

    def _output_prefix(self, dataset_id: str, recording_id: str, run_id: str) -> str:
        return normalize_key(
            f"datasets/{safe_entity(dataset_id)}/derived/{safe_entity(recording_id)}/model-runs/{run_id}"
        )

    def _upload_outputs(self, dataset_id: str, recording_id: str, run_id: str, root: Path) -> list[dict[str, Any]]:
        if not root.is_dir():
            raise FileNotFoundError(f"Model did not produce its results directory: {root}")
        prefix = self._output_prefix(dataset_id, recording_id, run_id)
        artifacts = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            key = normalize_key(f"{prefix}/{relative}")
            metadata = self.storage.put_file(path, key, sha256=sha256_file(path))
            artifacts.append({**asdict(metadata), "relative_path": relative})
            with self.session_factory.begin() as session:
                CatalogRepository(session).stored_file(
                    dataset_id,
                    f"derived/{recording_id}/model-runs/{run_id}/{relative}",
                    recording_id=f"{dataset_id}::{recording_id}", role="model-output",
                    modality="multimodal", storage_backend=self.settings.storage_backend,
                    storage_key=key, payload_state="stored", size_bytes=metadata.size_bytes,
                    sha256=metadata.sha256, media_type=None,
                    metadata_json={"run_id": run_id, "relative_path": relative},
                )
        return artifacts

    @staticmethod
    def _collect_metrics(root: Path) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for path in sorted(root.rglob("*metrics.json")):
            try:
                metrics[path.relative_to(root).as_posix()] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics[path.relative_to(root).as_posix()] = {"parse_error": True}
        return metrics

    def _create_run(
        self,
        run_id: str,
        dataset_id: str,
        recording_id: str,
        mode: str,
        config: dict[str, Any],
        command: list[str],
        *,
        require_labels: bool,
    ) -> None:
        with self.session_factory.begin() as session:
            CatalogRepository(session).model_run(
                run_id, name=f"seizure_multimodal {mode}", model_type="seizure_multimodal",
                model_version=None, modality=mode, status="running", git_commit=None,
                config_json={
                    "dataset_id": dataset_id,
                    "recording_id": recording_id,
                    "command": command,
                    "config": config,
                    "require_labels": require_labels,
                },
                metrics_json={}, artifact_file_id=None, started_at=datetime.now(timezone.utc), completed_at=None,
            )

    def _finish_run(self, run_id: str, status: str, metrics: dict[str, Any], artifacts: list[dict[str, Any]], error: str | None) -> None:
        with self.session_factory.begin() as session:
            row = session.get(ModelRun, run_id)
            if row is None:
                raise LookupError(f"Model run disappeared: {run_id}")
            row.status = status
            row.metrics_json = {"metrics": metrics, "artifacts": artifacts, "error": error}
            row.completed_at = datetime.now(timezone.utc)

    def _safe_cleanup(self, run_root: Path) -> None:
        work = self.settings.work_root.resolve()
        target = run_root.resolve()
        if work not in target.parents:
            raise ValueError(f"Refusing to clean outside work root: {target}")
        shutil.rmtree(target)
