"""Annotated expression/semiology data and masked multi-task training losses."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .movement_features import read_manifest


class MicroExpressionDataset(Dataset):
    """Load window annotations; blank task labels stay unknown (-1).

    The annotation schema is window_id,emotion,semiologies,annotator_id.
    ``semiologies=none`` is an explicit negative; a pipe-separated list denotes
    the complete set of observed signs. Seizure labels come only from the
    manifest's annotated 0/1 label, never its class folder or emotion label.
    """

    def __init__(
        self, manifest_path: str | Path, annotations_path: str | Path, split: str,
        *, emotion_classes: Sequence[str], semiology_classes: Sequence[str],
        minimum_face_detection: float = 0.5,
    ) -> None:
        self.root = Path(manifest_path).resolve().parent
        self.emotion_classes, self.semiology_classes = tuple(emotion_classes), tuple(semiology_classes)
        rows = read_manifest(manifest_path)
        with Path(annotations_path).open(newline="", encoding="utf-8") as stream:
            annotations = list(csv.DictReader(stream))
        by_id = {}
        known_ids = {row["window_id"] for row in rows}
        for annotation in annotations:
            identifier = annotation["window_id"]
            if identifier in by_id or identifier not in known_ids:
                raise ValueError(f"Duplicate or unknown annotated window_id: {identifier}")
            by_id[identifier] = annotation

        self.rows: list[dict[str, str]] = []
        self.targets: list[dict[str, torch.Tensor]] = []
        for row in rows:
            if row["split"] != split:
                continue
            fps = float(row["sampled_fps"])
            if not np.isfinite(fps) or fps < 25:
                raise ValueError("Micro-expression training requires at least 25 fps; re-preprocess these videos")
            fraction = float(row.get("face_region_fraction") or row.get("face_detection_fraction") or 0)
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Invalid face detection fraction")
            if fraction < minimum_face_detection:
                continue
            annotation = by_id.get(row["window_id"], {})
            emotion = str(annotation.get("emotion", "")).strip().lower()
            if emotion and emotion not in self.emotion_classes:
                raise ValueError(f"Unknown emotion: {emotion}")
            semiologies = str(annotation.get("semiologies", "")).strip().lower()
            signs = {part.strip() for part in semiologies.split("|")} if semiologies else set()
            if signs and signs != {"none"} and not signs <= set(self.semiology_classes):
                raise ValueError(f"Unknown or ambiguous semiologies: {semiologies}")
            seizure = str(row.get("label", "")).strip()
            if seizure not in {"", "0", "1"}:
                raise ValueError("Seizure labels must be blank, 0, or 1")
            if not emotion and not semiologies and not seizure:
                continue
            if not row.get("face_geometry_path"):
                raise ValueError(f"Missing face_geometry_path for {row['window_id']}")
            self.rows.append(row)
            self.targets.append({
                "emotion": torch.tensor(self.emotion_classes.index(emotion) if emotion else -1),
                "semiology": torch.tensor(
                    [float(name in signs) for name in self.semiology_classes]
                    if semiologies else [-1.] * len(self.semiology_classes),
                ),
                "seizure": torch.tensor(int(seizure) if seizure else -1),
            })
        if not self.rows:
            raise ValueError(f"No usable annotated windows for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        video = np.load(self.root / row["face_path"], allow_pickle=False)
        geometry = np.load(self.root / row["face_geometry_path"], allow_pickle=False)
        if video.ndim != 4 or video.shape[1] != 4 or min(video.shape[0], *video.shape[2:]) < 1:
            raise ValueError("Face window must have nonempty shape [T,4,H,W]")
        if geometry.shape != (video.shape[0], 20):
            raise ValueError("Geometry must have shape [T,20] aligned with the face window")
        if not np.isfinite(video).all() or not np.isfinite(geometry).all():
            raise ValueError(f"Nonfinite face/geometry data in {row['window_id']}")
        return {"video": torch.from_numpy(video).float(), "geometry": torch.from_numpy(geometry).float(),
                **self.targets[index]}


def multitask_loss(
    output: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum mean losses for observed tasks; unknown targets have zero gradient."""
    parts = {}
    for task in ("emotion", "seizure"):
        labels = targets[task]
        known = labels != -1
        if known.any():
            parts[f"{task}_loss"] = F.cross_entropy(output[f"{task}_logits"][known], labels[known].long())
    labels = targets["semiology"]
    if not torch.isin(labels, labels.new_tensor([-1, 0, 1])).all():
        raise ValueError("Semiology targets must be -1 (unknown), 0, or 1")
    known = labels != -1
    if known.any():
        parts["semiology_loss"] = F.binary_cross_entropy_with_logits(
            output["semiology_logits"][known], labels[known].float(),
        )
    if not parts:
        raise ValueError("Batch has no observed task labels")
    return torch.stack(list(parts.values())).sum(), parts


def run_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    """Train/evaluate all heads; aggregate each task over its observed targets."""
    model.train(optimizer is not None)
    # FaceNet's final BatchNorm receives B*T samples. Fixed pretrained running
    # statistics also make small training batches well-defined.
    if optimizer is not None:
        for module in model.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch["video"], batch["geometry"])
            loss, parts = multitask_loss(output, batch)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            samples += batch["video"].shape[0]
            for name, value in parts.items():
                count = int((batch[name.removesuffix("_loss")] != -1).sum().item())
                totals[name] = totals.get(name, 0.) + value.item() * count
                counts[name] = counts.get(name, 0) + count
    if not samples:
        raise ValueError("Cannot run an epoch on an empty dataset")
    metrics = {name: value / counts[name] for name, value in totals.items()}
    return {**metrics, "total_loss": sum(metrics.values()), "samples": samples}
