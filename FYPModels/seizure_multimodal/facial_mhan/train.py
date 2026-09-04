"""Train MHAN-BiGRU with explicit micro-expression/semiology annotations."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import DEFAULT_SEMIOLOGIES, EMOTION_CLASSES, MHANBiGRUSemiology
from ..facial_resnet.train import resolve_device


class MicroExpressionDataset(Dataset):
    """Join preprocessed windows to blinded, independently supplied labels."""

    def __init__(
        self,
        manifest: str | Path,
        annotations: str | Path,
        split: str,
        *,
        emotion_classes: Sequence[str] = EMOTION_CLASSES,
        semiology_classes: Sequence[str] = DEFAULT_SEMIOLOGIES,
        minimum_fps: float = 25.0,
        minimum_face_fraction: float = 0.8,
    ) -> None:
        self.root = Path(manifest).resolve().parent
        with Path(manifest).open(newline="", encoding="utf-8") as stream:
            manifest_rows = list(csv.DictReader(stream))
        with Path(annotations).open(newline="", encoding="utf-8") as stream:
            annotation_rows = list(csv.DictReader(stream))
        annotations_by_window = {row["window_id"]: row for row in annotation_rows}
        if len(annotations_by_window) != len(annotation_rows):
            raise ValueError("Micro-expression annotations contain duplicate window_id values")

        self.emotions = tuple(emotion_classes)
        self.semiologies = tuple(semiology_classes)
        self.rows: list[tuple[dict[str, str], dict[str, str]]] = []
        for row in manifest_rows:
            if row["split"] != split:
                continue
            fps = float(row["sampled_fps"])
            if fps < minimum_fps:
                raise ValueError(
                    f"{row['window_id']} has {fps:g} fps; micro-expression training "
                    f"requires at least {minimum_fps:g} fps"
                )
            quality = float(row.get("face_landmark_fraction") or row["face_detection_fraction"])
            annotation = annotations_by_window.get(row["window_id"])
            if quality >= minimum_face_fraction and annotation is not None:
                self.rows.append((row, annotation))
        if not self.rows:
            raise ValueError(f"No annotated, quality-controlled windows for split={split!r}")
        if not any(annotation.get("emotion", "").strip() for _, annotation in self.rows):
            raise ValueError(f"No emotion labels are available for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, annotation = self.rows[index]
        face = np.load(self.root / row["face_path"], allow_pickle=False)
        geometry = np.load(self.root / row["face_geometry_path"], allow_pickle=False)
        emotion_name = annotation.get("emotion", "").strip().lower()
        if emotion_name and emotion_name not in self.emotions:
            raise ValueError(f"Unknown emotion {emotion_name!r} for {row['window_id']}")
        emotion = self.emotions.index(emotion_name) if emotion_name else -1

        raw_semiologies = annotation.get("semiologies", "").strip().lower()
        semiology = np.full(len(self.semiologies), -1.0, dtype=np.float32)
        if raw_semiologies:
            semiology.fill(0.0)
            names = set() if raw_semiologies == "none" else set(raw_semiologies.split("|"))
            unknown = names.difference(self.semiologies)
            if unknown:
                raise ValueError(f"Unknown semiologies for {row['window_id']}: {sorted(unknown)}")
            for name in names:
                semiology[self.semiologies.index(name)] = 1.0
        return {
            "video": torch.from_numpy(face).float(),
            "geometry": torch.from_numpy(geometry).float(),
            "emotion": torch.tensor(emotion, dtype=torch.long),
            "semiology": torch.from_numpy(semiology),
            "seizure": torch.tensor(int(row["label"]), dtype=torch.long),
            "window_id": row["window_id"],
        }


def multitask_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    emotion_weight: float = 1.0,
    semiology_weight: float = 1.0,
    seizure_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked loss permits labels to be missing without inventing targets."""
    device = output["emotion_logits"].device
    total = torch.zeros((), device=device)
    parts: dict[str, float] = {}
    emotion_mask = batch["emotion"] >= 0
    if emotion_mask.any():
        loss = nn.functional.cross_entropy(
            output["emotion_logits"][emotion_mask], batch["emotion"][emotion_mask]
        )
        total = total + emotion_weight * loss
        parts["emotion_loss"] = float(loss.detach())
    semiology_mask = batch["semiology"] >= 0
    if semiology_mask.any():
        loss = nn.functional.binary_cross_entropy_with_logits(
            output["semiology_logits"][semiology_mask], batch["semiology"][semiology_mask]
        )
        total = total + semiology_weight * loss
        parts["semiology_loss"] = float(loss.detach())
    seizure_loss = nn.functional.cross_entropy(output["seizure_logits"], batch["seizure"])
    total = total + seizure_weight * seizure_loss
    parts["seizure_loss"] = float(seizure_loss.detach())
    parts["total_loss"] = float(total.detach())
    return total, parts


def run_epoch(
    model: MHANBiGRUSemiology,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    emotion_correct = emotion_count = seizure_correct = seizure_count = 0
    semiology_tp = semiology_fp = semiology_fn = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(tensor_batch["video"], tensor_batch["geometry"])
            loss, parts = multitask_loss(output, tensor_batch)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value
            emotion_mask = tensor_batch["emotion"] >= 0
            emotion_correct += int((output["emotion_logits"][emotion_mask].argmax(1) == tensor_batch["emotion"][emotion_mask]).sum())
            emotion_count += int(emotion_mask.sum())
            seizure_correct += int((output["seizure_logits"].argmax(1) == tensor_batch["seizure"]).sum())
            seizure_count += int(tensor_batch["seizure"].numel())
            semiology_mask = tensor_batch["semiology"] >= 0
            predicted = output["semiology_logits"] >= 0
            target = tensor_batch["semiology"] == 1
            semiology_tp += int((predicted & target & semiology_mask).sum())
            semiology_fp += int((predicted & ~target & semiology_mask).sum())
            semiology_fn += int((~predicted & target & semiology_mask).sum())
            batches += 1
    metrics = {name: value / batches for name, value in totals.items()}
    metrics["emotion_accuracy"] = emotion_correct / emotion_count if emotion_count else float("nan")
    metrics["seizure_accuracy"] = seizure_correct / seizure_count
    denominator = 2 * semiology_tp + semiology_fp + semiology_fn
    metrics["semiology_micro_f1"] = 2 * semiology_tp / denominator if denominator else float("nan")
    return metrics


def save_checkpoint(
    path: Path,
    model: MHANBiGRUSemiology,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": "MHANBiGRUSemiology",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "emotion_classes": model.emotion_classes,
        "semiology_classes": model.semiology_classes,
        "frame_feature_dim": model.frame_feature_dim,
        "gru_hidden_size": model.gru_hidden_size,
        "attention_heads": model.attention_heads,
        "geometry_dim": model.geometry_dim,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/fyp/facial/mhan_bigru_best.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-fps", type=float, default=25.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="Resume from an MHAN latest checkpoint")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    datasets = {
        split: MicroExpressionDataset(
            args.manifest, args.annotations, split, minimum_fps=args.minimum_fps
        ) for split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(data, batch_size=args.batch_size, shuffle=split == "train")
        for split, data in datasets.items()
    }
    model = MHANBiGRUSemiology().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    checkpoint = Path(args.checkpoint)
    best, stale = float("inf"), 0
    history = []
    start_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=True)
        if resumed.get("model_name") != "MHANBiGRUSemiology":
            raise ValueError("--resume checkpoint is not an MHANBiGRUSemiology model")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        best = float(resumed.get("validation_loss", best))
    for epoch in range(start_epoch, args.epochs + 1):
        train = run_epoch(model, loaders["train"], device, optimizer)
        validation = run_epoch(model, loaders["validation"], device, None)
        row = {"epoch": epoch, "train": train, "validation": validation}
        history.append(row); print(json.dumps(row))
        value = validation["total_loss"]
        save_checkpoint(checkpoint.with_name(checkpoint.stem + "_latest.pth"), model, optimizer, epoch, value)
        if value < best:
            best, stale = value, 0
            save_checkpoint(checkpoint, model, optimizer, epoch, value)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation loss={best:.6f}")
                break
    checkpoint.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
