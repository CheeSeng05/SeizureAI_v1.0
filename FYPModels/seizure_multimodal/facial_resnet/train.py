from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..evaluation.classification import binary_classification_metrics
from ..evaluation.thresholds import select_threshold

if TYPE_CHECKING:
    from .facial_model import InceptionResnetV1GRUAttention


CLASSES = ("normal", "seizure")


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FaceWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        minimum_face_detection: float = 0.5,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent

        with self.manifest_path.open(
            newline="",
            encoding="utf-8",
        ) as stream:
            rows = list(csv.DictReader(stream))

        if not rows:
            raise ValueError(
                f"Manifest contains no windows: {self.manifest_path}. "
                "The requested preprocessing window may be longer than the "
                "source videos. Regenerate using a shorter window."
            )

        self.rows = [
            row
            for row in rows
            if row["split"] == split
            and float(row.get("face_region_fraction") or row["face_detection_fraction"])
            >= minimum_face_detection
        ]

        if not self.rows:
            raise ValueError(
                f"No usable rows for split={split!r} "
                f"in {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]

        face_path = self.root / row["face_path"]
        window = np.load(face_path, allow_pickle=False)

        if window.ndim != 4 or window.shape[1] != 4:
            raise ValueError(
                f"{face_path} should have shape [T,4,H,W], "
                f"got {window.shape}"
            )

        video = torch.from_numpy(window).float()
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return video, label


def configure_trainable_layers(
    model: InceptionResnetV1GRUAttention,
) -> None:
    # Freeze pretrained InceptionResnetV1.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    # Train the adapted four-channel input layer.
    for parameter in model.backbone.conv2d_1a.conv.parameters():
        parameter.requires_grad = True

    # GRU, attention, and classifier are trained on the facial dataset.
    for parameter in model.gru.parameters():
        parameter.requires_grad = True

    for parameter in model.temporal_attention.parameters():
        parameter.requires_grad = True

    for parameter in model.attention_norm.parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def run_training_epoch(
    model: InceptionResnetV1GRUAttention,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    # Keep frozen InceptionResnetV1 BatchNorm statistics fixed.
    model.backbone.eval()
    model.backbone.conv2d_1a.conv.train()

    total_loss = 0.0
    total_samples = 0

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(videos)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_loss += loss.item() * count
        total_samples += count

    return total_loss / total_samples


def run_evaluation(
    model: InceptionResnetV1GRUAttention,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float | int | None | str], list[int], list[float]]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    targets: list[int] = []
    probabilities: list[float] = []

    with torch.inference_mode():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            loss = criterion(logits, labels)

            count = labels.size(0)
            total_loss += loss.item() * count
            total_samples += count

            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            targets.extend(labels.cpu().tolist())

    metrics = {
        "loss": total_loss / total_samples,
        **binary_classification_metrics(targets, probabilities),
    }
    return metrics, targets, probabilities


def save_checkpoint(
    path: Path,
    model: InceptionResnetV1GRUAttention,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | int | None | str],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": "InceptionResnetV1GRUAttention",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "gru_hidden_size": 256,
            "bidirectional": True,
            "attention_heads": model.attention_heads,
            "validation_metrics": metrics,
            "probability_threshold": threshold,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/fyp/facial/inception_resnet_v1_gru_attention_final.pth",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()

    # Dataset and device helpers do not need the optional facial backbone.
    from .facial_model import InceptionResnetV1GRUAttention

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_dataset = FaceWindowDataset(
        args.manifest,
        split="train",
    )
    validation_dataset = FaceWindowDataset(
        args.manifest,
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = InceptionResnetV1GRUAttention(
        num_classes=len(CLASSES),
        attention_heads=args.attention_heads,
        pretrained=True,
    ).to(device)

    configure_trainable_layers(model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.conv2d_1a.conv.parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.gru.parameters(),
                "lr": 1e-4,
            },
            {
                "params": list(model.temporal_attention.parameters())
                + list(model.attention_norm.parameters()),
                "lr": 1e-4,
            },
            {
                "params": model.classifier.parameters(),
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()
    checkpoint_path = Path(args.checkpoint)
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_training_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        history.append(result)
        print(json.dumps(result))

    # Select the operating threshold on validation patients. The test cohort is
    # never loaded by training and is reserved for the standalone evaluator.
    validation_metrics, validation_labels, validation_probabilities = run_evaluation(
        model,
        validation_loader,
        criterion,
        device,
    )
    threshold_result = select_threshold(
        validation_labels, validation_probabilities, split="validation"
    )
    selected_threshold = float(threshold_result["threshold"])
    validation_metrics = {
        **validation_metrics,
        **binary_classification_metrics(
            validation_labels, validation_probabilities, threshold=selected_threshold
        ),
    }
    print(json.dumps({
        "validation_metrics": validation_metrics,
        "selected_threshold": selected_threshold,
    }))
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs,
        validation_metrics,
        selected_threshold,
    )
    print(f"Saved final checkpoint: {checkpoint_path}")

    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
