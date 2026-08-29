from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .facial_model import ResNet50GRU


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
    model: ResNet50GRU,
) -> None:
    # Freeze pretrained ResNet50.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    # Train the adapted four-channel input layer.
    for parameter in model.backbone.conv1.parameters():
        parameter.requires_grad = True

    # GRU and classifier are trainable by default.
    for parameter in model.gru.parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def run_training_epoch(
    model: ResNet50GRU,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    # Keep frozen ResNet BatchNorm statistics fixed.
    model.backbone.eval()
    model.backbone.conv1.train()

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
    model: ResNet50GRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.inference_mode():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            loss = criterion(logits, labels)

            count = labels.size(0)
            total_loss += loss.item() * count
            total_samples += count

            predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )
            targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            targets,
            predictions,
        ),
    }


def save_checkpoint(
    path: Path,
    model: ResNet50GRU,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": "ResNet50GRU",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "gru_hidden_size": 256,
            "bidirectional": True,
            "test_metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/fyp/facial/resnet50_gru_best.pth",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_dataset = FaceWindowDataset(
        args.manifest,
        split="train",
    )
    test_dataset = FaceWindowDataset(
        args.manifest,
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = ResNet50GRU(
        num_classes=len(CLASSES),
        pretrained=True,
    ).to(device)

    configure_trainable_layers(model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.conv1.parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.gru.parameters(),
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

    # Evaluate exactly once after fixed-epoch training so the held-out test set
    # is not used to select a checkpoint or tune training duration.
    test_metrics = run_evaluation(
        model,
        test_loader,
        criterion,
        device,
    )
    print(json.dumps({f"test_{key}": value for key, value in test_metrics.items()}))
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs,
        test_metrics,
    )
    print(f"Saved final checkpoint: {checkpoint_path}")

    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
