"""Train the independent InceptionResnetV1-BiGRU expression/semiology model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .facial_model import InceptionResnetV1BiGRUSemiology
from .train import resolve_device
from ..facial_mhan.train import MicroExpressionDataset, run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/fyp/facial/inception_resnet_v1_emotions_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = InceptionResnetV1BiGRUSemiology(pretrained=True).to(device)
    datasets = {
        split: MicroExpressionDataset(
            args.manifest, args.annotations, split,
            emotion_classes=model.emotion_classes,
            semiology_classes=model.semiology_classes,
        )
        for split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(data, batch_size=args.batch_size, shuffle=split == "train")
        for split, data in datasets.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    checkpoint = Path(args.checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], device, optimizer)
        validation = run_epoch(model, loaders["validation"], device, None)
        row = {"epoch": epoch, "train": train, "validation": validation}
        history.append(row); print(json.dumps(row))
        if validation["total_loss"] < best:
            best, stale = validation["total_loss"], 0
            torch.save({
                "model_name": "InceptionResnetV1BiGRUSemiology",
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "emotion_classes": model.emotion_classes,
                "semiology_classes": model.semiology_classes,
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
