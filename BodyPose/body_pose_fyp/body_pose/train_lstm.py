from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .build_dataset import assert_patient_disjoint
from .models.dataset import PoseWindowDataset
from .models.lstm import PoseLSTMClassifier
from .utils import active_labels, ensure_parent, get_device, load_config, set_seed, window_manifest_path

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss_sum += loss.item() * y_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes)
    weights = len(y) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal LSTM body-pose classifier.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    set_seed(int(train_cfg["seed"]))
    device = get_device()
    LOGGER.info("Using device: %s", device)

    data = np.load(cfg["paths"]["dataset"], allow_pickle=False)
    x_all, y_all = data["X"], data["y"]
    feature_names = data["feature_names"].astype(str).tolist()
    metadata = pd.read_csv(window_manifest_path(cfg))
    if len(metadata) != len(x_all):
        raise ValueError("Metadata and window array lengths differ.")
    assert_patient_disjoint(metadata)

    label_names = active_labels(cfg)
    stored_label_names = data["label_names"].astype(str).tolist() if "label_names" in data else label_names
    if stored_label_names != label_names:
        raise ValueError(
            f"Dataset labels {stored_label_names} do not match config labels {label_names}. "
            "Rebuild the temporal windows."
        )

    train_idx = metadata.index[metadata["split"] == "train"].to_numpy()
    val_idx = metadata.index[metadata["split"] == "validation"].to_numpy()
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Training and validation splits must both contain windows.")

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, x_train.shape[-1]))
    x_train = scaler.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    ensure_parent(cfg["paths"]["scaler"])
    joblib.dump(scaler, cfg["paths"]["scaler"])

    train_loader = DataLoader(
        PoseWindowDataset(x_train, y_train),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PoseWindowDataset(x_val, y_val),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    num_classes = len(label_names)
    if np.any(y_all < 0) or np.any(y_all >= num_classes):
        raise ValueError("Dataset contains label IDs outside config.labels.active.")
    model = PoseLSTMClassifier(
        input_size=x_train.shape[-1],
        num_classes=num_classes,
        hidden_size=int(train_cfg["hidden_size"]),
        num_layers=int(train_cfg["num_layers"]),
        bidirectional=bool(train_cfg["bidirectional"]),
        dropout=float(train_cfg["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, num_classes).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    checkpoint_path = ensure_parent(cfg["paths"]["checkpoint"])
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x_batch, y_batch in progress:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
            optimizer.step()

            train_loss_sum += loss.item() * y_batch.size(0)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += y_batch.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total
        val_loss, val_accuracy = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        LOGGER.info(
            "Epoch %03d | train loss %.4f acc %.3f | val loss %.4f acc %.3f",
            epoch, train_loss, train_accuracy, val_loss, val_accuracy,
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "input_size": x_train.shape[-1],
                "num_classes": num_classes,
                "feature_names": feature_names,
                "label_names": label_names,
                "embedding_size": int(train_cfg["hidden_size"]) * (
                    2 if bool(train_cfg["bidirectional"]) else 1
                ),
                "model_config": {
                    "hidden_size": int(train_cfg["hidden_size"]),
                    "num_layers": int(train_cfg["num_layers"]),
                    "bidirectional": bool(train_cfg["bidirectional"]),
                    "dropout": float(train_cfg["dropout"]),
                },
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= int(train_cfg["patience"]):
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)
    LOGGER.info("Best checkpoint saved to %s", checkpoint_path)


if __name__ == "__main__":
    main()
