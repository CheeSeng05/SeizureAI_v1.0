from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .build_dataset import assert_patient_disjoint
from .models.dataset import PoseWindowDataset
from .models.lstm import PoseLSTMClassifier
from .utils import get_device, load_config, window_manifest_path

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def encode_batches(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    embeddings = []
    for x_batch, _ in loader:
        embeddings.append(model.encode(x_batch.to(device)).cpu().numpy())
    return np.concatenate(embeddings).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fusion-ready body LSTM embeddings.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--split", choices=["all", "train", "validation", "test"], default="all"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    data = np.load(cfg["paths"]["dataset"], allow_pickle=False)
    x_all, y_all = data["X"], data["y"]
    manifest = pd.read_csv(window_manifest_path(cfg))
    if len(manifest) != len(x_all):
        raise ValueError("Window manifest and array lengths differ.")
    assert_patient_disjoint(manifest)

    if args.split == "all":
        indices = manifest.index.to_numpy()
    else:
        indices = manifest.index[manifest["split"] == args.split].to_numpy()
    if len(indices) == 0:
        raise ValueError(f"No windows exist in split '{args.split}'.")

    scaler = joblib.load(cfg["paths"]["scaler"])
    x = x_all[indices]
    x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    y = y_all[indices]
    device = get_device()
    checkpoint = torch.load(cfg["paths"]["checkpoint"], map_location=device, weights_only=False)
    model = PoseLSTMClassifier(
        input_size=int(checkpoint["input_size"]),
        num_classes=int(checkpoint["num_classes"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(
        PoseWindowDataset(x, y),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )
    embeddings = encode_batches(model, loader, device)

    selected = manifest.iloc[indices].reset_index(drop=True)
    metadata_columns = [
        "window_id",
        "media_id",
        "patient_id",
        "start_sec",
        "end_sec",
        "label",
        "split",
    ]
    missing = set(metadata_columns) - set(selected.columns)
    if missing:
        raise ValueError(f"Window manifest is missing embedding metadata: {sorted(missing)}")
    output = selected[metadata_columns].copy()
    embedding_columns = [f"body_emb_{index:03d}" for index in range(embeddings.shape[1])]
    output[embedding_columns] = embeddings

    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg["paths"].get("body_embeddings_csv", results_dir / "body_embeddings.csv"))
    npz_path = Path(cfg["paths"].get("body_embeddings_npz", results_dir / "body_embeddings.npz"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        window_id=output["window_id"].to_numpy(str),
        media_id=output["media_id"].to_numpy(str),
        patient_id=output["patient_id"].to_numpy(str),
        start_sec=output["start_sec"].to_numpy(float),
        end_sec=output["end_sec"].to_numpy(float),
        label=output["label"].to_numpy(str),
        split=output["split"].to_numpy(str),
    )
    LOGGER.info("Saved %s body embeddings with dimension %d", len(output), embeddings.shape[1])
    LOGGER.info("CSV: %s", csv_path)
    LOGGER.info("NPZ: %s", npz_path)


if __name__ == "__main__":
    main()
