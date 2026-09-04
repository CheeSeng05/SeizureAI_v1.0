"""Inference for a trained MHAN-BiGRU micro-expression model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import MHANBiGRUSemiology
from ..facial_resnet.train import resolve_device


def load_mhan_checkpoint(
    path: str | Path, device: torch.device
) -> MHANBiGRUSemiology:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "MHANBiGRUSemiology":
        raise ValueError("Checkpoint is not an MHANBiGRUSemiology model")
    model = MHANBiGRUSemiology(
        emotion_classes=checkpoint["emotion_classes"],
        semiology_classes=checkpoint["semiology_classes"],
        frame_feature_dim=int(checkpoint.get("frame_feature_dim", 256)),
        gru_hidden_size=int(checkpoint.get("gru_hidden_size", 128)),
        attention_heads=int(checkpoint.get("attention_heads", 8)),
        geometry_dim=int(checkpoint.get("geometry_dim", 20)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def predict_window(
    model: MHANBiGRUSemiology,
    face_path: str | Path,
    geometry_path: str | Path,
    device: torch.device,
) -> dict:
    face = torch.from_numpy(np.load(face_path, allow_pickle=False)).float().unsqueeze(0)
    geometry = torch.from_numpy(np.load(geometry_path, allow_pickle=False)).float().unsqueeze(0)
    output = model.predict(face.to(device), geometry.to(device))
    emotion_probability = output["emotion_probability"][0].cpu()
    semiology_probability = output["semiology_probability"][0].cpu()
    return {
        "emotion": model.emotion_classes[int(emotion_probability.argmax())],
        "emotion_probabilities": dict(zip(model.emotion_classes, emotion_probability.tolist())),
        "semiology_probabilities": dict(zip(model.semiology_classes, semiology_probability.tolist())),
        "seizure_probability": float(output["seizure_probability"][0].cpu()),
        "attention_by_head": output["attention_weights"][0].cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_mhan_checkpoint(args.checkpoint, device)
    print(json.dumps(predict_window(model, args.face, args.geometry, device), indent=2))


if __name__ == "__main__":
    main()
