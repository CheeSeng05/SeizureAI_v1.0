from __future__ import annotations

import argparse

import numpy as np
import torch

from .facial_model import EMOTIONS, ResNet50GRU


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> ResNet50GRU:
    # Do not download ImageNet weights when loading your trained checkpoint.
    model = ResNet50GRU(pretrained=False)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def predict_window(
    model: ResNet50GRU,
    video: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()

    with torch.inference_mode():
        probabilities = model.predict_proba(video.to(device))

    confidence, class_index = probabilities.max(dim=1)
    index = class_index.item()

    return {
        "emotion": EMOTIONS[index],
        "confidence": confidence.item(),
        "probabilities": probabilities[0].cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Input file shape: [T, 4, 224, 224]
    face_window = np.load(args.input, allow_pickle=False)
    video = torch.from_numpy(face_window).float().unsqueeze(0)

    model = load_model(args.checkpoint, device)
    result = predict_window(model, video, device)

    print("Emotion:", result["emotion"])
    print("Confidence:", result["confidence"])
    print("All probabilities:", result["probabilities"])


if __name__ == "__main__":
    main()