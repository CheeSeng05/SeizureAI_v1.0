from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .facial_model import InceptionResnetV1Attention


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[InceptionResnetV1Attention, tuple[str, ...]]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'"
        )

    if "classes" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'classes'"
        )

    classes = tuple(checkpoint["classes"])

    model_name = checkpoint.get("model_name")
    if model_name != "InceptionResnetV1Attention":
        raise ValueError(
            "This inference path requires a InceptionResnetV1Attention checkpoint; "
            f"found {model_name!r}. Retrain the attention model; BiGRU checkpoints are incompatible."
        )

    model = InceptionResnetV1Attention(
        num_classes=len(classes),
        attention_heads=checkpoint.get("attention_heads", 8),
        max_frames=checkpoint.get("max_frames", 512),
        dropout=checkpoint.get("dropout", 0.3),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, classes


def load_face_window(
    input_path: str | Path,
) -> torch.Tensor:
    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Face window not found: {input_path}"
        )

    face_window = np.load(
        input_path,
        allow_pickle=False,
    )

    # Expected NumPy shape: [T, 4, 224, 224]
    if face_window.ndim != 4:
        raise ValueError(
            "Face window must have shape [T,4,H,W], "
            f"got {face_window.shape}"
        )

    if face_window.shape[1] != 4:
        raise ValueError(
            "Face window must contain four channels: "
            "RGB plus optical-flow magnitude. "
            f"Got {face_window.shape[1]} channels."
        )

    if not np.isfinite(face_window).all():
        raise ValueError(
            "Face window contains NaN or infinite values"
        )

    # Add batch dimension:
    # [T,4,H,W] -> [1,T,4,H,W]
    return torch.from_numpy(face_window).float().unsqueeze(0)


def predict_window(
    model: InceptionResnetV1Attention,
    video: torch.Tensor,
    classes: tuple[str, ...],
    device: torch.device,
) -> dict:
    video = video.to(device, non_blocking=True)

    with torch.inference_mode():
        probabilities = model.predict_proba(video)

    confidence, class_index = probabilities.max(dim=1)
    index = int(class_index.item())

    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(
            classes,
            probabilities[0].cpu().tolist(),
        )
    }

    return {
        "prediction": classes[index],
        "class_index": index,
        "confidence": float(confidence.item()),
        "probabilities": class_probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run InceptionResnetV1 multi-head attention inference on one face window"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a preprocessed [T,4,H,W] .npy file",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained .pth checkpoint",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    video = load_face_window(args.input)
    model, classes = load_model(
        args.checkpoint,
        device,
    )

    result = predict_window(
        model,
        video,
        classes,
        device,
    )

    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Class probabilities:")

    for class_name, probability in result["probabilities"].items():
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
