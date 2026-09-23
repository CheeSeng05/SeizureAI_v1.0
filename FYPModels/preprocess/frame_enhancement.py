"""Optional super-resolution and sharpening for uint8 BGR face crops.

Owns model selection, cached neural weights, enhancement validation and CLI
options. The facial preprocessing pipeline supplies crops and consumes the
results; this module does not depend on face detection or window extraction.
"""
from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SR_MODELS = {
    # OpenCV interpolation methods
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,

    # Supported deep-learning super-resolution models
    "edsr": "EDSR_x4.pb",
    "espcn": "ESPCN_x4.pb",
    "fsrcnn": "FSRCNN_x4.pb",
    "lapsrn": "LapSRN_x4.pb",
}
SR_SCALES = {
    "edsr": (2, 3, 4), "espcn": (2, 3, 4),
    "fsrcnn": (2, 3, 4), "lapsrn": (2, 4, 8),
}
DEFAULT_SR_MODEL_DIR = Path(__file__).resolve().parents[2] / "checkpoints/pretrained/super_resolution"


def _sr_model_path(model: str, scale: int, model_dir: str | Path) -> Path:
    if scale not in SR_SCALES[model]:
        raise ValueError(f"{model} supports scales {SR_SCALES[model]}, received {scale}")
    prefix = SR_MODELS[model].split("_x", 1)[0]
    return Path(model_dir).resolve() / f"{prefix}_x{scale}.pb"


@lru_cache(maxsize=4)
def _load_sr_model(model: str, scale: int, model_path: str) -> Any:
    """Reuse loaded networks for sequential frame processing in this process."""
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "Neural upscaling requires an OpenCV contrib build with dnn_superres. "
            "Install opencv-contrib-python or opencv-contrib-python-headless."
        )
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Super-resolution model not found: {model_path}. "
            "Run python -m tests.test_upscaling with --download-models to fetch the weights."
        )
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model, scale)
    return sr


def _upscaling(
    image: np.ndarray, model: str, *, scale: int = 4,
    model_dir: str | Path = DEFAULT_SR_MODEL_DIR,
) -> np.ndarray:
    """Upscale a uint8 BGR image by an integer factor (default: 4 for all six).

    Interpolation needs no weights. Neural methods use the checkpoint matching
    ``scale`` and cache up to four networks; do not share them across threads.
    """
    model = model.strip().lower()
    if model not in SR_MODELS:
        raise ValueError(f"Unknown upscaling model {model!r}; choose from {tuple(SR_MODELS)}")
    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or scale < 2:
        raise ValueError("scale must be an integer of at least 2")
    scale = int(scale)
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR image with shape [H, W, 3]")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 (pixel values 0 to 255)")
    if model in {"bilinear", "bicubic"}:
        return cv2.resize(
            image, (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=SR_MODELS[model],
        )
    model_path = _sr_model_path(model, scale, model_dir)
    sr = _load_sr_model(model, scale, str(model_path))
    return sr.upsample(np.ascontiguousarray(image))


def _validate_enhancement(
    *, upscale_model: str | None, upscale_scale: int,
    sharpen_amount: float, sharpen_sigma: float,
) -> None:
    if upscale_model is not None and upscale_model not in SR_MODELS:
        raise ValueError(f"Unknown upscaling model {upscale_model!r}; choose from {tuple(SR_MODELS)}")
    if isinstance(upscale_scale, bool) or not isinstance(upscale_scale, (int, np.integer)) or upscale_scale < 2:
        raise ValueError("upscale_scale must be an integer of at least 2")
    if upscale_model in SR_SCALES and upscale_scale not in SR_SCALES[upscale_model]:
        raise ValueError(f"{upscale_model} supports scales {SR_SCALES[upscale_model]}")
    if not math.isfinite(sharpen_amount) or sharpen_amount < 0:
        raise ValueError("sharpen_amount must be finite and nonnegative")
    if not math.isfinite(sharpen_sigma) or sharpen_sigma <= 0:
        raise ValueError("sharpen_sigma must be finite and positive")


def enhance_face_crop(
    image: np.ndarray, *, upscale_model: str | None = None,
    upscale_scale: int = 4, sr_model_dir: str | Path = DEFAULT_SR_MODEL_DIR,
    sharpen_amount: float = 0.0, sharpen_sigma: float = 1.0,
) -> np.ndarray:
    """Optionally upscale, then sharpen a uint8 BGR crop without modifying it.

    Reuses ``_upscaling`` and its cached neural weights. Unsharp masking adds
    ``amount * (image - GaussianBlur(image))`` to enhance existing edges; it
    cannot recover missing detail. Sigma is measured in pixels after upscaling.
    Defaults preserve the input pixels. Apply to a crop, before RGB conversion.
    """
    if upscale_model is not None:
        upscale_model = upscale_model.strip().lower()
    _validate_enhancement(
        upscale_model=upscale_model, upscale_scale=upscale_scale,
        sharpen_amount=sharpen_amount, sharpen_sigma=sharpen_sigma,
    )
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR image with shape [H, W, 3]")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 (pixel values 0 to 255)")
    result = image.copy() if upscale_model is None else _upscaling(
        image, upscale_model, scale=upscale_scale, model_dir=sr_model_dir,
    )
    if sharpen_amount > 0:
        pixels = result.astype(np.float32)
        blurred = cv2.GaussianBlur(pixels, (0, 0), sigmaX=sharpen_sigma)
        # Floating-point subtraction and clipping avoid uint8 wraparound.
        sharpened = cv2.addWeighted(pixels, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
        result = np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result)


def add_enhancement_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Face crop enhancement (disabled by default)")
    group.add_argument("--upscale-model", choices=tuple(SR_MODELS),
                       help="Upscale head/face crops before landmark detection and RGB extraction.")
    group.add_argument("--upscale-scale", type=int, default=4)
    group.add_argument("--sr-model-dir", type=Path, default=DEFAULT_SR_MODEL_DIR)
    group.add_argument("--sharpen-amount", type=float, default=0.0,
                       help="Unsharp-mask strength; 0 disables sharpening (try 0.5).")
    group.add_argument("--sharpen-sigma", type=float, default=1.0,
                       help="Gaussian blur sigma in crop pixels after upscaling.")


def enhancement_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in (
        "upscale_model", "upscale_scale", "sr_model_dir", "sharpen_amount", "sharpen_sigma",
    )}
