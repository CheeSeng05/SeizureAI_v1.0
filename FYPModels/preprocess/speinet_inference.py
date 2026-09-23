"""Inference adapter for https://github.com/yangt1013/SPEINet.

The upstream network and sharpness features live in ``_speinet``. This adapter
owns checkpoint loading, device selection, BGR/RGB conversion and tiled inference.
It imports PyTorch only when SPEINet is enabled. See ``_speinet/UPSTREAM.md`` for
the pinned source revision and differences from the research inference script.
"""
from __future__ import annotations

import json
from importlib.metadata import version
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from ..seizure_multimodal.checkpoints.provenance import sha256_file


DEFAULT_SPEINET_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "checkpoints/pretrained/speinet/model_best.pt"
)
_SOURCE_DIR = Path(__file__).parent / "_speinet"


@dataclass(frozen=True)
class SPEINetConfig:
    checkpoint: str | Path = DEFAULT_SPEINET_CHECKPOINT
    device: str = "auto"
    tile_size: int = 200
    tile_overlap: int = 40
    reference_radius: int = 7

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("SPEINet device must be auto, cpu, cuda, or mps")
        if type(self.tile_size) is not int or self.tile_size < 20 or self.tile_size % 20:
            raise ValueError("SPEINet tile_size must be a positive multiple of 20")
        if type(self.tile_overlap) is not int or not 0 <= self.tile_overlap < self.tile_size:
            raise ValueError("SPEINet tile_overlap must be in [0, tile_size)")
        if type(self.reference_radius) is not int or not 1 <= self.reference_radius <= 7:
            raise ValueError("SPEINet reference_radius must be an integer in [1, 7]")


def _tile_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


class SPEINetEnhancer:
    """Restore a center frame from native-rate neighbours and sharp references."""

    def __init__(self, config: SPEINetConfig):
        self.config = config
        checkpoint = Path(config.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SPEINet checkpoint not found: {checkpoint}. Download the author's "
                "speinet_model_best weights (see preprocess/README.md) and pass --speinet-checkpoint."
            )
        try:
            import torch
            from ._speinet.speinet import SPEINet
            from ._speinet.sharpness import generate_vars
        except ImportError as exc:
            raise RuntimeError(
                "SPEINet requires the enhancement extra: python -m pip install -e '.[enhancement]'"
            ) from exc
        self.torch = torch
        self.generate_vars = generate_vars
        device = config.device
        if device == "auto":
            # CPU is the portable fallback; MPS can be requested explicitly.
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SPEINet CUDA was requested but is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("SPEINet MPS was requested but is unavailable")
        self.device = torch.device(device)
        args = SimpleNamespace(
            patch_size=200, window_size=5, rgb_range=1,
            depths=[6] * 6, embed_dim=256, num_heads=[8] * 6,
            mlp_ratio=2, resi_connection="1conv",
        )
        self.model = SPEINet(
            in_channels=3, n_sequence=3, out_channels=3, n_resblock=3,
            n_feat=32, device=self.device, args=args,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or not state:
            raise ValueError("SPEINet checkpoint must contain a non-empty state dictionary")
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        # Do not silently accept another architecture or use untrained weights.
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.detector = json.loads((_SOURCE_DIR / "detector.json").read_text())
        self.record = {
            "model": "speinet", **asdict(config), "checkpoint": str(checkpoint),
            "resolved_device": str(self.device),
            "upstream": json.loads((_SOURCE_DIR / "source.json").read_text()),
            "detector_sha256": sha256_file(_SOURCE_DIR / "detector.json"),
            "implementation_sha256": {
                path.name: sha256_file(path) for path in sorted(_SOURCE_DIR.glob("*.py"))
            },
            "software_versions": {name: version(name) for name in ("torch", "torchvision", "timm", "ptwt", "PyWavelets")},
            "temporal_policy": "native-rate neighbours; nearest sharp references within radius; reflect boundaries",
            "spatial_policy": "overlapping tiles; replicate padding to multiples of 20; original output size",
        }

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("SPEINet expects non-empty BGR frames with shape [H, W, 3]")
        if frame.dtype != np.uint8:
            raise ValueError("SPEINet expects uint8 BGR frames")

    def is_sharp(self, frame: np.ndarray) -> bool:
        """Apply the upstream six-feature logistic detector without unpickling it."""
        self._validate_frame(frame)
        torch = self.torch
        rgb = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1))
        pixels = torch.from_numpy(rgb).unsqueeze(0).float().to(self.device)
        # The detector's DCT feature needs >=14 pixels. Normal video is unchanged.
        height, width = frame.shape[:2]
        if min(height, width) < 14:
            pixels = torch.nn.functional.pad(pixels, (0, max(0, 14 - width), 0, max(0, 14 - height)), mode="replicate")
        with torch.inference_mode():
            features = torch.stack(self.generate_vars(pixels, self.detector["kernel_size"], self.device), dim=1)
        values = features.cpu().numpy()[0].astype(np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("SPEINet sharpness detector produced non-finite features")
        return bool(values @ np.asarray(self.detector["coef"]) + self.detector["intercept"] > 0)

    def enhance(self, frames: Sequence[np.ndarray | None]) -> np.ndarray:
        """Restore the middle of three BGR frames plus two optional references.

        ``frames`` is [previous, current, next, previous_reference, next_reference].
        Missing references become zero tensors, as expected by SPEINet's routing.
        """
        if len(frames) != 5:
            raise ValueError("SPEINet requires three neighbours and two reference slots")
        for frame in frames[:3]:
            self._validate_frame(frame)
        center = frames[1]
        for frame in frames[3:]:
            if frame is not None:
                self._validate_frame(frame)
        if any(frame is not None and frame.shape != center.shape for frame in frames):
            raise ValueError("All SPEINet frames must have the same dimensions")
        height, width = center.shape[:2]
        size, overlap = self.config.tile_size, self.config.tile_overlap
        restored = np.zeros((height, width, 3), np.float32)
        weights = np.zeros((height, width, 1), np.float32)
        torch = self.torch
        with torch.inference_mode():
            for top in _tile_starts(height, size, overlap):
                for left in _tile_starts(width, size, overlap):
                    bottom, right = min(top + size, height), min(left + size, width)
                    h, w = bottom - top, right - left
                    tiles = [np.zeros((h, w, 3), np.uint8) if frame is None
                             else frame[top:bottom, left:right, ::-1] for frame in frames]
                    data = np.ascontiguousarray(np.stack(tiles).transpose(0, 3, 1, 2))
                    tensor = torch.from_numpy(data).float().to(self.device) / 255.0
                    tensor = torch.nn.functional.pad(tensor, (0, -w % 20, 0, -h % 20), mode="replicate")
                    output = self.model(tensor.unsqueeze(0))
                    expected = (1, 3, tensor.shape[-2], tensor.shape[-1])
                    if tuple(output.shape) != expected:
                        raise RuntimeError(f"SPEINet returned shape {tuple(output.shape)}; expected {expected}")
                    if not bool(torch.isfinite(output).all()):
                        raise RuntimeError(
                            "SPEINet produced non-finite pixels. Check the upstream model/checkpoint "
                            "compatibility (see preprocess/README.md). Invalid frames were not passed "
                            "to face preprocessing."
                        )
                    tile = output[0, :, :h, :w].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    blend = np.outer(np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1]).astype(np.float32)[:, :, None]
                    restored[top:bottom, left:right] += tile * blend
                    weights[top:bottom, left:right] += blend
        rgb = np.clip(np.rint(restored / weights * 255.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])
