from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from ..types import EncoderOutput


class VSViGAdapter:
    """Thin inference-only wrapper around the unmodified local VSViG source.

    The local factory contains a ``PATH_TO_DYNAMIC_PARTITIONS`` placeholder, so
    this adapter instantiates the original ``STViG`` class with the provided
    dynamic partition asset. It does not redefine or change the network.
    """

    embedding_dim = 384
    clip_frames = 30
    skeleton_points = 15
    patch_channels = 3
    patch_height = 32
    patch_width = 32
    keypoint_features = 3

    def __init__(
        self,
        *,
        source_path: str | Path,
        checkpoint_path: str | Path,
        dynamic_partition_path: str | Path,
        device: str = "cpu",
        strict: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "VSViG inference requires PyTorch, torchvision, and timm. "
                "Install the inference dependencies; the checkpoint was not modified."
            ) from exc
        self.torch = torch
        self.source_path = Path(source_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.dynamic_partition_path = Path(dynamic_partition_path).resolve()
        for description, path in (
            ("VSViG source", self.source_path),
            ("VSViG checkpoint", self.checkpoint_path),
            ("dynamic partition", self.dynamic_partition_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{description} not found: {path}")
        self.device = torch.device(device)
        module = self._load_original_module()
        dynamic_order = self._torch_load(self.dynamic_partition_path)

        class Options:
            dynamic = 1
            num_layer = [2, 2, 6, 2]
            output_channels = [24, 48, 96, 192]
            expansion = 2
            pos_emb = "stem"
            dynamic_point_order = dynamic_order

        self.model = module.STViG(Options())
        state = self._torch_load(self.checkpoint_path)
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise TypeError("VSViG checkpoint does not contain a state dictionary")
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        incompatible = self.model.load_state_dict(state, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise ValueError(
                "Non-strict VSViG loading still found incompatible keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.model.to(self.device)
        self.freeze()
        self._embedding: Any = None
        self._logits: Any = None
        self._hooks = (
            self.model.fc.register_forward_pre_hook(self._capture_embedding),
            self.model.fc[-1].register_forward_hook(self._capture_logits),
        )

    def _torch_load(self, path: Path) -> Any:
        try:
            return self.torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return self.torch.load(path, map_location="cpu")

    def _load_original_module(self) -> ModuleType:
        module_name = "fyp_original_vsvig"
        spec = importlib.util.spec_from_file_location(module_name, self.source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import VSViG source: {self.source_path}")
        module = importlib.util.module_from_spec(spec)
        source_directory = str(self.source_path.parent)
        sys.path.insert(0, source_directory)
        # timm's current register_model implementation resolves the decorated
        # function's module through sys.modules while the file is executing.
        # Register this normal import invariant explicitly for the local file.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            if sys.path and sys.path[0] == source_directory:
                sys.path.pop(0)
        return module

    def _capture_embedding(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        self._embedding = inputs[0].flatten(1)

    def _capture_logits(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        self._logits = output.flatten(1).squeeze(-1)

    def freeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @property
    def is_inference_only(self) -> bool:
        return not self.model.training and all(not p.requires_grad for p in self.model.parameters())

    def predict(self, video_window: Any, keypoints: Any) -> dict[str, Any]:
        """Predict from pre-extracted Gaussian keypoint patches and keypoints.

        ``video_window`` must be ``[B, 30, 15, 3, 32, 32]``. The original model's
        positional stem has three input channels, so keypoints must be
        ``[B, 30, 15, 3]`` (typically x, y, confidence). This corrects the
        two-coordinate comment in ``train.py``, which conflicts with the actual
        layer and checkpoint shape ``[24, 3, 1, 1]``.
        """

        torch = self.torch
        data = video_window if torch.is_tensor(video_window) else torch.as_tensor(video_window)
        kpts = keypoints if torch.is_tensor(keypoints) else torch.as_tensor(keypoints)
        expected = (self.clip_frames, self.skeleton_points, 3, 32, 32)
        if data.ndim != 6 or tuple(data.shape[1:]) != expected:
            raise ValueError(f"VSViG patches must have shape [B, {', '.join(map(str, expected))}]")
        if kpts.ndim != 4 or tuple(kpts.shape[1:]) != (
            self.clip_frames,
            self.skeleton_points,
            self.keypoint_features,
        ):
            raise ValueError("VSViG keypoints must have shape [B, 30, 15, 3]")
        if data.shape[0] != kpts.shape[0]:
            raise ValueError("Patch and keypoint batch sizes differ")
        self.freeze()
        self._embedding = self._logits = None
        with torch.no_grad():
            probability = self.model(
                data.to(self.device, dtype=torch.float32),
                kpts.to(self.device, dtype=torch.float32),
            )
        if self._embedding is None or self._logits is None:
            raise RuntimeError("VSViG feature hooks did not capture encoder outputs")
        return EncoderOutput(
            embedding=self._embedding.detach(),
            logits=self._logits.detach(),
            probability=probability.detach(),
        ).as_dict()

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def __enter__(self) -> "VSViGAdapter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
