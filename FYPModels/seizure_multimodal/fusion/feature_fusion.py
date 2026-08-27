from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureFusionSpec:
    video_embedding_dim: int
    eeg_embedding_dim: int
    projection_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.3

    def __post_init__(self) -> None:
        if min(
            self.video_embedding_dim,
            self.eeg_embedding_dim,
            self.projection_dim,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("All feature-fusion dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")

    @property
    def concatenated_dim(self) -> int:
        return self.projection_dim * 2

    def validate_inputs(self, video_embedding: Any, eeg_embedding: Any) -> tuple[np.ndarray, np.ndarray]:
        video = np.asarray(video_embedding)
        eeg = np.asarray(eeg_embedding)
        if video.ndim != 2 or eeg.ndim != 2:
            raise ValueError("Feature fusion expects 2-D [batch, embedding] arrays")
        if video.shape[0] != eeg.shape[0]:
            raise ValueError("Video and EEG batch sizes differ")
        if video.shape[1] != self.video_embedding_dim:
            raise ValueError(
                f"Expected video embedding dim {self.video_embedding_dim}, got {video.shape[1]}"
            )
        if eeg.shape[1] != self.eeg_embedding_dim:
            raise ValueError(f"Expected EEG embedding dim {self.eeg_embedding_dim}, got {eeg.shape[1]}")
        return video, eeg


def build_feature_fusion_model(spec: FeatureFusionSpec) -> Any:
    """Instantiate the untrained feature-level architecture.

    The model returns a single binary logit. Encoder embeddings may have
    different dimensions and are projected independently before concatenation.
    """

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to instantiate feature fusion") from exc

    class FeatureFusionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.video_projection = nn.Sequential(
                nn.LayerNorm(spec.video_embedding_dim),
                nn.Linear(spec.video_embedding_dim, spec.projection_dim),
                nn.ReLU(),
                nn.Dropout(spec.dropout),
            )
            self.eeg_projection = nn.Sequential(
                nn.LayerNorm(spec.eeg_embedding_dim),
                nn.Linear(spec.eeg_embedding_dim, spec.projection_dim),
                nn.ReLU(),
                nn.Dropout(spec.dropout),
            )
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(spec.concatenated_dim),
                nn.Linear(spec.concatenated_dim, spec.hidden_dim),
                nn.ReLU(),
                nn.Dropout(spec.dropout),
                nn.Linear(spec.hidden_dim, 1),
            )

        def forward(self, video_embedding: Any, eeg_embedding: Any) -> Any:
            if video_embedding.ndim != 2 or eeg_embedding.ndim != 2:
                raise ValueError("Feature fusion expects [batch, embedding] tensors")
            if video_embedding.shape[0] != eeg_embedding.shape[0]:
                raise ValueError("Video and EEG batch sizes differ")
            video = self.video_projection(video_embedding)
            eeg = self.eeg_projection(eeg_embedding)
            return self.fusion_head(torch.cat((video, eeg), dim=-1)).squeeze(-1)

    return FeatureFusionModel()
