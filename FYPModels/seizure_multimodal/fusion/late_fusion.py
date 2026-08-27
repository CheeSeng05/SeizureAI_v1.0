from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sigmoid(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


@dataclass(frozen=True)
class LateFusion:
    """Unvalidated decision-level fusion for interface/software testing.

    Weights must be fixed a priori until paired validation vEEG exists. They must
    never be optimized by matching unrelated video and EEG samples by label.
    """

    video_weight: float = 0.5
    eeg_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.video_weight < 0 or self.eeg_weight < 0:
            raise ValueError("Fusion weights cannot be negative")
        if not np.isclose(self.video_weight + self.eeg_weight, 1.0):
            raise ValueError("Fusion weights must sum to one")

    def predict(
        self,
        *,
        video_probability: Any | None = None,
        eeg_probability: Any | None = None,
        video_logit: Any | None = None,
        eeg_logit: Any | None = None,
    ) -> np.ndarray:
        if video_probability is None:
            if video_logit is None:
                raise ValueError("Video probability or logit is required; missing data is not zero-filled")
            video_probability = sigmoid(video_logit)
        if eeg_probability is None:
            if eeg_logit is None:
                raise ValueError("EEG probability or logit is required; missing data is not zero-filled")
            eeg_probability = sigmoid(eeg_logit)
        video = np.asarray(video_probability, dtype=float)
        eeg = np.asarray(eeg_probability, dtype=float)
        if video.shape != eeg.shape:
            raise ValueError(f"Modality probability shapes differ: video={video.shape}, EEG={eeg.shape}")
        if np.any(~np.isfinite(video)) or np.any(~np.isfinite(eeg)):
            raise ValueError("Fusion inputs must be finite")
        if np.any((video < 0) | (video > 1)) or np.any((eeg < 0) | (eeg > 1)):
            raise ValueError("Probabilities must be in [0, 1]")
        return self.video_weight * video + self.eeg_weight * eeg


def build_decision_fusion_head(*, hidden_dim: int = 8, dropout: float = 0.2) -> Any:
    """Build an untrained PyTorch MLP over the two unimodal logits/probabilities."""

    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to instantiate the trainable decision fusion head") from exc
    if hidden_dim <= 0 or not 0 <= dropout < 1:
        raise ValueError("Invalid decision fusion dimensions/dropout")
    return nn.Sequential(
        nn.LayerNorm(2),
        nn.Linear(2, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
