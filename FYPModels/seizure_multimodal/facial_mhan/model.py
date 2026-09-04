"""MHAN + BiGRU model for micro-expression and seizure-semiology learning."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


EMOTION_CLASSES = (
    "neutral", "fear", "anger", "sadness", "disgust", "surprise", "happiness"
)
DEFAULT_SEMIOLOGIES = (
    "fear", "crying", "laughter", "grimacing", "oral_automatism", "motor_asymmetry"
)


class MultiHeadAttentionPooling(nn.Module):
    """Independent attention heads that pool a temporal sequence.

    Unlike self-attention followed by mean pooling, each head exposes a
    normalized weight for every frame, making attended micro-expression frames
    available for later review.
    """

    def __init__(self, feature_dim: int, heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        if feature_dim <= 0 or heads <= 0:
            raise ValueError("feature_dim and heads must be positive")
        self.heads = heads
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, heads, bias=False),
        )

    def forward(
        self, sequence: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [B,T,D]")
        scores = self.score(sequence).transpose(1, 2)  # [B,H,T]
        if valid_mask is not None:
            if valid_mask.shape != sequence.shape[:2]:
                raise ValueError("valid_mask must have shape [B,T]")
            if not valid_mask.any(dim=1).all():
                raise ValueError("every sample requires at least one valid frame")
            scores = scores.masked_fill(~valid_mask[:, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        contexts = torch.einsum("bht,btd->bhd", weights, sequence)
        return contexts.flatten(1), weights


class MicroExpressionEncoder(nn.Module):
    """Compact frame encoder for RGB, optical flow, and facial geometry."""

    def __init__(self, geometry_dim: int = 20, output_dim: int = 256) -> None:
        super().__init__()
        self.geometry_dim = geometry_dim
        self.image = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.geometry = nn.Sequential(
            nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, 64), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )

    def forward(self, video: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 4:
            raise ValueError("video must have shape [B,T,4,H,W]")
        if geometry.ndim != 3 or geometry.shape[:2] != video.shape[:2]:
            raise ValueError("geometry must have shape [B,T,G] aligned with video")
        if geometry.shape[2] != self.geometry_dim:
            raise ValueError(f"geometry must contain {self.geometry_dim} features")
        batch, time = video.shape[:2]
        image = self.image(video.flatten(0, 1)).reshape(batch, time, -1)
        geometry_features = self.geometry(geometry)
        return self.fusion(torch.cat((image, geometry_features), dim=-1))


class MHANBiGRUSemiology(nn.Module):
    """Multi-task MHAN-BiGRU for emotion, semiology, and seizure outputs."""

    def __init__(
        self,
        *,
        emotion_classes: Sequence[str] = EMOTION_CLASSES,
        semiology_classes: Sequence[str] = DEFAULT_SEMIOLOGIES,
        frame_feature_dim: int = 256,
        gru_hidden_size: int = 128,
        attention_heads: int = 8,
        dropout: float = 0.3,
        geometry_dim: int = 20,
    ) -> None:
        super().__init__()
        if len(emotion_classes) < 2 or not semiology_classes:
            raise ValueError("At least two emotions and one semiology are required")
        self.emotion_classes = tuple(emotion_classes)
        self.semiology_classes = tuple(semiology_classes)
        self.frame_feature_dim = frame_feature_dim
        self.gru_hidden_size = gru_hidden_size
        self.attention_heads = attention_heads
        self.geometry_dim = geometry_dim
        self.encoder = MicroExpressionEncoder(geometry_dim, frame_feature_dim)
        self.bigru = nn.GRU(
            frame_feature_dim, gru_hidden_size, batch_first=True, bidirectional=True
        )
        temporal_dim = gru_hidden_size * 2
        self.mhan = MultiHeadAttentionPooling(temporal_dim, attention_heads, dropout)
        representation_dim = temporal_dim * attention_heads
        self.shared = nn.Sequential(
            nn.LayerNorm(representation_dim), nn.Dropout(dropout),
            nn.Linear(representation_dim, temporal_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(temporal_dim, len(self.emotion_classes))
        self.semiology_head = nn.Linear(temporal_dim, len(self.semiology_classes))
        self.seizure_head = nn.Linear(temporal_dim, 2)

    def forward(
        self,
        video: torch.Tensor,
        geometry: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        frame_features = self.encoder(video, geometry)
        temporal, _ = self.bigru(frame_features)
        attended, weights = self.mhan(temporal, valid_mask)
        representation = self.shared(attended)
        return {
            "emotion_logits": self.emotion_head(representation),
            "semiology_logits": self.semiology_head(representation),
            "seizure_logits": self.seizure_head(representation),
            "attention_weights": weights,
            "embedding": representation,
        }

    @torch.inference_mode()
    def predict(self, video: torch.Tensor, geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self(video, geometry)
        return {
            **output,
            "emotion_probability": torch.softmax(output["emotion_logits"], dim=-1),
            "semiology_probability": torch.sigmoid(output["semiology_logits"]),
            "seizure_probability": torch.softmax(output["seizure_logits"], dim=-1)[:, 1],
        }


__all__ = [
    "DEFAULT_SEMIOLOGIES", "EMOTION_CLASSES", "MHANBiGRUSemiology",
    "MicroExpressionEncoder", "MultiHeadAttentionPooling",
]
