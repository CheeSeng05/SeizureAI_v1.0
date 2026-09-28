from __future__ import annotations

import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1
from torch.nn import functional as F


EMOTIONS = (
    "neutral",
    "fear",
    "anger",
    "sadness",
    "disgust",
    "surprise",
    "happiness",
)


def _validate_attention(attention_dim: int, attention_heads: int, max_frames: int) -> None:
    if type(attention_heads) is not int or attention_heads <= 0 or attention_dim % attention_heads:
        raise ValueError(f"attention_heads must be positive and divide {attention_dim}")
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")


def _frame_backbone(pretrained: bool) -> nn.Module:
    """Build only the shared RGB/flow frame encoder, without temporal heads."""
    backbone = InceptionResnetV1(
        pretrained="vggface2" if pretrained else None, classify=False
    )
    if hasattr(backbone, "logits"):
        del backbone.logits
    original_conv = backbone.conv2d_1a.conv
    backbone.conv2d_1a.conv = nn.Conv2d(
        4, original_conv.out_channels, kernel_size=original_conv.kernel_size,
        stride=original_conv.stride, padding=original_conv.padding, bias=False,
    )
    if pretrained:
        with torch.no_grad():
            backbone.conv2d_1a.conv.weight[:, :3].copy_(original_conv.weight)
            backbone.conv2d_1a.conv.weight[:, 3:4].copy_(original_conv.weight.mean(1, keepdim=True))
    return backbone


class InceptionResnetV1Attention(nn.Module):
    """InceptionResnetV1 frame encoder and temporal multi-head attention.

    Input:
        [batch, time, 4, height, width]

    Channels:
        RGB plus optical-flow magnitude.

    Output:
        [batch, num_classes] logits.
    """

    def __init__(
        self,
        num_classes: int = len(EMOTIONS),
        dropout: float = 0.3,
        attention_heads: int = 8,
        pretrained: bool = True,
        max_frames: int = 512,
    ) -> None:
        super().__init__()

        self.feature_dim = 512
        attention_dim = self.feature_dim
        _validate_attention(attention_dim, attention_heads, max_frames)
        self.attention_heads = attention_heads
        self.max_frames = max_frames
        self.dropout = dropout
        self.backbone = _frame_backbone(pretrained)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.max_frames, attention_dim) * 0.02
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(attention_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, num_classes),
        )

        # VGGFace2 normalization values.
        self.register_buffer(
            "rgb_mean",
            torch.tensor(
                [127.5 / 255.0] * 3,
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(
                [128.0 / 255.0] * 3,
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def normalize_frames(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize RGB while retaining flow in approximately [0, 1]."""

        frames = F.interpolate(frames, size=(160, 160), mode="bilinear", align_corners=False)
        rgb = (frames[:, :3] - self.rgb_mean) / self.rgb_std
        flow = frames[:, 3:4]

        return torch.cat((rgb, flow), dim=1)

    def forward(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 4:
            raise ValueError(
                "Expected [B,T,4,H,W] RGB+flow input, "
                f"got {tuple(video.shape)}"
            )

        batch, time, channels, height, width = video.shape
        if not 1 <= time <= self.max_frames:
            raise ValueError(f"Window must contain 1–{self.max_frames} frames")
        if min(batch, height, width) < 1:
            raise ValueError("Batch and spatial dimensions must be nonempty")

        # Process all frames with InceptionResnetV1 as one large frame batch.
        frames = video.reshape(
            batch * time,
            channels,
            height,
            width,
        )
        frames = self.normalize_frames(frames)

        frame_features = self.backbone(frames)

        # Restore the frame sequence and encode its temporal order.
        sequence_features = frame_features.reshape(
            batch,
            time,
            self.feature_dim,
        )

        temporal_features = (
            sequence_features + self.position_embedding[:, :time]
        )

        # Every frame attends to the full window; retain its features through a residual.
        attended_features, _ = self.temporal_attention(
            temporal_features,
            temporal_features,
            temporal_features,
            need_weights=False,
        )
        attended_features = self.attention_norm(
            temporal_features + attended_features
        )

        # This path expects fixed-length windows without padding.
        sequence_embedding = attended_features.mean(dim=1)

        return self.classifier(sequence_embedding)

    def predict_proba(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        return torch.softmax(self(video), dim=1)


class InceptionResnetV1Semiology(nn.Module):
    """InceptionResnetV1 and temporal attention over visual/geometry features."""

    def __init__(
        self, pretrained: bool = True, *, attention_heads: int = 8,
        max_frames: int = 512, dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.emotion_classes = EMOTIONS
        self.semiology_classes = (
            "fear", "crying", "laughter", "grimacing", "oral_automatism", "motor_asymmetry"
        )
        self.feature_dim = 512
        output_dim = self.feature_dim + 128
        _validate_attention(output_dim, attention_heads, max_frames)
        self.attention_heads = attention_heads
        self.max_frames = max_frames
        self.dropout = dropout
        self.backbone = _frame_backbone(pretrained)
        self.register_buffer("rgb_mean", torch.full((1, 3, 1, 1), 127.5 / 255.0))
        self.register_buffer("rgb_std", torch.full((1, 3, 1, 1), 128.0 / 255.0))
        self.geometry = nn.Sequential(nn.LayerNorm(20), nn.Linear(20, 128), nn.GELU())

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.max_frames, output_dim) * 0.02
        )
        self.temporal_attention = nn.MultiheadAttention(
            output_dim, num_heads=attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(output_dim)

        self.emotion_head = nn.Linear(output_dim, len(self.emotion_classes))
        self.semiology_head = nn.Linear(output_dim, len(self.semiology_classes))
        self.seizure_head = nn.Linear(output_dim, 2)

    def forward(self, video: torch.Tensor, geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 5 or video.shape[2] != 4:
            raise ValueError("video must have shape [B,T,4,H,W]")
        if geometry.shape != (video.shape[0], video.shape[1], 20):
            raise ValueError("geometry must have shape [B,T,20]")
        batch, time = video.shape[:2]
        if not 1 <= time <= self.max_frames:
            raise ValueError(f"Window must contain 1–{self.max_frames} frames")
        if min(batch, *video.shape[3:]) < 1:
            raise ValueError("Batch and spatial dimensions must be nonempty")
        frames = video.flatten(0, 1)
        frames = InceptionResnetV1Attention.normalize_frames(self, frames)
        visual = self.backbone(frames).reshape(batch, time, -1)
        tokens = torch.cat((visual, self.geometry(geometry)), dim=-1)
        tokens = tokens + self.position_embedding[:, :time]
        attended, _ = self.temporal_attention(
            tokens, tokens, tokens, need_weights=False
        )
        embedding = self.attention_norm(tokens + attended).mean(dim=1)
        return {
            "emotion_logits": self.emotion_head(embedding),
            "semiology_logits": self.semiology_head(embedding),
            "seizure_logits": self.seizure_head(embedding),
            "embedding": embedding,
        }


__all__ = ["EMOTIONS", "InceptionResnetV1Semiology", "InceptionResnetV1Attention"]
