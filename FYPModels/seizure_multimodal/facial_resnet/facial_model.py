from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


EMOTIONS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)


class ResNet50GRUAttention(nn.Module):
    """ResNet50 frame encoder, BiGRU, and temporal multi-head attention.

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
        gru_hidden_size: int = 256,
        gru_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_heads: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Adapt ImageNet ResNet50 from RGB to RGB + optical flow.
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                # Preserve the original pretrained RGB filters.
                backbone.conv1.weight[:, :3].copy_(
                    original_conv.weight
                )

                # Initialize the optical-flow channel from the mean RGB filter.
                backbone.conv1.weight[:, 3:4].copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        self.feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_output_size = gru_hidden_size * (
            2 if bidirectional else 1
        )

        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if gru_output_size % attention_heads:
            raise ValueError(
                "GRU output size must be divisible by attention_heads: "
                f"{gru_output_size} is not divisible by {attention_heads}"
            )

        self.attention_heads = attention_heads
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=gru_output_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(gru_output_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, num_classes),
        )

        # ImageNet normalization values.
        self.register_buffer(
            "rgb_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def normalize_frames(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize RGB while retaining flow in approximately [0, 1]."""

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

        # Process all frames with ResNet50 as one large frame batch.
        frames = video.reshape(
            batch * time,
            channels,
            height,
            width,
        )
        frames = self.normalize_frames(frames)

        frame_features = self.backbone(frames)

        # Restore the temporal dimension for the GRU.
        sequence_features = frame_features.reshape(
            batch,
            time,
            self.feature_dim,
        )

        temporal_features, _ = self.gru(sequence_features)

        # Each time step attends to every other time step. The residual path
        # preserves the BiGRU representation while LayerNorm stabilizes it.
        attended_features, _ = self.temporal_attention(
            temporal_features,
            temporal_features,
            temporal_features,
            need_weights=False,
        )
        attended_features = self.attention_norm(
            temporal_features + attended_features
        )

        # All preprocessed windows contain a fixed number of valid frames, so
        # unmasked temporal mean pooling is appropriate here.
        sequence_embedding = attended_features.mean(dim=1)

        return self.classifier(sequence_embedding)

    def predict_proba(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        return torch.softmax(self(video), dim=1)


# Keep the old import name as an alias while making its architecture explicit.
ResNet50GRU = ResNet50GRUAttention

__all__ = ["EMOTIONS", "ResNet50GRU", "ResNet50GRUAttention"]
