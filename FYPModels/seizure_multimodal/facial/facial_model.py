from __future__ import annotations

import torch
from torch import nn


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.GELU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.GELU())


class HybridAttention(nn.Module):
    """Parallel channel, spatial, local-region, and multi-head token attention."""
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        squeeze = max(8, channels // 8)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, squeeze, 1),
                                     nn.GELU(), nn.Conv2d(squeeze, channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.tokens = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.fuse = nn.Conv2d(channels * 4, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x * self.channel(x)
        spatial_map = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        spatial = x * self.spatial(spatial_map)
        local = self.local(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens, _ = self.tokens(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens).transpose(1, 2).reshape(b, c, h, w)
        return x + self.fuse(torch.cat((channel, spatial, local, tokens), dim=1))


class FaceEmotionModel(nn.Module):
    """Compact FER network accepting RGB plus optical-flow magnitude (B,4,H,W)."""
    def __init__(self, num_classes: int = len(EMOTIONS), dropout: float = .3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(4, 32, 2), nn.MaxPool2d(2), ConvBlock(32, 64, 2),
            HybridAttention(64, 4), ConvBlock(64, 128, 2), HybridAttention(128, 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError(f"Expected [B,4,H,W] RGB+flow input, got {tuple(image.shape)}")
        return self.head(self.pool(self.features(image)))


__all__ = ["EMOTIONS", "ConvBlock", "HybridAttention", "FaceEmotionModel"]
