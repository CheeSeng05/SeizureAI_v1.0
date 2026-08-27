from __future__ import annotations

import torch
from torch import nn


class PoseLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(output_size, output_size // 2),
            nn.Tanh(),
            nn.Linear(output_size // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fusion-ready body embedding before classification."""
        sequence, _ = self.lstm(x)
        attention_scores = self.attention(sequence).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        return torch.sum(sequence * attention_weights, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))
