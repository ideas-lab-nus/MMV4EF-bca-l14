"""Shared LSTM encoder used by local-only and ECMWF-augmented models."""

from __future__ import annotations

import torch
from torch import nn


class WeatherLSTM(nn.Module):
    def __init__(
        self,
        local_features: int,
        future_features: int,
        hidden_dim: int = 32,
        targets: int = 5,
    ) -> None:
        super().__init__()
        self.config = {
            "local_features": local_features,
            "future_features": future_features,
            "hidden_dim": hidden_dim,
            "targets": targets,
        }
        self.encoder = nn.LSTM(
            local_features, hidden_dim, num_layers=1, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + future_features, 32),
            nn.SiLU(),
            nn.Linear(32, targets),
        )

    def forward(
        self, history: torch.Tensor, future_features: torch.Tensor
    ) -> torch.Tensor:
        if history.ndim != 3 or future_features.ndim != 3:
            raise ValueError("WeatherLSTM inputs must be rank-three tensors")
        if history.shape[0] != future_features.shape[0]:
            raise ValueError("History and future-feature batches must match")
        _, (hidden, _) = self.encoder(history)
        context = hidden[-1].unsqueeze(1).expand(
            -1, future_features.shape[1], -1
        )
        return self.head(torch.cat([context, future_features], dim=-1))
