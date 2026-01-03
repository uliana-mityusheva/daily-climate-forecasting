from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 6          # month, day, humidity, wind_speed, meanpressure, ratio
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0         # applied only if num_layers > 1
    bidirectional: bool = False

    out_size: int = 1            # meantemp per timestep


class ClimateLSTM(nn.Module):
    """
    Kaggle-like seq2seq model:
      X: [B, T, F]
      y_hat: [B, T, 1]
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        lstm_out = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.fc = nn.Linear(lstm_out, cfg.out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        return: [B, T, 1]
        """
        out, _ = self.lstm(x)     # [B, T, H]
        y_hat = self.fc(out)      # [B, T, 1]
        return y_hat
