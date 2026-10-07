"""Projection-head modules for retrieval finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    """LayerNorm-MLP projection head used by DINO/CLIP/ResNet wrappers."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and projection_dim must be positive")
        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, projection_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def to_config(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "projection_dim": self.projection_dim,
            "dropout": self.dropout,
            "use_layer_norm": self.use_layer_norm,
        }


def normalize_embedding(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable L2-normalization for retrieval descriptors."""
    return F.normalize(x, p=2, dim=1, eps=1e-12)
