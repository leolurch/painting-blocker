"""Pooling layers for trainable retrieval heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GeMPooling2d(nn.Module):
    """Generalized mean pooling for non-negative CNN feature maps."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError("p must be positive")
        self.eps = float(eps)
        value = torch.ones(1) * float(p)
        if learn_p:
            self.p = nn.Parameter(value)
        else:
            self.register_buffer("p", value)
        self.learn_p = bool(learn_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"GeMPooling2d expects [B, C, H, W], got {tuple(x.shape)}")
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p).flatten(1)

    def to_config(self) -> dict[str, object]:
        return {"type": "gem", "p": float(self.p.detach().cpu()[0]), "eps": self.eps, "learn_p": self.learn_p}


class TokenGeMPooling(nn.Module):
    """GeM-style pooling over non-negative projected token features."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError("p must be positive")
        self.eps = float(eps)
        value = torch.ones(1) * float(p)
        if learn_p:
            self.p = nn.Parameter(value)
        else:
            self.register_buffer("p", value)
        self.learn_p = bool(learn_p)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"TokenGeMPooling expects [B, N, D], got {tuple(tokens.shape)}")
        if tokens.shape[1] <= 0:
            raise ValueError("Cannot pool an empty token sequence")
        x = tokens.clamp(min=self.eps).pow(self.p)
        return x.mean(dim=1).pow(1.0 / self.p)


class TokenAggregationHead(nn.Module):
    """Project signed ViT tokens to non-negative features before token pooling."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        pooling: str = "gem",
        p_init: float = 3.0,
        learn_p: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or feature_dim <= 0:
            raise ValueError("input_dim and feature_dim must be positive")
        self.pre = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, feature_dim), nn.GELU(), nn.ReLU())
        if pooling == "gem":
            self.pool = TokenGeMPooling(p=p_init, learn_p=learn_p)
        elif pooling == "mean":
            self.pool = nn.Identity()
        else:
            raise ValueError("token pooling must be 'gem' or 'mean'")
        self.pooling = pooling

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.pre(tokens)
        if self.pooling == "mean":
            return features.mean(dim=1)
        return self.pool(features)
