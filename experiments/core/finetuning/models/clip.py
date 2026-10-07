"""Trainable CLIP image-encoder retrieval wrappers."""

from __future__ import annotations

import os

import torch
from torch import nn

from .base import TrainableRetrievalModel, set_module_trainable
from .pooling import TokenAggregationHead
from .projection import ProjectionHead, normalize_embedding


class ClipProjectionModel(TrainableRetrievalModel):
    """OpenAI-style CLIP vision encoder with CLS or patch-token aggregation."""

    def __init__(
        self,
        backbone_name: str,
        pooling: str = "cls",
        projection_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        token_feature_dim: int = 1024,
        token_gem_p: float = 3.0,
        token_gem_learn_p: bool = True,
        gem_after_projection: bool = True,
        hf_token: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.model_type = "clip_gem_projection" if pooling == "patch_gem" else "clip_projection"
        self.backbone_name = backbone_name
        self.pooling_type = pooling
        self.projection_dim = int(projection_dim)
        self.backbone_revision = revision
        self.gem_after_projection = bool(gem_after_projection)
        self.model = self._load_model(backbone_name, hf_token, revision)
        self.vision_model = self.model.vision_model
        self.visual_projection = self.model.visual_projection
        clip_dim = int(getattr(self.model.config, "projection_dim"))
        hidden_size = int(getattr(self.model.vision_model.config, "hidden_size"))
        if pooling == "cls":
            self.projection = ProjectionHead(clip_dim, projection_dim, hidden_dim, dropout)
            self.token_pool = None
            self.token_projection = None
        elif pooling in {"patch_mean", "patch_gem"}:
            mode = "mean" if pooling == "patch_mean" else "gem"
            token_dim = clip_dim if self.gem_after_projection else hidden_size
            self.token_pool = TokenAggregationHead(
                token_dim,
                token_feature_dim,
                pooling=mode,
                p_init=token_gem_p,
                learn_p=token_gem_learn_p,
            )
            self.token_projection = nn.Linear(token_feature_dim, projection_dim)
            self.projection = ProjectionHead(clip_dim, projection_dim, hidden_dim, dropout) if not self.gem_after_projection else nn.Identity()
        else:
            raise ValueError("CLIP pooling must be 'cls', 'patch_mean', or 'patch_gem'")
        if freeze_backbone:
            set_module_trainable(self.model, False)

    @staticmethod
    def _load_model(backbone_name: str, hf_token: str | None, revision: str | None) -> nn.Module:
        try:
            from transformers import CLIPModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("CLIP finetuning requires transformers") from exc
        token = hf_token or os.getenv("HF_TOKEN")
        kwargs = {"token": token} if token else {}
        if revision:
            kwargs["revision"] = revision
        return CLIPModel.from_pretrained(backbone_name, **kwargs)

    def _project_visual(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(x)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        if self.pooling_type == "cls":
            cls = self._project_visual(vision_outputs.pooler_output)
            return normalize_embedding(self.projection(cls))
        tokens = self.vision_model.post_layernorm(vision_outputs.last_hidden_state[:, 1:, :])
        if self.gem_after_projection:
            tokens = self._project_visual(tokens)
            pooled = self.token_pool(tokens)
            return normalize_embedding(self.token_projection(pooled))
        pooled_hidden = self.token_pool(tokens)
        clip_space = self._project_visual(pooled_hidden)
        return normalize_embedding(self.projection(clip_space))


def build_clip_model(config: dict[str, object]) -> ClipProjectionModel:
    pooling_cfg = dict(config.get("pooling") or {})
    return ClipProjectionModel(
        backbone_name=str(config.get("backbone")),
        pooling=str(pooling_cfg.get("type", config.get("pooling_type", "cls"))),
        projection_dim=int(config.get("projection_dim", 512)),
        hidden_dim=int(config.get("projection_hidden_dim", 1024)),
        dropout=float(config.get("dropout", 0.1)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        token_feature_dim=int(config.get("token_feature_dim", 1024)),
        token_gem_p=float(pooling_cfg.get("p_init", 3.0)),
        token_gem_learn_p=bool(pooling_cfg.get("learn_p", True)),
        gem_after_projection=bool(config.get("gem_after_projection", True)),
        hf_token=config.get("hf_token") if isinstance(config.get("hf_token"), str) else None,
        revision=str(config.get("revision") or config.get("backbone_revision") or "") or None,
    )
