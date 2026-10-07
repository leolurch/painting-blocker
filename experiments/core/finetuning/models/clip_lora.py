"""Trainable CLIP-family retrieval wrapper (OpenAI CLIP and LAION OpenCLIP in HF format)."""

from __future__ import annotations

import os

import torch
from torch import nn

from .base import TrainableRetrievalModel
from .dinov3 import (
    _layer_index,
    _matching_lora_modules,
    layer_span_settings,
    qv_lora_candidate_modules,
    resolve_qv_lora_targets,
)
from .projection import ProjectionHead, normalize_embedding


class ClipLoRAEmbeddingModel(TrainableRetrievalModel):
    """CLIP vision tower with q/v LoRA and a LayerNorm-MLP head.

    The head reads ``image_embeds``: the post-LayerNorm CLS token through the
    pretrained visual projection, which is the descriptor the frozen CLIP and
    OpenCLIP baselines score. The visual projection stays frozen.
    """

    checkpoint_trainable_only = True

    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int | None = None,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.0,
        lora_rank: int = 4,
        lora_alpha: int = 4,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_target_modules: object = None,
        adapt_last_n_layers: int | None = None,
        adapt_first_n_layers: int | None = None,
        lora_span: str | None = None,
        inspect_lora_modules: bool = False,
        attn_implementation: str = "sdpa",
        hf_token: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        if not revision:
            raise ValueError("clip_lora_qv_projection stores trainable tensors only and needs a pinned revision")
        self.model_type = "clip_lora_qv_projection"
        self.backbone_name = backbone_name
        self.backbone_revision = revision
        self.pooling_type = "image_embeds"

        base_backbone = self._load_backbone(backbone_name, hf_token, revision, attn_implementation)
        vision_config = base_backbone.config
        descriptor_dim = int(vision_config.projection_dim)
        self.native_image_size = int(vision_config.image_size)
        self.projection_dim = int(embedding_dim or descriptor_dim)
        self.head_hidden_dim = int(head_hidden_dim or 1024)

        self.lora_span = None if lora_span is None else str(lora_span)
        self.adapt_first_n_layers = int(adapt_first_n_layers) if adapt_first_n_layers is not None else None
        self.adapt_last_n_layers = int(adapt_last_n_layers) if adapt_last_n_layers is not None else None
        self.lora_target_modules = resolve_qv_lora_targets(
            base_backbone,
            lora_target_modules,
            adapt_first_n_layers=self.adapt_first_n_layers,
            adapt_last_n_layers=self.adapt_last_n_layers,
        )
        matched_modules = _matching_lora_modules(base_backbone, self.lora_target_modules)
        if inspect_lora_modules:
            print("CLIP LoRA q/v candidate modules:", flush=True)
            for name in qv_lora_candidate_modules(base_backbone):
                print(f"  {name}", flush=True)
            print("CLIP LoRA selected modules:", flush=True)
            for name in matched_modules:
                print(f"  {name}", flush=True)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("CLIP LoRA finetuning requires peft") from exc
        for parameter in base_backbone.parameters():
            parameter.requires_grad = False
        peft_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            target_modules=self.lora_target_modules,
            lora_dropout=float(lora_dropout),
            bias=str(lora_bias),
            modules_to_save=[],
        )
        self.backbone = get_peft_model(base_backbone, peft_config)

        self.projection = ProjectionHead(
            input_dim=descriptor_dim,
            projection_dim=self.projection_dim,
            hidden_dim=self.head_hidden_dim,
            dropout=float(head_dropout),
            use_layer_norm=True,
        )
        self.lora_config = {
            "rank": int(lora_rank),
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
            "bias": str(lora_bias),
            "target_modules": list(self.lora_target_modules),
            "matched_modules": matched_modules,
            "span": self.lora_span,
            "adapt_first_n_layers": self.adapt_first_n_layers,
            "adapt_last_n_layers": self.adapt_last_n_layers,
            "adapted_layer_indices": sorted(
                {index for name in matched_modules if (index := _layer_index(name)) is not None}
            ),
        }

    @staticmethod
    def _load_backbone(
        backbone_name: str,
        hf_token: str | None,
        revision: str,
        attn_implementation: str | None,
    ) -> nn.Module:
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("CLIP finetuning requires transformers") from exc
        token = hf_token or os.getenv("HF_TOKEN")
        kwargs: dict[str, object] = {"revision": revision}
        if token:
            kwargs["token"] = token
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        return CLIPVisionModelWithProjection.from_pretrained(backbone_name, **kwargs)

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata["lora_config"] = dict(self.lora_config)
        metadata["head_hidden_dim"] = int(self.head_hidden_dim)
        metadata["native_image_size"] = int(self.native_image_size)
        return metadata

    def backbone_descriptor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=pixel_values).image_embeds

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return normalize_embedding(self.projection(self.backbone_descriptor(pixel_values)))


def build_clip_lora_model(config: dict[str, object]) -> ClipLoRAEmbeddingModel:
    lora_cfg = dict(config.get("lora") or {})
    span, adapt_first_n_layers, adapt_last_n_layers = layer_span_settings(lora_cfg)
    head_cfg = dict(config.get("head") or {})
    head_type = str(head_cfg.get("type", "mlp_projection")).strip().lower()
    if head_type not in {"mlp", "mlp_projection"}:
        raise ValueError("CLIP LoRA head.type must be 'mlp_projection'")
    output_dim = config.get("projection_dim", config.get("embedding_dim", head_cfg.get("output_dim")))
    hidden_dim = head_cfg.get("hidden_dim")
    return ClipLoRAEmbeddingModel(
        backbone_name=str(config.get("backbone") or config.get("backbone_id")),
        embedding_dim=int(output_dim) if output_dim is not None else None,
        head_hidden_dim=int(hidden_dim) if hidden_dim is not None else None,
        head_dropout=float(head_cfg.get("dropout", 0.0)),
        lora_rank=int(lora_cfg.get("rank", lora_cfg.get("r", 4))),
        lora_alpha=int(lora_cfg.get("alpha", lora_cfg.get("lora_alpha", 4))),
        lora_dropout=float(lora_cfg.get("dropout", lora_cfg.get("lora_dropout", 0.05))),
        lora_bias=str(lora_cfg.get("bias", "none")),
        lora_target_modules=lora_cfg.get("target_modules"),
        adapt_first_n_layers=adapt_first_n_layers,
        adapt_last_n_layers=adapt_last_n_layers,
        lora_span=span,
        inspect_lora_modules=bool(lora_cfg.get("inspect_modules", False)),
        attn_implementation=str(config.get("attn_implementation") or "sdpa"),
        hf_token=config.get("hf_token") if isinstance(config.get("hf_token"), str) else None,
        revision=str(config.get("revision") or config.get("backbone_revision") or "") or None,
    )
