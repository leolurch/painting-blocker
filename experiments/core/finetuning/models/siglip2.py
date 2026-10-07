"""Trainable SigLIP2 retrieval wrapper."""

from __future__ import annotations

import os

import torch
from torch import nn

from .base import TrainableRetrievalModel, set_module_trainable
from .dinov3 import (
    _layer_index,
    _matching_lora_modules,
    layer_span_settings,
    qv_lora_candidate_modules,
    resolve_qv_lora_targets,
)
from .projection import ProjectionHead, normalize_embedding


_POOLING_ALIASES = {
    "map": "map",
    "pooler_output": "map",
    "default": "map",
    "patch_mean": "patch_mean",
    "mean_patch": "patch_mean",
}
POOLING_HEAD_PREFIX = "vision_model.head."


class Siglip2LoRAEmbeddingModel(TrainableRetrievalModel):
    """Fixed-resolution SigLIP2 vision tower with q/v LoRA and a LayerNorm-MLP head.

    ``map`` pooling reads ``pooler_output``, the pretrained attention-pooling
    head that the frozen SigLIP2 baseline scores. ``patch_mean`` averages the
    post-LayerNorm patch tokens instead. SigLIP has no CLS token.
    """

    def __init__(
        self,
        backbone_name: str,
        pooling: str = "map",
        train_pooling_head: bool = False,
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
        self.model_type = "siglip2_lora_qv_projection"
        self.backbone_name = backbone_name
        self.backbone_revision = revision
        key = str(pooling).strip().lower()
        if key not in _POOLING_ALIASES:
            raise ValueError("SigLIP2 LoRA pooling must be 'map' (pooler_output) or 'patch_mean'")
        self.pooling_type = _POOLING_ALIASES[key]
        self.train_pooling_head = bool(train_pooling_head)
        if self.train_pooling_head and self.pooling_type != "map":
            raise ValueError("pooling.trainable only applies to map pooling")

        base_backbone = self._load_backbone(backbone_name, hf_token, revision, attn_implementation)
        vision_config = base_backbone.config
        if self.pooling_type == "map" and not getattr(vision_config, "vision_use_head", True):
            raise ValueError(f"{backbone_name} has no attention-pooling head; use patch_mean pooling")
        hidden_size = int(vision_config.hidden_size)
        self.native_image_size = int(vision_config.image_size)
        self.projection_dim = int(embedding_dim or hidden_size)
        self.head_hidden_dim = int(head_hidden_dim or 2 * hidden_size)

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
            print("SigLIP2 LoRA q/v candidate modules:", flush=True)
            for name in qv_lora_candidate_modules(base_backbone):
                print(f"  {name}", flush=True)
            print("SigLIP2 LoRA selected modules:", flush=True)
            for name in matched_modules:
                print(f"  {name}", flush=True)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("SigLIP2 LoRA finetuning requires peft") from exc
        pooling_head = base_backbone.vision_model.head if self.pooling_type == "map" else None
        peft_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            target_modules=self.lora_target_modules,
            lora_dropout=float(lora_dropout),
            bias=str(lora_bias),
            modules_to_save=[],
        )
        self.backbone = get_peft_model(base_backbone, peft_config)
        if pooling_head is not None:
            set_module_trainable(pooling_head, self.train_pooling_head)
        self.backbone_trainable_prefixes = (POOLING_HEAD_PREFIX,) if self.train_pooling_head else ()

        self.dropout = nn.Identity()
        self.projection = ProjectionHead(
            input_dim=hidden_size,
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
        revision: str | None,
        attn_implementation: str | None,
    ) -> nn.Module:
        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("SigLIP2 finetuning requires transformers") from exc
        token = hf_token or os.getenv("HF_TOKEN")
        kwargs: dict[str, object] = {}
        if token:
            kwargs["token"] = token
        if revision:
            kwargs["revision"] = revision
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        backbone = SiglipVisionModel.from_pretrained(backbone_name, **kwargs)
        model_type = getattr(backbone.config, "model_type", "")
        if model_type != "siglip_vision_model":
            raise ValueError(
                f"{backbone_name} loaded as {model_type!r}; only fixed-resolution SigLIP/SigLIP2 "
                "vision towers are supported (NaFlex checkpoints need patch-aware inputs)"
            )
        return backbone

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata["lora_config"] = dict(self.lora_config)
        metadata["head_hidden_dim"] = int(self.head_hidden_dim)
        metadata["train_pooling_head"] = bool(self.train_pooling_head)
        metadata["native_image_size"] = int(self.native_image_size)
        return metadata

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        native = self.native_image_size
        interpolate = tuple(pixel_values.shape[-2:]) != (native, native)
        outputs = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=interpolate)
        if self.pooling_type == "map":
            x = outputs.pooler_output
        else:
            x = outputs.last_hidden_state.mean(dim=1)
        x = self.dropout(x)
        x = self.projection(x)
        return normalize_embedding(x)


def build_siglip2_lora_model(config: dict[str, object]) -> Siglip2LoRAEmbeddingModel:
    pooling_cfg = dict(config.get("pooling") or {})
    lora_cfg = dict(config.get("lora") or {})
    span, adapt_first_n_layers, adapt_last_n_layers = layer_span_settings(lora_cfg)
    head_cfg = dict(config.get("head") or {})
    head_type = str(head_cfg.get("type", "mlp_projection")).strip().lower()
    if head_type not in {"mlp", "mlp_projection"}:
        raise ValueError("SigLIP2 LoRA head.type must be 'mlp_projection'")
    output_dim = config.get("projection_dim", config.get("embedding_dim", head_cfg.get("output_dim")))
    hidden_dim = head_cfg.get("hidden_dim")
    return Siglip2LoRAEmbeddingModel(
        backbone_name=str(config.get("backbone") or config.get("backbone_id")),
        pooling=str(pooling_cfg.get("type", config.get("pooling_type", "map"))),
        train_pooling_head=bool(pooling_cfg.get("trainable", False)),
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
