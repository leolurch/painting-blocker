"""Trainable ResNet wrapper on the Hugging Face weights the frozen ResNet-50 baseline scores."""

from __future__ import annotations

import os

import torch
from torch import nn

from .base import TrainableRetrievalModel
from .projection import ProjectionHead, normalize_embedding

# In a bottleneck block, layer.0 reduces channels, layer.1 is the 3x3 spatial
# convolution, and layer.2 expands channels again.
DEFAULT_CONV_TARGETS = ("layer.1.convolution",)


def bottleneck_blocks(backbone: nn.Module) -> list[str]:
    """Names of the residual blocks in network order, e.g. ``encoder.stages.3.layers.2``."""
    names = []
    for stage_index, stage in enumerate(backbone.encoder.stages):
        for layer_index, _ in enumerate(stage.layers):
            names.append(f"encoder.stages.{stage_index}.layers.{layer_index}")
    return names


def resolve_conv_lora_targets(
    backbone: nn.Module,
    targets: tuple[str, ...],
    *,
    adapt_last_n_blocks: int | None,
    adapt_first_n_blocks: int | None,
) -> tuple[list[str], list[int]]:
    blocks = bottleneck_blocks(backbone)
    if adapt_first_n_blocks is not None and adapt_last_n_blocks is not None:
        raise ValueError("Set only one of the first-N and last-N block limits")
    count = adapt_last_n_blocks if adapt_last_n_blocks is not None else adapt_first_n_blocks
    if count is not None and not 0 < int(count) <= len(blocks):
        raise ValueError(f"LoRA block count {count} must be within 1 and {len(blocks)}")
    indices = list(range(len(blocks)))
    if adapt_last_n_blocks is not None:
        indices = indices[-int(adapt_last_n_blocks):]
    elif adapt_first_n_blocks is not None:
        indices = indices[: int(adapt_first_n_blocks)]
    modules = dict(backbone.named_modules())
    selected = []
    for index in indices:
        for target in targets:
            name = f"{blocks[index]}.{target}"
            if not isinstance(modules.get(name), nn.Conv2d):
                raise ValueError(f"LoRA target {name} is not a Conv2d in the loaded ResNet")
            selected.append(name)
    return selected, indices


class ResNetHFLoRAEmbeddingModel(TrainableRetrievalModel):
    """HF ResNet with convolution LoRA, frozen BatchNorm, and a LayerNorm-MLP head.

    The descriptor is the spatial mean of the last stage, which equals the
    ``pooler_output`` the frozen baseline scores; the mean has a deterministic
    backward pass, unlike CUDA adaptive average pooling.
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
        hf_token: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        if not revision:
            raise ValueError("resnet_hf_lora_conv_projection stores trainable tensors only and needs a pinned revision")
        self.model_type = "resnet_hf_lora_conv_projection"
        self.backbone_name = backbone_name
        self.backbone_revision = revision
        self.pooling_type = "avgpool"

        base_backbone = self._load_backbone(backbone_name, hf_token, revision)
        descriptor_dim = int(base_backbone.config.hidden_sizes[-1])
        self.projection_dim = int(embedding_dim or descriptor_dim)
        self.head_hidden_dim = int(head_hidden_dim or 1024)
        targets = tuple(str(t) for t in (lora_target_modules or DEFAULT_CONV_TARGETS))  # type: ignore[union-attr]
        self.lora_span = None if lora_span is None else str(lora_span)
        selected, block_indices = resolve_conv_lora_targets(
            base_backbone,
            targets,
            adapt_last_n_blocks=adapt_last_n_layers,
            adapt_first_n_blocks=adapt_first_n_layers,
        )
        if inspect_lora_modules:
            print("ResNet LoRA selected convolutions:", flush=True)
            for name in selected:
                print(f"  {name}", flush=True)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("ResNet LoRA finetuning requires peft") from exc
        for parameter in base_backbone.parameters():
            parameter.requires_grad = False
        peft_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            target_modules=selected,
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
            "target_modules": list(targets),
            "matched_modules": selected,
            "span": self.lora_span,
            "adapt_first_n_layers": adapt_first_n_layers,
            "adapt_last_n_layers": adapt_last_n_layers,
            "adapted_layer_indices": block_indices,
        }

    @staticmethod
    def _load_backbone(backbone_name: str, hf_token: str | None, revision: str) -> nn.Module:
        try:
            from transformers import ResNetModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("ResNet finetuning requires transformers") from exc
        token = hf_token or os.getenv("HF_TOKEN")
        kwargs: dict[str, object] = {"revision": revision}
        if token:
            kwargs["token"] = token
        return ResNetModel.from_pretrained(backbone_name, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata["lora_config"] = dict(self.lora_config)
        metadata["head_hidden_dim"] = int(self.head_hidden_dim)
        return metadata

    def backbone_descriptor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        return hidden.mean(dim=(2, 3))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return normalize_embedding(self.projection(self.backbone_descriptor(pixel_values)))


def build_resnet_hf_lora_model(config: dict[str, object]) -> ResNetHFLoRAEmbeddingModel:
    from .dinov3 import layer_span_settings

    lora_cfg = dict(config.get("lora") or {})
    span, adapt_first_n_layers, adapt_last_n_layers = layer_span_settings(lora_cfg)
    head_cfg = dict(config.get("head") or {})
    head_type = str(head_cfg.get("type", "mlp_projection")).strip().lower()
    if head_type not in {"mlp", "mlp_projection"}:
        raise ValueError("ResNet LoRA head.type must be 'mlp_projection'")
    output_dim = config.get("projection_dim", config.get("embedding_dim", head_cfg.get("output_dim")))
    hidden_dim = head_cfg.get("hidden_dim")
    return ResNetHFLoRAEmbeddingModel(
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
        hf_token=config.get("hf_token") if isinstance(config.get("hf_token"), str) else None,
        revision=str(config.get("revision") or config.get("backbone_revision") or "") or None,
    )
