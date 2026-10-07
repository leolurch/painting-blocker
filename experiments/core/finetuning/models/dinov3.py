"""Trainable DINOv3 retrieval wrappers."""

from __future__ import annotations

import os
import re

import torch
from torch import nn

from .base import TrainableRetrievalModel, set_module_trainable
from .pooling import TokenAggregationHead
from .projection import ProjectionHead, normalize_embedding


_QV_TARGET_PAIRS = (("query", "value"), ("q_proj", "v_proj"))
_SPAN_RE = re.compile(r"^(?:all|first[1-9][0-9]*|last[1-9][0-9]*)$")


def _linear_module_names(module: nn.Module) -> list[str]:
    return [name for name, child in module.named_modules() if isinstance(child, nn.Linear)]


def _matches_lora_target(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith(f".{target}") or module_name.split(".")[-1] == target


def _matching_lora_modules(backbone: nn.Module, targets: list[str]) -> list[str]:
    names = _linear_module_names(backbone)
    return [name for name in names if any(_matches_lora_target(name, target) for target in targets)]


def qv_lora_candidate_modules(backbone: nn.Module) -> list[str]:
    """Return discovered q/v-like Linear module names for one loaded DINOv3 backbone."""
    needles = ("query", "value", "q_proj", "v_proj")
    return [name for name in _linear_module_names(backbone) if any(needle in name for needle in needles)]


def _layer_index(module_name: str) -> int | None:
    # DINOv3 names blocks ``layer.N``; SigLIP names them ``layers.N``.
    match = re.search(r"(?:^|\.)layers?\.(\d+)(?:\.|$)", module_name)
    return int(match.group(1)) if match else None


def layer_span_settings(lora_cfg: dict[str, object]) -> tuple[str | None, int | None, int | None]:
    """Return ``(span, adapt_first_n_layers, adapt_last_n_layers)``.

    ``span: all`` adapts every discovered attention block. ``span: first16`` and
    ``span: last24`` adapt that end of the stack. An explicit ``adapt_*`` count
    must match the span. Omitting ``span`` keeps the older last-N behavior.
    """
    raw_span = lora_cfg.get("span")
    span = None if raw_span is None else str(raw_span).strip().lower()
    if span == "":
        raise ValueError("lora.span must not be empty")
    if span is not None and _SPAN_RE.fullmatch(span) is None:
        raise ValueError("lora.span must be 'all', 'first<N>', or 'last<N>'")
    raw_first = lora_cfg.get("adapt_first_n_layers")
    raw_last = lora_cfg.get("adapt_last_n_layers")
    first = None if raw_first is None else int(raw_first)
    last = None if raw_last is None else int(raw_last)
    if span == "all":
        if first is not None or last is not None:
            raise ValueError(
                "lora.span=all adapts every attention block, so omit "
                "adapt_first_n_layers and adapt_last_n_layers"
            )
        return span, None, None
    if span is not None and span.startswith("first"):
        count = int(span.removeprefix("first"))
        if first is not None and first != count:
            raise ValueError(
                f"lora.span={span} does not match adapt_first_n_layers={first}"
            )
        if last is not None:
            raise ValueError(f"lora.span={span} cannot be combined with adapt_last_n_layers")
        return span, count, None
    if span is not None and span.startswith("last"):
        count = int(span.removeprefix("last"))
        if last is not None and last != count:
            raise ValueError(
                f"lora.span={span} does not match adapt_last_n_layers={last}"
            )
        if first is not None:
            raise ValueError(f"lora.span={span} cannot be combined with adapt_first_n_layers")
        return span, None, count
    if first is not None and last is not None:
        raise ValueError("Set only one of lora.adapt_first_n_layers and lora.adapt_last_n_layers")
    return None, first, last


def resolve_qv_lora_targets(
    backbone: nn.Module,
    configured_targets: object = None,
    *,
    adapt_last_n_layers: int | None = None,
    adapt_first_n_layers: int | None = None,
) -> list[str]:
    """Resolve PEFT target_modules for DINOv3 attention query/value LoRA.

    When explicit targets are supplied, this verifies that they match at least one
    Linear module in the loaded backbone. Otherwise, it infers the common HF DINO
    q/v naming pair (``query``/``value`` or ``q_proj``/``v_proj``) from the actual
    module tree and fails loudly if neither pair is present.

    With neither layer limit, every matching attention block is adapted. ``adapt_first_n_layers``
    keeps the earliest blocks and ``adapt_last_n_layers`` keeps the latest ones.
    """
    if configured_targets:
        targets = [str(target) for target in configured_targets]  # type: ignore[arg-type]
        missing = [target for target in targets if not _matching_lora_modules(backbone, [target])]
        if missing:
            candidates = ", ".join(qv_lora_candidate_modules(backbone)[:40]) or "<none>"
            raise ValueError(
                "Configured LoRA target_modules did not match DINOv3 Linear modules: "
                f"{missing}. Discovered q/v-like candidates: {candidates}"
            )
    else:
        leaf_names = {name.split(".")[-1] for name in _linear_module_names(backbone)}
        targets = []
        for query_name, value_name in _QV_TARGET_PAIRS:
            if query_name in leaf_names and value_name in leaf_names:
                targets = [query_name, value_name]
                break
        if not targets:
            candidates = ", ".join(qv_lora_candidate_modules(backbone)[:40]) or "<none>"
            raise ValueError(
                "Could not infer DINOv3 query/value LoRA targets. Inspect backbone.named_modules(); "
                f"q/v-like candidates: {candidates}"
            )

    if adapt_first_n_layers is not None and adapt_last_n_layers is not None:
        raise ValueError("Set only one of lora.adapt_first_n_layers and lora.adapt_last_n_layers")
    if adapt_first_n_layers is None and adapt_last_n_layers is None:
        return targets
    if adapt_first_n_layers is not None:
        count = int(adapt_first_n_layers)
        field = "lora.adapt_first_n_layers"
    else:
        count = int(adapt_last_n_layers)
        field = "lora.adapt_last_n_layers"
    if count <= 0:
        raise ValueError(f"{field} must be positive")
    matched = _matching_lora_modules(backbone, targets)
    indexed = [(name, _layer_index(name)) for name in matched]
    if any(index is None for _, index in indexed):
        unindexed = [name for name, index in indexed if index is None]
        raise ValueError(
            "Could not infer transformer layer indices for LoRA modules: "
            + ", ".join(unindexed[:20])
        )
    layer_indices = sorted({int(index) for _, index in indexed if index is not None})
    if count > len(layer_indices):
        raise ValueError(
            f"{field}={count} exceeds discovered layer count {len(layer_indices)}"
        )
    selected_indices = set(layer_indices[:count] if adapt_first_n_layers is not None else layer_indices[-count:])
    selected = [name for name, index in indexed if int(index) in selected_indices]  # type: ignore[arg-type]
    expected_per_layer = len(targets)
    counts = {index: sum(_layer_index(name) == index for name in selected) for index in selected_indices}
    incomplete = {index: found for index, found in counts.items() if found != expected_per_layer}
    if incomplete:
        raise ValueError(
            "Selected DINOv3 LoRA layers do not contain every configured target module: "
            f"{incomplete}; expected {expected_per_layer} per layer"
        )
    return selected


class DinoV3ProjectionModel(TrainableRetrievalModel):
    """DINOv3 backbone with CLS or patch-token aggregation and projection head."""

    def __init__(
        self,
        backbone_name: str,
        pooling: str = "cls",
        projection_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        token_feature_dim: int = 1024,
        token_gem_p: float = 3.0,
        token_gem_learn_p: bool = True,
        hf_token: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.model_type = "dinov3_gem_projection" if pooling == "patch_gem" else "dinov3_projection"
        self.backbone_name = backbone_name
        self.pooling_type = pooling
        self.projection_dim = int(projection_dim)
        self.backbone_revision = revision
        self.backbone = self._load_backbone(backbone_name, hf_token, revision)
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        if pooling == "cls":
            self.projection = ProjectionHead(hidden_size, projection_dim, hidden_dim, dropout)
            self.token_pool = None
            self.token_projection = None
        elif pooling in {"patch_mean", "patch_gem"}:
            mode = "mean" if pooling == "patch_mean" else "gem"
            self.token_pool = TokenAggregationHead(
                hidden_size,
                token_feature_dim,
                pooling=mode,
                p_init=token_gem_p,
                learn_p=token_gem_learn_p,
            )
            self.token_projection = nn.Linear(token_feature_dim, projection_dim)
            self.projection = nn.Identity()
        else:
            raise ValueError("DINOv3 pooling must be 'cls', 'patch_mean', or 'patch_gem'")
        if freeze_backbone:
            set_module_trainable(self.backbone, False)
        if unfreeze_last_n_blocks > 0:
            self._unfreeze_last_blocks(unfreeze_last_n_blocks)

    @staticmethod
    def _load_backbone(backbone_name: str, hf_token: str | None, revision: str | None) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("DINOv3 finetuning requires transformers") from exc
        token = hf_token or os.getenv("HF_TOKEN")
        kwargs = {"trust_remote_code": True}
        if token:
            kwargs["token"] = token
        if revision:
            kwargs["revision"] = revision
        return AutoModel.from_pretrained(backbone_name, **kwargs)

    def _patch_token_start(self) -> int:
        register_count = getattr(self.backbone.config, "num_register_tokens", 0) or 0
        return 1 + int(register_count)

    def _candidate_blocks(self) -> list[nn.Module]:
        for path in ("encoder.layer", "encoder.layers", "layers", "blocks"):
            obj: object = self.backbone
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, (nn.ModuleList, list, tuple)) and obj:
                return list(obj)
        return []

    def _unfreeze_last_blocks(self, count: int) -> None:
        blocks = self._candidate_blocks()
        if not blocks:
            raise ValueError("Could not find DINOv3 transformer blocks to unfreeze")
        for block in blocks[-int(count):]:
            set_module_trainable(block, True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        if self.pooling_type == "cls":
            return normalize_embedding(self.projection(hidden[:, 0, :]))
        patch_tokens = hidden[:, self._patch_token_start() :, :]
        pooled = self.token_pool(patch_tokens)
        return normalize_embedding(self.token_projection(pooled))


class DinoV3LoRAEmbeddingModel(TrainableRetrievalModel):
    """Parameter-efficient DINOv3 embedding model with q/v LoRA and linear head."""

    def __init__(
        self,
        backbone_name: str,
        pooling: str = "cls",
        embedding_dim: int = 512,
        head_dropout: float = 0.2,
        head_hidden_dim: int = 0,
        lora_rank: int = 8,
        lora_alpha: int = 16,
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
        self.model_type = "dinov3_lora_qv_projection"
        self.backbone_name = backbone_name
        self.backbone_revision = revision
        self.pooling_type = "patch_mean" if pooling == "mean_patch" else pooling
        self.projection_dim = int(embedding_dim)
        base_backbone = DinoV3ProjectionModel._load_backbone(backbone_name, hf_token, revision)
        hidden_size = int(getattr(base_backbone.config, "hidden_size"))
        self.num_register_tokens = int(getattr(base_backbone.config, "num_register_tokens", 4) or 0)
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
            print("DINOv3 LoRA q/v candidate modules:", flush=True)
            for name in qv_lora_candidate_modules(base_backbone):
                print(f"  {name}", flush=True)
            print("DINOv3 LoRA selected modules:", flush=True)
            for name in matched_modules:
                print(f"  {name}", flush=True)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("DINOv3 LoRA finetuning requires peft") from exc
        peft_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            target_modules=self.lora_target_modules,
            lora_dropout=float(lora_dropout),
            bias=str(lora_bias),
            modules_to_save=[],
        )
        self.backbone = get_peft_model(base_backbone, peft_config)
        self.head_hidden_dim = int(head_hidden_dim)
        if self.head_hidden_dim > 0:
            # LayerNorm-MLP projection head (same block used by the frozen-CLS
            # DinoV3ProjectionModel). Its internal dropout owns regularization,
            # so the pre-projection dropout collapses to identity.
            self.dropout = nn.Identity()
            self.projection = ProjectionHead(
                input_dim=hidden_size,
                projection_dim=self.projection_dim,
                hidden_dim=self.head_hidden_dim,
                dropout=float(head_dropout),
                use_layer_norm=True,
            )
        else:
            self.dropout = nn.Dropout(float(head_dropout))
            self.projection = nn.Linear(hidden_size, self.projection_dim)
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
        if self.pooling_type not in {"cls", "patch_mean"}:
            raise ValueError("DINOv3 LoRA pooling must be 'cls', 'mean_patch', or 'patch_mean'")

    def _patch_token_start(self) -> int:
        return 1 + int(self.num_register_tokens)

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata["lora_config"] = dict(self.lora_config)
        metadata["head_hidden_dim"] = int(self.head_hidden_dim)
        return metadata

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        if self.pooling_type == "cls":
            x = hidden[:, 0, :]
        else:
            x = hidden[:, self._patch_token_start() :, :].mean(dim=1)
        x = self.dropout(x)
        x = self.projection(x)
        return normalize_embedding(x)


def build_dinov3_model(config: dict[str, object]) -> DinoV3ProjectionModel:
    pooling_cfg = dict(config.get("pooling") or {})
    return DinoV3ProjectionModel(
        backbone_name=str(config.get("backbone")),
        pooling=str(pooling_cfg.get("type", config.get("pooling_type", "cls"))),
        projection_dim=int(config.get("projection_dim", 512)),
        hidden_dim=int(config.get("projection_hidden_dim", 1024)),
        dropout=float(config.get("dropout", 0.1)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        unfreeze_last_n_blocks=int(config.get("unfreeze_last_n_blocks", 0)),
        token_feature_dim=int(config.get("token_feature_dim", 1024)),
        token_gem_p=float(pooling_cfg.get("p_init", 3.0)),
        token_gem_learn_p=bool(pooling_cfg.get("learn_p", True)),
        hf_token=config.get("hf_token") if isinstance(config.get("hf_token"), str) else None,
        revision=str(config.get("revision") or config.get("backbone_revision") or "") or None,
    )


def build_dinov3_lora_model(config: dict[str, object]) -> DinoV3LoRAEmbeddingModel:
    pooling_cfg = dict(config.get("pooling") or {})
    lora_cfg = dict(config.get("lora") or {})
    span, adapt_first_n_layers, adapt_last_n_layers = layer_span_settings(lora_cfg)
    head_cfg = dict(config.get("head") or {})
    head_type = str(head_cfg.get("type", "linear_projection")).strip().lower()
    head_hidden_dim = int(head_cfg.get("hidden_dim", head_cfg.get("projection_hidden_dim", config.get("projection_hidden_dim", 0)) or 0))
    if head_type in {"mlp", "mlp_projection"} and head_hidden_dim <= 0:
        head_hidden_dim = 1024
    return DinoV3LoRAEmbeddingModel(
        backbone_name=str(config.get("backbone") or config.get("backbone_id")),
        pooling=str(pooling_cfg.get("type", config.get("pooling_type", "cls"))),
        embedding_dim=int(config.get("projection_dim", config.get("embedding_dim", head_cfg.get("output_dim", 512)))),
        head_dropout=float(head_cfg.get("dropout", config.get("dropout", 0.2))),
        head_hidden_dim=head_hidden_dim,
        lora_rank=int(lora_cfg.get("rank", lora_cfg.get("r", 8))),
        lora_alpha=int(lora_cfg.get("alpha", lora_cfg.get("lora_alpha", 16))),
        lora_dropout=float(lora_cfg.get("dropout", lora_cfg.get("lora_dropout", 0.05))),
        lora_bias=str(lora_cfg.get("bias", "none")),
        lora_target_modules=lora_cfg.get("target_modules"),
        adapt_first_n_layers=adapt_first_n_layers,
        adapt_last_n_layers=adapt_last_n_layers,
        lora_span=span,
        inspect_lora_modules=bool(lora_cfg.get("inspect_modules", config.get("inspect_lora_modules", False))),
        hf_token=config.get("hf_token") if isinstance(config.get("hf_token"), str) else None,
        revision=str(config.get("revision") or config.get("backbone_revision") or "") or None,
    )
