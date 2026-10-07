"""Checkpoint save/load helpers for finetuned retrieval models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models.base import TrainableRetrievalModel
from .models.clip import build_clip_model
from .models.clip_lora import build_clip_lora_model
from .models.dinov3 import build_dinov3_lora_model, build_dinov3_model
from .models.resnet import build_resnet_model
from .models.resnet_lora import build_resnet_hf_lora_model
from .models.siglip2 import build_siglip2_lora_model


def build_trainable_model(model_config: dict[str, Any]) -> TrainableRetrievalModel:
    """Build a trainable retrieval model from the finetuning model config."""
    cfg = dict(model_config)
    model_type = str(cfg.get("type") or cfg.get("model_type") or "")
    if model_type in {
        "resnet50_projection",
        "resnet50_gem_projection",
        "resnet_gem_mlp",
    }:
        return build_resnet_model(cfg)
    if model_type in {"dinov3_projection", "dinov3_gem_projection"}:
        return build_dinov3_model(cfg)
    if model_type in {"dinov3_lora_qv_projection", "dinov3_lora_projection"}:
        return build_dinov3_lora_model(cfg)
    if model_type == "siglip2_lora_qv_projection":
        return build_siglip2_lora_model(cfg)
    if model_type in {"clip_projection", "clip_gem_projection"}:
        return build_clip_model(cfg)
    if model_type == "clip_lora_qv_projection":
        return build_clip_lora_model(cfg)
    if model_type == "resnet_hf_lora_conv_projection":
        return build_resnet_hf_lora_model(cfg)
    raise ValueError(f"Unsupported finetuning model.type: {model_type!r}")


def save_checkpoint(
    path: Path | str,
    model: TrainableRetrievalModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    training_config: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    normalization_config: dict[str, Any] | None = None,
    validation_threshold: float | None = None,
    validation_metrics: dict[str, Any] | None = None,
    reproducibility_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint matching the checkpoint-backed adapter contract."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = model.checkpoint_metadata()
    payload = {
        **metadata,
        "epoch": int(epoch),
        "model_state_dict": model.checkpoint_state_dict(),
        "state_dict_scope": "trainable" if model.checkpoint_trainable_only else "full",
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "preprocessing_config": dict(preprocessing_config or {}),
        "normalization_config": dict(normalization_config or {}),
        "training_config": dict(training_config or {}),
        "validation_threshold": validation_threshold,
        "validation_metrics": dict(validation_metrics or {}),
        "reproducibility": dict(reproducibility_metadata or {}),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch
        return torch.load(checkpoint_path, map_location=map_location)


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    training = dict(checkpoint.get("training_config") or {})
    model_cfg = dict(training.get("model") or training.get("finetuning_model") or {})
    if not model_cfg:
        model_cfg = {
            "type": checkpoint.get("model_type"),
            "backbone": checkpoint.get("backbone_name"),
            "revision": checkpoint.get("backbone_revision"),
            "pooling": {"type": checkpoint.get("pooling_type")},
            "projection_dim": checkpoint.get("projection_dim"),
        }
    model_cfg.setdefault("type", checkpoint.get("model_type"))
    model_cfg.setdefault("backbone", checkpoint.get("backbone_name"))
    model_cfg.setdefault("revision", checkpoint.get("backbone_revision"))
    if "pooling" not in model_cfg:
        model_cfg["pooling"] = {"type": checkpoint.get("pooling_type")}
    model_cfg.setdefault("projection_dim", checkpoint.get("projection_dim", 512))
    return model_cfg


def load_model_from_checkpoint(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[TrainableRetrievalModel, dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    model = build_trainable_model(model_config_from_checkpoint(checkpoint))
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain model_state_dict")
    if checkpoint.get("state_dict_scope") == "trainable":
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        unknown = sorted(set(state) - set(model.state_dict()))
        absent = sorted(trainable - set(state))
        if unknown or absent:
            raise ValueError(
                f"Trainable-only checkpoint does not match the rebuilt model: unknown={unknown[:10]} absent={absent[:10]}"
            )
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)
    model.eval()
    return model, checkpoint
