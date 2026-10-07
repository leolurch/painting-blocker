"""Base contract for trainable image-retrieval models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class TrainableRetrievalModel(nn.Module, ABC):
    """Model that maps preprocessed images to normalized retrieval embeddings."""

    model_type: str
    backbone_name: str
    pooling_type: str
    projection_dim: int
    # Large frozen backbones store only trainable tensors; the rest is rebuilt
    # from the pretrained weights at the pinned revision.
    checkpoint_trainable_only: bool = False

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return an L2-normalized retrieval embedding with shape ``[B, D]``."""

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        if not self.checkpoint_trainable_only:
            return state
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value for name, value in state.items() if name in trainable}

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return model-identifying metadata stored beside the state dict."""
        return {
            "model_type": getattr(self, "model_type", self.__class__.__name__),
            "backbone_name": getattr(self, "backbone_name", ""),
            "backbone_revision": getattr(self, "backbone_revision", None),
            "pooling_type": getattr(self, "pooling_type", ""),
            "projection_dim": int(getattr(self, "projection_dim", 0)),
        }


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    """Toggle gradients for every parameter in ``module``."""
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def trainable_parameter_groups(
    model: nn.Module,
    head_keywords: tuple[str, ...] = ("projection", "pool", "head."),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split model parameters into backbone and head groups by name.

    ``model.backbone_trainable_prefixes`` keeps pretrained modules whose names
    contain a head keyword (for example SigLIP's ``vision_model.head``) in the
    backbone group.
    """
    backbone_prefixes = tuple(getattr(model, "backbone_trainable_prefixes", ()))
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(prefix in name for prefix in backbone_prefixes):
            backbone.append(parameter)
            continue
        target = head if any(keyword in name for keyword in head_keywords) else backbone
        target.append(parameter)
    return backbone, head
