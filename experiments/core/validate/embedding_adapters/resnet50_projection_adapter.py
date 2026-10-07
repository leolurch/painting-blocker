"""Checkpoint-backed ResNet-50 projection adapter."""

from __future__ import annotations

from .checkpoint_utils import CheckpointProjectionAdapter


class ResNet50ProjectionAdapter(CheckpointProjectionAdapter):
    """Expose ResNet-50 avgpool/GeM checkpoints through BaseEmbeddingModel."""

    ALLOWED_MODEL_TYPES = frozenset(
        {
            "resnet50_projection",
            "resnet50_gem_projection",
            "resnet_gem_mlp",
            "resnet_gem_whitening",
        }
    )
