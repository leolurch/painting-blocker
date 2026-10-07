"""Checkpoint-backed DINOv3 CLS projection adapter."""

from __future__ import annotations

from .checkpoint_utils import CheckpointProjectionAdapter


class DinoV3ProjectionAdapter(CheckpointProjectionAdapter):
    """Expose DINOv3 CLS projection checkpoints through BaseEmbeddingModel."""

    ALLOWED_MODEL_TYPES = frozenset({"dinov3_projection", "dinov3_gem_projection", "dinov3_lora_qv_projection"})
