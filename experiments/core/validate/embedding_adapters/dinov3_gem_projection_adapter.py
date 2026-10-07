"""Checkpoint-backed DINOv3 patch-GeM projection adapter."""

from __future__ import annotations

from .checkpoint_utils import CheckpointProjectionAdapter


class DinoV3GemProjectionAdapter(CheckpointProjectionAdapter):
    """Expose DINOv3 patch-token aggregation checkpoints through BaseEmbeddingModel."""

    ALLOWED_MODEL_TYPES = frozenset({"dinov3_projection", "dinov3_gem_projection", "dinov3_lora_qv_projection"})
