"""Checkpoint-backed CLIP patch-GeM projection adapter."""

from __future__ import annotations

from .checkpoint_utils import CheckpointProjectionAdapter


class ClipGemProjectionAdapter(CheckpointProjectionAdapter):
    """Expose CLIP patch-token aggregation checkpoints through BaseEmbeddingModel."""

    ALLOWED_MODEL_TYPES = frozenset({"clip_projection", "clip_gem_projection"})
