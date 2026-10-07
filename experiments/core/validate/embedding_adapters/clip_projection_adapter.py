"""Checkpoint-backed CLIP CLS projection adapter."""

from __future__ import annotations

from .checkpoint_utils import CheckpointProjectionAdapter


class ClipProjectionAdapter(CheckpointProjectionAdapter):
    """Expose CLIP projection checkpoints through BaseEmbeddingModel."""

    ALLOWED_MODEL_TYPES = frozenset({"clip_projection", "clip_gem_projection"})
