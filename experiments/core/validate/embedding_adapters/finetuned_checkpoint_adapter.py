"""Generic adapter for checkpoints produced by the finetuning package."""

from __future__ import annotations

from .checkpoint_utils import CheckpointProjectionAdapter


class FinetunedCheckpointAdapter(CheckpointProjectionAdapter):
    """Reconstruct any supported finetuning model from its stored training config.

    ``load_model_from_checkpoint`` remains the source of truth for supported model
    types.  This adapter deliberately adds no second architecture registry.
    """

    ALLOWED_MODEL_TYPES = frozenset()
