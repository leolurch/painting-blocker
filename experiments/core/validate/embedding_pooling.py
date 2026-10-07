"""Pooling helpers for image embedding adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import AbstractSet

import torch

TOKEN_POOLING_MODES = frozenset({"avg", "median", "min", "max"})
POOLING_ALIASES = {"average": "avg", "mean": "avg"}


def normalize_pooling_mode(mode: str) -> str:
    """Return the canonical lowercase pooling mode name."""
    value = str(mode).strip().lower()
    if not value:
        raise ValueError("Pooling mode must not be empty")
    return POOLING_ALIASES.get(value, value)


def normalize_pooling_modes(
    pooling: Sequence[str] | str, supported_poolings: AbstractSet[str]
) -> tuple[str, ...]:
    """Validate and canonicalize requested pooling modes."""
    values = [pooling] if isinstance(pooling, str) else list(pooling)
    modes = [normalize_pooling_mode(mode) for mode in values]
    if not modes:
        raise ValueError("At least one pooling mode must be supplied")
    unsupported = [mode for mode in modes if mode not in supported_poolings]
    if unsupported:
        raise ValueError(
            f"Unsupported pooling mode(s): {', '.join(unsupported)}. "
            f"Expected one of: {', '.join(sorted(supported_poolings))}"
        )
    return tuple(dict.fromkeys(modes))


def aggregate_tokens(
    tokens: torch.Tensor, mode: str, token_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Aggregate token embeddings into one embedding per image."""
    if tokens.ndim != 3:
        raise ValueError(
            "Expected tokens with shape [batch, tokens, dim], "
            f"got {tuple(tokens.shape)}"
        )
    if tokens.shape[1] <= 0:
        raise ValueError("Cannot aggregate an empty token sequence")
    normalized = normalize_pooling_mode(mode)
    if normalized not in TOKEN_POOLING_MODES:
        raise ValueError(f"Unsupported token pooling mode: {mode!r}")
    if token_mask is None:
        return _aggregate_unmasked_tokens(tokens, normalized)
    return _aggregate_masked_tokens(tokens, normalized, token_mask)


def _aggregate_unmasked_tokens(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "avg":
        return tokens.mean(dim=1)
    if mode == "median":
        return tokens.median(dim=1).values
    if mode == "min":
        return tokens.min(dim=1).values
    if mode == "max":
        return tokens.max(dim=1).values
    raise AssertionError(f"Unhandled pooling mode: {mode}")


def _aggregate_masked_tokens(
    tokens: torch.Tensor, mode: str, token_mask: torch.Tensor
) -> torch.Tensor:
    if token_mask.shape != tokens.shape[:2]:
        raise ValueError(
            "Expected token_mask with shape [batch, tokens], "
            f"got {tuple(token_mask.shape)} for tokens {tuple(tokens.shape)}"
        )
    mask = token_mask.to(device=tokens.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    if torch.any(counts <= 0):
        raise ValueError("Cannot aggregate token rows with no valid tokens")
    expanded = mask.unsqueeze(-1)
    if mode == "avg":
        return (tokens * expanded.to(dtype=tokens.dtype)).sum(dim=1) / counts.clamp_min(1).unsqueeze(-1)
    if mode == "min":
        fill = torch.finfo(tokens.dtype).max
        return tokens.masked_fill(~expanded, fill).min(dim=1).values
    if mode == "max":
        fill = torch.finfo(tokens.dtype).min
        return tokens.masked_fill(~expanded, fill).max(dim=1).values
    if mode == "median":
        return torch.stack(
            [row[row_mask].median(dim=0).values for row, row_mask in zip(tokens, mask)],
            dim=0,
        )
    raise AssertionError(f"Unhandled pooling mode: {mode}")
