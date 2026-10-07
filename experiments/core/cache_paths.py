"""Centralized resolution of the shared cross-run experiment cache layout.

The shared cache is a single content-addressed directory tree used by both
ordinary evaluation runs and the frozen-reference prewarming path::

    <root>/
      embeddings/
      reference-evaluations/
      locks/
      logs/

Resolution of ``<root>``:

1. ``EXPERIMENTS_CACHE_DIR`` when set (required for Slurm prewarming, which must
   point at a shared filesystem), otherwise
2. a documented local default ``<repo>/cache/experiment-cache`` for non-Slurm
   execution.

Note: an experiment's explicit ``embedding.cache_dir`` still takes precedence
for the embeddings location where applicable; that is handled at the call site
in ``embedding_pipeline``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .env import env_experiments_cache_dir, env_repo_root

LOCAL_DEFAULT_SUBDIR = "cache/experiment-cache"


def experiments_cache_root() -> Path:
    """Return the shared cache root, honoring ``EXPERIMENTS_CACHE_DIR``."""
    configured = env_experiments_cache_dir()
    if configured is not None:
        return configured
    return env_repo_root() / LOCAL_DEFAULT_SUBDIR


def embeddings_dir() -> Path:
    return experiments_cache_root() / "embeddings"


def reference_evaluations_dir() -> Path:
    return experiments_cache_root() / "reference-evaluations"


def locks_dir() -> Path:
    return experiments_cache_root() / "locks"


def logs_dir() -> Path:
    return experiments_cache_root() / "logs"


def ensure_cache_layout() -> Path:
    """Create the shared cache subdirectories and return the root."""
    root = experiments_cache_root()
    for sub in (embeddings_dir(), reference_evaluations_dir(), locks_dir(), logs_dir()):
        sub.mkdir(parents=True, exist_ok=True)
    return root


def require_shared_cache_writable(*, require_env: bool = True) -> Path:
    """Validate the shared cache is configured and writable; return the root.

    Used by the Slurm prewarm path, which must fail early when
    ``EXPERIMENTS_CACHE_DIR`` is unset or the target is not writable.
    """
    if require_env and env_experiments_cache_dir() is None:
        raise RuntimeError(
            "EXPERIMENTS_CACHE_DIR is not set. Prewarming requires it to point at a "
            "shared filesystem so every sweep task reads the same reference cache."
        )
    root = ensure_cache_layout()
    probe = root / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"Shared experiment cache is not writable: {root} ({exc})") from exc
    return root


def is_shared_cache_configured() -> bool:
    return env_experiments_cache_dir() is not None or "EXPERIMENTS_CACHE_DIR" in os.environ
