"""Central environment loading helpers for blocking experiments."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
_IMAGE_FILE_ROLES = frozenset({"auction", "lost"})
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def env_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def env_load() -> bool:
    """Load the repository-root .env file once with python-dotenv."""
    env_path = env_repo_root() / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        if env_path.is_file():
            raise RuntimeError(
                "python-dotenv is required to load the repository .env file. "
                "Install the python-dotenv package."
            ) from exc
        return False
    return bool(load_dotenv(env_path, override=False))


def env_str(name: str, default: str | None = None) -> str | None:
    env_load()
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_bool(name: str, default: bool = False) -> bool:
    env_load()
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def env_int(name: str, default: int | None = None) -> int | None:
    value = env_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: Invalid integer for {name}={value!r}. Using default {default}.")
        return default


def env_float(name: str, default: float | None = None) -> float | None:
    value = env_str(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: Invalid float for {name}={value!r}. Using default {default}.")
        return default


def env_positive_int(name: str, default: int) -> int:
    value = env_str(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"Environment variable {name} must be a positive integer")
    return parsed


def env_path(name: str, default: Path = None) -> Path:
    value = env_str(name)
    if value is None:
        return default
    return Path(value).expanduser().resolve()


def env_cache_dir() -> Path:
    value = env_str("CACHE_DIR")
    if value is None:
        return env_repo_root() / "cache"
    return Path(value).expanduser().resolve()


def env_experiments_cache_dir() -> Path | None:
    """Return the shared cross-run experiment cache root, or None if unset.

    Reads ``EXPERIMENTS_CACHE_DIR``. This is the shared, content-addressed cache
    root that holds embeddings, reference evaluations, locks, and logs. It is
    kept separate from ``CACHE_DIR``/``env_cache_dir()`` (image-blocking cache).
    """
    value = env_str("EXPERIMENTS_CACHE_DIR")
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def env_image_blocking_dir() -> Path:
    return env_cache_dir() / "image_blocking"


def env_image_files_parquet_path(role: str) -> Path:
    if role not in _IMAGE_FILE_ROLES:
        raise ValueError(f"Invalid image-file artifact role: {role!r}")
    return env_image_blocking_dir() / role / "image_files.parquet"


def env_auction_to_lost_rankings_dir() -> Path:
    return env_image_blocking_dir() / "auction_to_lost_candidates"


def env_hf_token(cli_token: str | None = None) -> str | None:
    if cli_token:
        return cli_token
    value = env_str("HF_TOKEN")
    if value:
        return value
    return None


def env_dinov3_model_id() -> str:
    return env_str("DINOV3_MODEL_ID", DEFAULT_DINOV3_MODEL_ID) or DEFAULT_DINOV3_MODEL_ID


def env_non_gpu_inference_allowed() -> bool:
    return env_bool("ALLOW_NON_GPU_INFERENCE")
