"""Reproducibility and provenance helpers for blocking experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import random
import sys
from typing import Any, Iterable

import numpy as np

from experiments.core.env import env_load

from .artifacts import sha256_json

SECRET_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
BEHAVIOR_ENV_VARS = (
    "ALLOW_NON_GPU_INFERENCE",
    "DINOV3_MODEL_ID",
    "DINOV3_SIZE_KEY",
    "OPENAI_CLIP_MODEL_ID",
    "OPENAI_CLIP_SIZE_KEY",
    "OPEN_CLIP_MODEL_ID",
    "OPEN_CLIP_SIZE_KEY",
    "QWEN3_VL_MODEL_ID",
    "QWEN3_VL_SIZE_KEY",
    "SIGLIP2_MODEL_ID",
    "SIGLIP2_SIZE_KEY",
)
_ADAPTER_ENV_VARS = {
    "dino_adapter": ("ALLOW_NON_GPU_INFERENCE", "DINOV3_MODEL_ID", "DINOV3_SIZE_KEY"),
    "open_clip_adapter": ("OPEN_CLIP_MODEL_ID", "OPEN_CLIP_SIZE_KEY"),
    "openai_clip_adapter": ("OPENAI_CLIP_MODEL_ID", "OPENAI_CLIP_SIZE_KEY"),
    "qwen3_vl_adapter": ("QWEN3_VL_MODEL_ID", "QWEN3_VL_SIZE_KEY"),
    "siglip2_adapter": ("SIGLIP2_MODEL_ID", "SIGLIP2_SIZE_KEY"),
}
_VERSION_PACKAGES = (
    "numpy",
    "Pillow",
    "torch",
    "torchvision",
    "transformers",
    "qwen-vl-utils",
    "huggingface_hub",
    "open_clip_torch",
    "peft",
    "safetensors",
    "albumentations",
    "opencv-python-headless",
    "flash-attn",
)


def dependency_versions() -> dict[str, Any]:
    """Return lightweight dependency/version metadata for run manifests."""
    packages: dict[str, str] = {}
    for name in _VERSION_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def environment_provenance(models: Iterable[Any] = ()) -> dict[str, Any]:
    """Capture env-derived behavior without serializing secret values."""
    env_load()
    variables = {name: os.getenv(name) for name in BEHAVIOR_ENV_VARS if os.getenv(name) is not None}
    checkpoint_paths: dict[str, str | None] = {}
    for model in models:
        adapter = getattr(model, "adapter", None)
        env_name = (getattr(adapter, "kwargs", {}) or {}).get("checkpoint_path_env") if adapter is not None else None
        if env_name:
            checkpoint_paths[str(env_name)] = os.getenv(str(env_name))
    return {
        "variables": variables,
        "secret_presence": {name: os.getenv(name) is not None for name in SECRET_ENV_VARS},
        "checkpoint_path_env": checkpoint_paths,
    }


def model_environment_cache_inputs(model: Any) -> dict[str, Any]:
    """Return model-specific env inputs that can affect generated embeddings."""
    env_load()
    adapter = getattr(model, "adapter", None)
    adapter_name = getattr(adapter, "adapter_name", "")
    names = set(_ADAPTER_ENV_VARS.get(adapter_name, ()))
    checkpoint_env = (getattr(adapter, "kwargs", {}) or {}).get("checkpoint_path_env") if adapter is not None else None
    if checkpoint_env:
        names.add(str(checkpoint_env))
    return {
        "adapter_name": adapter_name,
        "variables": {name: os.getenv(name) for name in sorted(names) if os.getenv(name) is not None},
        "secret_presence": {name: os.getenv(name) is not None for name in SECRET_ENV_VARS},
    }


def rng_state_fingerprints(torch_module: Any | None = None) -> dict[str, Any]:
    """Hash current RNG states without storing bulky raw state blobs."""
    state = {
        "python_random_state_sha256": sha256_json(repr(random.getstate())),
        "numpy_random_state_sha256": _hash_numpy_state(np.random.get_state()),
    }
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            torch_module = None
    if torch_module is not None:
        state["torch_cpu_rng_state_sha256"] = _hash_bytes(torch_module.get_rng_state().cpu().numpy().tobytes())
        if torch_module.cuda.is_available():
            state["torch_cuda_rng_state_sha256"] = [
                _hash_bytes(item.cpu().numpy().tobytes()) for item in torch_module.cuda.get_rng_state_all()
            ]
    return state


def _hash_numpy_state(state: tuple[Any, ...]) -> str:
    name, keys, pos, has_gauss, cached_gaussian = state
    payload = {
        "name": name,
        "keys_sha256": _hash_bytes(np.asarray(keys).tobytes()),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }
    return sha256_json(payload)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
