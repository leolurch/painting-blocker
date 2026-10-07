"""Model-config driven adapter construction for blocking experiments."""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.core.validate.embedding_adapters.base import BaseEmbeddingModel
from experiments.core.validate.embedding_pooling import normalize_pooling_modes
from experiments.core.env import env_load

_ADAPTER_PACKAGE = "experiments.core.validate.embedding_adapters"
_ADAPTER_DIR = Path(__file__).resolve().parent / "validate" / "embedding_adapters"


@dataclass(frozen=True)
class AdapterConfig:
    """YAML-owned adapter module and constructor arguments."""

    adapter_name: str
    model_id: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None

    def identity(self) -> str:
        suffix = f"@{self.revision}" if self.revision else ""
        return f"{self.adapter_name}:{self.model_id}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "model_id": self.model_id,
            "revision": self.revision,
            "kwargs": dict(self.kwargs),
        }


def known_adapter_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.stem
            for path in _ADAPTER_DIR.glob("*_adapter.py")
            if path.stem not in {"base", "__init__"}
        )
    )


def resolve_hf_token(configured: str | None = None) -> str | None:
    if configured:
        return configured
    env_load()
    return os.getenv("HF_TOKEN")


def _validate_adapter_name(adapter_name: str) -> str:
    normalized = adapter_name.strip()
    if not normalized.isidentifier():
        raise ValueError(f"adapter_name must be a Python module filename stem: {adapter_name!r}")
    if normalized not in known_adapter_names():
        raise ValueError(
            f"Unknown adapter_name {adapter_name!r}. Expected one of: "
            f"{', '.join(known_adapter_names())}"
        )
    return normalized


def _load_adapter_class(adapter_name: str) -> type[BaseEmbeddingModel]:
    normalized = _validate_adapter_name(adapter_name)
    module = importlib.import_module(f"{_ADAPTER_PACKAGE}.{normalized}")
    candidates = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module.__name__
        and issubclass(value, BaseEmbeddingModel)
        and value is not BaseEmbeddingModel
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Adapter module {adapter_name!r} must define exactly one "
            f"BaseEmbeddingModel subclass; found {len(candidates)}"
        )
    return candidates[0]


def _constructor_accepts(adapter_cls: type, name: str) -> bool:
    params = inspect.signature(adapter_cls).parameters
    if name in params:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _validate_constructor_kwargs(adapter_cls: type, kwargs: dict[str, Any], adapter_name: str) -> None:
    params = inspect.signature(adapter_cls).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return
    unknown = sorted(key for key in kwargs if key not in params)
    if unknown:
        raise ValueError(
            f"Adapter {adapter_name!r} does not accept constructor arg(s): "
            f"{', '.join(unknown)}"
        )


def _adapter_kwargs(
    adapter_cls: type,
    adapter: AdapterConfig,
    geometry: object | None,
    embedding_options: dict[str, Any],
) -> dict[str, Any]:
    kwargs = dict(adapter.kwargs)
    if "model_id" in kwargs:
        raise ValueError("adapter kwargs must not contain model_id; use model_id")
    if "size_key" in kwargs:
        raise ValueError("adapter kwargs must not contain size_key; use model_id")
    if not _constructor_accepts(adapter_cls, "model_id"):
        raise ValueError(f"Adapter {adapter.adapter_name!r} must accept model_id")
    kwargs["model_id"] = adapter.model_id
    token = resolve_hf_token(embedding_options.get("hf_token"))
    if geometry is not None and "geometry" not in kwargs and _constructor_accepts(adapter_cls, "geometry"):
        kwargs["geometry"] = geometry
    if token and "hf_token" not in kwargs and _constructor_accepts(adapter_cls, "hf_token"):
        kwargs["hf_token"] = token
    if adapter.revision and "revision" not in kwargs and _constructor_accepts(adapter_cls, "revision"):
        kwargs["revision"] = adapter.revision
    if "use_compile" not in kwargs and _constructor_accepts(adapter_cls, "use_compile"):
        kwargs["use_compile"] = not bool(embedding_options.get("no_compile", False))
    _validate_constructor_kwargs(adapter_cls, kwargs, adapter.adapter_name)
    return kwargs


def _pooling_modes(adapter_cls: type, requested: str) -> tuple[str, ...]:
    supported = getattr(adapter_cls, "SUPPORTED_POOLINGS", frozenset({"default"}))
    return normalize_pooling_modes((requested,), supported)


def instantiate_adapter(
    adapter: AdapterConfig,
    geometry: object | None,
    pooling: str,
    embedding_options: dict[str, Any],
) -> tuple[object, tuple[str, ...], dict[str, Any]]:
    """Instantiate the configured adapter and return runtime metadata."""
    adapter_cls = _load_adapter_class(adapter.adapter_name)
    modes = _pooling_modes(adapter_cls, pooling)
    kwargs = _adapter_kwargs(adapter_cls, adapter, geometry, embedding_options)
    instance = adapter_cls(**kwargs)
    return instance, modes, {
        "adapter_name": adapter.adapter_name,
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "class": f"{adapter_cls.__module__}.{adapter_cls.__name__}",
        "kwargs": dict(adapter.kwargs),
    }
