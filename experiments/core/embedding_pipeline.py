"""Embedding generation and strongly-keyed cache management for experiments."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.core.adapter_registry import instantiate_adapter
from experiments.core.descriptor_postprocessing import (
    DESCRIPTOR_DTYPE,
    DESCRIPTOR_POSTPROCESSOR_VERSION,
    postprocess_descriptors,
    validate_descriptors,
)
from experiments.core.env import env_load
from experiments.core.validate.embedding_geometry import (
    RESIZE,
    RESIZE_AND_PAD,
    RESIZE_LONGEST_SIDE,
    ImageGeometry,
)

from . import EXPERIMENTS_VERSION
from .artifacts import sha256_file, sha256_json, short_hash
from .cache_lock import DEFAULT_TIMEOUT_SECONDS, key_lock
from .cache_paths import embeddings_dir
from .config_schema import DatasetConfig, ModelConfig
from .resource_metrics import ResourceMonitor, model_runtime_profile, release_torch_cuda_memory
from .reproducibility import model_environment_cache_inputs
from .split_schema import SplitImage


# Bump this whenever embedding-generation semantics change, including adapter
# preprocessing, pooling, normalization, or checkpoint-loading behavior.
EMBEDDING_EVALUATOR_VERSION = 4
DESCRIPTOR_POSTPROCESS_DTYPE = DESCRIPTOR_DTYPE
DESCRIPTOR_CACHE_POLICY = {
    "descriptor_postprocess_dtype": "float32",
    "normalized": True,
    "normalization": "l2",
    "normalization_axis": 1,
    "normalization_dtype": "float32",
    "descriptor_storage_dtype": "float32",
    "postprocessor_version": DESCRIPTOR_POSTPROCESSOR_VERSION,
}


def image_ids_hash(image_ids: list[str]) -> str:
    return sha256_json(image_ids)


def model_config_hash(model: ModelConfig) -> str:
    return sha256_json(
        {
            "model_id": model.model_id,
            "revision": model.revision,
            "adapter": model.adapter.to_dict(),
            "raw": model.identity_raw,
        }
    )


def _checkpoint_path_cache_inputs(model: ModelConfig) -> dict[str, Any] | None:
    """Fingerprint a local checkpoint file that can affect model embeddings."""
    env_load()
    kwargs = dict(model.adapter.kwargs)
    configured_path = kwargs.get("checkpoint_path")
    checkpoint_path_env = kwargs.get("checkpoint_path_env")
    if configured_path and checkpoint_path_env:
        raise ValueError("Configure only one of checkpoint_path or checkpoint_path_env")

    source: dict[str, Any]
    if checkpoint_path_env:
        env_name = str(checkpoint_path_env).strip()
        configured_path = os.getenv(env_name)
        source = {"kind": "environment", "name": env_name}
        if not configured_path:
            return {"source": source, "status": "unset"}
    elif configured_path:
        source = {"kind": "adapter_kwargs"}
    else:
        # CheckpointProjectionAdapter also permits a checkpoint file as model_id.
        configured_path = model.model_id
        source = {"kind": "model_id"}

    expanded = os.path.expandvars(str(configured_path))
    if "$" in expanded:
        return {"source": source, "configured_path": str(configured_path), "status": "unresolved"}
    path = Path(expanded).expanduser().resolve()
    if not path.is_file():
        # A normal Hub model ID is not a local checkpoint input. Explicitly
        # configured missing paths remain in the key and fail during loading.
        if source["kind"] == "model_id":
            return None
        return {"source": source, "path": str(path), "status": "missing"}
    return {
        "source": source,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": "available",
    }


def embedding_cache_inputs(
    dataset: DatasetConfig, db_sha256: str, image_ids: list[str], model: ModelConfig
) -> dict[str, Any]:
    """Return the canonical, auditable inputs used by the embedding cache key."""
    _validate_model_descriptor_policy(model)
    return {
        "schema_version": 1,
        "embedding_evaluator_version": EMBEDDING_EVALUATOR_VERSION,
        "dataset_id": dataset.dataset_id,
        "dataset_db_sha256": db_sha256,
        "image_ids_hash": image_ids_hash(image_ids),
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_config_hash": model_config_hash(model),
        "descriptor_postprocessing": dict(DESCRIPTOR_CACHE_POLICY),
        "local_checkpoint": _checkpoint_path_cache_inputs(model),
        "environment": model_environment_cache_inputs(model),
        "code_version": EXPERIMENTS_VERSION,
    }


def embedding_cache_key(
    dataset: DatasetConfig, db_sha256: str, image_ids: list[str], model: ModelConfig
) -> str:
    return sha256_json(embedding_cache_inputs(dataset, db_sha256, image_ids, model))


def instantiate_model_adapter(
    model: ModelConfig,
    embedding_options: dict[str, Any],
) -> tuple[object, tuple[str, ...], dict[str, Any]]:
    """Instantiate the configured adapter without running image inference."""
    return instantiate_adapter(
        model.adapter,
        geometry=_geometry(model),
        pooling=_pooling(model),
        embedding_options=embedding_options,
    )


@dataclass(frozen=True)
class EmbeddingArtifact:
    """A shared, content-addressed embedding artifact in the cross-run cache."""

    key: str
    artifact_dir: Path
    embeddings_path: Path
    metadata_path: Path
    reused: bool
    metadata: dict[str, Any]


def _embedding_cache_root(embedding_options: dict[str, Any], cache_dir: str | Path | None) -> Path:
    resolved = cache_dir if cache_dir is not None else embedding_options.get("cache_dir")
    if resolved:
        return Path(resolved).expanduser().resolve()
    return embeddings_dir()


def get_or_create_embedding_artifact(
    dataset: DatasetConfig,
    model: ModelConfig,
    image_ids: list[str],
    records: dict[str, SplitImage],
    run_options: dict[str, Any],
    embedding_options: dict[str, Any],
    db_sha256: str,
    *,
    cache_dir: str | Path | None = None,
    cache_policy: str | None = None,
    lock_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> EmbeddingArtifact:
    """Return the shared embedding artifact for these inputs, generating on miss.

    Uses the hardened embedding cache key, acquires a per-key lock, re-checks the
    cache under the lock, generates only when missing (writing through unique
    temporary files that are atomically published), validates metadata + image
    IDs, and returns the shared artifact **without copying it** into any run dir.
    """
    cache_inputs = embedding_cache_inputs(dataset, db_sha256, image_ids, model)
    key = sha256_json(cache_inputs)
    cache_root = _embedding_cache_root(embedding_options, cache_dir)
    artifact_dir = cache_root / dataset.dataset_id / model.storage_key / short_hash(key)
    metadata_path = artifact_dir / "metadata.json"
    embeddings_path = artifact_dir / "embeddings.npy"
    policy = str(cache_policy or embedding_options.get("cache_policy") or "refresh")
    if policy not in {"reuse_if_config_hash_matches", "overwrite", "refresh"}:
        raise ValueError(f"Unsupported embedding cache_policy: {policy}")

    def _reusable() -> bool:
        return policy == "reuse_if_config_hash_matches" and _cache_is_valid(
            metadata_path, embeddings_path, key, image_ids
        )

    if _reusable():
        reused = True
    else:
        with key_lock(key, timeout=lock_timeout):
            # Re-check after acquiring the lock: a concurrent worker may have
            # published a valid artifact while we waited.
            if _reusable():
                reused = True
            else:
                _generate_embeddings(
                    artifact_dir,
                    dataset,
                    model,
                    image_ids,
                    records,
                    run_options,
                    embedding_options,
                    key,
                    db_sha256,
                    cache_inputs,
                )
                reused = False
    if not _cache_is_valid(metadata_path, embeddings_path, key, image_ids):
        raise RuntimeError(f"Embedding artifact invalid after preparation: {artifact_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return EmbeddingArtifact(
        key=key,
        artifact_dir=artifact_dir,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        reused=reused,
        metadata=metadata,
    )


def copy_or_link_artifact_into_run(artifact: EmbeddingArtifact, model_dir: Path) -> Path:
    """Materialize a shared embedding artifact inside a run directory.

    The large ``embeddings.npy`` is symlinked (falling back to a copy across
    filesystems). The metadata is always **copied** (never symlinked) because the
    per-run flow annotates it in place and must not mutate the shared cache file.
    Returns the run-dir metadata path.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    run_embeddings = model_dir / "embeddings.npy"
    if run_embeddings.exists() or run_embeddings.is_symlink():
        run_embeddings.unlink()
    try:
        run_embeddings.symlink_to(artifact.embeddings_path)
    except OSError:
        shutil.copy2(artifact.embeddings_path, run_embeddings)
    run_metadata = model_dir / "embeddings_metadata.json"
    shutil.copy2(artifact.metadata_path, run_metadata)
    return run_metadata


def prepare_embeddings(
    dataset: DatasetConfig,
    model: ModelConfig,
    image_ids: list[str],
    records: dict[str, SplitImage],
    model_dir: Path,
    run_options: dict[str, Any],
    embedding_options: dict[str, Any],
    db_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = get_or_create_embedding_artifact(
        dataset,
        model,
        image_ids,
        records,
        run_options,
        embedding_options,
        db_sha256,
    )
    run_metadata_path = copy_or_link_artifact_into_run(artifact, model_dir)
    embeddings = np.load(model_dir / "embeddings.npy")
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    metadata["cache_reused_for_run"] = artifact.reused
    metadata["cache_artifact_dir"] = str(artifact.artifact_dir)
    run_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return embeddings, metadata


def _cache_is_valid(metadata_path: Path, embeddings_path: Path, key: str, image_ids: list[str]) -> bool:
    if not metadata_path.is_file() or not embeddings_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") != key or metadata.get("image_ids") != image_ids:
            return False
        if metadata.get("descriptor_storage_dtype") != "float32":
            return False
        if metadata.get("embedding_sha256") != sha256_file(embeddings_path):
            return False
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if list(embeddings.shape) != metadata.get("embedding_shape"):
            return False
        if embeddings.shape[0] != len(image_ids):
            return False
        validate_descriptors(
            embeddings,
            name="cached descriptor",
            require_unit_norm=True,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _geometry(model: ModelConfig) -> ImageGeometry | None:
    resize = ((model.raw.get("preprocessing") or {}).get("resize") or {})
    mode = str(resize.get("mode", "model_default"))
    if mode == "model_default":
        return None
    max_width = resize.get("max_width", resize.get("size"))
    width = int(max_width) if max_width is not None else None
    if mode == "letterbox":
        return ImageGeometry(mode=RESIZE_AND_PAD, max_width=width)
    if mode in {"resize", "square"}:
        return ImageGeometry(mode=RESIZE, max_width=width)
    if mode == "max_side":
        return ImageGeometry(mode=RESIZE_LONGEST_SIDE, max_width=width)
    raise ValueError(f"Unsupported resize mode for {model.model_id}: {mode}")


def _validate_model_descriptor_policy(model: ModelConfig) -> None:
    embedding = dict(model.raw.get("embedding") or {})
    configured_dtype = str(embedding.get("dtype") or "float32")
    if configured_dtype != "float32":
        raise ValueError(
            f"Model {model.model_id} configures embedding.dtype={configured_dtype!r}, "
            "but descriptor post-processing and storage are fixed to 'float32'"
        )
    if embedding.get("normalize") is False:
        raise ValueError(
            f"Model {model.model_id} disables normalization, but the canonical "
            "evaluation descriptor contract requires normalized descriptors"
        )


def _pooling(model: ModelConfig) -> str:
    value = (model.raw.get("embedding") or {}).get("pooling", "default")
    if value in (None, "last_hidden_or_configured"):
        return "default"
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Model {model.model_id} must configure exactly one pooling mode")
        return str(value[0])
    return str(value)


def _validate_output_dim(model: ModelConfig, embeddings: np.ndarray) -> None:
    expected = (model.raw.get("embedding") or {}).get("output_dim")
    if expected is None:
        return
    actual = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    if int(expected) != actual:
        raise ValueError(
            f"Model {model.model_id} embedding.output_dim={expected} does not "
            f"match generated dimension {actual}"
        )


def _generate_embeddings(
    artifact_dir: Path,
    dataset: DatasetConfig,
    model: ModelConfig,
    image_ids: list[str],
    records: dict[str, SplitImage],
    run_options: dict[str, Any],
    embedding_options: dict[str, Any],
    cache_key: str,
    db_sha256: str,
    cache_inputs: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    adapter = None
    load_started = time.perf_counter()
    try:
        adapter, pooling_modes, adapter_meta = instantiate_model_adapter(
            model,
            embedding_options,
        )
        model_profile = model_runtime_profile(adapter)
        model_load_seconds = time.perf_counter() - load_started
        with ResourceMonitor() as monitor:
            embeddings = _run_batches(
                adapter, image_ids, records, dataset, model, pooling_modes, run_options, embedding_options
            )
        model_profile["descriptor_tensor_dtypes"] = list(
            getattr(adapter, "descriptor_tensor_dtypes", [])
        )
        model_profile["intrinsic_normalization"] = bool(
            getattr(adapter, "intrinsic_normalization", False)
        )
        model_profile["intrinsic_normalization_dtype"] = getattr(
            adapter, "intrinsic_normalization_dtype", None
        )
        embedding_resources = monitor.summary()
    finally:
        if adapter is not None:
            _close_adapter(adapter)
            del adapter
        release_torch_cuda_memory(reset_compiler=True)
    _validate_output_dim(model, embeddings)
    adapter_output_dtype = str(np.asarray(embeddings).dtype)
    embeddings = postprocess_descriptors(embeddings)
    if embeddings.ndim == 2:
        model_profile["embedding_dim"] = int(embeddings.shape[1])
    emb_path = artifact_dir / "embeddings.npy"
    _atomic_save_npy(emb_path, embeddings)
    embedding_seconds = float(embedding_resources["wall_time_seconds"])
    metadata = {
        "schema_version": 1,
        "dataset_id": dataset.dataset_id,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_storage_key": model.storage_key,
        "adapter": adapter_meta,
        "num_images": len(image_ids),
        "image_ids": image_ids,
        "embedding_shape": list(embeddings.shape),
        "normalized": True,
        "normalization": "l2",
        "normalization_axis": 1,
        "normalization_dtype": "float32",
        "descriptor_postprocess_dtype": "float32",
        "descriptor_storage_dtype": str(embeddings.dtype),
        "postprocessor_version": DESCRIPTOR_POSTPROCESSOR_VERSION,
        "precision": {
            "parameter_dtypes": model_profile.get("parameter_dtypes", []),
            "autocast_enabled": model_profile.get("autocast_enabled", False),
            "autocast_dtype": model_profile.get("autocast_dtype"),
            "descriptor_tensor_dtypes": model_profile.get("descriptor_tensor_dtypes", []),
            "adapter_output_dtype": adapter_output_dtype,
            "intrinsic_normalization": model_profile.get("intrinsic_normalization", False),
            "intrinsic_normalization_dtype": model_profile.get("intrinsic_normalization_dtype"),
            "descriptor_postprocess_dtype": "float32",
            "normalization_dtype": "float32",
            "stored_descriptor_dtype": str(embeddings.dtype),
        },
        "cache_key": cache_key,
        "cache_inputs": cache_inputs,
        "embedding_evaluator_version": EMBEDDING_EVALUATOR_VERSION,
        "config_sha256": model_config_hash(model),
        "dataset_db_sha256": db_sha256,
        "image_ids_hash": image_ids_hash(image_ids),
        "embedding_sha256": sha256_file(emb_path),
        "resources": {
            "model": model_profile,
            "model_load_seconds": float(model_load_seconds),
            "embedding": {
                **embedding_resources,
                "num_images": len(image_ids),
                "time_per_100_images_seconds": embedding_seconds / max(len(image_ids), 1) * 100.0,
            },
        },
    }
    # Publish metadata last and atomically: a reader treats a valid metadata
    # cache_key as proof the embeddings.npy (written above) is complete.
    meta_path = artifact_dir / "metadata.json"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + f".{os.getpid()}.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(meta_tmp, meta_path)


def _close_adapter(adapter: object) -> None:
    close = getattr(adapter, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            print(f"Warning: failed to close {adapter.__class__.__name__}: {exc}")


def _run_batches(adapter: object, image_ids: list[str], records: dict[str, SplitImage], dataset: DatasetConfig, model: ModelConfig, pooling_modes: tuple[str, ...], run_options: dict[str, Any], embedding_options: dict[str, Any]) -> np.ndarray:
    batch_size = int((model.raw.get("embedding") or {}).get("batch_size") or embedding_options.get("batch_size") or 32)
    arrays: list[np.ndarray] = []
    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start : start + batch_size]
        images = [_load_image(records[file_id], dataset, bool(run_options.get("fail_on_missing_images", True))) for file_id in batch_ids]
        batch = adapter.generate_embeddings_batch_from_pil(images, pooling_modes)
        missing = [mode for mode in pooling_modes if mode not in batch]
        if missing:
            raise ValueError(f"Model {model.model_id} did not return pooling output(s): {', '.join(missing)}")
        values = np.asarray(batch[pooling_modes[0]])
        if values.shape[0] != len(batch_ids):
            raise ValueError(
                f"Model {model.model_id} returned {values.shape[0]} embeddings "
                f"for a batch of {len(batch_ids)} images"
            )
        arrays.extend(np.asarray(value) for value in values)
    if not arrays:
        raise RuntimeError(f"No embeddings generated for {model.model_id}")
    return np.stack(arrays, axis=0)


def _load_image(record: SplitImage, dataset: DatasetConfig, fail_on_missing: bool) -> Image.Image:
    path = record.absolute_path(dataset.image_root)
    if not path.is_file():
        if fail_on_missing:
            raise FileNotFoundError(f"Image for file_id={record.file_id} not found: {path}")
        return Image.new("RGB", (1, 1), (0, 0, 0))
    return Image.open(path).convert("RGB")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


