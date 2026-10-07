"""Stage 2 — calibrated DINOv3 embeddings with resumable batch parts.

Heavy imports are deliberately lazy: ingestion, review, clustering, and export remain usable on
machines without the GPU environment. The image geometry and pooling contract mirrors the
calibration adapter, while failing rather than using an uncalibrated preprocessing fallback.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from . import file_io, manifest
from .config import Config, ModelConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
PAD_COLOR = tuple(round(value * 255) for value in IMAGENET_MEAN)

try:
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - old Pillow
    _BICUBIC = Image.BICUBIC


class BatchEmbedder(Protocol):
    output_dim: int

    def embed_paths(self, paths: list[Path]) -> np.ndarray: ...

    def close(self) -> None: ...


def letterbox_image(image: Image.Image, size: int, pad_color: tuple[int, int, int] = PAD_COLOR) -> Image.Image:
    """Convert to RGB and resize/pad to an exact square using bicubic interpolation."""
    if size <= 0:
        raise ValueError("letterbox size must be positive")
    prepared = image if image.mode == "RGB" else image.convert("RGB")
    # Mirror ImageGeometry._apply_max_width before RESIZE_AND_PAD. Keeping this seemingly
    # redundant first resize preserves the adapter's exact rounding on tall images.
    if prepared.width > size:
        first_height = max(1, int(round(prepared.height * (size / float(prepared.width)))))
        prepared = prepared.resize((size, first_height), _BICUBIC)
    scale = min(size / float(prepared.width), size / float(prepared.height))
    width = max(1, int(round(prepared.width * scale)))
    height = max(1, int(round(prepared.height * scale)))
    resized = prepared.resize((width, height), _BICUBIC)
    if resized.size == (size, size):
        return resized
    canvas = Image.new("RGB", (size, size), pad_color)
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding array, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Embedding output contains non-finite or zero vectors")
    return values / norms


class DinoEmbedder:
    """Small, exact adapter for the one calibrated DINOv3 configuration."""

    def __init__(self, config: ModelConfig):
        if config.resize_mode != "letterbox" or config.pooling not in {"default", "cls"}:
            raise ValueError("Calibrated DINO adapter requires letterbox geometry and CLS pooling")
        if not config.normalize:
            raise ValueError("Calibrated DINO adapter requires L2 normalization")
        if not config.revision:
            raise ValueError("A pinned DINO revision is required")

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            forward_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif os.environ.get("DEDUP_ALLOW_CPU") == "1":
            self.device = torch.device("cpu")
            forward_dtype = None
        else:
            raise RuntimeError(
                "CUDA GPU not detected. Run DINOv3 embedding on the GPU host, or set "
                "DEDUP_ALLOW_CPU=1 only for a small test model."
            )

        common = {"revision": config.revision, "trust_remote_code": True}
        self.processor = AutoImageProcessor.from_pretrained(
            config.model_id, use_fast=True, **common
        )
        model_kwargs = dict(common)
        if forward_dtype is not None:
            model_kwargs["dtype"] = forward_dtype
        self.model = AutoModel.from_pretrained(config.model_id, **model_kwargs).to(self.device)
        self.model.eval()
        self.size = config.resize_size
        self._preprocess_pool = ThreadPoolExecutor(
            max_workers=config.preprocess_workers, thread_name_prefix="dino-preprocess"
        )
        self.output_dim = int(getattr(self.model.config, "hidden_size"))
        if config.output_dim is not None and self.output_dim != config.output_dim:
            raise ValueError(
                f"Configured DINO output_dim={config.output_dim}, model reports {self.output_dim}"
            )

    def _prepare_path(self, path: Path) -> Image.Image:
        with Image.open(path) as source:
            return letterbox_image(source, self.size)

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        # Decode/resize concurrently. map preserves input order and bounds resident decoded images
        # to one inference batch rather than the complete corpus.
        images = list(self._preprocess_pool.map(self._prepare_path, paths))
        batch = self.processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {}
        for key, value in batch.items():
            if not hasattr(value, "to"):
                inputs[key] = value
            elif self.device.type == "cuda" and hasattr(value, "pin_memory"):
                inputs[key] = value.pin_memory().to(self.device, non_blocking=True)
            else:
                inputs[key] = value.to(self.device)
        with self._torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
            pooled = hidden[:, 0, :]
            pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
        return pooled.float().cpu().numpy()

    def close(self) -> None:
        self._preprocess_pool.shutdown(wait=True, cancel_futures=True)
        del self.model
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _part_key(config: Config, entries: list[manifest.ImageEntry]) -> str:
    return file_io.sha256_json(
        {
            "model_fingerprint": config.model.fingerprint(),
            "images": [[entry.image_id, entry.sha256] for entry in entries],
        }
    )


def _valid_part(array_path: Path, metadata_path: Path, key: str, expected_rows: int) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = file_io.read_json(metadata_path)
        if metadata.get("part_key") != key or metadata.get("array_sha256") != file_io.sha256_file(array_path):
            return False
        values = np.load(array_path, allow_pickle=False, mmap_mode="r")
        return values.ndim == 2 and values.shape[0] == expected_rows and np.isfinite(values).all()
    except (OSError, ValueError, KeyError):
        return False


def _reuse_embedding_index(
    config: Config, reuse_work_dirs: list[Path]
) -> tuple[dict[str, tuple[np.ndarray, int]], list[dict[str, object]]]:
    index: dict[str, tuple[np.ndarray, int]] = {}
    sources: list[dict[str, object]] = []
    dropped: set[str] = set()
    for raw_work in reuse_work_dirs:
        work = Path(raw_work).expanduser().resolve()
        metadata_path = work / "embeddings" / "metadata.json"
        values_path = work / "embeddings" / "embeddings.npy"
        ids_path = work / "embeddings" / "image_ids.json"
        images_path = work / "images.csv"
        if not all(path.is_file() for path in (metadata_path, values_path, ids_path, images_path)):
            raise FileNotFoundError(f"Reusable DINO work directory is incomplete: {work}")
        metadata = file_io.read_json(metadata_path)
        if metadata.get("model_fingerprint_sha256") != config.model.fingerprint_sha256():
            raise ValueError(f"Reusable DINO model fingerprint mismatch: {work}")
        if metadata.get("embeddings_sha256") != file_io.sha256_file(values_path):
            raise ValueError(f"Reusable DINO embedding hash mismatch: {work}")
        values = np.load(values_path, allow_pickle=False, mmap_mode="r")
        image_ids = file_io.read_json(ids_path)
        old_images = manifest.load_images(images_path)
        if len(image_ids) != len(values) or any(image_id not in old_images for image_id in image_ids):
            raise ValueError(f"Reusable DINO image manifest mismatch: {work}")
        for row_index, image_id in enumerate(image_ids):
            sha = old_images[image_id].sha256
            if sha in dropped:
                continue
            previous = index.get(sha)
            if previous is not None and not np.allclose(previous[0][previous[1]], values[row_index], atol=2e-5):
                dropped.add(sha)
                del index[sha]
                continue
            index.setdefault(sha, (values, row_index))
        sources.append({"work_dir": str(work), "image_count": len(image_ids)})
    if dropped:
        print(
            f"Dropped {len(dropped)} reusable DINO images whose caches disagree: "
            + ", ".join(sorted(dropped)),
            flush=True,
        )
    return index, sources


def embed(
    config: Config,
    *,
    embedder_factory: Callable[[ModelConfig], BatchEmbedder] = DinoEmbedder,
    reuse_work_dirs: list[Path] | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    merge_only: bool = False,
) -> dict[str, object]:
    """Embed every current valid image and atomically merge resumable batch parts.

    ``num_shards > 1`` computes only this shard's batches and leaves the merged
    matrix to a later ``merge_only`` call. Part filenames stay global, so the
    shards do not overlap.
    """
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("embed shard index must lie in [0, num_shards)")
    if merge_only and num_shards != 1:
        raise ValueError("embed --merge-only reads every part and does not take a shard")
    work = file_io.work_dir()
    images = manifest.load_images(work / "images.csv")
    entries = [images[image_id] for image_id in sorted(images)]
    out_dir = work / "embeddings"
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    batch_size = config.model.batch_size
    specs: list[tuple[Path, Path, str, list[manifest.ImageEntry]]] = []
    owned: list[tuple[Path, Path, str, list[manifest.ImageEntry]]] = []
    for batch_index, start in enumerate(range(0, len(entries), batch_size)):
        batch_entries = entries[start : start + batch_size]
        stem = f"part_{start:08d}_{start + len(batch_entries):08d}"
        spec = (parts_dir / f"{stem}.npy", parts_dir / f"{stem}.json", _part_key(config, batch_entries), batch_entries)
        specs.append(spec)
        if num_shards == 1 or batch_index % num_shards == shard_index:
            owned.append(spec)

    if merge_only:
        pending = []
        reuse_index: dict = {}
        reuse_sources: list = []
        reused_count = 0
        computed_count = 0
    else:
        pending = [spec for spec in owned if not _valid_part(spec[0], spec[1], spec[2], len(spec[3]))]
        reuse_index, reuse_sources = _reuse_embedding_index(config, reuse_work_dirs or [])
        reused_count = 0
        computed_count = 0
    reused_count = 0
    computed_count = 0
    runner: BatchEmbedder | None = None
    try:
        for array_path, metadata_path, key, batch_entries in pending:
            missing = [entry for entry in batch_entries if entry.sha256 not in reuse_index]
            computed_by_id: dict[str, np.ndarray] = {}
            if missing:
                if runner is None:
                    runner = embedder_factory(config.model)
                computed = normalize_rows(runner.embed_paths([entry.absolute_path() for entry in missing]))
                if computed.shape[0] != len(missing):
                    raise ValueError("DINO adapter returned the wrong batch length")
                computed_by_id = {entry.image_id: computed[i] for i, entry in enumerate(missing)}
                computed_count += len(missing)
            rows: list[np.ndarray] = []
            for entry in batch_entries:
                cached = reuse_index.get(entry.sha256)
                if cached is not None:
                    rows.append(np.asarray(cached[0][cached[1]], dtype=np.float32))
                    reused_count += 1
                else:
                    rows.append(computed_by_id[entry.image_id])
            values = normalize_rows(np.asarray(rows, dtype=np.float32))
            if values.shape[0] != len(batch_entries):
                raise ValueError("DINO adapter returned the wrong batch length")
            if config.model.output_dim is not None and values.shape[1] != config.model.output_dim:
                raise ValueError(
                    f"Expected embedding dimension {config.model.output_dim}, got {values.shape[1]}"
                )
            _atomic_save_npy(array_path, values.astype(np.float32, copy=False))
            file_io.atomic_write_json(
                metadata_path,
                {
                    "part_key": key,
                    "image_ids": [entry.image_id for entry in batch_entries],
                    "image_sha256s": [entry.sha256 for entry in batch_entries],
                    "shape": list(values.shape),
                    "array_sha256": file_io.sha256_file(array_path),
                },
            )
    finally:
        if runner is not None:
            runner.close()

    if num_shards > 1:
        shard_summary = {
            "schema_version": 1,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "owned_part_count": len(owned),
            "pending_part_count": len(pending),
            "reused_embedding_count": reused_count,
            "computed_embedding_count": computed_count,
            "complete": all(_valid_part(spec[0], spec[1], spec[2], len(spec[3])) for spec in owned),
        }
        file_io.atomic_write_json(out_dir / f"shard_{shard_index}.json", shard_summary)
        return shard_summary

    incomplete = [spec[0].name for spec in specs if not _valid_part(spec[0], spec[1], spec[2], len(spec[3]))]
    if incomplete:
        raise RuntimeError(f"Embedding merge is missing {len(incomplete)} parts, including {incomplete[:3]}")

    if specs:
        arrays = [np.load(spec[0], allow_pickle=False) for spec in specs]
        merged = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    else:
        merged = np.empty((0, config.model.output_dim or 0), dtype=np.float32)
    if merged.shape[0] != len(entries) or not np.isfinite(merged).all():
        raise ValueError("Merged embeddings failed shape/finiteness validation")
    if len(entries) and not np.allclose(np.linalg.norm(merged, axis=1), 1.0, atol=2e-5):
        raise ValueError("Merged embeddings are not L2-normalized")

    merged_path = out_dir / "embeddings.npy"
    _atomic_save_npy(merged_path, merged)
    image_ids_path = out_dir / "image_ids.json"
    file_io.atomic_write_json(image_ids_path, [entry.image_id for entry in entries])
    metadata = {
        "schema_version": 1,
        "model_fingerprint": config.model.fingerprint(),
        "model_fingerprint_sha256": config.model.fingerprint_sha256(),
        "images_manifest_sha256": file_io.sha256_file(work / "images.csv"),
        "image_ids_sha256": file_io.sha256_json([entry.image_id for entry in entries]),
        "num_images": len(entries),
        "shape": list(merged.shape),
        "dtype": str(merged.dtype),
        "normalized": True,
        "batch_size": config.model.batch_size,
        "preprocess_workers": config.model.preprocess_workers,
        "embeddings_sha256": file_io.sha256_file(merged_path),
        "part_hashes": {spec[0].name: file_io.sha256_file(spec[0]) for spec in specs},
        "reused_embedding_count": reused_count,
        "computed_embedding_count": computed_count,
        "reuse_sources": reuse_sources,
    }
    file_io.atomic_write_json(out_dir / "metadata.json", metadata)
    return metadata
