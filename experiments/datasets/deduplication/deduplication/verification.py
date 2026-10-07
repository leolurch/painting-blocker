"""Stage 4 — scalable SuperPoint/LightGlue verification.

Candidate verification can be split deterministically across nodes/GPUs with ``num_shards`` and
``shard_index``. Each node may run several model workers on its visible GPU. Workers share an
atomic, content-keyed on-disk SuperPoint cache and keep only a bounded CPU feature LRU.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import shutil
import socket
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

from . import file_io, manifest
from .config import Config

DEFAULT_PART_SIZE = 250
DEFAULT_FEATURE_CACHE_SIZE = 32

_WORKER_CONFIG: Config | None = None
_WORKER_RUNTIME: tuple | None = None
_WORKER_IMAGES: dict[str, manifest.ImageEntry] | None = None
_PRECOMPUTE_STORE: BoundedFeatureStore | None = None
_PRECOMPUTE_FINGERPRINT: str | None = None

# CUDA may already have been initialized by embedding or retrieval. A forked child would inherit
# that unusable runtime state, so CUDA workers must start in fresh interpreters. The context object
# itself starts no processes; workers remain alive for the complete pool lifetime.
_CUDA_MP_CONTEXT = multiprocessing.get_context("spawn")


def _cuda_process_pool(
    *,
    max_workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initializer,
        initargs=initargs,
        mp_context=_CUDA_MP_CONTEXT,
    )


def feature_cache_key(image_sha256: str, fingerprint: str) -> str:
    """Content-addressed identity, independent of source path/image id."""
    return file_io.sha256_json([image_sha256, fingerprint])


def superpoint_fingerprint(config: Config) -> str:
    return file_io.sha256_json(
        {
            "extractor": "lightglue.SuperPoint",
            "max_num_keypoints": config.superpoint_max_num_keypoints,
            "matcher": "lightglue.LightGlue(features=superpoint)",
            "reused_module": "lightglue_score_pseudo_pairs",
        }
    )


def _atomic_save_feature(path: Path, features: object) -> None:
    """Save a torch feature tree atomically with a cross-node-unique temporary name."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().replace("/", "_")
    tmp = path.with_name(f".{path.name}.{host}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(features, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class BoundedFeatureStore:
    """Atomic disk-backed SuperPoint features with a bounded per-worker CPU LRU."""

    def __init__(self, extractor, device, feature_dir: Path, capacity: int):
        self.extractor = extractor
        self.device = device
        self.feature_dir = feature_dir
        self.capacity = max(0, int(capacity))
        self.memory: OrderedDict[str, object] = OrderedDict()
        self.device_memory: dict[str, object] = {}
        self.transient_device: tuple[str, object] | None = None

        file_io.ensure_lightglue_verified_importable()
        from image_matching.keypoint.lightglue_score_pseudo_pairs import (
            _load_feature_file,
            _tensor_tree_to_cpu,
            _tensor_tree_to_device,
        )

        self._atomic_save = _atomic_save_feature
        self._load = _load_feature_file
        self._to_cpu = _tensor_tree_to_cpu
        self._to_device = _tensor_tree_to_device

    def _remember(self, key: str, value: object) -> None:
        if self.capacity == 0:
            return
        self.memory[key] = value
        self.memory.move_to_end(key)
        while len(self.memory) > self.capacity:
            self.memory.popitem(last=False)

    def get(self, key: str, image_path: Path, *, resident: bool = False):
        if key in self.device_memory:
            return self.device_memory[key]
        if self.transient_device is not None and self.transient_device[0] == key:
            return self.transient_device[1]
        if key in self.memory:
            value = self.memory.pop(key)
            self.memory[key] = value
            result = self._to_device(value, self.device)
            if resident:
                self.device_memory[key] = result
            else:
                self.transient_device = (key, result)
            return result

        feature_path = self.feature_dir / f"{key}.pt"
        if feature_path.is_file():
            try:
                value = self._load(feature_path)
                self._remember(key, value)
                result = self._to_device(value, self.device)
                if resident:
                    self.device_memory[key] = result
                else:
                    self.transient_device = (key, result)
                return result
            except Exception:
                # A damaged cache is recomputable. Atomic writers ensure readers never see a
                # normal in-progress file, but this also recovers from interrupted old versions.
                feature_path.unlink(missing_ok=True)

        import torch
        from lightglue.utils import load_image

        image = load_image(image_path).to(self.device, non_blocking=True)
        with torch.no_grad():
            features_gpu = self.extractor.extract(image)
        features_cpu = self._to_cpu(features_gpu)
        # Multiple nodes may compute the same missing feature. PID-specific temp files and atomic
        # replace make that benign: both complete values have the same content contract.
        self._atomic_save(feature_path, features_cpu)
        self._remember(key, features_cpu)
        if resident:
            self.device_memory[key] = features_gpu
        else:
            self.transient_device = (key, features_gpu)
        return features_gpu

    def preload_resident(self, entries: list[manifest.ImageEntry], fingerprint: str) -> None:
        for entry in entries:
            self.get(feature_cache_key(entry.sha256, fingerprint), entry.absolute_path(), resident=True)


def _load_feature_store(config: Config, feature_dir: Path) -> BoundedFeatureStore:
    file_io.ensure_lightglue_verified_importable()
    import torch
    from lightglue import SuperPoint

    if not torch.cuda.is_available():
        raise RuntimeError("SuperPoint precomputation requires a visible CUDA GPU")
    device = torch.device("cuda")
    extractor = SuperPoint(max_num_keypoints=config.superpoint_max_num_keypoints).eval().to(device)
    return BoundedFeatureStore(extractor, device, feature_dir, capacity=0)


def _precompute_initialize(config: Config, images_csv: str, feature_dir: str) -> None:
    global _WORKER_IMAGES, _PRECOMPUTE_STORE, _PRECOMPUTE_FINGERPRINT
    file_io.configure_work_dir(Path(images_csv).resolve().parent)
    _WORKER_IMAGES = manifest.load_images(images_csv)
    _PRECOMPUTE_STORE = _load_feature_store(config, Path(feature_dir))
    _PRECOMPUTE_FINGERPRINT = superpoint_fingerprint(config)


def _precompute_chunk(image_ids: list[str]) -> int:
    assert _WORKER_IMAGES is not None and _PRECOMPUTE_STORE is not None and _PRECOMPUTE_FINGERPRINT
    for image_id in image_ids:
        entry = _WORKER_IMAGES[image_id]
        _PRECOMPUTE_STORE.get(
            feature_cache_key(entry.sha256, _PRECOMPUTE_FINGERPRINT), entry.absolute_path()
        )
    return len(image_ids)


def _migrate_feature_reuse(
    config: Config, feature_dir: Path, reuse_work_dirs: list[Path]
) -> int:
    fingerprint = superpoint_fingerprint(config)
    migrated = 0
    feature_dir.mkdir(parents=True, exist_ok=True)
    for raw_work in reuse_work_dirs:
        old_work = Path(raw_work).expanduser().resolve()
        old_images_path = old_work / "images.csv"
        old_features = old_work / "verification" / "features"
        if not old_images_path.is_file() or not old_features.is_dir():
            raise FileNotFoundError(f"Reusable SuperPoint work directory is incomplete: {old_work}")
        for entry in manifest.load_images(old_images_path).values():
            target = feature_dir / f"{feature_cache_key(entry.sha256, fingerprint)}.pt"
            if target.is_file():
                continue
            # Accept both the new content key and the previous image-id-coupled key.
            sources = [
                old_features / target.name,
                old_features / f"{file_io.sha256_json([entry.image_id, entry.sha256, fingerprint])}.pt",
            ]
            source = next((path for path in sources if path.is_file()), None)
            if source is None:
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
            migrated += 1
    return migrated


def precompute_superpoint_features(
    config: Config,
    *,
    workers: int = 1,
    feature_dir: Path | None = None,
    reuse_work_dirs: list[Path] | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Materialize one content-addressed feature artifact per unique image SHA."""
    if workers <= 0:
        raise ValueError("SuperPoint precompute workers must be positive")
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("precompute shard index must lie in [0, num_shards)")
    work = file_io.work_dir()
    images = manifest.load_images(work / "images.csv")
    selected_dir = (feature_dir or (work / "verification" / "features")).expanduser().resolve()
    selected_dir.mkdir(parents=True, exist_ok=True)
    migrated = _migrate_feature_reuse(config, selected_dir, reuse_work_dirs or [])
    fingerprint = superpoint_fingerprint(config)
    unique_by_sha: dict[str, manifest.ImageEntry] = {}
    for entry in images.values():
        unique_by_sha.setdefault(entry.sha256, entry)
    pending_entries = [
        entry
        for sha, entry in sorted(unique_by_sha.items())
        if not (selected_dir / f"{feature_cache_key(sha, fingerprint)}.pt").is_file()
    ]
    if num_shards > 1:
        pending_entries = [
            entry
            for entry in pending_entries
            if int(entry.sha256[:8], 16) % num_shards == shard_index
        ]
    pending = [entry.image_id for entry in pending_entries]
    computed = 0
    if pending:
        chunks = [pending[index::workers] for index in range(workers)]
        chunks = [chunk for chunk in chunks if chunk]
        if workers == 1:
            global _PRECOMPUTE_STORE
            _precompute_initialize(config, str(work / "images.csv"), str(selected_dir))
            computed = _precompute_chunk(chunks[0])
            _PRECOMPUTE_STORE = None
            import torch

            torch.cuda.empty_cache()
        else:
            with _cuda_process_pool(
                # Avoid loading a GPU model in idle processes on nearly complete resumed runs.
                max_workers=len(chunks),
                initializer=_precompute_initialize,
                initargs=(config, str(work / "images.csv"), str(selected_dir)),
            ) as executor:
                computed = sum(executor.map(_precompute_chunk, chunks))
    if num_shards > 1:
        owned_missing = [
            entry.sha256
            for entry in pending_entries
            if not (selected_dir / f"{feature_cache_key(entry.sha256, fingerprint)}.pt").is_file()
        ]
        if owned_missing:
            raise RuntimeError(
                f"SuperPoint shard {shard_index} left {len(owned_missing)} missing features"
            )
        summary = {
            "schema_version": 1,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "feature_dir": str(selected_dir),
            "computed_count": computed,
            "complete": True,
        }
        file_io.atomic_write_json(
            work / "verification" / f"feature_precompute_shard_{shard_index}.json", summary
        )
        return summary
    missing = [
        sha
        for sha in unique_by_sha
        if not (selected_dir / f"{feature_cache_key(sha, fingerprint)}.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"SuperPoint precomputation left {len(missing)} missing features")
    summary = {
        "schema_version": 1,
        "feature_key": "sha256+superpoint_fingerprint",
        "feature_dir": str(selected_dir),
        "superpoint_fingerprint": fingerprint,
        "images_manifest_sha256": file_io.sha256_file(work / "images.csv"),
        "unique_content_count": len(unique_by_sha),
        "computed_count": computed,
        "migrated_count": migrated,
        "reused_count": len(unique_by_sha) - computed,
        "workers": workers,
        "complete": True,
    }
    file_io.atomic_write_json(work / "verification" / "feature_precompute_summary.json", summary)
    return summary


def _load_runtime(
    config: Config,
    feature_cache_size: int = DEFAULT_FEATURE_CACHE_SIZE,
    feature_dir: Path | None = None,
):
    actual_hash = file_io.sha256_file(config.classifier_path)
    if actual_hash != config.classifier_sha256:
        raise ValueError(
            f"Classifier hash mismatch: expected {config.classifier_sha256}, got {actual_hash}"
        )
    file_io.ensure_lightglue_verified_importable()
    import joblib
    import torch
    from lightglue import LightGlue, SuperPoint
    from image_matching.keypoint.lightglue_score_pseudo_pairs import (
        _match_feature_tensors,
        classify_scores,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("LightGlue verification requires a visible CUDA GPU")
    device = torch.device("cuda")
    extractor = SuperPoint(max_num_keypoints=config.superpoint_max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    classifier = joblib.load(config.classifier_path)
    store = BoundedFeatureStore(
        extractor,
        device,
        feature_dir or (file_io.work_dir() / "verification" / "features"),
        feature_cache_size,
    )
    return torch, device, matcher, classifier, store, _match_feature_tensors, classify_scores


def _process_chunk(
    chunk: list[dict[str, str]],
    config: Config,
    runtime: tuple,
    images: dict[str, manifest.ImageEntry],
) -> list[dict[str, Any]]:
    _torch, _device, matcher, classifier, store, match_fn, classify_fn = runtime
    fingerprint = superpoint_fingerprint(config)
    rows: list[dict[str, Any]] = []
    for candidate in chunk:
        pair_id = candidate["pair_id"]
        try:
            a = images[candidate["image_a_id"]]
            b = images[candidate["image_b_id"]]
            key_a = feature_cache_key(a.sha256, fingerprint)
            key_b = feature_cache_key(b.sha256, fingerprint)
            features_a = store.get(key_a, a.absolute_path())
            features_b = store.get(key_b, b.absolute_path())
            scores = match_fn(matcher, features_a, features_b)
            prediction, confidence = classify_fn(classifier, scores)
            rows.append(
                {
                    "pair_id": pair_id,
                    "status": "ok",
                    "lightglue_prediction": int(prediction),
                    "lightglue_confidence": repr(float(confidence)),
                    "match_count": len(scores),
                    "classifier_sha256": config.classifier_sha256,
                    "superpoint_fingerprint": fingerprint,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "pair_id": pair_id,
                    "status": "failed",
                    "lightglue_prediction": "",
                    "lightglue_confidence": "",
                    "match_count": "",
                    "classifier_sha256": config.classifier_sha256,
                    "superpoint_fingerprint": fingerprint,
                    "error": repr(exc).replace("\n", " ")[:2000],
                }
            )
    return rows


def _worker_initialize(
    config: Config,
    images_csv: str,
    feature_cache_size: int,
    feature_dir: str,
    resident_query_ids: list[str],
) -> None:
    global _WORKER_CONFIG, _WORKER_RUNTIME, _WORKER_IMAGES
    # Multiprocessing spawn imports file_io with its default work path; select the parent's path.
    file_io.configure_work_dir(Path(images_csv).resolve().parent)
    _WORKER_CONFIG = config
    _WORKER_IMAGES = manifest.load_images(images_csv)
    _WORKER_RUNTIME = _load_runtime(config, feature_cache_size, Path(feature_dir))
    store = _WORKER_RUNTIME[4]
    fingerprint = superpoint_fingerprint(config)
    store.preload_resident([_WORKER_IMAGES[image_id] for image_id in resident_query_ids], fingerprint)


def _worker_process(chunk: list[dict[str, str]]) -> list[dict[str, Any]]:
    assert _WORKER_CONFIG is not None and _WORKER_RUNTIME is not None and _WORKER_IMAGES is not None
    return _process_chunk(chunk, _WORKER_CONFIG, _WORKER_RUNTIME, _WORKER_IMAGES)


def _candidate_shard(pair_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def _iter_shard_chunks(
    candidates_path: Path,
    num_shards: int,
    shard_index: int,
    part_size: int,
) -> Iterator[tuple[int, list[dict[str, str]]]]:
    chunk: list[dict[str, str]] = []
    part_number = 0
    for row in file_io.iter_csv_rows(candidates_path):
        if _candidate_shard(row["pair_id"], num_shards) != shard_index:
            continue
        chunk.append(row)
        if len(chunk) == part_size:
            yield part_number, chunk
            part_number += 1
            chunk = []
    if chunk:
        yield part_number, chunk


def _part_key(config: Config, chunk: list[dict[str, str]], num_shards: int, shard_index: int) -> str:
    return file_io.sha256_json(
        {
            "rows": [
                [row["pair_id"], row["image_a_id"], row["image_b_id"], row["cosine_similarity"]]
                for row in chunk
            ],
            "classifier_sha256": config.classifier_sha256,
            "superpoint_fingerprint": superpoint_fingerprint(config),
            "num_shards": num_shards,
            "shard_index": shard_index,
        }
    )


def _valid_part(part_path: Path, meta_path: Path, key: str, expected_rows: int) -> bool:
    if not part_path.is_file() or not meta_path.is_file():
        return False
    try:
        metadata = file_io.read_json(meta_path)
        rows = file_io.read_csv_rows(part_path)
        return (
            metadata.get("part_key") == key
            and metadata.get("csv_sha256") == file_io.sha256_file(part_path)
            and len(rows) == expected_rows
            and all(row.get("status") == "ok" for row in rows)
        )
    except (OSError, ValueError, KeyError):
        return False


def _write_part(part_path: Path, meta_path: Path, key: str, rows: list[dict[str, Any]]) -> None:
    file_io.atomic_write_csv(part_path, manifest.VERIFICATION_FIELDS, rows)
    file_io.atomic_write_json(
        meta_path,
        {"part_key": key, "csv_sha256": file_io.sha256_file(part_path), "rows": len(rows)},
    )


def _update_counts(counts: dict[str, int], rows: list[dict[str, str]] | list[dict[str, Any]]) -> None:
    for row in rows:
        counts["candidate_count"] += 1
        if row.get("status") == "ok":
            counts["completed_count"] += 1
            if str(row.get("lightglue_prediction")) in {"1", "true", "True"}:
                counts["positive_count"] += 1
            else:
                counts["negative_count"] += 1
        else:
            counts["failed_count"] += 1


def _candidate_context(config: Config) -> tuple[Path, dict[str, Any], dict[str, manifest.ImageEntry]]:
    from .retrieval import retrieval_config_sha256

    work = file_io.work_dir()
    candidates_path = work / "candidates" / "candidates.csv"
    candidate_summary = file_io.read_json(work / "candidates" / "summary.json")
    if candidate_summary.get("candidates_sha256") != file_io.sha256_file(candidates_path):
        raise ValueError("Candidate CSV hash does not match its summary")
    embedding_metadata = file_io.read_json(work / "embeddings" / "metadata.json")
    images = manifest.load_images(work / "images.csv")
    if "mode" in candidate_summary:
        image_ids = file_io.read_json(work / "embeddings" / "image_ids.json")
        mode = str(candidate_summary["mode"])
        if mode == "bipartite":
            query_ids = [image_id for image_id in image_ids if images[image_id].source_name == "set_a"]
            candidate_ids = [image_id for image_id in image_ids if images[image_id].source_name == "set_b"]
        else:
            query_ids = list(image_ids)
            candidate_ids = list(image_ids)
        expected_retrieval_hash = retrieval_config_sha256(
            config,
            embedding_metadata,
            mode=mode,
            query_image_ids=query_ids,
            candidate_image_ids=candidate_ids,
            scoring_backend=candidate_summary.get("scoring_backend"),
        )
    else:
        # Compatibility for hand-built/legacy test artifacts without partition metadata.
        expected_retrieval_hash = retrieval_config_sha256(config, embedding_metadata)
    if candidate_summary.get("retrieval_config_sha256") != expected_retrieval_hash:
        raise ValueError(
            "Candidate retrieval configuration does not match the current config. "
            "Rerun retrieve with this exact config before verification."
        )
    return candidates_path, candidate_summary, images


def verify_shard(
    config: Config,
    *,
    workers: int = 1,
    num_shards: int = 1,
    shard_index: int = 0,
    part_size: int = DEFAULT_PART_SIZE,
    feature_cache_size: int = DEFAULT_FEATURE_CACHE_SIZE,
    feature_dir: Path | None = None,
) -> dict[str, Any]:
    if workers <= 0 or num_shards <= 0 or not 0 <= shard_index < num_shards or part_size <= 0:
        raise ValueError("Invalid workers/shard/part-size configuration")
    candidates_path, candidate_summary, images = _candidate_context(config)
    work = file_io.work_dir()
    precompute_summary_path = work / "verification" / "feature_precompute_summary.json"
    precomputed_dir = (
        Path(file_io.read_json(precompute_summary_path)["feature_dir"])
        if precompute_summary_path.is_file()
        else work / "verification" / "features"
    )
    selected_feature_dir = (feature_dir or precomputed_dir).expanduser().resolve()
    mode = str(candidate_summary.get("mode", "symmetric"))
    resident_query_ids = (
        sorted(image_id for image_id, entry in images.items() if entry.source_name == "set_a")
        if mode == "bipartite"
        else []
    )
    schedule_summary: dict[str, Any] | None = None
    if mode == "bipartite":
        from .verification_schedule import iter_reference_chunks, prepare_bipartite_schedules

        all_schedule_paths, schedule_summary = prepare_bipartite_schedules(
            candidates_path,
            candidate_summary["candidates_sha256"],
            images,
            num_shards=num_shards,
            workers=workers,
        )
        marker = f"shard_{shard_index:05d}_worker_"
        schedule_paths = sorted(path for path in all_schedule_paths if marker in path.name)

        def chunk_iterator():
            for worker_number, schedule_path in enumerate(schedule_paths):
                for local_part, chunk in iter_reference_chunks(schedule_path, part_size):
                    yield worker_number * 10_000_000 + local_part, chunk
    else:
        def chunk_iterator():
            yield from _iter_shard_chunks(candidates_path, num_shards, shard_index, part_size)
    layout = f"shards_{num_shards:05d}"
    parts_dir = work / "verification" / "parts" / layout / f"shard_{shard_index:05d}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = work / "verification" / f"shard_{shard_index:05d}_of_{num_shards:05d}.summary.json"

    counts = {key: 0 for key in ("candidate_count", "completed_count", "failed_count", "positive_count", "negative_count")}
    part_hashes: dict[str, str] = {}
    started = time.monotonic()
    completed_now = 0
    runtime = None
    executor: ProcessPoolExecutor | None = None
    worker_executors: list[ProcessPoolExecutor] = []
    in_flight: dict[Any, tuple[Path, Path, str, int]] = {}

    def record(part_path: Path, rows: list[dict[str, Any]] | list[dict[str, str]]) -> None:
        nonlocal completed_now
        _update_counts(counts, rows)
        part_hashes[part_path.name] = file_io.sha256_file(part_path)
        completed_now += len(rows)
        if completed_now and completed_now % 1000 < len(rows):
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"shard {shard_index}/{num_shards}: processed_now={completed_now:,} "
                f"rate={completed_now / elapsed:.2f} pairs/s",
                flush=True,
            )

    def finish_one() -> None:
        if not in_flight:
            return
        done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
        for future in done:
            part_path, meta_path, key, expected = in_flight.pop(future)
            rows = future.result()
            if len(rows) != expected:
                raise RuntimeError(f"Worker returned {len(rows)} rows; expected {expected}")
            _write_part(part_path, meta_path, key, rows)
            record(part_path, rows)

    try:
        if workers > 1:
            initargs = (
                config,
                str(work / "images.csv"),
                feature_cache_size,
                str(selected_feature_dir),
                resident_query_ids,
            )
            if mode == "bipartite":
                # One single-process executor per worker gives every reference hash bucket a
                # stable process owner and makes the worker schedule an actual affinity contract.
                worker_executors = [
                    _cuda_process_pool(
                        max_workers=1, initializer=_worker_initialize, initargs=initargs
                    )
                    for _ in range(workers)
                ]
            else:
                executor = _cuda_process_pool(
                    max_workers=workers,
                    initializer=_worker_initialize,
                    initargs=initargs,
                )
        for part_number, chunk in chunk_iterator():
            part_path = parts_dir / f"part_{part_number:08d}.csv"
            meta_path = part_path.with_suffix(".json")
            key = _part_key(config, chunk, num_shards, shard_index)
            if _valid_part(part_path, meta_path, key, len(chunk)):
                rows = file_io.read_csv_rows(part_path)
                _update_counts(counts, rows)
                part_hashes[part_path.name] = file_io.sha256_file(part_path)
                continue
            if workers == 1:
                if runtime is None:
                    runtime = _load_runtime(config, feature_cache_size, selected_feature_dir)
                    if resident_query_ids:
                        runtime[4].preload_resident(
                            [images[image_id] for image_id in resident_query_ids],
                            superpoint_fingerprint(config),
                        )
                rows = _process_chunk(chunk, config, runtime, images)
                _write_part(part_path, meta_path, key, rows)
                record(part_path, rows)
            else:
                while len(in_flight) >= workers * 2:
                    finish_one()
                if mode == "bipartite":
                    worker_number = part_number // 10_000_000
                    future = worker_executors[worker_number].submit(_worker_process, chunk)
                else:
                    assert executor is not None
                    future = executor.submit(_worker_process, chunk)
                in_flight[future] = (part_path, meta_path, key, len(chunk))
        while in_flight:
            finish_one()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for worker_executor in worker_executors:
            worker_executor.shutdown(wait=True, cancel_futures=True)
        if runtime is not None:
            torch = runtime[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    summary = {
        "schema_version": 2,
        "complete": True,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "workers": workers,
        "part_size": part_size,
        "feature_cache_size": feature_cache_size,
        "feature_dir": str(selected_feature_dir),
        "sharding_strategy": "set_b_reference" if mode == "bipartite" else "pair_id",
        "schedule_key": schedule_summary.get("schedule_key") if schedule_summary else None,
        **counts,
        "elapsed_seconds": elapsed,
        "processed_now": completed_now,
        "processed_now_pairs_per_second": completed_now / elapsed if elapsed and completed_now else 0.0,
        "classifier_sha256": config.classifier_sha256,
        "superpoint_fingerprint": superpoint_fingerprint(config),
        "candidates_sha256": candidate_summary["candidates_sha256"],
        "part_hashes": part_hashes,
    }
    file_io.atomic_write_json(summary_path, summary)
    return summary


def merge_verification(config: Config, *, num_shards: int) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    candidates_path, candidate_summary, _images = _candidate_context(config)
    work = file_io.work_dir()
    verification_dir = work / "verification"
    shard_summaries: list[tuple[Path, dict[str, Any]]] = []
    for shard_index in range(num_shards):
        path = verification_dir / f"shard_{shard_index:05d}_of_{num_shards:05d}.summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Verification shard summary is missing: {path}")
        summary = file_io.read_json(path)
        if (
            not summary.get("complete")
            or summary.get("num_shards") != num_shards
            or summary.get("shard_index") != shard_index
            or summary.get("candidates_sha256") != candidate_summary["candidates_sha256"]
            or summary.get("classifier_sha256") != config.classifier_sha256
            or summary.get("superpoint_fingerprint") != superpoint_fingerprint(config)
        ):
            raise ValueError(f"Incompatible verification shard summary: {path}")
        shard_summaries.append((path, summary))

    candidate_count = sum(int(summary["candidate_count"]) for _, summary in shard_summaries)
    expected_count = int(candidate_summary["unique_unordered_pair_count"])
    if candidate_count != expected_count:
        raise ValueError(
            f"Verification shards cover {candidate_count} pairs; candidates contain {expected_count}"
        )

    layout = f"shards_{num_shards:05d}"

    def rows() -> Iterator[dict[str, str]]:
        for shard_index, (_summary_path, summary) in enumerate(shard_summaries):
            parts_dir = verification_dir / "parts" / layout / f"shard_{shard_index:05d}"
            for part_name, expected_hash in sorted(summary["part_hashes"].items()):
                path = parts_dir / part_name
                if file_io.sha256_file(path) != expected_hash:
                    raise ValueError(f"Verification part hash mismatch: {path}")
                yield from file_io.iter_csv_rows(path)

    output = verification_dir / "verification.csv"
    file_io.atomic_write_csv(output, manifest.VERIFICATION_FIELDS, rows())
    summary = {
        "schema_version": 2,
        "candidate_count": candidate_count,
        "completed_count": sum(int(item[1]["completed_count"]) for item in shard_summaries),
        "failed_count": sum(int(item[1]["failed_count"]) for item in shard_summaries),
        "positive_count": sum(int(item[1]["positive_count"]) for item in shard_summaries),
        "negative_count": sum(int(item[1]["negative_count"]) for item in shard_summaries),
        "num_shards": num_shards,
        "classifier_sha256": config.classifier_sha256,
        "superpoint_fingerprint": superpoint_fingerprint(config),
        "candidates_sha256": file_io.sha256_file(candidates_path),
        "verification_sha256": file_io.sha256_file(output),
        "shard_summary_hashes": {
            path.name: file_io.sha256_file(path) for path, _summary in shard_summaries
        },
    }
    file_io.atomic_write_json(verification_dir / "summary.json", summary)
    return summary


def verify(
    config: Config,
    *,
    workers: int = 1,
    num_shards: int = 1,
    shard_index: int = 0,
    part_size: int = DEFAULT_PART_SIZE,
    feature_cache_size: int = DEFAULT_FEATURE_CACHE_SIZE,
    feature_dir: Path | None = None,
    merge: bool = True,
) -> dict[str, Any]:
    summary = verify_shard(
        config,
        workers=workers,
        num_shards=num_shards,
        shard_index=shard_index,
        part_size=part_size,
        feature_cache_size=feature_cache_size,
        feature_dir=feature_dir,
    )
    if merge and num_shards == 1:
        return merge_verification(config, num_shards=1)
    return summary
