"""MLPerf-inspired, batch-one single-stream embedding latency benchmark.

This module deliberately reuses the quality-evaluation model adapters while
keeping latency execution separate from the batched embedding-cache pipeline.
The experimental repetitions are complete counterbalanced blocks, not images.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from .artifacts import atomic_write_json, git_info, make_run_id, sha256_file, sha256_json
from .config_schema import ModelConfig, find_repo_root, load_model_reference
from .db import fetch_same_painting_records
from .env import env_hf_token
from .reproducibility import dependency_versions

BENCHMARK_SCHEMA_VERSION = 1
SINGLE_STREAM_IMPLEMENTATION_VERSION = 15
A100_RESULT_NAME = "Measured single-stream embedding latency on one A100 80 GB system"
# Backwards-compatible public name used by existing A100 tests/importers.
RESULT_NAME = A100_RESULT_NAME


@dataclass(frozen=True)
class SingleStreamConfig:
    path: Path
    repo_root: Path
    benchmark_id: str
    result_name: str
    dataset_config: Path
    dataset_id: str
    dataset_db: Path
    image_root: Path
    image_count: int
    selection_seed: int
    models: tuple[ModelConfig, ...]
    protocol: dict[str, Any]
    run: dict[str, Any]
    raw: dict[str, Any]


def _expand_path(value: str | Path, base: Path) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"Path contains an unset environment variable: {value}")
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_single_stream_config(path: str | Path) -> SingleStreamConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported single-stream benchmark schema in {source}")
    benchmark_id = str(raw.get("benchmark_id") or "").strip()
    if not benchmark_id:
        raise ValueError(f"benchmark_id is required in {source}")
    result_name = str(raw.get("result_name") or "").strip()
    if not result_name.startswith("Measured single-stream embedding latency on one ") or not result_name.endswith(" system"):
        raise ValueError(
            "result_name must identify the measured system as "
            "'Measured single-stream embedding latency on one <system> system'"
        )

    repo = find_repo_root(source.parent)
    dataset_raw = raw.get("dataset")
    if not isinstance(dataset_raw, dict):
        raise ValueError("dataset must be a mapping")
    dataset_config = _expand_path(dataset_raw.get("config", ""), source.parent)
    dataset_yaml = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    if not isinstance(dataset_yaml, dict) or dataset_yaml.get("schema_version") != 1:
        raise ValueError(f"Invalid dataset config: {dataset_config}")
    paths = dataset_yaml.get("paths") or {}
    dataset_db = _expand_path(dataset_raw.get("dataset_db") or paths.get("dataset_db", ""), repo)
    image_root = _expand_path(dataset_raw.get("image_root") or paths.get("image_root", ""), repo)
    image_count = int(dataset_raw.get("image_count", 1024))
    if image_count < 1024:
        raise ValueError("Final single-stream workload requires at least 1,024 unique images")
    require_all = dataset_raw.get("require_all_available_images", False)
    if not isinstance(require_all, bool):
        raise ValueError("dataset.require_all_available_images must be boolean")

    model_block = raw.get("models") or {}
    includes = model_block.get("include") if isinstance(model_block, dict) else None
    if not isinstance(includes, list) or len(includes) < 1:
        raise ValueError("models.include must contain at least one model configuration")
    models = tuple(load_model_reference(item, source.parent, source) for item in includes)
    if len({model.model_id for model in models}) != len(models):
        raise ValueError("Every benchmark model must have a unique model_id/variant_id")
    for model in models:
        embedding = model.raw.get("embedding") or {}
        if embedding.get("normalize") is False:
            raise ValueError(f"Benchmark model {model.model_id} must produce normalized descriptors")

    protocol = dict(raw.get("protocol") or {})
    cycles = int(protocol.get("counterbalancing_cycles", 1))
    cycle_seeds = protocol.get("cycle_seeds")
    if cycles < 1 or not isinstance(cycle_seeds, list) or len(cycle_seeds) != cycles:
        raise ValueError("protocol.cycle_seeds must contain one predetermined seed per cycle")
    if len({int(seed) for seed in cycle_seeds}) != cycles:
        raise ValueError("protocol.cycle_seeds must be unique")
    if int(protocol.get("warmup_queries", 50)) < 1:
        raise ValueError("protocol.warmup_queries must be positive")
    if not isinstance(protocol.get("skip_warmup_pilot", False), bool):
        raise ValueError("protocol.skip_warmup_pilot must be boolean")
    if not isinstance(protocol.get("warmup_stability_required", True), bool):
        raise ValueError("protocol.warmup_stability_required must be boolean")
    pilot_queries = int(protocol.get("pilot_warmup_queries", 100))
    pilot_window = int(protocol.get("pilot_stability_window_queries", 20))
    warmup_queries = int(protocol.get("warmup_queries", 50))
    if pilot_window < 10:
        raise ValueError("protocol.pilot_stability_window_queries must be at least 10")
    if pilot_queries < warmup_queries + 3 * pilot_window:
        raise ValueError(
            "protocol.pilot_warmup_queries must cover warmup_queries plus "
            "three complete stability windows"
        )
    if str(protocol.get("precision", "")).lower() not in {"bfloat16", "float16", "float32"}:
        raise ValueError("protocol.precision must be bfloat16, float16, or float32")
    if int(protocol.get("batch_size", 1)) != 1:
        raise ValueError("Single-stream protocol requires batch_size: 1")
    if protocol.get("torch_compile") not in {True, False}:
        raise ValueError("protocol.torch_compile must be explicitly true or false")
    if not str(protocol.get("attention_implementation") or "").strip():
        raise ValueError("protocol.attention_implementation must be explicit")
    required_gpu_name = str(protocol.get("required_gpu_name") or "").strip()
    if not required_gpu_name:
        raise ValueError("protocol.required_gpu_name must be explicit")
    if required_gpu_name.lower() not in result_name.lower():
        raise ValueError(
            f"result_name {result_name!r} does not identify required GPU {required_gpu_name!r}"
        )
    if int(protocol.get("minimum_gpu_memory_gib", 0)) < 1:
        raise ValueError("protocol.minimum_gpu_memory_gib must be positive")
    lora_merged = protocol.get("lora_weights_merged")
    if str(lora_merged).lower() not in {"true", "false", "not_applicable"}:
        raise ValueError("protocol.lora_weights_merged must be explicit")

    return SingleStreamConfig(
        path=source,
        repo_root=repo,
        benchmark_id=benchmark_id,
        result_name=result_name,
        dataset_config=dataset_config,
        dataset_id=str(dataset_yaml.get("dataset_id") or dataset_config.stem),
        dataset_db=dataset_db,
        image_root=image_root,
        image_count=image_count,
        selection_seed=int(dataset_raw.get("selection_seed", 20260727)),
        models=models,
        protocol=protocol,
        run=dict(raw.get("run") or {}),
        raw=raw,
    )


def validate_single_stream_setup(config_path: str | Path) -> dict[str, Any]:
    """Validate non-GPU inputs before submitting a long-waiting Slurm job."""
    config = load_single_stream_config(config_path)
    source_hashes = source_code_hashes(config.repo_root)
    configured_output = _expand_path(
        config.run.get("output_dir", f"experiments/runs/{config.benchmark_id}"),
        config.repo_root,
    )
    schedule = build_schedule(
        [model.model_id for model in config.models],
        cycle_seeds=[int(value) for value in config.protocol["cycle_seeds"]],
        image_order_seed=int(config.protocol.get("image_order_seed", 20260728)),
    )
    validate_schedule_balance(schedule)

    missing: set[str] = set()
    if not config.dataset_db.is_file():
        missing.add(str(config.dataset_db))
    if not config.image_root.is_dir():
        missing.add(str(config.image_root))

    available_images = 0
    selected_workload: list[dict[str, Any]] = []
    if config.dataset_db.is_file() and config.image_root.is_dir():
        records = fetch_same_painting_records(config.dataset_db)
        for record in records.values():
            path = record.absolute_path(config.dataset_db, config.image_root)
            if path is not None and path.is_file():
                available_images += 1
        require_all = bool((config.raw.get("dataset") or {}).get("require_all_available_images", False))
        if available_images < config.image_count:
            missing.add(
                f"dataset has only {available_images} available images; requires {config.image_count}"
            )
        elif require_all and available_images != config.image_count:
            missing.add(
                f"full-dataset protocol requires exactly {config.image_count} available images; found {available_images}"
            )
        else:
            # Exercise deterministic selection, hashing, and complete image
            # decoding before reserving a GPU. Images are released one by one.
            selected_workload = select_workload(config)
            for item in selected_workload:
                with Image.open(item["path"]) as source:
                    source.convert("RGB").load()
    decoded_rgb_bytes = sum(
        int(item["width"]) * int(item["height"]) * 3 for item in selected_workload
    )
    decoded_budget_gib = int(config.protocol.get("decoded_image_memory_budget_gib", 0))
    if selected_workload and decoded_budget_gib < 1:
        raise ValueError("protocol.decoded_image_memory_budget_gib must be positive")
    if decoded_rgb_bytes > decoded_budget_gib * 1024**3:
        raise MemoryError(
            f"Selected RGB workload requires at least {decoded_rgb_bytes / 1024**3:.2f} GiB "
            f"before PIL overhead, exceeding the {decoded_budget_gib} GiB budget"
        )

    hf_home = Path(os.path.expandvars(os.getenv("HF_HOME", ""))).expanduser()
    model_inputs: list[dict[str, Any]] = []
    for model in config.models:
        checkpoint = model.adapter.kwargs.get("checkpoint_path")
        if checkpoint:
            path = _expand_path(checkpoint, config.repo_root)
            exists = path.is_file()
            if not exists:
                missing.add(str(path))
            if exists:
                # Detect truncated/incompatible payloads before submission; the
                # same helper also freezes the checkpoint SHA-256 and metadata.
                _checkpoint_manifest(model, config.repo_root)
            model_inputs.append({"model_id": model.model_id, "kind": "checkpoint", "path": str(path), "available": exists})
            continue
        repo_dir = hf_home / "hub" / f"models--{model.adapter.model_id.replace('/', '--')}"
        snapshot = repo_dir / "snapshots" / str(model.revision)
        available = snapshot.is_dir()
        if not available:
            missing.add(str(snapshot))
        model_inputs.append(
            {
                "model_id": model.model_id,
                "kind": "huggingface_snapshot",
                "revision": model.revision,
                "path": str(snapshot),
                "available": available,
            }
        )
    if missing:
        raise FileNotFoundError(
            "Single-stream preflight missing input(s):\n- " + "\n- ".join(sorted(missing))
        )
    return {
        "status": "ok",
        "implementation_version": SINGLE_STREAM_IMPLEMENTATION_VERSION,
        "benchmark_id": config.benchmark_id,
        "result_name": config.result_name,
        "repo_root": str(config.repo_root),
        "output_dir": str(configured_output),
        "source_file_count": len(source_hashes),
        "source_code_sha256": sha256_json(source_hashes),
        "required_gpu_name": config.protocol["required_gpu_name"],
        "minimum_gpu_memory_gib": config.protocol["minimum_gpu_memory_gib"],
        "model_count": len(config.models),
        "block_count": schedule["block_count"],
        "image_count": config.image_count,
        "available_images": available_images,
        "selected_workload_sha256": sha256_json(selected_workload) if selected_workload else None,
        "decoded_rgb_bytes_estimate": decoded_rgb_bytes,
        "decoded_image_memory_budget_gib": decoded_budget_gib,
        "dataset_db": str(config.dataset_db),
        "image_root": str(config.image_root),
        "hf_home": str(hf_home),
        "model_inputs": model_inputs,
    }


def prepare_single_stream_model_cache(config_path: str | Path) -> dict[str, Any]:
    """Download any missing exact Hub revisions before Slurm submission."""
    config = load_single_stream_config(config_path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to prepare benchmark models") from exc

    hf_home = Path(os.path.expandvars(os.getenv("HF_HOME", ""))).expanduser()
    if not str(hf_home) or str(hf_home) == ".":
        raise ValueError("HF_HOME must point to the benchmark Hugging Face cache")
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    # Central env_hf_token() loads <repo>/.env without overriding an explicitly
    # exported token. Do not source .env from Bash or print the secret.
    token = env_hf_token()
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for model in config.models:
        if model.adapter.kwargs.get("checkpoint_path"):
            continue
        key = (model.adapter.model_id, str(model.revision))
        if key in seen:
            continue
        seen.add(key)
        repo_id, revision = key
        expected = hub_cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        existed_before = expected.is_dir()
        try:
            # Always call snapshot_download: an existing directory can be the
            # residue of an interrupted download. The Hub client verifies the
            # pinned revision and fills any missing files idempotently.
            downloaded = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=str(hub_cache),
                    token=token,
                )
            ).resolve()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {repo_id}@{revision} into {hub_cache}. "
                "For gated DINOv3 models, set HF_TOKEN in the repository .env "
                "or export it explicitly."
            ) from exc
        if not downloaded.is_dir():
            raise RuntimeError(f"snapshot_download returned a missing directory: {downloaded}")
        if not expected.is_dir():
            raise RuntimeError(f"Pinned snapshot was not materialized at expected path: {expected}")
        results.append(
            {
                "model_id": repo_id,
                "revision": revision,
                "status": "verified_cached" if existed_before else "downloaded",
                "path": str(expected),
            }
        )
    return {
        "status": "ok",
        "implementation_version": SINGLE_STREAM_IMPLEMENTATION_VERSION,
        "hf_home": str(hf_home),
        "snapshots": results,
    }


def validate_single_stream_gpu_setup(config_path: str | Path) -> dict[str, Any]:
    """Run a minimal exact-policy CUDA/SDPA smoke test inside the allocation."""
    config = load_single_stream_config(config_path)
    _assert_allocation(config)
    import torch
    import torch.nn.functional as torch_functional

    if not all(
        hasattr(torch.backends.cuda, name)
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp")
    ):
        raise RuntimeError("Installed PyTorch does not expose the required SDPA backend controls")
    torch.cuda.set_device(0)
    attention_policy = str(config.protocol["attention_implementation"]).lower()
    if attention_policy == "pytorch_sdpa_auto_where_applicable":
        enabled = {"flash": True, "memory_efficient": True, "math": True, "cudnn": True}
    elif attention_policy == "pytorch_sdpa_flash":
        enabled = {"flash": True, "memory_efficient": False, "math": False, "cudnn": False}
    else:
        enabled = {"flash": False, "memory_efficient": True, "math": False, "cudnn": False}
    torch.backends.cuda.enable_flash_sdp(enabled["flash"])
    torch.backends.cuda.enable_mem_efficient_sdp(enabled["memory_efficient"])
    torch.backends.cuda.enable_math_sdp(enabled["math"])
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(enabled["cudnn"])
    else:
        enabled["cudnn"] = False
    query = torch.randn((1, 8, 64, 64), device="cuda", dtype=torch.bfloat16)
    output = torch_functional.scaled_dot_product_attention(query, query, query)
    torch.cuda.synchronize()
    if output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"BF16 PyTorch SDPA smoke test returned invalid {output.dtype} output")
    props = torch.cuda.get_device_properties(0)
    return {
        "status": "ok",
        "implementation_version": SINGLE_STREAM_IMPLEMENTATION_VERSION,
        "gpu_name": props.name,
        "gpu_memory_bytes": props.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "sdpa_policy": attention_policy,
        "pytorch_sdpa_backends_enabled": enabled,
        "external_flash_attention": False,
        "smoke_output_dtype": str(output.dtype).removeprefix("torch."),
    }


def canonical_williams_rows(model_count: int) -> list[list[int]]:
    """Return one complete first-order counterbalancing cycle.

    Even ``m`` uses ``m`` Williams rows. Odd ``m`` includes the reversals and
    therefore uses ``2m`` rows.
    """
    if model_count < 1:
        raise ValueError("Schedule requires at least one model")
    if model_count == 1:
        return [[0]]
    first: list[int] = []
    low, high = 0, model_count - 1
    while low <= high:
        first.append(low)
        low += 1
        if low <= high:
            first.append(high)
            high -= 1
    rows = [[(value + shift) % model_count for value in first] for shift in range(model_count)]
    if model_count % 2:
        rows += [list(reversed(row)) for row in rows]
    return rows


def build_schedule(
    model_ids: Iterable[str], *, cycle_seeds: Iterable[int], image_order_seed: int
) -> dict[str, Any]:
    ids = list(model_ids)
    seeds = [int(value) for value in cycle_seeds]
    rows = canonical_williams_rows(len(ids))
    blocks: list[dict[str, Any]] = []
    image_rng = random.Random(int(image_order_seed))
    mappings: list[dict[str, Any]] = []
    for cycle_index, seed_value in enumerate(seeds):
        seed = int(seed_value)
        rng = random.Random(seed)
        mapped_ids = list(ids)
        rng.shuffle(mapped_ids)
        row_order = list(range(len(rows)))
        rng.shuffle(row_order)
        mapping = {str(symbol): mapped_ids[symbol] for symbol in range(len(ids))}
        mappings.append({"cycle": cycle_index + 1, "seed": seed, "symbol_to_model": mapping, "row_order": row_order})
        for within_cycle, row_index in enumerate(row_order):
            sequence = [mapped_ids[symbol] for symbol in rows[row_index]]
            blocks.append(
                {
                    "block": len(blocks) + 1,
                    "cycle": cycle_index + 1,
                    "block_within_cycle": within_cycle + 1,
                    "canonical_row": row_index,
                    "image_order_seed": image_rng.randrange(0, 2**63),
                    "models": sequence,
                }
            )
    return {
        "schema_version": 1,
        "design": (
            "Single-model repeated full-workload blocks"
            if len(ids) == 1
            else "Williams balanced Latin square with reversals for odd model counts"
        ),
        "model_count": len(ids),
        "cycle_count": len(seeds),
        "block_count": len(blocks),
        "mappings": mappings,
        "blocks": blocks,
    }


def validate_schedule_balance(schedule: dict[str, Any]) -> None:
    blocks = schedule["blocks"]
    by_cycle: dict[int, list[dict[str, Any]]] = {}
    for block in blocks:
        by_cycle.setdefault(int(block["cycle"]), []).append(block)
    for cycle, cycle_blocks in by_cycle.items():
        sequences = [list(block["models"]) for block in cycle_blocks]
        models = sorted(sequences[0])
        m = len(models)
        expected_blocks = 1 if m == 1 else (m if m % 2 == 0 else 2 * m)
        if len(sequences) != expected_blocks:
            raise ValueError(f"Cycle {cycle} has {len(sequences)} blocks, expected {expected_blocks}")
        position_counts = {(model, pos): 0 for model in models for pos in range(m)}
        predecessor_counts = {(left, right): 0 for left in models for right in models if left != right}
        for sequence in sequences:
            if sorted(sequence) != models:
                raise ValueError(f"Cycle {cycle} contains an incomplete model row")
            for pos, model in enumerate(sequence):
                position_counts[(model, pos)] += 1
            for left, right in zip(sequence, sequence[1:]):
                predecessor_counts[(left, right)] += 1
        expected_position = 1 if m <= 1 or m % 2 == 0 else 2
        if set(position_counts.values()) != {expected_position}:
            raise ValueError(f"Cycle {cycle} is not position balanced")
        if m > 1:
            expected_pair = 1 if m % 2 == 0 else 2
            if set(predecessor_counts.values()) != {expected_pair}:
                raise ValueError(f"Cycle {cycle} is not immediate-predecessor balanced")


def select_workload(config: SingleStreamConfig) -> list[dict[str, Any]]:
    if not config.dataset_db.is_file():
        raise FileNotFoundError(f"Dataset DB not found: {config.dataset_db}")
    if not config.image_root.is_dir():
        raise FileNotFoundError(f"Dataset image root not found: {config.image_root}")
    records = fetch_same_painting_records(config.dataset_db)
    candidates: list[tuple[str, Path]] = []
    for file_id, record in records.items():
        path = record.absolute_path(config.dataset_db, config.image_root)
        if path is not None and path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(config.dataset_db.parent.resolve()):
                raise RuntimeError(f"Dataset record resolves outside the configured dataset root: {resolved}")
            candidates.append((file_id, resolved))
    if len(candidates) < config.image_count:
        raise ValueError(f"Only {len(candidates)} unique local images are available; need {config.image_count}")
    require_all = bool((config.raw.get("dataset") or {}).get("require_all_available_images", False))
    if require_all and len(candidates) != config.image_count:
        raise ValueError(
            f"Full-dataset workload expected exactly {config.image_count} images, found {len(candidates)}"
        )
    candidates.sort(key=lambda item: item[0])
    rng = random.Random(config.selection_seed)
    chosen = rng.sample(candidates, config.image_count)
    workload: list[dict[str, Any]] = []
    for file_id, path in chosen:
        with Image.open(path) as image:
            width, height = image.size
        workload.append(
            {
                "image_id": file_id,
                "path": str(path),
                "width": int(width),
                "height": int(height),
                "sha256": sha256_file(path),
            }
        )
    return workload


def _checkpoint_manifest(model: ModelConfig, repo_root: Path) -> dict[str, Any]:
    kwargs = dict(model.adapter.kwargs)
    configured = kwargs.get("checkpoint_path")
    if configured:
        path = _expand_path(configured, repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint for {model.model_id} not found: {path}")
        try:
            import torch

            try:
                payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            training = payload.get("training_config") or {}
            model_config = training.get("model") or training.get("finetuning_model") or {}
            inference_metadata = {
                "model_type": payload.get("model_type"),
                "backbone_name": payload.get("backbone_name"),
                "backbone_revision": payload.get("backbone_revision"),
                "pooling_type": payload.get("pooling_type"),
                "projection_dim": payload.get("projection_dim"),
                "model_config": model_config,
                "preprocessing_config": payload.get("preprocessing_config") or {},
                "normalization_config": payload.get("normalization_config") or {},
                "lora_weights_merged": False if str(payload.get("model_type") or "").startswith("dinov3_lora") else "not_applicable",
            }
            del payload
        except Exception as exc:
            raise RuntimeError(f"Could not inspect checkpoint metadata for {model.model_id}: {path}") from exc
        return {
            "kind": "local_checkpoint",
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "inference_metadata": inference_metadata,
        }
    if not model.revision:
        raise ValueError(f"Frozen Hub model {model.model_id} must pin an exact revision")
    return {
        "kind": "huggingface_revision",
        "model_id": model.adapter.model_id,
        "revision": model.revision,
        "checksum": model.revision,
        "lora_weights_merged": "not_applicable",
    }


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip() or None
    except Exception:
        return None


def source_code_hashes(repo_root: Path) -> dict[str, str]:
    paths = sorted((repo_root / "experiments/core").rglob("*.py"))
    paths.append(repo_root / "experiments/cli.py")
    return {str(path.relative_to(repo_root)): sha256_file(path) for path in paths}


def build_manifest(config: SingleStreamConfig, workload: list[dict[str, Any]], schedule: dict[str, Any]) -> dict[str, Any]:
    models = []
    for model in config.models:
        checkpoint = _checkpoint_manifest(model, config.repo_root)
        checkpoint_inference = checkpoint.get("inference_metadata") or {}
        models.append(
            {
                "model_id": model.model_id,
                "source_model_id": model.adapter.model_id,
                "label": model.raw.get("label") or model.raw.get("display_name") or model.model_id,
                "model_config_path": str(model.path),
                "model_config_sha256": sha256_file(model.path),
                "adapter": model.adapter.to_dict(),
                "checkpoint": checkpoint,
                "preprocessing": checkpoint_inference.get("preprocessing_config") or model.raw.get("preprocessing") or {},
                "descriptor_pooling": checkpoint_inference.get("pooling_type") or (model.raw.get("embedding") or {}).get("pooling", "default"),
                "descriptor_projection": {
                    "model_type": checkpoint_inference.get("model_type"),
                    "projection_dim": checkpoint_inference.get("projection_dim") or (model.raw.get("embedding") or {}).get("output_dim"),
                },
                "descriptor_normalization": "model intrinsic when applicable, then canonical FP32 L2 normalization",
                "embedding": model.raw.get("embedding") or {},
                "lora_weights_merged": checkpoint_inference.get("lora_weights_merged", "not_applicable"),
            }
        )
    try:
        import torch
        torch_runtime = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    except ImportError:
        torch_runtime = {}
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip().split(",")[0].strip() or "0"
    selected_gpu_raw = _command_output([
        "nvidia-smi", "-i", visible,
        "--query-gpu=uuid,name,memory.total,compute_mode",
        "--format=csv,noheader,nounits",
    ])
    selected_gpu_values = [part.strip() for part in selected_gpu_raw.split(",", 3)] if selected_gpu_raw else []
    selected_gpu = {
        "uuid": selected_gpu_values[0] if len(selected_gpu_values) == 4 else None,
        "name": selected_gpu_values[1] if len(selected_gpu_values) == 4 else None,
        "memory_total_mib": selected_gpu_values[2] if len(selected_gpu_values) == 4 else None,
        "compute_mode": selected_gpu_values[3] if len(selected_gpu_values) == 4 else None,
        "raw": selected_gpu_raw,
    }
    manifest = {
        "schema_version": 1,
        "frozen": True,
        "result_name": config.result_name,
        "interpretation": "Complete model-software-hardware system runtime; not hardware-independent computational complexity.",
        "mlperf_status": "MLPerf-inspired Single Stream semantics; not MLPerf-compliant.",
        "benchmark_id": config.benchmark_id,
        "config_path": str(config.path),
        "config_sha256": sha256_file(config.path),
        "git": git_info(config.repo_root),
        "source_code_sha256": source_code_hashes(config.repo_root),
        "protocol": config.protocol,
        "models": models,
        "dataset": {
            "dataset_id": config.dataset_id,
            "dataset_config": str(config.dataset_config),
            "dataset_config_sha256": sha256_file(config.dataset_config),
            "dataset_db": str(config.dataset_db),
            "dataset_db_sha256": sha256_file(config.dataset_db),
            "image_root": str(config.image_root),
            "image_count": len(workload),
            "workload_sha256": sha256_json(workload),
        },
        "schedule_sha256": sha256_json(schedule),
        "software": {**dependency_versions(), **torch_runtime},
        "hardware": {
            "hostname": platform.node(),
            "selected_gpu": selected_gpu,
            "driver_version": _command_output(["nvidia-smi", "-i", visible, "--query-gpu=driver_version", "--format=csv,noheader"]),
            "nvidia_smi": _command_output(["nvidia-smi", "-q"]),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_job_details": _command_output(
                ["scontrol", "show", "job", "-o", os.getenv("SLURM_JOB_ID", "")]
            ),
            "slurm_job_gpus": os.getenv("SLURM_JOB_GPUS"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "cpu_thread_count": int(config.protocol.get("cpu_threads", 1)),
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def image_permutation(image_count: int, seed: int) -> list[int]:
    order = list(range(image_count))
    random.Random(int(seed)).shuffle(order)
    return order


def summarize_execution(latencies_ns: list[int]) -> dict[str, float]:
    values = np.asarray(latencies_ns, dtype=np.float64) / 1_000_000.0
    if values.size == 0:
        raise ValueError("Cannot summarize empty latency values")
    total_seconds = float(values.sum() / 1000.0)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "sequential_images_per_second": float(values.size / total_seconds),
        "total_timed_seconds": total_seconds,
    }


def aggregate_results(
    run_dir: str | Path,
    schedule: dict[str, Any],
    models: Iterable[ModelConfig],
    reference_model: str | None = None,
    result_name: str = RESULT_NAME,
) -> dict[str, Any]:
    root = Path(run_dir)
    accepted: list[dict[str, Any]] = []
    for block in schedule["blocks"]:
        marker = root / "blocks" / f"block-{int(block['block']):03d}" / "accepted.json"
        if not marker.is_file():
            raise FileNotFoundError(f"Missing accepted block: {marker}")
        accepted.append(json.loads(marker.read_text(encoding="utf-8")))
    model_map = {model.model_id: model for model in models}
    grouped: dict[str, list[dict[str, Any]]] = {model_id: [] for model_id in model_map}
    for block in accepted:
        for execution in block["executions"]:
            grouped[execution["model_id"]].append(execution)
    rows: list[dict[str, Any]] = []
    means_by_block: dict[tuple[int, str], float] = {}
    for model_id, executions in grouped.items():
        block_means = [float(item["metrics"]["mean_ms"]) for item in executions]
        pooled_ns: list[int] = []
        for item in executions:
            latency_path = root / item["latencies_path"]
            with latency_path.open(newline="", encoding="utf-8") as handle:
                pooled_ns.extend(int(row["latency_ns"]) for row in csv.DictReader(handle))
            means_by_block[(int(item["block"]), model_id)] = float(item["metrics"]["mean_ms"])
        pooled = summarize_execution(pooled_ns)
        block_min, block_max = min(block_means), max(block_means)
        rows.append(
            {
                "model_id": model_id,
                "label": model_map[model_id].raw.get("label") or model_id,
                "input_configuration": model_map[model_id].raw.get("preprocessing") or {},
                "precision": str(json.loads((root / executions[0]["execution_path"]).read_text(encoding="utf-8"))["precision"]),
                "batch_size": 1,
                "median_block_mean_ms": float(statistics.median(block_means)),
                "block_mean_sample_sd_ms": float(statistics.stdev(block_means)) if len(block_means) > 1 else None,
                "block_mean_min_ms": block_min,
                "block_mean_max_ms": block_max,
                "block_mean_range_ms": block_max - block_min,
                "pooled_median_ms": pooled["median_ms"],
                "pooled_p90_ms": pooled["p90_ms"],
                "pooled_mean_ms": pooled["mean_ms"],
                "pooled_iqr_ms": pooled["iqr_ms"],
                "sequential_images_per_second": pooled["sequential_images_per_second"],
                "peak_allocated_bytes": max(int(item["peak_allocated_bytes"]) for item in executions),
                "peak_reserved_bytes": max(int(item["peak_reserved_bytes"]) for item in executions),
                "block_points_ms": block_means,
            }
        )
    reference = reference_model or (rows[0]["model_id"] if rows else None)
    ratios: list[dict[str, Any]] = []
    if reference:
        for row in rows:
            values = [
                means_by_block[(int(block["block"]), row["model_id"])] / means_by_block[(int(block["block"]), reference)]
                for block in schedule["blocks"]
            ]
            ratios.append(
                {
                    "model_id": row["model_id"],
                    "reference_model_id": reference,
                    "ratio_definition": "model block mean / reference block mean",
                    "median_ratio": float(statistics.median(values)),
                    "min_ratio": min(values),
                    "max_ratio": max(values),
                    "block_ratios": values,
                }
            )
    pilot_path = root / "warmup_pilot.json"
    pilot_document = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.is_file() else {}
    pilot_summary = {
        "passed": pilot_document.get("passed"),
        "skipped": bool(pilot_document.get("skipped", False)),
        "skip_reason": pilot_document.get("skip_reason"),
        "stability_required": pilot_document.get("stability_required"),
        "measurement_allowed": pilot_document.get("measurement_allowed"),
        "failed_model_ids": pilot_document.get("failed_model_ids", []),
        "maximum_adjacent_relative_change_by_model": {
            item["model_id"]: item.get("maximum_adjacent_relative_change")
            for item in pilot_document.get("models", [])
        },
        "artifact": "warmup_pilot.json",
    }
    result = {
        "schema_version": 1,
        "result_name": result_name,
        "warmup_pilot": pilot_summary,
        "primary_metric": "90th-percentile end-to-end embedding latency in milliseconds at B=1",
        "experimental_unit": "complete counterbalanced block",
        "models": rows,
        "paired_block_ratios": ratios,
    }
    atomic_write_json(root / "summary.json", result)
    _write_summary_csv(root / "summary.csv", rows)
    return result


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_id", "label", "input_configuration", "precision", "batch_size", "median_block_mean_ms",
        "block_mean_sample_sd_ms", "block_mean_min_ms", "block_mean_max_ms",
        "pooled_median_ms", "pooled_p90_ms", "pooled_mean_ms",
        "sequential_images_per_second", "peak_allocated_bytes", "peak_reserved_bytes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["input_configuration"] = json.dumps(row["input_configuration"], sort_keys=True)
            writer.writerow(serializable)


def _write_readme(run_dir: Path, result_name: str) -> None:
    text = f"""# {result_name}\n\nThis is an MLPerf-inspired Single Stream benchmark, **not an MLPerf-compliant run**.\nIt measures each complete model–software–hardware configuration and must not be\nreported as hardware-independent computational complexity. The primary metric is\npooled p90 end-to-end embedding latency at batch size one. Complete blocks are\nthe experimental repetitions. See `manifest.json`, `schedule.json`, `workload.json`,\n`summary.json`, and per-image CSV files for the auditable result.\n"""
    (run_dir / "README.md").write_text(text, encoding="utf-8")


def _assert_allocation(config: SingleStreamConfig) -> None:
    if os.getenv("SINGLE_STREAM_ALLOW_UNSAFE_SYSTEM") == "1":
        return
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Final benchmark must run inside one Slurm allocation")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Final benchmark requires exactly one visible CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    if str(config.protocol.get("precision")).lower() == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen benchmark requests BF16, but the selected GPU does not support BF16")
    expected = str(config.protocol.get("required_gpu_name") or "").strip()
    if not expected:
        raise RuntimeError("protocol.required_gpu_name must identify the frozen GPU model")
    minimum_gib = int(config.protocol.get("minimum_gpu_memory_gib", 1))
    minimum = minimum_gib * 1024**3
    if expected.lower() not in props.name.lower() or props.total_memory < minimum:
        raise RuntimeError(
            f"Expected one {expected} GPU with at least {minimum_gib} GiB, "
            f"got {props.name} ({props.total_memory} bytes)"
        )
    if os.getenv("SLURM_JOB_NUM_NODES") not in {None, "1"}:
        raise RuntimeError("Benchmark requires exactly one allocated node")
    if bool(config.protocol.get("require_exclusive_node", False)):
        job_id = os.getenv("SLURM_JOB_ID", "")
        job_details = _command_output(["scontrol", "show", "job", "-o", job_id])
        normalized = (job_details or "").lower()
        if "oversubscribe=exclusive" not in normalized and "shared=0" not in normalized:
            raise RuntimeError(
                "Frozen protocol requires an exclusive node, but Slurm did not "
                f"report an exclusive allocation: {job_details or 'unavailable'}"
            )
    if bool(config.protocol.get("require_node_local_storage", True)):
        tmp_value = os.getenv("SLURM_TMPDIR")
        if not tmp_value:
            raise RuntimeError("Node-local benchmark requires SLURM_TMPDIR")
        tmp = Path(tmp_value).resolve()
        required_paths = [config.dataset_db, config.image_root, Path(os.path.expandvars(os.getenv("HF_HOME", ""))).expanduser()]
        for model in config.models:
            checkpoint = model.adapter.kwargs.get("checkpoint_path")
            if checkpoint:
                required_paths.append(_expand_path(checkpoint, config.repo_root))
        outside = [str(path) for path in required_paths if not path.resolve().is_relative_to(tmp)]
        if outside:
            raise RuntimeError(f"Dataset, checkpoints, and model cache must be staged under SLURM_TMPDIR: {outside}")


def run_single_stream_benchmark(config_path: str | Path, *, run_id: str | None = None, resume: bool = False) -> dict[str, Any]:
    config = load_single_stream_config(config_path)
    _assert_allocation(config)
    configured_output = _expand_path(config.run.get("output_dir", f"experiments/runs/{config.benchmark_id}"), config.repo_root)
    root = configured_output / (run_id or make_run_id())
    if root.exists() and not resume:
        raise FileExistsError(f"Benchmark run already exists: {root}; pass --resume to continue it")
    root.mkdir(parents=True, exist_ok=True)
    workload_path, schedule_path, manifest_path = root / "workload.json", root / "schedule.json", root / "manifest.json"
    if resume and manifest_path.is_file():
        workload = json.loads(workload_path.read_text(encoding="utf-8"))["images"]
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != sha256_file(config.path):
            raise RuntimeError("Cannot resume: benchmark config differs from frozen manifest")
        if manifest.get("source_code_sha256") != source_code_hashes(config.repo_root):
            raise RuntimeError("Cannot resume: benchmark source code differs from frozen manifest")
        for model, frozen in zip(config.models, manifest.get("models") or []):
            if frozen.get("model_id") != model.model_id or frozen.get("model_config_sha256") != sha256_file(model.path):
                raise RuntimeError("Cannot resume: a referenced model configuration differs from the frozen manifest")
    else:
        workload = select_workload(config)
        schedule = build_schedule(
            [model.model_id for model in config.models],
            cycle_seeds=[int(value) for value in config.protocol["cycle_seeds"]],
            image_order_seed=int(config.protocol.get("image_order_seed", 20260728)),
        )
        validate_schedule_balance(schedule)
        atomic_write_json(workload_path, {"schema_version": 1, "images": workload})
        atomic_write_json(schedule_path, schedule)
        manifest = build_manifest(config, workload, schedule)
        atomic_write_json(manifest_path, manifest)
        try:
            manifest_path.chmod(0o444)
        except OSError:
            pass
        _write_readme(root, config.result_name)

    pilot_path = root / "warmup_pilot.json"
    skip_pilot = bool(config.protocol.get("skip_warmup_pilot", False))
    if not pilot_path.is_file() and skip_pilot:
        pilot = {
            "schema_version": 1,
            "passed": None,
            "skipped": True,
            "skip_reason": "predeclared protocol configuration",
            "stability_required": False,
            "measurement_allowed": True,
            "failed_model_ids": [],
            "models": [],
        }
        atomic_write_json(pilot_path, pilot)
    elif not pilot_path.is_file():
        pilot_results = []
        for model in config.models:
            output = root / "pilot" / model.storage_key
            result = _launch_worker(config, root, model.model_id, mode="pilot", output_dir=output)
            pilot_results.append(result)
        stability_passed = all(item["passed"] for item in pilot_results)
        stability_required = bool(config.protocol.get("warmup_stability_required", True))
        pilot = {
            "schema_version": 1,
            "passed": stability_passed,
            "skipped": False,
            "stability_required": stability_required,
            "measurement_allowed": stability_passed or not stability_required,
            "failed_model_ids": [item["model_id"] for item in pilot_results if not item["passed"]],
            "models": pilot_results,
        }
        atomic_write_json(pilot_path, pilot)
        if not pilot["measurement_allowed"]:
            raise RuntimeError("Warm-up pilot failed its predeclared stability condition; increase warmup_queries for all models and start a new frozen run")
    else:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        stability_required = bool(config.protocol.get("warmup_stability_required", True))
        if not pilot.get("skipped") and not pilot.get("passed") and stability_required:
            raise RuntimeError("Recorded warm-up pilot did not pass")

    max_attempts = int(config.protocol.get("max_block_attempts", 2))
    for block in schedule["blocks"]:
        block_root = root / "blocks" / f"block-{int(block['block']):03d}"
        accepted_path = block_root / "accepted.json"
        if accepted_path.is_file():
            continue
        failures: list[dict[str, Any]] = []
        accepted = None
        for attempt in range(1, max_attempts + 1):
            attempt_root = block_root / f"attempt-{attempt:02d}"
            executions = []
            try:
                for position, model_id in enumerate(block["models"], start=1):
                    model = next(item for item in config.models if item.model_id == model_id)
                    output = attempt_root / f"position-{position:02d}_{model.storage_key}"
                    result = _launch_worker(
                        config, root, model_id, mode="measure", output_dir=output,
                        block=int(block["block"]), position=position,
                        predecessor=block["models"][position - 2] if position > 1 else None,
                        image_order_seed=int(block["image_order_seed"]),
                    )
                    executions.append(result)
                accepted = {"block": block["block"], "attempt": attempt, "image_order_seed": block["image_order_seed"], "models": block["models"], "executions": executions}
                break
            except Exception as exc:
                failures.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}", "completed_executions": executions})
                atomic_write_json(attempt_root / "invalid.json", failures[-1])
        if accepted is None:
            atomic_write_json(block_root / "failures.json", failures)
            raise RuntimeError(f"Block {block['block']} failed {max_attempts} complete attempts")
        atomic_write_json(accepted_path, accepted)
    result = aggregate_results(
        root,
        schedule,
        config.models,
        config.run.get("reference_model_id"),
        result_name=config.result_name,
    )
    atomic_write_json(root / "run.json", {"status": "completed", "result_name": config.result_name, "manifest_sha256": manifest["manifest_sha256"], "completed_at": datetime.now(timezone.utc).isoformat(), "summary": "summary.json"})
    return {"run_dir": str(root), "summary": result}


def _launch_worker(
    config: SingleStreamConfig,
    run_dir: Path,
    model_id: str,
    *,
    mode: str,
    output_dir: Path,
    block: int | None = None,
    position: int | None = None,
    predecessor: str | None = None,
    image_order_seed: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "experiments.core.single_stream_worker",
        "--config", str(config.path), "--run-dir", str(run_dir),
        "--model", model_id, "--mode", mode, "--output-dir", str(output_dir),
    ]
    if block is not None:
        command += ["--block", str(block), "--position", str(position), "--image-order-seed", str(image_order_seed)]
    if predecessor:
        command += ["--predecessor", predecessor]
    env = os.environ.copy()
    threads = str(int(config.protocol.get("cpu_threads", 1)))
    env.update({"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads, "TOKENIZERS_PARALLELISM": "false"})
    completed = subprocess.run(command, cwd=config.repo_root, env=env, text=True, capture_output=True)
    (output_dir / "worker.stdout").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "worker.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Worker failed for {model_id} (exit {completed.returncode}); see {output_dir}")
    result_path = output_dir / ("pilot.json" if mode == "pilot" else "execution.json")
    if not result_path.is_file():
        raise RuntimeError(f"Worker did not produce {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mode == "measure":
        result["execution_path"] = str(result_path.relative_to(run_dir))
        result["latencies_path"] = str((output_dir / "latencies.csv").relative_to(run_dir))
    return result
