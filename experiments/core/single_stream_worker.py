"""Fresh-process worker for one single-stream model execution."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .artifacts import atomic_write_json, sha256_file, sha256_json
from .descriptor_postprocessing import validate_descriptors
from .embedding_pipeline import instantiate_model_adapter
from .single_stream_benchmark import image_permutation, load_single_stream_config, source_code_hashes, summarize_execution


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("pilot", "measure"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--block", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--predecessor")
    parser.add_argument("--image-order-seed", type=int)
    return parser


def _nvidia_target() -> str:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    first = visible.split(",")[0].strip() if visible else ""
    return first if first and first not in {"NoDevFiles", "none", "void"} else "0"


def _nvidia_query(fields: list[str]) -> dict[str, Any]:
    command = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", "-i", _nvidia_target()]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()[0]
        values = next(csv.reader([output], skipinitialspace=True))
        return {field: value.strip() for field, value in zip(fields, values)}
    except Exception as exc:
        return {"query_error": f"{type(exc).__name__}: {exc}"}


def _gpu_processes() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits", "-i", _nvidia_target()]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return []
    rows = []
    for line in output.splitlines():
        values = next(csv.reader([line], skipinitialspace=True))
        if len(values) >= 3 and values[0].strip().isdigit():
            rows.append({"pid": int(values[0]), "process_name": values[1].strip(), "used_memory_mib": values[2].strip()})
    return rows


def _system_state() -> dict[str, Any]:
    fields = [
        "uuid", "name", "temperature.gpu", "clocks.gr", "clocks.mem",
        "power.draw", "power.limit", "utilization.gpu", "utilization.memory",
        "memory.used", "memory.total", "compute_mode",
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": _nvidia_query(fields),
        "ecc": _nvidia_query([
            "ecc.errors.corrected.aggregate.total",
            "ecc.errors.uncorrected.aggregate.total",
        ]),
        "gpu_compute_processes": _gpu_processes(),
        "cpu_load_1_5_15": list(os.getloadavg()),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def _assert_no_other_gpu_process(state: dict[str, Any]) -> None:
    other = [item for item in state["gpu_compute_processes"] if int(item["pid"]) != os.getpid()]
    if other:
        raise RuntimeError(f"Another compute process is using the benchmark GPU: {other}")


def _ecc_total(state: dict[str, Any]) -> int | None:
    values = []
    for value in (state.get("ecc") or {}).values():
        try:
            values.append(int(str(value).strip()))
        except ValueError:
            continue
    return sum(values) if values else None


def _decode_images(workload: list[dict[str, Any]]) -> list[Image.Image]:
    decoded: list[Image.Image] = []
    for item in workload:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Benchmark image changed after manifest freeze: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
            decoded.append(image.copy())
    return decoded


def _representative_indices(workload: list[dict[str, Any]], count: int) -> list[int]:
    by_area = sorted(range(len(workload)), key=lambda index: int(workload[index]["width"]) * int(workload[index]["height"]))
    quantile_count = min(10, len(by_area), count)
    anchors = [by_area[int(round(value))] for value in np.linspace(0, len(by_area) - 1, quantile_count)]
    return [anchors[index % len(anchors)] for index in range(count)]


def _configure_pytorch_sdpa(policy: str) -> dict[str, bool]:
    """Freeze the allowed built-in PyTorch SDPA kernel set for this worker."""
    if not hasattr(torch.backends.cuda, "enable_flash_sdp"):
        raise RuntimeError("Installed PyTorch does not expose CUDA SDPA backend controls")
    normalized = str(policy).lower()
    if normalized == "pytorch_sdpa_flash":
        flash, efficient, math, cudnn = True, False, False, False
    elif normalized in {
        "pytorch_sdpa_mem_efficient",
        "pytorch_sdpa_mem_efficient_where_applicable",
    }:
        flash, efficient, math, cudnn = False, True, False, False
    elif normalized == "pytorch_sdpa_auto_where_applicable":
        # These are PyTorch's built-in kernels. This does not import or select
        # the external flash-attn package used by attn_implementation=flash_attention_2.
        flash, efficient, math, cudnn = True, True, True, True
    else:
        raise ValueError(f"Unsupported attention_implementation policy: {policy!r}")
    torch.backends.cuda.enable_flash_sdp(flash)
    torch.backends.cuda.enable_mem_efficient_sdp(efficient)
    torch.backends.cuda.enable_math_sdp(math)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(cudnn)
    return {
        "flash": flash,
        "memory_efficient": efficient,
        "math": math,
        "cudnn": cudnn if hasattr(torch.backends.cuda, "enable_cudnn_sdp") else False,
    }


def _precision_context(precision: str):
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _normalize_fp32_unchecked(raw: np.ndarray) -> np.ndarray:
    """Apply the canonical scale-first FP32 L2 math without timed assertions."""
    values = np.array(raw, dtype=np.float32, order="C", copy=True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scales = np.max(np.abs(values), axis=1, keepdims=True)
        scaled = values / scales
        norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True, dtype=np.float32))
        return scaled / norms


def _query(adapter: object, image: Image.Image, pooling: tuple[str, ...], precision: str) -> np.ndarray:
    # Passing a one-element list is intentional: it exercises the exact adapter
    # preprocessing/forward/pooling path while proving that no larger batch can
    # be formed internally by this worker.
    with torch.inference_mode(), _precision_context(precision):
        result = adapter.generate_embeddings_batch_from_pil([image], pooling)
        missing = [mode for mode in pooling if mode not in result]
        if missing:
            raise ValueError(f"Adapter omitted pooling outputs: {missing}")
        raw = np.asarray(result[pooling[0]])
        embedding = _normalize_fp32_unchecked(raw)
    torch.cuda.synchronize()
    return embedding


def _lora_merge_state(adapter: object) -> bool | str:
    model = getattr(adapter, "model", None)
    if not str(getattr(model, "model_type", "")).startswith("dinov3_lora"):
        return "not_applicable"
    merged = []
    for module in model.modules():
        value = getattr(module, "merged", None)
        if isinstance(value, bool):
            merged.append(value)
    if any(merged):
        return True
    return False


def _attention_implementations(adapter: object) -> list[str]:
    model = getattr(adapter, "model", None)
    if model is None:
        return []
    values: set[str] = set()
    modules = model.modules() if hasattr(model, "modules") else (model,)
    for module in modules:
        config = getattr(module, "config", None)
        if config is None:
            continue
        for name in ("_attn_implementation", "_attn_implementation_internal"):
            value = getattr(config, name, None)
            if value:
                values.add(str(value).lower())
    return sorted(values)


def _correctness_check(adapter: object, image: Image.Image, pooling: tuple[str, ...], precision: str) -> dict[str, Any]:
    embedding = _query(adapter, image, pooling, precision)
    expected_dim = int(adapter.get_dimension())
    if embedding.shape != (1, expected_dim):
        raise ValueError(f"Expected descriptor shape (1, {expected_dim}), got {embedding.shape}")
    validate_descriptors(embedding, name="correctness descriptor", require_unit_norm=True)
    return {"descriptor_count": 1, "descriptor_dim": expected_dim, "finite": True, "unit_norm": True}


def _time_queries(
    adapter: object,
    images: list[Image.Image],
    indices: list[int],
    pooling: tuple[str, ...],
    precision: str,
) -> list[int]:
    torch.cuda.synchronize()
    latencies: list[int] = []
    for index in indices:
        started = time.perf_counter_ns()
        embedding = _query(adapter, images[index], pooling, precision)
        ended = time.perf_counter_ns()
        latencies.append(ended - started)
        # Correctness validation is intentionally outside the measured interval.
        expected_dim = int(adapter.get_dimension())
        if embedding.shape != (1, expected_dim):
            raise ValueError(f"Expected descriptor shape (1, {expected_dim}), got {embedding.shape}")
        validate_descriptors(embedding, name="timed descriptor", require_unit_norm=True)
        del embedding
    return latencies


def _pilot_result(latencies_ns: list[int], tolerance: float, window_queries: int) -> dict[str, Any]:
    values = np.asarray(latencies_ns, dtype=np.float64) / 1_000_000.0
    required = 3 * int(window_queries)
    if values.size < required:
        raise ValueError(f"Pilot requires at least {required} latency samples")
    start = int(values.size) - required
    windows: list[dict[str, Any]] = []
    for index in range(3):
        left = start + index * window_queries
        right = left + window_queries
        windows.append(
            {
                "window": index + 1,
                "query_start": left + 1,
                "query_end": right,
                "median_ms": float(np.median(values[left:right])),
            }
        )
    changes = [
        abs(windows[index + 1]["median_ms"] - windows[index]["median_ms"])
        / windows[index]["median_ms"]
        for index in range(2)
    ]
    return {
        "stability_window_queries": window_queries,
        "stability_windows": windows,
        "adjacent_relative_changes": changes,
        "maximum_adjacent_relative_change": max(changes),
        "tolerance": tolerance,
        "passed": max(changes) < tolerance,
    }


def _write_latencies(path: Path, workload: list[dict[str, Any]], order: list[int], latencies_ns: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("query_index", "workload_index", "image_id", "latency_ns", "latency_ms"))
        writer.writeheader()
        for query_index, (workload_index, latency) in enumerate(zip(order, latencies_ns), start=1):
            writer.writerow({
                "query_index": query_index,
                "workload_index": workload_index,
                "image_id": workload[workload_index]["image_id"],
                "latency_ns": latency,
                "latency_ms": latency / 1_000_000.0,
            })
    os.replace(temporary, path)


def main() -> None:
    args = _parser().parse_args()
    config = load_single_stream_config(args.config)
    model = next((item for item in config.models if item.model_id == args.model), None)
    if model is None:
        raise ValueError(f"Unknown benchmark model: {args.model}")
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    recorded_manifest_hash = manifest.get("manifest_sha256")
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_sha256", None)
    if recorded_manifest_hash != sha256_json(manifest_payload):
        raise RuntimeError("Frozen manifest content hash is invalid")
    if manifest["config_sha256"] != sha256_file(config.path):
        raise RuntimeError("Worker configuration differs from frozen manifest")
    if manifest.get("source_code_sha256") != source_code_hashes(config.repo_root):
        raise RuntimeError("Worker source code differs from frozen manifest")
    frozen_models = manifest.get("models") or []
    for current, frozen in zip(config.models, frozen_models):
        if frozen.get("model_id") != current.model_id or frozen.get("model_config_sha256") != sha256_file(current.path):
            raise RuntimeError("A referenced model configuration differs from frozen manifest")
    if len(frozen_models) != len(config.models):
        raise RuntimeError("Frozen manifest model count differs from current configuration")
    workload = json.loads((args.run_dir / "workload.json").read_text(encoding="utf-8"))["images"]
    schedule = json.loads((args.run_dir / "schedule.json").read_text(encoding="utf-8"))
    if manifest["dataset"]["workload_sha256"] != sha256_json(workload):
        raise RuntimeError("Workload differs from frozen manifest")
    if manifest["schedule_sha256"] != sha256_json(schedule):
        raise RuntimeError("Schedule differs from frozen manifest")
    if args.mode == "measure":
        if args.block is None or args.position is None or args.image_order_seed is None:
            raise ValueError("Measurement worker requires block, position, and image-order seed")
        scheduled = next((item for item in schedule["blocks"] if int(item["block"]) == args.block), None)
        valid_position = scheduled is not None and 1 <= args.position <= len(scheduled["models"])
        expected_model = scheduled["models"][args.position - 1] if valid_position else None
        expected_predecessor = scheduled["models"][args.position - 2] if valid_position and args.position > 1 else None
        if (
            not valid_position
            or expected_model != args.model
            or int(scheduled["image_order_seed"]) != args.image_order_seed
            or expected_predecessor != args.predecessor
        ):
            raise RuntimeError("Worker assignment differs from the frozen counterbalanced schedule")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    threads = int(config.protocol.get("cpu_threads", 1))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(threads, 4)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    sdpa_backends_enabled = _configure_pytorch_sdpa(
        str(config.protocol["attention_implementation"])
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_state = _system_state()
    _assert_no_other_gpu_process(started_state)
    expected_uuid = ((manifest.get("hardware") or {}).get("selected_gpu") or {}).get("uuid")
    if not expected_uuid or (started_state.get("gpu") or {}).get("uuid") != expected_uuid:
        raise RuntimeError(
            f"Worker GPU UUID {(started_state.get('gpu') or {}).get('uuid')!r} "
            f"differs from frozen manifest UUID {expected_uuid!r}"
        )
    load_started = time.perf_counter_ns()
    adapter, pooling, adapter_meta = instantiate_model_adapter(
        model,
        {"no_compile": not bool(config.protocol["torch_compile"])},
    )
    model_load_ns = time.perf_counter_ns() - load_started
    decoded_started = time.perf_counter_ns()
    images = _decode_images(workload)
    decode_ns = time.perf_counter_ns() - decoded_started
    precision = str(config.protocol["precision"]).lower()
    representative = _representative_indices(workload, max(int(config.protocol["pilot_warmup_queries"]), int(config.protocol["warmup_queries"])))
    correctness = _correctness_check(adapter, images[representative[0]], pooling, precision)
    descriptor_tensor_dtypes = list(getattr(adapter, "descriptor_tensor_dtypes", []))
    if not descriptor_tensor_dtypes:
        raise RuntimeError("Adapter did not report the pre-FP32 descriptor tensor dtype")
    actual_attention = _attention_implementations(adapter)
    actual_lora_merged = _lora_merge_state(adapter)
    wrapped_model = getattr(adapter, "model", None)
    compile_actual = wrapped_model is not None and hasattr(wrapped_model, "_orig_mod")
    if bool(config.protocol["torch_compile"]) != compile_actual:
        raise RuntimeError(
            f"Frozen manifest requested torch_compile={config.protocol['torch_compile']}, "
            f"but adapter actual state was {compile_actual}"
        )
    if actual_lora_merged is True or (actual_lora_merged is False and config.protocol["lora_weights_merged"] is not False):
        raise RuntimeError(
            f"Runtime LoRA merge state {actual_lora_merged!r} differs from the frozen unmerged policy"
        )
    adapter_name = model.adapter.adapter_name
    attention_policy = str(config.protocol["attention_implementation"]).lower()
    is_resnet = adapter_name == "resnet50_hf_adapter"
    is_open_clip = adapter_name == "open_clip_adapter"
    if is_open_clip:
        if attention_policy not in {
            "pytorch_sdpa_mem_efficient_where_applicable",
            "pytorch_sdpa_auto_where_applicable",
        }:
            raise RuntimeError(
                "OpenCLIP's evaluated native adapter requires an explicit "
                "where-applicable PyTorch SDPA policy"
            )
        if not actual_attention:
            actual_attention = ["open_clip_native"]
    elif not is_resnet:
        if attention_policy in {
            "pytorch_sdpa_mem_efficient_where_applicable",
            "pytorch_sdpa_auto_where_applicable",
        }:
            valid_attention = "sdpa" in actual_attention
        else:
            valid_attention = actual_attention == ["sdpa"]
        if not valid_attention:
            raise RuntimeError(
                "Frozen benchmark requires PyTorch SDPA for applicable Hugging Face attention modules, "
                f"but runtime model configs reported {actual_attention or ['unknown']}"
            )

    if args.mode == "pilot":
        count = int(config.protocol["pilot_warmup_queries"])
        latencies = _time_queries(adapter, images, representative[:count], pooling, precision)
        pilot_ended_state = _system_state()
        _assert_no_other_gpu_process(pilot_ended_state)
        if (started_state.get("gpu") or {}).get("uuid") != (pilot_ended_state.get("gpu") or {}).get("uuid"):
            raise RuntimeError("GPU UUID changed during warm-up pilot")
        ecc_start, ecc_end = _ecc_total(started_state), _ecc_total(pilot_ended_state)
        if ecc_start is not None and ecc_end is not None and ecc_end > ecc_start:
            raise RuntimeError(f"ECC error count increased during warm-up pilot: {ecc_start} -> {ecc_end}")
        result = {
            "model_id": model.model_id,
            "label": model.raw.get("label") or model.model_id,
            "query_count": count,
            "latencies_ns": latencies,
            "correctness": correctness,
            "precision": precision,
            "descriptor_tensor_dtypes": descriptor_tensor_dtypes,
            "runtime_attention_implementations": actual_attention or ["not_applicable"],
            "pytorch_sdpa_backends_enabled": sdpa_backends_enabled,
            "external_flash_attention": "flash_attention_2" in actual_attention,
            "torch_compile": compile_actual,
            "lora_weights_merged": actual_lora_merged,
            "system_state_start": started_state,
            "system_state_end": pilot_ended_state,
            **_pilot_result(
                latencies,
                float(config.protocol.get("warmup_stability_tolerance", 0.01)),
                int(config.protocol.get("pilot_stability_window_queries", 20)),
            ),
        }
        atomic_write_json(output / "pilot.json", result)
        return

    warmup_count = int(config.protocol["warmup_queries"])
    _time_queries(adapter, images, representative[:warmup_count], pooling, precision)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    order = image_permutation(len(images), args.image_order_seed)
    measured_started = datetime.now(timezone.utc).isoformat()
    latencies = _time_queries(adapter, images, order, pooling, precision)
    measured_ended = datetime.now(timezone.utc).isoformat()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    ended_state = _system_state()
    _assert_no_other_gpu_process(ended_state)
    if (started_state.get("gpu") or {}).get("uuid") != (ended_state.get("gpu") or {}).get("uuid"):
        raise RuntimeError("GPU UUID changed during one model execution")
    ecc_start, ecc_end = _ecc_total(started_state), _ecc_total(ended_state)
    if ecc_start is not None and ecc_end is not None and ecc_end > ecc_start:
        raise RuntimeError(f"ECC error count increased during execution: {ecc_start} -> {ecc_end}")
    _write_latencies(output / "latencies.csv", workload, order, latencies)
    metrics = summarize_execution(latencies)
    result = {
        "schema_version": 1,
        "model_id": model.model_id,
        "label": model.raw.get("label") or model.model_id,
        "block": args.block,
        "position": args.position,
        "predecessor": args.predecessor,
        "image_order_seed": args.image_order_seed,
        "query_count": len(latencies),
        "batch_size": 1,
        "precision": precision,
        "descriptor_tensor_dtypes": descriptor_tensor_dtypes,
        "attention_implementation": config.protocol["attention_implementation"],
        "runtime_attention_implementations": actual_attention or ["not_applicable"],
        "pytorch_sdpa_backends_enabled": sdpa_backends_enabled,
        "external_flash_attention": "flash_attention_2" in actual_attention,
        "torch_compile": compile_actual,
        "lora_weights_merged": actual_lora_merged,
        "adapter": adapter_meta,
        "correctness": correctness,
        "model_load_ns": model_load_ns,
        "image_decode_ns_excluded": decode_ns,
        "warmup_queries_excluded": warmup_count,
        "measured_started": measured_started,
        "measured_ended": measured_ended,
        "metrics": metrics,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "system_state_start": started_state,
        "system_state_end": ended_state,
        "invalidation_checks": {
            "other_gpu_process": False,
            "finite_descriptors": True,
            "descriptor_count_and_dimension": True,
            "configuration_matches_manifest": True,
            "gpu_uuid_stable": True,
            "ecc_error_count_increased": False,
        },
    }
    atomic_write_json(output / "execution.json", result)


if __name__ == "__main__":
    main()
