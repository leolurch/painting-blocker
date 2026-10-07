"""Resource and runtime metric helpers for blocking experiments."""

from __future__ import annotations

import gc
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight environments
    torch = None


class ResourceMonitor:
    """Sample wall time, PyTorch CUDA peaks, and nvidia-smi GPU memory."""

    def __init__(self, sample_interval_seconds: float = 0.5):
        self.sample_interval_seconds = sample_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[list[dict[str, Any]]] = []
        self._started = 0.0
        self._ended = 0.0

    def __enter__(self) -> "ResourceMonitor":
        self._started = time.perf_counter()
        _reset_torch_peaks()
        self._samples.append(query_gpu_snapshot())
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 2))
        self._samples.append(query_gpu_snapshot())
        self._ended = time.perf_counter()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._samples.append(query_gpu_snapshot())

    def summary(self) -> dict[str, Any]:
        samples = [gpu for snapshot in self._samples for gpu in snapshot]
        peaks: dict[int, dict[str, Any]] = {}
        for gpu in samples:
            index = int(gpu.get("index", 0))
            current = peaks.setdefault(index, dict(gpu))
            if int(gpu.get("memory_used_bytes") or 0) > int(current.get("memory_used_bytes") or 0):
                peaks[index] = dict(gpu)
        peak_used = max((int(gpu.get("memory_used_bytes") or 0) for gpu in peaks.values()), default=None)
        return {
            "wall_time_seconds": max(0.0, self._ended - self._started),
            "gpu_memory": {
                "nvidia_smi_peak_used_bytes": peak_used,
                "nvidia_smi_peak_per_gpu": list(peaks.values()),
                "nvidia_smi_sample_count": len(self._samples),
                **torch_cuda_memory(),
            },
        }


def _reset_torch_peaks() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass


def torch_cuda_memory() -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {"torch_cuda_available": False}
    per_gpu = []
    for idx in range(torch.cuda.device_count()):
        per_gpu.append(
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(idx)),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(idx)),
            }
        )
    return {"torch_cuda_available": True, "torch_peak_per_gpu": per_gpu}


def query_gpu_snapshot() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=3)
    except Exception:
        return _torch_gpu_snapshot()
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_bytes": int(float(parts[2])) * 1024 * 1024,
                "memory_total_bytes": int(float(parts[3])) * 1024 * 1024,
            }
        )
    return rows


def _torch_gpu_snapshot() -> list[dict[str, Any]]:
    if torch is None or not torch.cuda.is_available():
        return []
    return [
        {
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "memory_used_bytes": int(torch.cuda.memory_reserved(idx)),
            "memory_total_bytes": None,
        }
        for idx in range(torch.cuda.device_count())
    ]


def system_profile() -> dict[str, Any]:
    return {
        "cpu_cores": os.cpu_count(),
        "gpu": query_gpu_snapshot(),
        "memory_total_bytes": _system_memory_total_bytes(),
        "worker_threads": _worker_threads(),
        "environment": {
            key: os.getenv(key)
            for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            if os.getenv(key) is not None
        },
    }


def _system_memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (ValueError, OSError, AttributeError):
        return None


def _worker_threads() -> dict[str, int | None]:
    return {
        "torch_num_threads": torch.get_num_threads() if torch is not None else None,
        "torch_num_interop_threads": torch.get_num_interop_threads() if torch is not None else None,
        "slurm_cpus_per_task": _int_env("SLURM_CPUS_PER_TASK"),
        "omp_num_threads": _int_env("OMP_NUM_THREADS"),
        "mkl_num_threads": _int_env("MKL_NUM_THREADS"),
    }


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value and value.isdigit() else None


def release_torch_cuda_memory(reset_compiler: bool = False) -> None:
    """Best-effort release of idle PyTorch CUDA memory between model runs."""
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    if reset_compiler:
        _reset_torch_compiler()
        gc.collect()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    for idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _reset_torch_compiler() -> None:
    compiler_reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(compiler_reset):
        try:
            compiler_reset()
            return
        except Exception:
            pass
    dynamo_reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
    if callable(dynamo_reset):
        try:
            dynamo_reset()
        except Exception:
            pass


def model_runtime_profile(adapter: object) -> dict[str, Any]:
    cls = adapter.__class__
    size_key = getattr(adapter, "size_key", None)
    size_label = _size_label(adapter, size_key)
    exact_params = _parameter_count(getattr(adapter, "model", None))
    estimated_params = _parse_parameter_count(size_label or "") or _parse_parameter_count(getattr(adapter, "model_id", ""))
    autocast_dtype = getattr(adapter, "autocast_dtype", None)
    if autocast_dtype is not None:
        autocast_dtype = str(autocast_dtype).removeprefix("torch.")
    return {
        "inference_engine": _inference_engine(cls),
        "adapter_class": f"{cls.__module__}.{cls.__name__}",
        "model_size_label": size_label,
        "model_size_key": size_key,
        "parameter_count": exact_params or estimated_params,
        "parameter_count_source": "torch_parameters" if exact_params else ("size_label_estimate" if estimated_params else None),
        "parameter_dtypes": _parameter_dtypes(getattr(adapter, "model", None)),
        "autocast_enabled": bool(getattr(adapter, "autocast_enabled", False)),
        "autocast_dtype": autocast_dtype,
        "intrinsic_normalization": bool(getattr(adapter, "intrinsic_normalization", False)),
        "intrinsic_normalization_dtype": getattr(adapter, "intrinsic_normalization_dtype", None),
        "embedding_dim": _safe_dimension(adapter),
    }


def _inference_engine(cls: type) -> str:
    name = f"{cls.__module__}.{cls.__name__}".lower()
    if "vllm" in name:
        return "vLLM"
    if "open_clip" in name or "openclip" in name:
        return "open_clip_torch"
    return "PyTorch/Transformers"


def _size_label(adapter: object, size_key: Any) -> str | None:
    if size_key is not None and hasattr(adapter.__class__, "size_label"):
        try:
            return str(adapter.__class__.size_label(size_key))
        except Exception:
            return None
    return None


def _parameter_dtypes(model: object | None) -> list[str]:
    if model is None or not hasattr(model, "parameters"):
        return []
    model = getattr(model, "_orig_mod", model)
    try:
        return sorted({str(param.dtype).removeprefix("torch.") for param in model.parameters()})
    except Exception:
        return []


def _parameter_count(model: object | None) -> int | None:
    if model is None or not hasattr(model, "parameters"):
        return None
    model = getattr(model, "_orig_mod", model)
    try:
        return int(sum(param.numel() for param in model.parameters()))
    except Exception:
        return None


def _parse_parameter_count(value: str) -> int | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])\b", value)
    if not match:
        return None
    scale = 1_000_000_000 if match.group(2).lower() == "b" else 1_000_000
    return int(float(match.group(1)) * scale)


def _safe_dimension(adapter: object) -> int | None:
    if not hasattr(adapter, "get_dimension"):
        return None
    try:
        return int(adapter.get_dimension())
    except Exception:
        return None
