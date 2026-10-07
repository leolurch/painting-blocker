"""Live console logging helpers for finetuning runs."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, TextIO

import torch


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ConsoleTrainingLogger:
    """Emit concise live progress logs suitable for Slurm stdout files."""

    def __init__(
        self,
        *,
        enabled: bool,
        heartbeat_seconds: float = 120.0,
        device: str = "cpu",
        log_gpu: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self.device = str(device)
        self.log_gpu = bool(log_gpu)
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = time.monotonic()
        self._epoch_start_time: float | None = None
        self._epoch_durations: list[float] = []
        self._loss_ema: float | None = None
        self._state: dict[str, Any] = {
            "phase": "setup",
            "epoch": 0,
            "epochs": 0,
            "batch": 0,
            "batches_per_epoch": 0,
            "val_batch": 0,
            "val_batches": 0,
            "global_step": 0,
            "total_steps": 0,
            "samples_seen": 0,
            "last_loss": None,
            "loss_epoch_sum": 0.0,
            "loss_epoch_count": 0,
            "lr": None,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, *, device: str) -> "ConsoleTrainingLogger":
        cfg = dict(config or {})
        enabled_default = bool(os.environ.get("SLURM_JOB_ID"))
        return cls(
            enabled=bool(cfg.get("console", enabled_default)),
            heartbeat_seconds=float(cfg.get("heartbeat_seconds", 120.0)),
            device=device,
            log_gpu=bool(cfg.get("log_gpu_memory", cfg.get("log_gpu", True))),
        )

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="finetune-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def configure(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = " ".join(f"{key}={self._format_value(value)}" for key, value in fields.items() if value is not None)
        line = f"[{now}] {event}" + (f" {payload}" if payload else "")
        print(line, file=self.stream, flush=True)

    def epoch_start(self, epoch: int, epochs: int, lr: float | None) -> None:
        with self._lock:
            self._epoch_start_time = time.monotonic()
            self._loss_ema = None
            self._state.update(
                {
                    "phase": "train",
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "batch": 0,
                    "val_batch": 0,
                    "loss_epoch_sum": 0.0,
                    "loss_epoch_count": 0,
                    "lr": lr,
                }
            )
        self.emit("epoch_start", epoch=f"{epoch}/{epochs}", global_step=self._state_snapshot()["global_step"], lr=lr)

    def update_train(self, *, epoch: int, batch: int, global_step: int, loss: float, lr: float | None, samples: int) -> None:
        with self._lock:
            self._loss_ema = loss if self._loss_ema is None else 0.9 * self._loss_ema + 0.1 * loss
            self._state.update(
                {
                    "phase": "train",
                    "epoch": int(epoch),
                    "batch": int(batch),
                    "global_step": int(global_step),
                    "samples_seen": int(self._state.get("samples_seen", 0)) + int(samples),
                    "last_loss": float(loss),
                    "loss_ema": float(self._loss_ema),
                    "loss_epoch_sum": float(self._state.get("loss_epoch_sum", 0.0)) + float(loss),
                    "loss_epoch_count": int(self._state.get("loss_epoch_count", 0)) + 1,
                    "lr": lr,
                }
            )

    def validation_start(self, epoch: int, val_batches: int, *, name: str | None = None) -> None:
        self.configure(phase="validation", epoch=int(epoch), val_batch=0, val_batches=int(val_batches), validation_set=name)
        self.emit("validation_start", epoch=epoch, validation_set=name, batches=val_batches)

    def update_validation(self, batch: int) -> None:
        self.configure(phase="validation", val_batch=int(batch))

    def checkpoint_start(self, epoch: int) -> None:
        self.configure(phase="checkpoint", epoch=int(epoch))

    def epoch_end(self, *, epoch: int, epochs: int, train_loss: float, metrics: dict[str, Any], new_best: bool) -> None:
        now = time.monotonic()
        with self._lock:
            duration = now - self._epoch_start_time if self._epoch_start_time is not None else None
            if duration is not None:
                self._epoch_durations.append(duration)
            self._state.update({"phase": "epoch_end", "epoch": int(epoch), "batch": self._state.get("batches_per_epoch", 0)})
        calibrated = dict(metrics.get("calibrated_threshold") or {})
        recalls = {int(row["k"]): row for row in metrics.get("recall_at_k", []) if "k" in row}
        self.emit(
            "epoch_end",
            epoch=f"{epoch}/{epochs}",
            duration=format_duration(duration),
            eta_total=format_duration(self._eta_seconds()),
            train_loss=train_loss,
            threshold=calibrated.get("threshold"),
            pc=calibrated.get("pair_completeness"),
            pq=calibrated.get("pair_quality"),
            rr=calibrated.get("reduction_ratio"),
            candidates=calibrated.get("candidate_pairs"),
            **{f"r@{k}": recalls[k].get("recall") for k in (1, 5, 10, 100) if k in recalls},
            **self._gpu_fields(),
            new_best=new_best,
        )

    def train_done(self, **fields: Any) -> None:
        self.configure(phase="done")
        self.emit("train_done", duration=format_duration(time.monotonic() - self._start_time), **fields)

    def train_failed(self, exc: BaseException) -> None:
        self.configure(phase="failed")
        self.emit("train_failed", error=exc.__class__.__name__, message=str(exc))

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.heartbeat()

    def heartbeat(self) -> None:
        state = self._state_snapshot()
        elapsed = time.monotonic() - self._start_time
        fields: dict[str, Any] = {
            "phase": state.get("phase"),
            "epoch": self._progress(state.get("epoch"), state.get("epochs")),
            "batch": self._progress(state.get("batch"), state.get("batches_per_epoch")) if state.get("phase") == "train" else None,
            "val_batch": self._progress(state.get("val_batch"), state.get("val_batches")) if state.get("phase") == "validation" else None,
            "global_step": self._progress(state.get("global_step"), state.get("total_steps")),
            "elapsed": format_duration(elapsed),
            "eta_total": format_duration(self._eta_seconds()),
            "loss_ema": state.get("loss_ema"),
            "loss_epoch": self._epoch_loss(state),
            "lr": state.get("lr"),
            "samples_s": self._samples_per_second(state, elapsed),
        }
        fields.update(self._gpu_fields())
        self.emit("heartbeat", **fields)

    def _state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _eta_seconds(self) -> float | None:
        state = self._state_snapshot()
        now = time.monotonic()
        epochs = int(state.get("epochs") or 0)
        if epochs <= 0:
            return None
        completed_epochs = len(self._epoch_durations)
        if self._epoch_durations:
            avg_epoch = sum(self._epoch_durations) / len(self._epoch_durations)
            current_elapsed = 0.0 if state.get("phase") in {"epoch_end", "done"} else max(0.0, now - (self._epoch_start_time or now))
            return max(0.0, avg_epoch * (epochs - completed_epochs) - current_elapsed)
        total_steps = int(state.get("total_steps") or 0)
        global_step = int(state.get("global_step") or 0)
        elapsed = now - self._start_time
        if global_step > 0 and total_steps > 0 and elapsed > 0:
            return max(0.0, (total_steps - global_step) / (global_step / elapsed))
        return None

    def _gpu_fields(self) -> dict[str, Any]:
        if not self.log_gpu or not self.device.startswith("cuda"):
            return {}
        fields: dict[str, Any] = {}
        if torch.cuda.is_available():
            fields.update(
                cuda_mem_alloc_gb=torch.cuda.memory_allocated() / 1e9,
                cuda_mem_reserved_gb=torch.cuda.memory_reserved() / 1e9,
                cuda_mem_max_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
        fields.update(_nvidia_smi_fields())
        return fields

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        text = str(value)
        return text.replace("\n", " ").replace("\r", " ")

    @staticmethod
    def _progress(value: Any, total: Any) -> str | None:
        if not total:
            return str(value) if value else None
        return f"{int(value or 0)}/{int(total)}"

    @staticmethod
    def _epoch_loss(state: dict[str, Any]) -> float | None:
        count = int(state.get("loss_epoch_count") or 0)
        return float(state.get("loss_epoch_sum", 0.0)) / count if count else None

    @staticmethod
    def _samples_per_second(state: dict[str, Any], elapsed: float) -> float | None:
        samples = int(state.get("samples_seen") or 0)
        return samples / elapsed if samples and elapsed > 0 else None


def _nvidia_smi_fields() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    target = _nvidia_smi_target()
    if target:
        command.extend(["-i", target])
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).strip()
    except Exception:
        return {}
    if not output:
        return {}
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(parts) < 4:
        return {}
    try:
        gpu_util, mem_util, mem_used, mem_total = (float(part) for part in parts[:4])
    except ValueError:
        return {}
    return {
        "gpu_util_pct": gpu_util,
        "gpu_mem_util_pct": mem_util,
        "nvidia_mem_used_gb": mem_used / 1024.0,
        "nvidia_mem_total_gb": mem_total / 1024.0,
    }


def _nvidia_smi_target() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "void", "none"}:
        return visible.split(",")[0].strip() or None
    if torch.cuda.is_available():
        try:
            return str(torch.cuda.current_device())
        except Exception:
            return None
    return None
