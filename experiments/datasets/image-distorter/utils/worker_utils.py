"""Worker-local utilities for process-based image processing."""

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DEFAULT_STATIC_SEED = 1337
_base_seed = DEFAULT_STATIC_SEED
_worker_index = 1
_worker_rng: random.Random | None = None
_worker_np_rng: np.random.Generator | None = None
_worker_seed: int | None = None


def set_rng_base_seed(base_seed: int) -> None:
    """Set the deterministic base seed used by future worker-local RNGs."""
    global _base_seed
    _base_seed = int(base_seed)
    reset_worker_rng()


def set_worker_rng_index(worker_index: int) -> None:
    """Set the one-based process worker index for deterministic seeding."""
    global _worker_index
    _worker_index = max(1, int(worker_index))
    reset_worker_rng()


def reset_worker_rng() -> None:
    """Clear current worker-local RNGs so new seeds take effect."""
    global _worker_rng, _worker_np_rng, _worker_seed
    _worker_rng = None
    _worker_np_rng = None
    _worker_seed = None


def _get_worker_seed() -> int:
    """Return the seed for this worker: base_seed * one_based_worker_index."""
    return _base_seed * _worker_index


def get_worker_rng() -> random.Random:
    """Get or create the process-local random.Random generator."""
    global _worker_rng, _worker_seed
    if _worker_rng is None:
        seed = _get_worker_seed()
        _worker_rng = random.Random(seed)
        _worker_seed = seed
    return _worker_rng


def get_worker_np_rng() -> np.random.Generator:
    """Get or create the process-local NumPy generator."""
    global _worker_np_rng, _worker_seed
    if _worker_np_rng is None:
        seed = _get_worker_seed()
        _worker_np_rng = np.random.default_rng(seed)
        _worker_seed = seed
    return _worker_np_rng


def get_worker_rng_seed() -> int:
    """Return the deterministic seed for the current process worker."""
    global _worker_seed
    if _worker_seed is None:
        _worker_seed = _get_worker_seed()
    return _worker_seed


@dataclass
class ProgressTracker:
    """Progress tracking with throughput calculation."""

    total: int
    _counter: int = field(default=0, init=False)
    _start_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._start_time = time.time()

    def increment(self, message: Optional[str] = None) -> tuple[int, float]:
        """Increment counter and optionally print progress.

        Returns (current_count, throughput).
        """
        self._counter += 1
        elapsed = time.time() - self._start_time
        throughput = self._counter / elapsed if elapsed > 0 else 0

        if message:
            print(f"[{self._counter}/{self.total}] {message} ({throughput:.1f}/s)")

        return self._counter, throughput

    @property
    def count(self) -> int:
        return self._counter

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    @property
    def throughput(self) -> float:
        elapsed = self.elapsed
        return self.count / elapsed if elapsed > 0 else 0
