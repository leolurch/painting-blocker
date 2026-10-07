"""Per-key filesystem locking for the shared cross-run experiment cache.

A cache key uniquely identifies a shared artifact (an embedding set or a
reference evaluation). Before generating an artifact, a worker acquires the lock
for its full key so that concurrent sweep tasks never generate the same artifact
twice or observe a partially written one.

Primary mechanism is ``fcntl.flock`` on a per-key lock file. Some shared
filesystems (certain NFS mounts) do not support ``flock``; on ``OSError`` we
fall back to an atomic ``O_CREAT | O_EXCL`` lock file polled until a timeout.
"""

from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .cache_paths import locks_dir

DEFAULT_TIMEOUT_SECONDS = 1800.0
_POLL_INTERVAL_SECONDS = 0.5


class LockTimeout(TimeoutError):
    """Raised when a cache lock cannot be acquired within the timeout."""


def _lock_path(key: str) -> Path:
    directory = locks_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # Keys are hex digests; guard against unexpected separators anyway.
    safe = key.replace("/", "_").replace(os.sep, "_")
    return directory / f"{safe}.lock"


@contextmanager
def key_lock(key: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Iterator[Path]:
    """Hold an exclusive lock for ``key`` for the duration of the context.

    Yields the lock-file path. Always released in ``finally``.
    """
    path = _lock_path(key)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        fcntl = None

    if fcntl is not None:
        handle = open(path, "w", encoding="utf-8")
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        if time.monotonic() >= deadline:
                            raise LockTimeout(f"Timed out acquiring cache lock for {key}") from exc
                        time.sleep(_POLL_INTERVAL_SECONDS)
                        continue
                    # flock unsupported on this filesystem: fall back.
                    handle.close()
                    with _lockfile_lock(path, timeout=timeout):
                        yield path
                    return
            try:
                yield path
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
            return
        except Exception:
            handle.close()
            raise

    with _lockfile_lock(path, timeout=timeout):
        yield path


@contextmanager
def _lockfile_lock(path: Path, *, timeout: float) -> Iterator[Path]:
    """Atomic-create lock-file fallback for filesystems without ``flock``."""
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise LockTimeout(f"Timed out acquiring cache lock file {path}")
            time.sleep(_POLL_INTERVAL_SECONDS)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        fd = None
        yield path
    finally:
        if fd is not None:
            os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
