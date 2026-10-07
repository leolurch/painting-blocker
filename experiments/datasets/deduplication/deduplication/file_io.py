"""Filesystem, hashing, CSV, and path helpers for the deduplication pipeline.

These helpers are vendored (small copies of ``experiments/core/artifacts.py`` behaviour) so the
compute stages run as a standalone ``python -m deduplication`` tool without importing the parent
``experiments`` package. Only the final export stage reaches into ``experiments`` for validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# ---------------------------------------------------------------------------
# Project / repository paths
# ---------------------------------------------------------------------------

# file_io.py lives at:
#   <repo>/experiments/datasets/deduplication/deduplication/file_io.py
# parents[1] = the deduplication project dir; parents[4] = the repository root.
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_DIR = PROJECT_DIR / "work"
LIGHTGLUE_VERIFIED_DIR = REPO_ROOT / "experiments" / "datasets" / "lightglue-verified"


def configure_work_dir(path: Path | str) -> Path:
    """Select one caller-provided path as this process's work tree."""
    selected = Path(path).expanduser().resolve()
    if selected.exists() and not selected.is_dir():
        raise ValueError(f"work path exists but is not a directory: {selected}")
    global WORK_DIR
    WORK_DIR = selected
    return WORK_DIR


def project_dir() -> Path:
    return PROJECT_DIR


def work_dir() -> Path:
    return WORK_DIR


def rel_to_project(path: Path | str) -> str:
    """Return a portable logical path, including files under the active external work tree."""
    p = Path(path).resolve()
    # Preserve logical work/... paths when the physical work tree is external, so review handovers
    # remain portable rather than recording a cluster-specific absolute target.
    try:
        work_relative = p.relative_to(WORK_DIR.resolve())
        return (Path("work") / work_relative).as_posix()
    except ValueError:
        pass
    try:
        return p.relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_from_project(rel_path: str) -> Path:
    """Resolve project paths, treating portable ``work/...`` paths as the active work tree."""
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if parts and parts[0] == "work":
        return (WORK_DIR / Path(*parts[1:])).resolve()
    return (PROJECT_DIR / candidate).resolve()


def ensure_experiments_importable() -> None:
    """Put the repository root on ``sys.path`` so ``import experiments`` works (export only)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_lightglue_verified_importable() -> None:
    """Put the ``lightglue-verified`` dir on ``sys.path`` for its keypoint helpers."""
    d = str(LIGHTGLUE_VERIFIED_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


# ---------------------------------------------------------------------------
# Time / run identity
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "".join(random.choice(string.hexdigits.lower()[:16]) for _ in range(6))
    return f"{stamp}_{suffix}"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CSV helpers (explicit ordered fieldnames, atomic writes)
# ---------------------------------------------------------------------------


def atomic_write_csv(
    path: Path | str,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(path: Path | str) -> Iterator[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------


def git_info(repo_root: Path | str = REPO_ROOT) -> dict[str, object]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    commit = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain"])
    return {"git_commit": commit, "git_dirty": bool(status)}
