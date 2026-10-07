"""Small artifact, hashing, and run-identity helpers for experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import string
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORE_DIR = Path(__file__).resolve().parent
# Preserve the historical meaning: the directory containing configs, datasets,
# runs, and artifacts—not the Python implementation directory.
PACKAGE_DIR = CORE_DIR.parent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "".join(random.choice(string.hexdigits.lower()[:16]) for _ in range(6))
    return f"{stamp}_{suffix}"


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


def short_hash(value: str, length: int = 18) -> str:
    return value[:length]


def atomic_write_json(path: Path | str, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)


_SLURM_LOG_RE = re.compile(r"slurm-[A-Za-z0-9_.-]+\.(out|err)")


def prepare_run_dir(path: Path | str) -> Path:
    """Create a run directory, allowing Slurm's pre-created log files only.

    Slurm stdout/stderr redirection must create the destination files before the
    Python runner starts. Keep normal duplicate-run protection by rejecting any
    pre-existing run directory that contains files other than ``slurm-*.out`` /
    ``slurm-*.err``.
    """
    run_dir = Path(path)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    except FileExistsError:
        pass

    entries = list(run_dir.iterdir())
    if entries and all(entry.is_file() and _SLURM_LOG_RE.fullmatch(entry.name) for entry in entries):
        return run_dir
    if not entries and os.environ.get("EXPERIMENTS_ALLOW_PRECREATED_RUN_DIR") == "1":
        return run_dir
    raise FileExistsError(f"run directory already exists and is not Slurm-log-only: {run_dir}")


def atomic_write_text(path: Path | str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def git_info(repo_root: Path) -> dict[str, object]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    commit = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain"])
    return {"git_commit": commit, "git_dirty": bool(status)}
