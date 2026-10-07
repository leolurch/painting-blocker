"""One-command, resumable set matching orchestration."""

from __future__ import annotations

import fcntl
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from . import file_io
from .config import Config, write_resolved_config, write_run_metadata
from .ingest import SourceSpec, ingest


def detect_set_mode(set_a_path: Path | str, set_b_path: Path | str) -> tuple[str, Path, Path]:
    a = Path(set_a_path).expanduser().resolve()
    b = Path(set_b_path).expanduser().resolve()
    if not a.is_dir():
        raise FileNotFoundError(f"Set A directory does not exist: {a}")
    if not b.is_dir():
        raise FileNotFoundError(f"Set B directory does not exist: {b}")
    same = os.path.samefile(a, b)
    if same:
        return "symmetric", a, b
    try:
        a.relative_to(b)
        nested = True
    except ValueError:
        try:
            b.relative_to(a)
            nested = True
        except ValueError:
            nested = False
    if nested:
        raise ValueError("Set A and set B must not contain one another unless they are the same directory")
    return "bipartite", a, b


@contextmanager
def _pipeline_lock(work: Path):
    work.mkdir(parents=True, exist_ok=True)
    path = work / ".match_sets.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            raise RuntimeError(f"Another match-sets command owns {path}: {owner or 'unknown'}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"host={socket.gethostname()} pid={os.getpid()} started={file_io.utc_now_iso()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _state_path() -> Path:
    return file_io.work_dir() / "pipeline_state.json"


def _update_stage(name: str, status: str, **details: Any) -> None:
    path = _state_path()
    state = file_io.read_json(path) if path.is_file() else {"schema_version": 1, "stages": {}}
    previous = dict(state["stages"].get(name, {}))
    row = {**previous, "status": status, "updated_at": file_io.utc_now_iso(), **details}
    if status == "running":
        row["started_at"] = file_io.utc_now_iso()
        row.pop("last_error", None)
    if status == "completed":
        row["completed_at"] = file_io.utc_now_iso()
    state["stages"][name] = row
    state["updated_at"] = file_io.utc_now_iso()
    file_io.atomic_write_json(path, state)


def _run_stage(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    _update_stage(name, "running")
    try:
        summary = function()
    except Exception as exc:
        _update_stage(name, "failed", last_error=repr(exc))
        raise
    output_fingerprint = file_io.sha256_json(summary)
    _update_stage(name, "completed", output_fingerprint=output_fingerprint, summary=summary)
    return summary


def _write_input_snapshot(mode: str, a: Path, b: Path, ingest_summary: dict[str, Any]) -> dict[str, Any]:
    work = file_io.work_dir()
    source_path = work / "source_files.csv"
    rows = file_io.read_csv_rows(source_path)
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["source_name"]] = source_counts.get(row["source_name"], 0) + 1
    snapshot = {
        "schema_version": 1,
        "mode": mode,
        "set_a_path": str(a),
        "set_b_path": str(b),
        "set_paths_samefile": mode == "symmetric",
        "source_counts": source_counts,
        "source_files_sha256": file_io.sha256_file(source_path),
        "images_manifest_sha256": file_io.sha256_file(work / "images.csv"),
        "ingest_summary": ingest_summary,
    }
    file_io.atomic_write_json(work / "input_snapshot.json", snapshot)
    return snapshot


def _prepare_review_safely() -> dict[str, Any]:
    from .review import prepare_review

    work = file_io.work_dir()
    output = work / "review" / "review_pairs.csv"
    decisions = work / "review" / "review_decisions.csv"
    temporary = work / "review" / ".review_pairs.next.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = prepare_review(temporary)
    if decisions.is_file() and decisions.stat().st_size and output.is_file():
        if file_io.sha256_file(output) != file_io.sha256_file(temporary):
            temporary.unlink(missing_ok=True)
            raise ValueError(
                "Candidate review queue changed while review decisions exist. Archive or remove "
                "the decisions explicitly before regenerating the queue."
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    return {**summary, "review_pairs_sha256": file_io.sha256_file(output)}


def match_sets(
    config: Config,
    set_a_path: Path | str,
    set_b_path: Path | str,
    *,
    workers: int = 1,
    part_size: int = 250,
    feature_cache_size: int = 32,
    reuse_dino_work_dirs: list[Path] | None = None,
    reuse_superpoint_work_dirs: list[Path] | None = None,
    node_local_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all potential-match stages in one resumable, single-allocation command."""
    if workers <= 0 or part_size <= 0 or feature_cache_size < 0:
        raise ValueError("workers and part_size must be positive; feature_cache_size must be non-negative")
    mode, a, b = detect_set_mode(set_a_path, set_b_path)
    work = file_io.work_dir()
    with _pipeline_lock(work):
        write_resolved_config(config, work)
        specs = [SourceSpec("set", a)] if mode == "symmetric" else [SourceSpec("set_a", a), SourceSpec("set_b", b)]
        ingest_summary = _run_stage("ingest", lambda: ingest(config, specs))
        snapshot = _run_stage("input_snapshot", lambda: _write_input_snapshot(mode, a, b, ingest_summary))

        from .retrieval import retrieve, write_exact_hash_artifacts

        exact_summary = _run_stage(
            "exact_hash_precheck", lambda: write_exact_hash_artifacts(mode=mode)
        )

        from .embedding import embed

        embedding_summary = _run_stage(
            "embed",
            lambda: embed(config, reuse_work_dirs=reuse_dino_work_dirs or []),
        )
        retrieval_summary = _run_stage("retrieve", lambda: retrieve(config, mode=mode))

        from .verification import precompute_superpoint_features, verify

        feature_dir = (
            Path(node_local_cache_dir).expanduser().resolve()
            / "deduplication_superpoint_features"
            / config.model.fingerprint_sha256()
            if node_local_cache_dir is not None
            else work / "verification" / "features"
        )
        feature_summary = _run_stage(
            "precompute_superpoint",
            lambda: precompute_superpoint_features(
                config,
                workers=workers,
                feature_dir=feature_dir,
                reuse_work_dirs=reuse_superpoint_work_dirs or [],
            ),
        )
        verification_summary = _run_stage(
            "verify",
            lambda: verify(
                config,
                workers=workers,
                num_shards=1,
                shard_index=0,
                part_size=part_size,
                feature_cache_size=feature_cache_size,
                feature_dir=feature_dir,
                merge=True,
            ),
        )
        review_summary = _run_stage("prepare_review", _prepare_review_safely)
        summary = {
            "mode": mode,
            "set_a_path": str(a),
            "set_b_path": str(b),
            "input_snapshot": snapshot,
            "exact_hash_summary": exact_summary,
            "embedding_metadata": embedding_summary,
            "retrieval_summary": retrieval_summary,
            "feature_precompute_summary": feature_summary,
            "verification_summary": verification_summary,
            "review_summary": review_summary,
        }
        write_run_metadata(config, work, {"match_sets": summary})
        return summary
