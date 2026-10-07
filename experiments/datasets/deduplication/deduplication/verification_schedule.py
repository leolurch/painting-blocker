"""Deterministic reference-grouped verification schedules for bipartite candidates."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import heapq
import os
import shutil
from pathlib import Path
from typing import Iterator

from . import file_io, manifest

REFERENCE_FIELD = "reference_image_id"
QUERY_FIELD = "query_image_id"
FIELDS = manifest.CANDIDATES_FIELDS + [REFERENCE_FIELD, QUERY_FIELD]


def _bucket(value: str, count: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _atomic_rows(path: Path, rows: Iterator[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sort_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row[REFERENCE_FIELD], row[QUERY_FIELD], row["pair_id"]


def _merge_runs(paths: list[Path], output: Path) -> None:
    iterators = [file_io.iter_csv_rows(path) for path in paths]
    _atomic_rows(output, heapq.merge(*iterators, key=_sort_key))


def _prepare_bipartite_schedules_unlocked(
    candidates_path: Path,
    candidates_sha256: str,
    images: dict[str, manifest.ImageEntry],
    *,
    num_shards: int,
    workers: int,
    chunk_rows: int = 100_000,
) -> tuple[list[Path], dict[str, object]]:
    """Create this node/shard layout's reference-local worker schedules.

    References are first hash-sharded across nodes and then across workers. Every reference and all
    of its incident pairs therefore have exactly one owner.
    """
    if num_shards <= 0 or workers <= 0:
        raise ValueError("num_shards and workers must be positive")
    root = file_io.work_dir() / "verification" / "schedules" / f"shards_{num_shards:05d}_workers_{workers:05d}"
    summary_path = root / "summary.json"
    expected_key = file_io.sha256_json(
        {
            "candidates_sha256": candidates_sha256,
            "num_shards": num_shards,
            "workers": workers,
            "strategy": "set_b_reference_sha256_v1",
        }
    )
    if summary_path.is_file():
        summary = file_io.read_json(summary_path)
        paths = [root / rel for rel in summary.get("worker_schedules", [])]
        if (
            summary.get("schedule_key") == expected_key
            and paths
            and all(path.is_file() for path in paths)
            and all(summary.get("hashes", {}).get(path.relative_to(root).as_posix()) == file_io.sha256_file(path) for path in paths)
        ):
            return paths, summary

    if root.exists():
        shutil.rmtree(root)
    runs_root = root / "runs"
    buffers: list[list[dict[str, str]]] = [[] for _ in range(num_shards)]
    run_paths: list[list[Path]] = [[] for _ in range(num_shards)]

    def flush(shard: int) -> None:
        rows = buffers[shard]
        if not rows:
            return
        rows.sort(key=_sort_key)
        path = runs_root / f"shard_{shard:05d}" / f"run_{len(run_paths[shard]):06d}.csv"
        _atomic_rows(path, iter(rows))
        run_paths[shard].append(path)
        buffers[shard] = []

    for candidate in file_io.iter_csv_rows(candidates_path):
        a = images[candidate["image_a_id"]]
        b = images[candidate["image_b_id"]]
        if a.source_name == "set_b" and b.source_name == "set_a":
            reference, query = a.image_id, b.image_id
        elif b.source_name == "set_b" and a.source_name == "set_a":
            reference, query = b.image_id, a.image_id
        else:
            raise ValueError(f"Bipartite candidate is not set_a↔set_b: {candidate['pair_id']}")
        row = {**candidate, REFERENCE_FIELD: reference, QUERY_FIELD: query}
        shard = _bucket(reference, num_shards)
        buffers[shard].append(row)
        if sum(len(buffer) for buffer in buffers) >= chunk_rows:
            for shard_index in range(num_shards):
                flush(shard_index)
    for shard_index in range(num_shards):
        flush(shard_index)

    shard_paths: list[Path] = []
    for shard_index, paths in enumerate(run_paths):
        target = root / f"shard_{shard_index:05d}.csv"
        if paths:
            _merge_runs(paths, target)
        else:
            _atomic_rows(target, iter(()))
        shard_paths.append(target)

    worker_paths: list[Path] = []
    counts: dict[str, int] = {}
    for shard_index, shard_path in enumerate(shard_paths):
        paths = [root / f"shard_{shard_index:05d}_worker_{worker:05d}.csv" for worker in range(workers)]
        handles = []
        writers = []
        try:
            for path in paths:
                handle = path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
                writer.writeheader()
                handles.append(handle)
                writers.append(writer)
                counts[path.relative_to(root).as_posix()] = 0
            for row in file_io.iter_csv_rows(shard_path):
                worker = _bucket(row[REFERENCE_FIELD], workers)
                writers[worker].writerow({field: row.get(field, "") for field in FIELDS})
                counts[paths[worker].relative_to(root).as_posix()] += 1
        finally:
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        worker_paths.extend(paths)
    shutil.rmtree(runs_root, ignore_errors=True)
    for path in shard_paths:
        path.unlink(missing_ok=True)

    summary = {
        "schema_version": 1,
        "schedule_key": expected_key,
        "strategy": "set_b_reference_sha256_v1",
        "num_shards": num_shards,
        "workers": workers,
        "worker_schedules": [path.relative_to(root).as_posix() for path in worker_paths],
        "counts": counts,
        "hashes": {path.relative_to(root).as_posix(): file_io.sha256_file(path) for path in worker_paths},
    }
    file_io.atomic_write_json(summary_path, summary)
    return worker_paths, summary


def prepare_bipartite_schedules(
    candidates_path: Path,
    candidates_sha256: str,
    images: dict[str, manifest.ImageEntry],
    *,
    num_shards: int,
    workers: int,
    chunk_rows: int = 100_000,
) -> tuple[list[Path], dict[str, object]]:
    lock_dir = file_io.work_dir() / "verification" / "schedules"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _prepare_bipartite_schedules_unlocked(
                candidates_path,
                candidates_sha256,
                images,
                num_shards=num_shards,
                workers=workers,
                chunk_rows=chunk_rows,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def iter_reference_chunks(schedule_path: Path, part_size: int) -> Iterator[tuple[int, list[dict[str, str]]]]:
    """Yield bounded chunks without ever splitting one reference group."""
    part_number = 0
    chunk: list[dict[str, str]] = []
    current_reference: str | None = None
    current_group: list[dict[str, str]] = []
    for row in file_io.iter_csv_rows(schedule_path):
        reference = row[REFERENCE_FIELD]
        if current_reference is not None and reference != current_reference:
            if chunk and len(chunk) + len(current_group) > part_size:
                yield part_number, chunk
                part_number += 1
                chunk = []
            chunk.extend(current_group)
            current_group = []
        current_reference = reference
        current_group.append(row)
    if current_group:
        if chunk and len(chunk) + len(current_group) > part_size:
            yield part_number, chunk
            part_number += 1
            chunk = []
        chunk.extend(current_group)
    if chunk:
        yield part_number, chunk
