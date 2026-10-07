"""Stage 3 — exhaustive-threshold plus minimum-window cosine retrieval.

The default low-level mode remains symmetric. ``match-sets`` selects bipartite mode when its two
input directories differ. Candidate inclusion is deliberately an OR: every score passing the
calibrated threshold is retained, and every query retains at least its configured nearest-neighbour
window. Byte-identical pairs are retained independently of both DINO rules.
"""

from __future__ import annotations

import csv
import heapq
import os
import shutil
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from . import file_io, manifest
from .config import Config

_SCORE_TOLERANCE = 1e-5
_REASON_ORDER = ("exact_sha256", "dino_threshold", "dino_window")


def _available_scoring_backend() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "torch_cuda_float32_no_tf32"
    except Exception:
        pass
    return "numpy_cpu_float32"


def retrieval_config_sha256(
    config: Config,
    embedding_metadata: dict[str, Any],
    *,
    mode: str = "symmetric",
    query_image_ids: list[str] | None = None,
    candidate_image_ids: list[str] | None = None,
    scoring_backend: str | None = None,
) -> str:
    return file_io.sha256_json(
        {
            "model_fingerprint_sha256": config.model.fingerprint_sha256(),
            "embeddings_sha256": embedding_metadata.get("embeddings_sha256"),
            "image_ids_sha256": embedding_metadata.get("image_ids_sha256"),
            "min_retrieval_window": config.retrieval.min_retrieval_window,
            "cosine_threshold": config.retrieval.cosine_threshold,
            "comparison": config.retrieval.comparison,
            "mode": mode,
            "query_image_ids_sha256": file_io.sha256_json(query_image_ids or []),
            "candidate_image_ids_sha256": file_io.sha256_json(candidate_image_ids or []),
            "scoring_backend": scoring_backend or _available_scoring_backend(),
        }
    )


def _candidate_row(
    query_id: str,
    other_id: str,
    image_sha256s: dict[str, str],
    score: float,
    rank: int,
    threshold_match: bool,
    window_match: bool,
    exact_match: bool,
    config: Config,
    retrieval_hash: str,
) -> dict[str, Any]:
    a_id, a_sha, b_id, b_sha = manifest.order_endpoints(
        query_id, image_sha256s[query_id], other_id, image_sha256s[other_id]
    )
    reasons = [
        reason
        for reason, enabled in (
            ("exact_sha256", exact_match),
            ("dino_threshold", threshold_match),
            ("dino_window", window_match),
        )
        if enabled
    ]
    from_a = query_id == a_id
    return {
        "pair_id": manifest.make_pair_id(a_id, a_sha, b_id, b_sha),
        "image_a_id": a_id,
        "image_b_id": b_id,
        "cosine_similarity": repr(score),
        "rank_a_to_b": rank if from_a else "",
        "rank_b_to_a": "" if from_a else rank,
        "retrieved_from_a": int(from_a),
        "retrieved_from_b": int(not from_a),
        "candidate_reasons": ";".join(reasons),
        "dino_threshold_match": int(threshold_match),
        "dino_window_match": int(window_match),
        "exact_sha256_match": int(exact_match),
        "model_fingerprint": config.model.fingerprint_sha256(),
        "retrieval_config_sha256": retrieval_hash,
    }


def _rows_for_query(
    scores: np.ndarray,
    query_index: int,
    candidate_indices: list[int],
    image_ids: list[str],
    image_sha256s: dict[str, str],
    config: Config,
    retrieval_hash: str,
    *,
    mask_self: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    query_id = image_ids[query_index]
    eligible = [idx for idx in candidate_indices if not (mask_self and idx == query_index)]
    if not eligible:
        return [], 0, 0

    # candidate_indices and image_ids are deterministic. Stable sort therefore resolves equal
    # cosine scores by image id without including every boundary tie in the minimum window.
    eligible.sort(key=lambda idx: image_ids[idx])
    ranked = sorted(eligible, key=lambda idx: -float(scores[idx]))
    window = min(config.retrieval.min_retrieval_window, len(ranked))
    rows: list[dict[str, Any]] = []
    threshold_count = 0
    window_only_count = 0
    query_sha = image_sha256s[query_id]
    for rank, other_index in enumerate(ranked, start=1):
        score = float(scores[other_index])
        threshold_match = config.retrieval.compare(score)
        window_match = rank <= window
        exact_match = query_sha == image_sha256s[image_ids[other_index]]
        if not (threshold_match or window_match or exact_match):
            continue
        threshold_count += int(threshold_match)
        window_only_count += int(window_match and not threshold_match)
        rows.append(
            _candidate_row(
                query_id,
                image_ids[other_index],
                image_sha256s,
                score,
                rank,
                threshold_match,
                window_match,
                exact_match,
                config,
                retrieval_hash,
            )
        )
    rows.sort(key=lambda row: row["pair_id"])
    return rows, threshold_count, window_only_count


def _part_rows(
    embeddings: np.ndarray,
    image_ids: list[str],
    image_sha256s: dict[str, str],
    start: int,
    end: int,
    config: Config,
    retrieval_hash: str,
    *,
    query_indices: list[int] | None = None,
    candidate_indices: list[int] | None = None,
    mask_self: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    """Small in-memory reference/helper used by tests.

    Production retrieval writes one sorted query part at a time and uses an external merge.
    """
    query_indices = query_indices if query_indices is not None else list(range(len(image_ids)))
    candidate_indices = candidate_indices if candidate_indices is not None else list(range(len(image_ids)))
    selected_queries = query_indices[start:end]
    rows: list[dict[str, Any]] = []
    threshold_total = 0
    window_only_total = 0
    if not selected_queries:
        return rows, 0, 0
    scores_batch = embeddings[selected_queries] @ embeddings.T
    for local, query_index in enumerate(selected_queries):
        query_rows, threshold_count, window_only = _rows_for_query(
            np.asarray(scores_batch[local], dtype=np.float32),
            query_index,
            candidate_indices,
            image_ids,
            image_sha256s,
            config,
            retrieval_hash,
            mask_self=mask_self,
        )
        rows.extend(query_rows)
        threshold_total += threshold_count
        window_only_total += window_only
    return _merge_rows_in_memory(rows), threshold_total, window_only_total


def _merge_one_group(rows: list[dict[str, str]]) -> dict[str, Any]:
    first = dict(rows[0])
    pair_id = first["pair_id"]
    if first["image_a_id"] >= first["image_b_id"]:
        raise ValueError(f"Malformed endpoint ordering for candidate {pair_id}")
    scores = [float(row["cosine_similarity"]) for row in rows]
    if max(scores) - min(scores) > _SCORE_TOLERANCE:
        raise ValueError(f"Conflicting cosine scores while merging {pair_id}")
    for row in rows[1:]:
        for key in ("image_a_id", "image_b_id", "model_fingerprint", "retrieval_config_sha256"):
            if row[key] != first[key]:
                raise ValueError(f"Conflicting {key} while merging {pair_id}")
    first["cosine_similarity"] = repr(float(sum(scores) / len(scores)))
    for direction in ("a", "b"):
        rank_key = f"rank_{direction}_to_{'b' if direction == 'a' else 'a'}"
        ranks = [int(row[rank_key]) for row in rows if row.get(rank_key)]
        first[rank_key] = min(ranks) if ranks else ""
        first[f"retrieved_from_{direction}"] = int(bool(ranks))
    for flag in ("dino_threshold_match", "dino_window_match", "exact_sha256_match"):
        first[flag] = int(any(int(row.get(flag, 0) or 0) for row in rows))
    present = {reason for row in rows for reason in row.get("candidate_reasons", "").split(";") if reason}
    first["candidate_reasons"] = ";".join(reason for reason in _REASON_ORDER if reason in present)
    return first


def _merge_rows_in_memory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: row["pair_id"])
    return [_merge_one_group(list(group)) for _, group in groupby(rows, key=lambda row: row["pair_id"])]


def _atomic_write_candidate_iter(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest.CANDIDATES_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in manifest.CANDIDATES_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _merge_sorted_files(inputs: list[Path], output: Path) -> None:
    iterators = [file_io.iter_csv_rows(path) for path in inputs]
    merged = heapq.merge(*iterators, key=lambda row: row["pair_id"])

    def combined() -> Iterator[dict[str, Any]]:
        for _, group in groupby(merged, key=lambda row: row["pair_id"]):
            yield _merge_one_group(list(group))

    _atomic_write_candidate_iter(output, combined())


def _external_merge(part_paths: list[Path], output: Path, *, fan_in: int = 64) -> None:
    if not part_paths:
        _atomic_write_candidate_iter(output, [])
        return
    merge_root = output.parent / "merge_runs"
    if merge_root.exists():
        shutil.rmtree(merge_root)
    merge_root.mkdir(parents=True)
    current = list(part_paths)
    level = 0
    while len(current) > fan_in:
        next_paths: list[Path] = []
        level_dir = merge_root / f"level_{level:03d}"
        for group_index, start in enumerate(range(0, len(current), fan_in)):
            target = level_dir / f"run_{group_index:06d}.csv"
            _merge_sorted_files(current[start : start + fan_in], target)
            next_paths.append(target)
        current = next_paths
        level += 1
    _merge_sorted_files(current, output)
    shutil.rmtree(merge_root)


def write_exact_hash_artifacts(
    images: dict[str, manifest.ImageEntry] | None = None, mode: str = "symmetric"
) -> dict[str, Any]:
    if images is None:
        images = manifest.load_images(file_io.work_dir() / "images.csv")
    hashing_dir = file_io.work_dir() / "hashing"
    by_sha: dict[str, list[manifest.ImageEntry]] = {}
    for image in images.values():
        by_sha.setdefault(image.sha256, []).append(image)
    rows: list[dict[str, Any]] = []
    for sha, entries in sorted(by_sha.items()):
        entries.sort(key=lambda entry: entry.image_id)
        if mode == "bipartite":
            left = [entry for entry in entries if entry.source_name == "set_a"]
            right = [entry for entry in entries if entry.source_name == "set_b"]
            pairs = ((a, b) for a in left for b in right)
        else:
            pairs = (
                (entries[i], entries[j])
                for i in range(len(entries))
                for j in range(i + 1, len(entries))
            )
        for a, b in pairs:
            a_id, a_sha, b_id, b_sha = manifest.order_endpoints(a.image_id, sha, b.image_id, sha)
            rows.append(
                {
                    "pair_id": manifest.make_pair_id(a_id, a_sha, b_id, b_sha),
                    "image_a_id": a_id,
                    "image_b_id": b_id,
                    "sha256": sha,
                    "reason": "exact_sha256",
                }
            )
    rows.sort(key=lambda row: row["pair_id"])
    path = hashing_dir / "exact_hash_candidates.csv"
    file_io.atomic_write_csv(
        path, ["pair_id", "image_a_id", "image_b_id", "sha256", "reason"], rows
    )
    summary = {
        "schema_version": 1,
        "mode": mode,
        "image_count": len(images),
        "exact_hash_pair_count": len(rows),
        "exact_hash_candidates_sha256": file_io.sha256_file(path),
        "perceptual_hashing": "deferred",
    }
    file_io.atomic_write_json(hashing_dir / "summary.json", summary)
    return summary


def retrieve(config: Config, *, mode: str = "symmetric") -> dict[str, Any]:
    if mode not in {"symmetric", "bipartite"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    work = file_io.work_dir()
    embedding_dir = work / "embeddings"
    metadata = file_io.read_json(embedding_dir / "metadata.json")
    if metadata.get("model_fingerprint_sha256") != config.model.fingerprint_sha256():
        raise ValueError("Embedding model fingerprint does not match the current configuration")
    embeddings_path = embedding_dir / "embeddings.npy"
    if metadata.get("embeddings_sha256") != file_io.sha256_file(embeddings_path):
        raise ValueError("Embedding artifact hash does not match metadata")
    embeddings = np.load(embeddings_path, allow_pickle=False, mmap_mode="r")
    image_ids: list[str] = file_io.read_json(embedding_dir / "image_ids.json")
    images = manifest.load_images(work / "images.csv")
    if len(image_ids) != embeddings.shape[0] or set(image_ids) != set(images):
        raise ValueError("Embeddings/image_ids do not match the current images manifest")
    if metadata.get("images_manifest_sha256") != file_io.sha256_file(work / "images.csv"):
        raise ValueError("Embeddings were generated from a different images manifest")

    if mode == "bipartite":
        unknown = sorted({entry.source_name for entry in images.values()} - {"set_a", "set_b"})
        if unknown:
            raise ValueError(f"Bipartite retrieval found unexpected source groups: {unknown}")
        query_indices = [i for i, image_id in enumerate(image_ids) if images[image_id].source_name == "set_a"]
        candidate_indices = [i for i, image_id in enumerate(image_ids) if images[image_id].source_name == "set_b"]
        if not query_indices or not candidate_indices:
            raise ValueError("Bipartite retrieval requires at least one valid image in each set")
        mask_self = False
    else:
        query_indices = list(range(len(image_ids)))
        candidate_indices = list(range(len(image_ids)))
        mask_self = True

    query_ids = [image_ids[i] for i in query_indices]
    candidate_ids = [image_ids[i] for i in candidate_indices]
    scoring_backend = _available_scoring_backend()
    retrieval_hash = retrieval_config_sha256(
        config,
        metadata,
        mode=mode,
        query_image_ids=query_ids,
        candidate_image_ids=candidate_ids,
        scoring_backend=scoring_backend,
    )
    exact_summary = write_exact_hash_artifacts(images, mode)

    candidates_dir = work / "candidates"
    candidates_path = candidates_dir / "candidates.csv"
    summary_path = candidates_dir / "summary.json"
    if candidates_path.is_file() and summary_path.is_file():
        existing_summary = file_io.read_json(summary_path)
        if (
            existing_summary.get("retrieval_config_sha256") == retrieval_hash
            and existing_summary.get("mode") == mode
            and existing_summary.get("candidates_sha256") == file_io.sha256_file(candidates_path)
            and existing_summary.get("exact_hash_summary", {}).get(
                "exact_hash_candidates_sha256"
            )
            == exact_summary["exact_hash_candidates_sha256"]
        ):
            return existing_summary

    parts_dir = candidates_dir / "parts" / retrieval_hash
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    threshold_total = 0
    window_only_total = 0
    per_query_counts: list[int] = []
    image_sha256s = {image_id: images[image_id].sha256 for image_id in image_ids}
    batch_size = config.retrieval.query_batch_size
    embeddings_gpu = None
    torch_module = None
    if scoring_backend.startswith("torch_cuda"):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        embeddings_gpu = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to("cuda")
        torch_module = torch

    for batch_start in range(0, len(query_indices), batch_size):
        selected_queries = query_indices[batch_start : batch_start + batch_size]
        missing_positions: list[tuple[int, int, Path, Path, str]] = []
        for offset, query_index in enumerate(selected_queries):
            query_position = batch_start + offset
            part_path = parts_dir / f"query_{query_position:08d}.csv"
            meta_path = part_path.with_suffix(".json")
            part_key = file_io.sha256_json(
                {
                    "retrieval_config_sha256": retrieval_hash,
                    "query_position": query_position,
                    "query_image_id": image_ids[query_index],
                }
            )
            valid = False
            if part_path.is_file() and meta_path.is_file():
                part_meta = file_io.read_json(meta_path)
                valid = (
                    part_meta.get("part_key") == part_key
                    and part_meta.get("csv_sha256") == file_io.sha256_file(part_path)
                )
            if valid:
                part_meta = file_io.read_json(meta_path)
                threshold_total += int(part_meta["threshold_match_count"])
                window_only_total += int(part_meta["window_only_count"])
                per_query_counts.append(int(part_meta["candidate_count"]))
                part_paths.append(part_path)
            else:
                missing_positions.append((offset, query_position, part_path, meta_path, part_key))

        if missing_positions:
            if embeddings_gpu is not None:
                assert torch_module is not None
                with torch_module.inference_mode():
                    query_tensor = embeddings_gpu[
                        torch_module.as_tensor(selected_queries, device="cuda", dtype=torch_module.long)
                    ]
                    scores_batch = (query_tensor @ embeddings_gpu.T).float().cpu().numpy()
            else:
                scores_batch = np.asarray(embeddings[selected_queries] @ embeddings.T, dtype=np.float32)
            for offset, _position, part_path, meta_path, part_key in missing_positions:
                query_index = selected_queries[offset]
                rows, threshold_count, window_only = _rows_for_query(
                    scores_batch[offset],
                    query_index,
                    candidate_indices,
                    image_ids,
                    image_sha256s,
                    config,
                    retrieval_hash,
                    mask_self=mask_self,
                )
                file_io.atomic_write_csv(part_path, manifest.CANDIDATES_FIELDS, rows)
                file_io.atomic_write_json(
                    meta_path,
                    {
                        "part_key": part_key,
                        "csv_sha256": file_io.sha256_file(part_path),
                        "threshold_match_count": threshold_count,
                        "window_only_count": window_only,
                        "candidate_count": len(rows),
                    },
                )
                threshold_total += threshold_count
                window_only_total += window_only
                per_query_counts.append(len(rows))
                part_paths.append(part_path)

    # Query parts are named by their stable query position; restoring this order makes summaries
    # deterministic regardless of which parts were reused in each batch.
    part_paths.sort()
    _external_merge(part_paths, candidates_path)

    unique_count = 0
    two_direction = 0
    threshold_unique = 0
    window_only_unique = 0
    exact_unique = 0
    multi_reason = 0
    # Fixed-memory score histogram keeps summary generation streaming even for tens of millions
    # of threshold candidates. Interior quantiles are approximate to 1e-4 cosine units; extrema
    # are tracked exactly.
    histogram = np.zeros(20001, dtype=np.int64)
    score_min = float("inf")
    score_max = float("-inf")
    for row in file_io.iter_csv_rows(candidates_path):
        unique_count += 1
        if mode == "bipartite":
            sources = {images[row["image_a_id"]].source_name, images[row["image_b_id"]].source_name}
            if sources != {"set_a", "set_b"}:
                raise ValueError(f"Non-cross-set pair emitted in bipartite mode: {row['pair_id']}")
        two_direction += int(bool(int(row["retrieved_from_a"])) and bool(int(row["retrieved_from_b"])))
        threshold_unique += int(row["dino_threshold_match"])
        window_only_unique += int(
            bool(int(row["dino_window_match"])) and not bool(int(row["dino_threshold_match"]))
        )
        exact_unique += int(row["exact_sha256_match"])
        multi_reason += int(len(row["candidate_reasons"].split(";")) > 1)
        score = float(row["cosine_similarity"])
        score_min = min(score_min, score)
        score_max = max(score_max, score)
        bin_index = max(0, min(len(histogram) - 1, int(round((score + 1.0) * 10000))))
        histogram[bin_index] += 1

    if unique_count:
        cumulative = np.cumsum(histogram)
        quantiles: dict[str, float] = {}
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            if q == 0.0:
                value = score_min
            elif q == 1.0:
                value = score_max
            else:
                target = max(1, int(np.ceil(q * unique_count)))
                value = float(np.searchsorted(cumulative, target, side="left") / 10000.0 - 1.0)
            quantiles[str(q)] = float(value)
    else:
        quantiles = {}
    maximum_pairs = (
        len(query_indices) * len(candidate_indices)
        if mode == "bipartite"
        else len(image_ids) * max(0, len(image_ids) - 1) // 2
    )
    query_values = np.asarray(per_query_counts, dtype=np.int64)
    summary = {
        "schema_version": 2,
        "mode": mode,
        "valid_image_count": len(image_ids),
        "set_a_count": len(query_indices),
        "set_b_count": len(candidate_indices) if mode == "bipartite" else len(image_ids),
        "maximum_possible_pair_count": maximum_pairs,
        "min_retrieval_window": config.retrieval.min_retrieval_window,
        "cosine_threshold": config.retrieval.cosine_threshold,
        "comparison": config.retrieval.comparison,
        "model_fingerprint": config.model.fingerprint_sha256(),
        "retrieval_config_sha256": retrieval_hash,
        "scoring_backend": scoring_backend,
        "directed_threshold_match_count": threshold_total,
        "directed_window_only_count": window_only_total,
        "unique_unordered_pair_count": unique_count,
        "threshold_candidate_count": threshold_unique,
        "window_only_candidate_count": window_only_unique,
        "exact_hash_candidate_count": exact_unique,
        "multiple_reason_candidate_count": multi_reason,
        "one_direction_count": unique_count - two_direction,
        "two_direction_count": two_direction,
        "per_query_candidate_count": {
            "min": int(query_values.min()) if len(query_values) else 0,
            "median": float(np.median(query_values)) if len(query_values) else 0.0,
            "max": int(query_values.max()) if len(query_values) else 0,
        },
        "score_quantiles": quantiles,
        "query_image_ids_sha256": file_io.sha256_json(query_ids),
        "candidate_image_ids_sha256": file_io.sha256_json(candidate_ids),
        "exact_hash_summary": exact_summary,
        "part_count": len(part_paths),
        "candidates_sha256": file_io.sha256_file(candidates_path),
    }
    file_io.atomic_write_json(summary_path, summary)
    return summary
