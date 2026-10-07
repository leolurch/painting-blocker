#!/usr/bin/env python3
"""Write the rank of every true pair for Wikidata evaluations.

Each modern query's historic targets are ranked by cosine similarity, as in
the evaluation (stable descending order, self-pairs last). For every true
pair the file records the query, the 1-based rank of its true target, and
its cosine similarity, and for every query its painting. Pairs completeness
and per-query recall at any k or any similarity threshold, and the smallest
k that reaches a completeness target, follow from these without the
similarity matrix. The similarities are checked against the true-pair counts
of the threshold curve in curves.json. Partition runs get the calibration half
and the test half. Full-catalog runs get the test split. Full-catalog runs
keep embeddings instead of a similarity matrix; their cosine similarities are
recomputed from the L2-normalized FP32 descriptors. Results are written
beside eval.json as rank_positions.json. The stored test row at k = 40 is
checked before writing. Numpy only: this does not import the evaluation
package.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "wiki_threshold_margin",
    Path(__file__).with_name("compute_wikidata_threshold_margin.py"),
)
_MARGIN = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MARGIN)

DATA = _MARGIN.DATA
TOLERANCE = _MARGIN.TOLERANCE
_class_of = _MARGIN._class_of
_sha256_json = _MARGIN._sha256_json
_split_ids = _MARGIN._split_ids

OUTPUT_NAME = "rank_positions.json"
METHOD = "rank_positions_v2"
ROOTS = (
    "eval_wikidata1_5_lostart_distractors_frozen_full_v1",
    "eval_wikidata1_5_lostart_distractors_frozen_multisplit_v1",
    "eval_wikidata1_5_lostart_distractors_hplus_full_v1",
    "eval_wikidata1_5_lostart_distractors_hplus_multisplit_v1",
    "eval_wikidata1_5_lostart_distractors_siglip2_full_v1",
    "eval_wikidata1_5_lostart_distractors_siglip2_multisplit_v1",
    "finetune_dinov3_vithplus_met_paint_winner_followup_v1",
    "finetune_dinov3_vithplus_met_paint_head_only_v1",
    "finetune_dinov3_vithplus_met_paint_modern_only_v1",
    "finetune_dinov3_vithplus_met_paint_originals_aug_v1",
)
MAX_DEPTH = 4


def iter_run_dirs(roots: list[Path]):
    """Evaluation run directories on Wikidata that hold a split snapshot."""
    for root in roots:
        if not root.is_dir():
            print(f"missing {root}", flush=True)
            continue
        base_depth = len(root.parts)
        for current, dirnames, filenames in os.walk(root):
            path = Path(current)
            depth = len(path.parts) - base_depth
            dirnames[:] = sorted(name for name in dirnames if name not in {"models", "figures", "bar_charts", "latex_tables", "resource_metrics"})
            if depth >= MAX_DEPTH:
                dirnames[:] = []
            if "split.snapshot.json" in filenames and (path / "models").is_dir():
                if "wikidata" in str(path.relative_to(DATA)):
                    yield path


def _model_dirs(run_dir: Path):
    for model_dir in sorted(path for path in (run_dir / "models").iterdir() if path.is_dir()):
        if not (model_dir / "eval.json").is_file():
            continue
        if (model_dir / "similarity_cache.npy").is_file() or (model_dir / "embeddings.npy").is_file():
            yield model_dir


def _similarity_source(model_dir: Path):
    """Image ids and a function that returns the query-by-candidate cosine block."""
    if (model_dir / "similarity_cache.npy").is_file():
        meta = json.loads((model_dir / "similarity_cache_metadata.json").read_text(encoding="utf-8"))
        image_ids = [str(value) for value in meta["image_ids"]]
        sims = np.load(model_dir / "similarity_cache.npy", mmap_mode="r")
        if sims.shape != (len(image_ids), len(image_ids)):
            raise ValueError(f"Similarity cache shape {sims.shape} does not match {len(image_ids)} ids")
        return image_ids, lambda rows, cols: np.asarray(sims[np.ix_(rows, cols)], dtype=np.float32)
    meta = json.loads((model_dir / "embeddings_metadata.json").read_text(encoding="utf-8"))
    image_ids = [str(value) for value in meta["image_ids"]]
    embeddings = np.load(model_dir / "embeddings.npy", mmap_mode="r")
    if embeddings.shape[0] != len(image_ids):
        raise ValueError(f"Embedding rows {embeddings.shape[0]} do not match {len(image_ids)} ids")

    def block(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        queries = np.asarray(embeddings[rows], dtype=np.float32)
        candidates = np.asarray(embeddings[cols], dtype=np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        return queries @ candidates.T

    return image_ids, block


def _has_roles(split: dict, subset: str) -> bool:
    roles = ((split.get("subsets") or {}).get(subset) or {}).get("roles") or {}
    return bool(roles.get("modern")) and bool(roles.get("historic"))


def rank_view(
    block,
    image_ids: list[str],
    classes: dict[str, int],
    query_ids: list[str],
    candidate_ids: list[str],
) -> dict:
    """Rank of each true pair, with the same view as eval_pipeline._task_matrix."""
    index = {file_id: position for position, file_id in enumerate(image_ids)}
    query_index = np.asarray([index[file_id] for file_id in query_ids], dtype=np.int64)
    candidate_index = np.asarray([index[file_id] for file_id in candidate_ids], dtype=np.int64)
    viewed = np.array(block(query_index, candidate_index), dtype=np.float32, copy=True)
    query_class = np.asarray([classes[file_id] for file_id in query_ids], dtype=np.int64)
    candidate_class = np.asarray([classes[file_id] for file_id in candidate_ids], dtype=np.int64)
    positives = query_class[:, None] == candidate_class[None, :]
    candidate_position = {file_id: position for position, file_id in enumerate(candidate_ids)}
    for query_position, file_id in enumerate(query_ids):
        match = candidate_position.get(file_id)
        if match is None:
            continue
        viewed[query_position, match] = -np.inf
        positives[query_position, match] = False
    keep = positives.sum(axis=1) > 0
    viewed = viewed[keep]
    positives = positives[keep]
    query_class = query_class[keep]
    order = np.argsort(-viewed, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(order.shape[1]), order.shape), axis=1)
    positive_query, positive_column = np.nonzero(positives)
    positive_rank = ranks[positive_query, positive_column] + 1
    positive_score = viewed[positive_query, positive_column]
    return {
        "num_queries": int(viewed.shape[0]),
        "num_candidates": int(viewed.shape[1]),
        "possible_pairs": int(np.isfinite(viewed).sum()),
        "num_positive_pairs": int(positives.sum()),
        "query_class": query_class.tolist(),
        "query_positives": positives.sum(axis=1).astype(int).tolist(),
        "positive_query": positive_query.astype(int).tolist(),
        "positive_rank": positive_rank.astype(int).tolist(),
        "positive_score": [float(value) for value in positive_score],
        "query_ids_hash": _sha256_json(sorted(query_ids)),
    }


def pc_at_k(view: dict, k: int) -> float:
    ranks = np.asarray(view["positive_rank"])
    return float((ranks <= int(k)).sum() / view["num_positive_pairs"]) if view["num_positive_pairs"] else 0.0


def _test_task(eval_doc: dict, query_hash: str) -> dict | None:
    for row in eval_doc.get("tasks") or []:
        if row.get("query_ids_hash") == query_hash and "modern_to_historic" in str(row.get("task_id")):
            return row
    return None


def _check_stored(view: dict, task_row: dict | None, label: str) -> None:
    if task_row is None:
        raise ValueError(f"{label}: no eval.json task matches the rebuilt query ids")
    for row in task_row.get("metrics") or []:
        if row.get("k") is not None and int(row["k"]) == 40 and view["num_candidates"] >= 40:
            delta = abs(pc_at_k(view, 40) - float(row["pair_completeness"]))
            if delta > TOLERANCE:
                raise ValueError(f"{label}: stored PC@40 {row['pair_completeness']} disagrees with ranks {pc_at_k(view, 40)}")
            return


def _check_scores(view: dict, curves: dict, task_row: dict | None, label: str) -> None:
    """True pairs with similarity >= t must match the stored threshold curve at every t."""
    if task_row is None:
        return
    curve = next((row for row in curves.get("tasks") or [] if row.get("task_id") == task_row.get("task_id")), None)
    points = (curve or {}).get("precision_recall") or []
    if not points:
        return
    thresholds = np.asarray([row["threshold"] for row in points], dtype=np.float64)
    stored = np.asarray([row["tp"] for row in points], dtype=np.int64)
    scores = np.sort(np.asarray(view["positive_score"], dtype=np.float32))
    rebuilt = len(scores) - np.searchsorted(scores, thresholds, side="left")
    worst = int(np.max(np.abs(rebuilt - stored)))
    if worst > 0:
        raise ValueError(f"{label}: true-pair similarities disagree with curves.json by up to {worst} pairs")


def _current(output: Path, eval_path: Path, run_path: Path) -> bool:
    if not output.is_file() or output.stat().st_mtime < max(eval_path.stat().st_mtime, run_path.stat().st_mtime):
        return False
    try:
        return json.loads(output.read_text(encoding="utf-8")).get("method") == METHOD
    except json.JSONDecodeError:
        return False


def score_model(run_dir: Path, model_dir: Path) -> str:
    output = model_dir / OUTPUT_NAME
    # run.json is written after the model's eval.json; without it the run is still going.
    if not (run_dir / "run.json").is_file():
        return "unfinished"
    if _current(output, model_dir / "eval.json", run_dir / "run.json"):
        return "kept"
    split = json.loads((run_dir / "split.snapshot.json").read_text(encoding="utf-8"))
    image_ids, block = _similarity_source(model_dir)
    eval_doc = json.loads((model_dir / "eval.json").read_text(encoding="utf-8"))
    curves_path = model_dir / "curves.json"
    curves = json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.is_file() else {}
    classes = _class_of(split)
    tasks: dict[str, dict] = {}
    if _has_roles(split, "test"):
        test_queries = _split_ids(split, "test", "modern")
        test_candidates = _split_ids(split, "test", "historic")
        view = rank_view(block, image_ids, classes, test_queries, test_candidates)
        task_row = _test_task(eval_doc, view["query_ids_hash"])
        _check_stored(view, task_row, str(model_dir))
        _check_scores(view, curves, task_row, str(model_dir))
        tasks["test"] = view
        wiki_candidates = [file_id for file_id in test_candidates if not str(file_id).startswith("LA")]
        if len(wiki_candidates) != len(test_candidates):
            tasks["test_wikidata"] = rank_view(block, image_ids, classes, test_queries, wiki_candidates)
    if _has_roles(split, "val"):
        val_queries = _split_ids(split, "val", "modern")
        val_candidates = _split_ids(split, "val", "historic")
        tasks["calibration"] = rank_view(block, image_ids, classes, val_queries, val_candidates)
        wiki_candidates = [file_id for file_id in val_candidates if not str(file_id).startswith("LA")]
        if len(wiki_candidates) != len(val_candidates):
            tasks["calibration_wikidata"] = rank_view(block, image_ids, classes, val_queries, wiki_candidates)
    if not tasks:
        return "no tasks"
    run_doc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model_runs = run_doc.get("model_runs") or [{}]
    payload = {
        "method": METHOD,
        "run_dir": str(run_dir.relative_to(DATA)),
        "model_dir": model_dir.name,
        "display_name": model_runs[0].get("display_name"),
        "model_id": model_runs[0].get("model_id"),
        "parent_training": run_doc.get("parent_training"),
        "split_id": (run_doc.get("split") or {}).get("split_id"),
        "split_seed": (run_doc.get("split") or {}).get("split_seed"),
        "tasks": tasks,
    }
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return "wrote"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=None, help="Run root under experiments/runs; repeatable")
    args = parser.parse_args()
    roots = [DATA / name for name in (args.root or ROOTS)]
    counts: dict[str, int] = {}
    problems = 0
    for run_dir in iter_run_dirs(roots):
        for model_dir in _model_dirs(run_dir):
            try:
                status = score_model(run_dir, model_dir)
            except Exception as error:  # noqa: BLE001
                status = "problem"
                problems += 1
                print(f"problem {model_dir}: {error}", flush=True)
            counts[status] = counts.get(status, 0) + 1
            if status == "wrote":
                print(f"wrote {model_dir.relative_to(DATA)}", flush=True)
    print(f"done {counts}", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
