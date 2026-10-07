"""Analytical random-selection baseline for blocking experiment retrieval tasks.

The baseline does not generate embeddings or scores. It reports the expected
metrics of uniformly random candidate selection from the task labels and split
structure, making it a cheap chance-level reference for real embedding models.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .artifacts import sha256_json
from .retrieval_metrics import query_coverage
from .split_schema import SplitImage

RANDOM_BASELINE_MODEL_ID = "random_selection_expected"
RANDOM_BASELINE_STORAGE_KEY = "random_selection_expected"
RANDOM_BASELINE_DISPLAY_NAME = "Random selection (expected)"


@dataclass(frozen=True)
class RandomBaselineConfig:
    model_id: str = RANDOM_BASELINE_MODEL_ID
    storage_key: str = RANDOM_BASELINE_STORAGE_KEY
    display_name: str = RANDOM_BASELINE_DISPLAY_NAME


@dataclass(frozen=True)
class RandomTaskCounts:
    """Task-local label counts after self-match and empty-query filtering."""

    available_per_query: np.ndarray
    positives_per_query: np.ndarray
    num_candidates: int
    excluded_pairs: int
    original_queries: int
    dropped_queries: int

    @property
    def num_queries(self) -> int:
        return int(len(self.positives_per_query))

    @property
    def possible_pairs(self) -> int:
        return int(self.available_per_query.sum())

    @property
    def positive_pairs(self) -> int:
        return int(self.positives_per_query.sum())


def parse_random_baseline_config(raw: Any) -> RandomBaselineConfig | None:
    """Parse ``evaluation.random_baseline`` from an experiment YAML block."""
    if raw in (None, False):
        return None
    if raw is True:
        return RandomBaselineConfig()
    if not isinstance(raw, dict):
        raise ValueError("evaluation.random_baseline must be a boolean or mapping")
    if not bool(raw.get("enabled", True)):
        return None
    model_id = str(raw.get("model_id") or RANDOM_BASELINE_MODEL_ID)
    storage_key = str(raw.get("storage_key") or RANDOM_BASELINE_STORAGE_KEY)
    display_name = str(raw.get("display_name") or RANDOM_BASELINE_DISPLAY_NAME)
    if not model_id.strip() or not storage_key.strip() or not display_name.strip():
        raise ValueError("random_baseline model_id, storage_key, and display_name must be non-empty")
    storage_key = storage_key.strip()
    if storage_key in {".", ".."} or "/" in storage_key or "\\" in storage_key:
        raise ValueError("random_baseline storage_key must be a single path segment")
    return RandomBaselineConfig(model_id.strip(), storage_key, display_name.strip())


def evaluate_random_task(
    task: dict[str, Any],
    records: dict[str, SplitImage],
    top_k: list[int],
    fail_on_empty_positives: bool,
    target_pc: Sequence[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return expected random-selection metrics for one retrieval task."""
    counts = _task_counts(task, records)
    if counts.positive_pairs == 0:
        if fail_on_empty_positives:
            raise ValueError(f"Task {task['task_id']} has no positive pairs")
        return _empty_task_result(task, counts), _empty_curves(task["task_id"])

    result = {
        "task_id": task["task_id"],
        "num_queries": counts.num_queries,
        "num_original_queries": int(counts.original_queries),
        "num_queries_dropped_without_positives": int(counts.dropped_queries),
        "num_candidates": int(counts.num_candidates),
        "num_cartesian_pairs": int(counts.num_queries * counts.num_candidates),
        "num_possible_pairs": counts.possible_pairs,
        "num_excluded_pairs": int(counts.excluded_pairs),
        "num_positive_pairs": counts.positive_pairs,
        "query_ids_hash": sha256_json(list(task["query_ids"])),
        "candidate_ids_hash": sha256_json(list(task["candidate_ids"])),
        "positive_pairs_hash": sha256_json(_positive_pairs(task, records)),
        "metrics": _summarize_random_top_k(task["task_id"], counts, top_k),
        "target_pc_metrics": _summarize_random_target_pc(counts, target_pc or []),
        "artifacts": {},
        "baseline": {"type": "random_selection_expected", "analytical": True},
    }
    return result, _empty_curves(task["task_id"])


def _task_counts(task: dict[str, Any], records: dict[str, SplitImage]) -> RandomTaskCounts:
    q_ids, c_ids = list(task["query_ids"]), list(task["candidate_ids"])
    missing = sorted((set(q_ids) | set(c_ids)) - set(records))
    if missing:
        raise ValueError(f"Task {task['task_id']} references unknown image IDs: {missing[:10]}")
    candidate_class_counts = Counter(records[file_id].class_id for file_id in c_ids)
    candidate_ids = set(c_ids)
    exclude_self = bool(task.get("exclude_self", False))
    available_per_query, positives_per_query = [], []
    excluded_pairs = 0
    dropped_queries = 0
    for q_id in q_ids:
        self_excluded = 1 if exclude_self and q_id in candidate_ids else 0
        available = len(c_ids) - self_excluded
        positives = int(candidate_class_counts[records[q_id].class_id]) - self_excluded
        if positives <= 0:
            dropped_queries += 1
            continue
        available_per_query.append(available)
        positives_per_query.append(positives)
        excluded_pairs += self_excluded
    return RandomTaskCounts(
        available_per_query=np.asarray(available_per_query, dtype=np.int64),
        positives_per_query=np.asarray(positives_per_query, dtype=np.int64),
        num_candidates=len(c_ids),
        excluded_pairs=excluded_pairs,
        original_queries=len(q_ids),
        dropped_queries=dropped_queries,
    )


def _summarize_random_top_k(
    task_id: str,
    counts: RandomTaskCounts,
    top_k: list[int],
) -> list[dict[str, Any]]:
    if not top_k or any(int(k) <= 0 for k in top_k):
        raise ValueError(f"Task {task_id} top_k must contain positive integers")
    rows = []
    available = counts.available_per_query.astype(np.float64)
    positives = counts.positives_per_query.astype(np.float64)
    for raw_k in top_k:
        k = int(raw_k)
        selected_per_query = np.minimum(k, counts.available_per_query).astype(np.float64)
        expected_hits = selected_per_query * positives / available
        precision = expected_hits / float(k)
        recall = expected_hits / positives
        f_scores = np.asarray(
            [
                _expected_query_f_scores(int(n), int(p), int(min(k, n)), k)
                for n, p in zip(counts.available_per_query, counts.positives_per_query)
            ],
            dtype=np.float64,
        )
        candidate_pairs = int(selected_per_query.sum())
        true_positives = float(expected_hits.sum())
        rows.append(
            {
                "k": k,
                "precision": float(np.mean(precision)),
                "recall": float(np.mean(recall)),
                "f0_5": float(np.mean(f_scores[:, 0])),
                "f1": float(np.mean(f_scores[:, 1])),
                "f2": float(np.mean(f_scores[:, 2])),
                "candidate_pairs": candidate_pairs,
                "true_positives": true_positives,
                "false_positives": float(candidate_pairs - true_positives),
                "false_negatives": float(counts.positive_pairs - true_positives),
                "pair_quality": float(true_positives / candidate_pairs) if candidate_pairs else 0.0,
                "pair_completeness": float(true_positives / counts.positive_pairs),
                "reduction_ratio": (
                    float(1.0 - candidate_pairs / counts.possible_pairs)
                    if counts.possible_pairs
                    else 0.0
                ),
                "query_coverage": query_coverage(selected_per_query > 0),
            }
        )
    return rows


def _summarize_random_target_pc(
    counts: RandomTaskCounts,
    targets: Sequence[float],
) -> list[dict[str, float | int | None]]:
    pc_targets = [float(target) for target in targets]
    if any(target <= 0.0 or target > 1.0 for target in pc_targets):
        raise ValueError("evaluation.target_pc values must be in (0, 1]")
    possible_pairs = counts.possible_pairs
    positive_pairs = counts.positive_pairs
    if not pc_targets or possible_pairs == 0 or positive_pairs == 0:
        return []
    prevalence = positive_pairs / float(possible_pairs)
    rows: list[dict[str, float | int | None]] = []
    for target in pc_targets:
        selected = min(possible_pairs, int(math.ceil(target * possible_pairs)))
        expected_tp = float(selected * prevalence)
        pc = float(expected_tp / positive_pairs)
        rows.append(
            {
                "target_pair_completeness": target,
                "pair_completeness": pc,
                "pair_quality": float(prevalence),
                "reduction_ratio": float(1.0 - selected / possible_pairs),
                "threshold": None,
                "candidate_pairs": selected,
                "true_positives": expected_tp,
                "false_positives": float(selected - expected_tp),
                "false_negatives": float(positive_pairs - expected_tp),
            }
        )
    return rows


@lru_cache(maxsize=65536)
def _expected_query_f_scores(
    population: int,
    positives: int,
    selected: int,
    requested_k: int,
) -> tuple[float, float, float]:
    if population <= 0 or positives <= 0 or selected <= 0 or requested_k <= 0:
        return 0.0, 0.0, 0.0
    min_hits = max(0, selected - (population - positives))
    max_hits = min(positives, selected)
    normalizer = _log_comb(population, selected)
    totals = [0.0, 0.0, 0.0]
    for hits in range(min_hits, max_hits + 1):
        log_prob = _log_comb(positives, hits) + _log_comb(population - positives, selected - hits) - normalizer
        probability = math.exp(log_prob)
        precision = hits / float(requested_k)
        recall = hits / float(positives)
        for index, beta in enumerate((0.5, 1.0, 2.0)):
            totals[index] += probability * _f_score(precision, recall, beta)
    return float(totals[0]), float(totals[1]), float(totals[2])


@lru_cache(maxsize=65536)
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _f_score(precision: float, recall: float, beta: float) -> float:
    denom = beta * beta * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta * beta) * precision * recall / denom


def _positive_pairs(task: dict[str, Any], records: dict[str, SplitImage]) -> list[list[str]]:
    candidates_by_class: dict[int, list[str]] = defaultdict(list)
    for c_id in task["candidate_ids"]:
        candidates_by_class[records[c_id].class_id].append(c_id)
    exclude_self = bool(task.get("exclude_self"))
    pairs: list[list[str]] = []
    for q_id in task["query_ids"]:
        for c_id in candidates_by_class[records[q_id].class_id]:
            if exclude_self and q_id == c_id:
                continue
            pairs.append([q_id, c_id])
    return pairs


def _empty_task_result(task: dict[str, Any], counts: RandomTaskCounts | None = None) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "num_queries": counts.num_queries if counts is not None else 0,
        "num_original_queries": (
            int(counts.original_queries) if counts is not None else len(task["query_ids"])
        ),
        "num_queries_dropped_without_positives": (
            int(counts.dropped_queries) if counts is not None else len(task["query_ids"])
        ),
        "num_candidates": int(counts.num_candidates) if counts is not None else len(task["candidate_ids"]),
        "num_cartesian_pairs": (
            int(counts.num_queries * counts.num_candidates) if counts is not None else 0
        ),
        "num_possible_pairs": counts.possible_pairs if counts is not None else 0,
        "num_excluded_pairs": int(counts.excluded_pairs) if counts is not None else 0,
        "num_positive_pairs": 0,
        "metrics": [],
        "target_pc_metrics": [],
        "artifacts": {},
        "baseline": {"type": "random_selection_expected", "analytical": True},
    }


def _empty_curves(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "histogram": {
            "bin_edges": [],
            "match_hist": [],
            "neg_hist": [],
            "match_total": 0,
            "neg_total": 0,
        },
        "precision_recall": [],
        "baseline": {"type": "random_selection_expected", "analytical": True},
    }
