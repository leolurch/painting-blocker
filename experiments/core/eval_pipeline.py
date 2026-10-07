"""Explicit-ID retrieval evaluation for experiment split JSON files.

The public entry points are deliberately small:
- ``cosine_cache`` builds the all-image similarity matrix used by a model run.
- ``evaluate_task`` aligns one split task to that matrix and writes the unchanged
  eval.json task schema consumed by aggregation and LaTeX rendering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .retrieval_metrics import cosine_similarity_matrix, f_score_vector, query_coverage

from .artifacts import sha256_json
from .split_schema import SplitImage


@dataclass(frozen=True)
class TaskMatrix:
    """Task-local similarity/label matrices after self-match and empty-query filtering."""

    similarities: np.ndarray
    positives: np.ndarray
    positives_per_query: np.ndarray
    excluded_pairs: int
    original_queries: int
    dropped_queries: int

    @property
    def possible_pairs(self) -> int:
        return int(self.similarities.shape[0] * self.similarities.shape[1] - self.excluded_pairs)

    @property
    def positive_pairs(self) -> int:
        return int(self.positives.sum())


def _hash_ids(ids: Sequence[str]) -> str:
    return sha256_json(list(ids))


def _task_matrix(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
) -> TaskMatrix:
    row_by_id = {file_id: idx for idx, file_id in enumerate(image_ids)}
    q_ids, c_ids = list(task["query_ids"]), list(task["candidate_ids"])
    q_idx = np.asarray([row_by_id[file_id] for file_id in q_ids], dtype=np.int64)
    c_idx = np.asarray([row_by_id[file_id] for file_id in c_ids], dtype=np.int64)
    sims = similarities[np.ix_(q_idx, c_idx)].astype(np.float32, copy=True)

    q_class = np.asarray([records[file_id].class_id for file_id in q_ids], dtype=np.int64)
    c_class = np.asarray([records[file_id].class_id for file_id in c_ids], dtype=np.int64)
    positives = q_class[:, None] == c_class[None, :]

    excluded_by_query = _mask_self_matches(task, q_ids, c_ids, sims, positives)
    positives_per_query = positives.sum(axis=1)
    valid = positives_per_query > 0
    return TaskMatrix(
        similarities=sims[valid],
        positives=positives[valid],
        positives_per_query=positives_per_query[valid],
        excluded_pairs=int(excluded_by_query[valid].sum()),
        original_queries=len(q_ids),
        dropped_queries=int(len(q_ids) - valid.sum()),
    )


def _mask_self_matches(
    task: dict[str, Any],
    q_ids: list[str],
    c_ids: list[str],
    sims: np.ndarray,
    positives: np.ndarray,
) -> np.ndarray:
    excluded_by_query = np.zeros(len(q_ids), dtype=np.int64)
    if not task.get("exclude_self", False):
        return excluded_by_query
    c_pos = {file_id: idx for idx, file_id in enumerate(c_ids)}
    for q_pos, file_id in enumerate(q_ids):
        c_pos_idx = c_pos.get(file_id)
        if c_pos_idx is None:
            continue
        sims[q_pos, c_pos_idx] = -np.inf
        positives[q_pos, c_pos_idx] = False
        excluded_by_query[q_pos] = 1
    return excluded_by_query


def evaluate_task(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    top_k: list[int],
    fail_on_empty_positives: bool,
    target_pc: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _task_matrix(task, image_ids, records, similarities)
    if matrix.positive_pairs == 0:
        if fail_on_empty_positives:
            raise ValueError(f"Task {task['task_id']} has no positive pairs")
        return _empty_task_result(task, matrix), {
            "task_id": task["task_id"],
            "warnings": ["no_positive_pairs"],
        }

    result = {
        "task_id": task["task_id"],
        "num_queries": int(matrix.similarities.shape[0]),
        "num_original_queries": int(matrix.original_queries),
        "num_queries_dropped_without_positives": int(matrix.dropped_queries),
        "num_candidates": int(matrix.similarities.shape[1]),
        "num_cartesian_pairs": int(matrix.similarities.shape[0] * matrix.similarities.shape[1]),
        "num_possible_pairs": matrix.possible_pairs,
        "num_excluded_pairs": matrix.excluded_pairs,
        "num_positive_pairs": matrix.positive_pairs,
        "query_ids_hash": _hash_ids(task["query_ids"]),
        "candidate_ids_hash": _hash_ids(task["candidate_ids"]),
        "positive_pairs_hash": sha256_json(_positive_pairs(task, records)),
        "metrics": _summarize_top_k(task["task_id"], matrix, top_k),
        "target_pc_metrics": summarize_target_pc(
            matrix.similarities,
            matrix.positives,
            target_pc or [],
        ),
        "artifacts": {},
    }
    curves = build_curves(task["task_id"], matrix.similarities, matrix.positives)
    return result, curves


def _summarize_top_k(task_id: str, matrix: TaskMatrix, top_k: list[int]) -> list[dict[str, Any]]:
    sims, positives = matrix.similarities, matrix.positives
    max_k = min(max(top_k), sims.shape[1])
    if max_k <= 0:
        raise ValueError(f"Task {task_id} has no candidate embeddings")
    top_idx = np.argpartition(-sims, kth=max_k - 1, axis=1)[:, :max_k]
    top_scores = np.take_along_axis(sims, top_idx, axis=1)
    top_idx = np.take_along_axis(top_idx, np.argsort(-top_scores, axis=1), axis=1)
    relevant_top = np.take_along_axis(positives, top_idx, axis=1)
    finite_candidates_per_query = np.isfinite(sims).sum(axis=1)

    rows = []
    for k in top_k:
        k_eff = min(k, max_k)
        hits = relevant_top[:, :k_eff].sum(axis=1).astype(np.float64)
        # Preserve the historical eval.json semantics: precision divides by the
        # requested k even when fewer than k candidates exist, while pair counts
        # use the effective available k.
        precision = hits / float(k)
        recall = hits / matrix.positives_per_query.astype(np.float64)
        candidate_pairs = int(len(hits) * k_eff)
        tp = int(hits.sum())
        rows.append(
            {
                "k": int(k),
                "precision": float(np.mean(precision)),
                "recall": float(np.mean(recall)),
                "f0_5": float(np.mean(f_score_vector(precision, recall, 0.5))),
                "f1": float(np.mean(f_score_vector(precision, recall, 1.0))),
                "f2": float(np.mean(f_score_vector(precision, recall, 2.0))),
                "candidate_pairs": candidate_pairs,
                "true_positives": tp,
                "false_positives": int(candidate_pairs - tp),
                "false_negatives": int(matrix.positive_pairs - tp),
                "pair_quality": float(tp / candidate_pairs) if candidate_pairs else 0.0,
                "pair_completeness": float(tp / matrix.positive_pairs),
                "reduction_ratio": (
                    float(1.0 - candidate_pairs / matrix.possible_pairs)
                    if matrix.possible_pairs
                    else 0.0
                ),
                "query_coverage": query_coverage(
                    np.minimum(k_eff, finite_candidates_per_query) > 0
                ),
            }
        )
    return rows


def calibrate_task_top_k(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    top_k: Sequence[int],
    target_pc: Sequence[float],
    *,
    on_unattained: str = "report_only",
) -> list[dict[str, Any]]:
    """Choose the smallest predeclared k satisfying pooled validation PC.

    If the target is unattainable, the failure is explicit.  The optional
    least-restrictive fallback is marked as such and is never counted as target
    attainment.
    """
    if on_unattained not in {"report_only", "least_restrictive"}:
        raise ValueError("top-k calibration on_unattained must be report_only or least_restrictive")
    matrix = _task_matrix(task, image_ids, records, similarities)
    candidates = _summarize_top_k(task["task_id"], matrix, sorted({int(value) for value in top_k}))
    rows: list[dict[str, Any]] = []
    for raw_target in target_pc:
        target = float(raw_target)
        feasible = [row for row in candidates if float(row["pair_completeness"]) >= target - 1e-12]
        maximum = max((float(row["pair_completeness"]) for row in candidates), default=0.0)
        if feasible:
            selected = min(feasible, key=lambda row: int(row["k"]))
            rows.append({
                **selected,
                "target_pair_completeness": target,
                "selection_status": "target_attained",
                "target_attained": True,
                "maximum_pair_completeness": maximum,
                "fallback_policy": "not_needed",
            })
        elif on_unattained == "least_restrictive" and candidates:
            selected = max(candidates, key=lambda row: int(row["k"]))
            rows.append({
                **selected,
                "target_pair_completeness": target,
                "selection_status": "target_not_attained_fallback",
                "target_attained": False,
                "maximum_pair_completeness": maximum,
                "fallback_policy": "least_restrictive_evaluated_k",
            })
        else:
            rows.append({
                "k": None,
                "target_pair_completeness": target,
                "selection_status": "target_not_attained",
                "target_attained": False,
                "maximum_pair_completeness": maximum,
                "fallback_policy": "none",
            })
    return rows


def evaluate_task_at_calibrated_top_k(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    calibration_rows: Sequence[dict[str, Any]],
    calibration_id: str,
) -> list[dict[str, Any]]:
    """Apply each validation-selected k unchanged to the held-out test task."""
    matrix = _task_matrix(task, image_ids, records, similarities)
    selected_ks = sorted({int(row["k"]) for row in calibration_rows if row.get("k") is not None})
    test_by_k = {
        int(row["k"]): row
        for row in (_summarize_top_k(task["task_id"], matrix, selected_ks) if selected_ks else [])
    }
    outputs: list[dict[str, Any]] = []
    for calibration in calibration_rows:
        target = calibration["target_pair_completeness"]
        k = calibration.get("k")
        if k is None:
            outputs.append({
                "k": None,
                "calibration_id": calibration_id,
                "calibration_target_pair_completeness": target,
                "calibration_selection_status": calibration["selection_status"],
                "calibration_target_attained": False,
                "calibration_maximum_pair_completeness": calibration.get("maximum_pair_completeness"),
                "fallback_policy": calibration.get("fallback_policy"),
                "test_evaluated": False,
            })
            continue
        test = test_by_k[int(k)]
        outputs.append({
            **test,
            "calibration_id": calibration_id,
            "calibration_target_pair_completeness": target,
            "calibration_selection_status": calibration["selection_status"],
            "calibration_target_attained": calibration["target_attained"],
            "calibration_pair_completeness": calibration.get("pair_completeness"),
            "calibration_pair_quality": calibration.get("pair_quality"),
            "calibration_reduction_ratio": calibration.get("reduction_ratio"),
            "calibration_candidate_pairs": calibration.get("candidate_pairs"),
            "calibration_maximum_pair_completeness": calibration.get("maximum_pair_completeness"),
            "fallback_policy": calibration.get("fallback_policy"),
            "test_evaluated": True,
            "test_target_attained": float(test["pair_completeness"]) >= float(target) - 1e-12,
            "calibration_error": float(test["pair_completeness"]) - float(target),
        })
    return outputs


def _positive_pairs(task: dict[str, Any], records: dict[str, SplitImage]) -> list[list[str]]:
    pairs: list[list[str]] = []
    for q_id in task["query_ids"]:
        for c_id in task["candidate_ids"]:
            if task.get("exclude_self") and q_id == c_id:
                continue
            if records[q_id].class_id == records[c_id].class_id:
                pairs.append([q_id, c_id])
    return pairs


def _empty_task_result(task: dict[str, Any], matrix: TaskMatrix | None = None) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "num_queries": int(matrix.similarities.shape[0]) if matrix is not None else 0,
        "num_original_queries": int(matrix.original_queries) if matrix is not None else len(task["query_ids"]),
        "num_queries_dropped_without_positives": (
            int(matrix.dropped_queries) if matrix is not None else len(task["query_ids"])
        ),
        "num_candidates": int(matrix.similarities.shape[1]) if matrix is not None else len(task["candidate_ids"]),
        "num_cartesian_pairs": (
            int(matrix.similarities.shape[0] * matrix.similarities.shape[1])
            if matrix is not None
            else 0
        ),
        "num_possible_pairs": matrix.possible_pairs if matrix is not None else 0,
        "num_excluded_pairs": int(matrix.excluded_pairs) if matrix is not None else 0,
        "num_positive_pairs": 0,
        "metrics": [],
        "target_pc_metrics": [],
        "artifacts": {},
    }


def summarize_fixed_threshold(
    sims: np.ndarray,
    positives: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Report pair metrics for all finite scores at one similarity threshold."""
    valid = np.isfinite(sims)
    selected = (sims >= float(threshold)) & valid
    positive_valid = positives & valid
    selected_pairs = int(selected.sum())
    true_positives = int((selected & positive_valid).sum())
    total_positive_pairs = int(positive_valid.sum())
    possible_pairs = int(valid.sum())
    return {
        "threshold": float(threshold),
        "pair_completeness": float(true_positives / total_positive_pairs) if total_positive_pairs else 0.0,
        "pair_quality": float(true_positives / selected_pairs) if selected_pairs else 0.0,
        "reduction_ratio": float(1.0 - selected_pairs / possible_pairs) if possible_pairs else 0.0,
        "query_coverage": query_coverage(np.any(selected, axis=1)),
        "candidate_pairs": selected_pairs,
        "true_positives": true_positives,
        "false_positives": int(selected_pairs - true_positives),
        "false_negatives": int(total_positive_pairs - true_positives),
        "total_positive_pairs": total_positive_pairs,
        "possible_pairs": possible_pairs,
    }


def summarize_fixed_threshold_with_top_k_floor(
    sims: np.ndarray,
    positives: np.ndarray,
    threshold: float,
    minimum_top_k: int,
) -> dict[str, float | int]:
    """Apply a frozen threshold while retaining at least ``minimum_top_k`` per query.

    The candidate rule is ``score >= threshold OR rank <= minimum_top_k``.  A stable
    descending sort makes the existing candidate-column order the deterministic
    tie-break; task candidate IDs are resolved in sorted order by the split loader.
    The floor is clamped to the number of finite candidates for each query.
    """
    if isinstance(minimum_top_k, bool) or not isinstance(minimum_top_k, int) or minimum_top_k <= 0:
        raise ValueError("minimum_top_k must be a positive integer")
    valid = np.isfinite(sims)
    threshold_selected = (sims >= float(threshold)) & valid
    selected = threshold_selected.copy()
    if sims.shape[1]:
        order = np.argsort(-sims, axis=1, kind="stable")
        for query_index in range(sims.shape[0]):
            finite_order = order[query_index, valid[query_index, order[query_index]]]
            selected[query_index, finite_order[:minimum_top_k]] = True

    positive_valid = positives & valid
    selected_pairs = int(selected.sum())
    threshold_pairs = int(threshold_selected.sum())
    true_positives = int((selected & positive_valid).sum())
    threshold_true_positives = int((threshold_selected & positive_valid).sum())
    total_positive_pairs = int(positive_valid.sum())
    possible_pairs = int(valid.sum())
    threshold_sizes = threshold_selected.sum(axis=1)
    available_sizes = valid.sum(axis=1)
    effective_floor = np.minimum(minimum_top_k, available_sizes)
    floor_activated = threshold_sizes < effective_floor
    return {
        "threshold": float(threshold),
        "minimum_top_k": int(minimum_top_k),
        "pair_completeness": float(true_positives / total_positive_pairs) if total_positive_pairs else 0.0,
        "pair_quality": float(true_positives / selected_pairs) if selected_pairs else 0.0,
        "reduction_ratio": float(1.0 - selected_pairs / possible_pairs) if possible_pairs else 0.0,
        "query_coverage": query_coverage(np.any(selected, axis=1)),
        "candidate_pairs": selected_pairs,
        "true_positives": true_positives,
        "false_positives": int(selected_pairs - true_positives),
        "false_negatives": int(total_positive_pairs - true_positives),
        "total_positive_pairs": total_positive_pairs,
        "possible_pairs": possible_pairs,
        "threshold_candidate_pairs": threshold_pairs,
        "floor_added_candidate_pairs": int(selected_pairs - threshold_pairs),
        "threshold_true_positives": threshold_true_positives,
        "floor_rescued_true_positives": int(true_positives - threshold_true_positives),
        "queries_requiring_floor": int(floor_activated.sum()),
        "floor_activation_rate": float(floor_activated.mean()) if len(floor_activated) else 0.0,
    }


def evaluate_task_at_threshold(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate an explicit-ID retrieval task at a fixed similarity threshold."""
    matrix = _task_matrix(task, image_ids, records, similarities)
    return summarize_fixed_threshold(matrix.similarities, matrix.positives, threshold)


def calibrate_task_thresholds(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    target_pc: Sequence[float],
) -> list[dict[str, float | int]]:
    """Select empirical similarity thresholds for a calibration retrieval task.

    Labels of the calibration task may be used here to choose the operating
    points. The returned rows are the frozen thresholds later applied unchanged
    to held-out evaluation tasks via ``evaluate_task_at_calibrated_thresholds``.
    """
    matrix = _task_matrix(task, image_ids, records, similarities)
    return summarize_target_pc(matrix.similarities, matrix.positives, target_pc)


def evaluate_task_at_calibrated_thresholds(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    calibration_rows: Sequence[dict[str, float | int]],
    calibration_id: str,
) -> list[dict[str, Any]]:
    """Apply calibration thresholds to an evaluation task without re-searching.

    Each output row reports the achieved test metrics alongside the calibration
    objective that produced the threshold. The ``calibration_*`` fields describe
    what was requested/achieved on the calibration set; the bare metric fields
    (``pair_completeness`` etc.) are the actual test-set results and must never be
    relabeled as the calibration target.
    """
    rows: list[dict[str, Any]] = []
    for calibration in calibration_rows:
        threshold = calibration.get("threshold")
        if threshold is None:
            # Infeasible calibration target (e.g. requested PC unreachable on the
            # calibration set) yields no threshold; skip rather than fabricate one.
            continue
        test_metrics = evaluate_task_at_threshold(
            task,
            image_ids,
            records,
            similarities,
            float(threshold),
        )
        target = calibration.get("target_pair_completeness")
        test_pc = test_metrics.get("pair_completeness")
        rows.append(
            {
                **test_metrics,
                "calibration_id": calibration_id,
                "calibration_selection_status": calibration.get("selection_status", "target_attained"),
                "calibration_target_attained": calibration.get("target_attained", True),
                "calibration_target_pair_completeness": target,
                "test_target_attained": (
                    bool(float(test_pc) >= float(target) - 1e-12)
                    if test_pc is not None and target is not None
                    else None
                ),
                "calibration_error": (
                    float(test_pc) - float(target)
                    if test_pc is not None and target is not None
                    else None
                ),
                "calibration_pair_completeness": calibration.get("pair_completeness"),
                "calibration_pair_quality": calibration.get("pair_quality"),
                "calibration_reduction_ratio": calibration.get("reduction_ratio"),
                "calibration_query_coverage": calibration.get("query_coverage"),
            }
        )
    return rows


def evaluate_task_at_calibrated_threshold_top_k_floors(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    calibration_rows: Sequence[dict[str, float | int]],
    calibration_id: str,
    minimum_top_k: Sequence[int],
) -> list[dict[str, Any]]:
    """Apply frozen pure-threshold calibrations with predeclared top-k floors."""
    matrix = _task_matrix(task, image_ids, records, similarities)
    rows: list[dict[str, Any]] = []
    for calibration in calibration_rows:
        threshold = calibration.get("threshold")
        if threshold is None:
            continue
        target = calibration.get("target_pair_completeness")
        for floor in minimum_top_k:
            metrics = summarize_fixed_threshold_with_top_k_floor(
                matrix.similarities,
                matrix.positives,
                float(threshold),
                int(floor),
            )
            test_pc = metrics.get("pair_completeness")
            rows.append(
                {
                    **metrics,
                    "calibration_id": calibration_id,
                    "candidate_rule": "calibrated_threshold_or_top_k_floor",
                    "threshold_source": "pure_calibrated_threshold",
                    "calibration_selection_status": calibration.get(
                        "selection_status", "target_attained"
                    ),
                    "calibration_target_attained": calibration.get("target_attained", True),
                    "calibration_target_pair_completeness": target,
                    "test_target_attained": (
                        bool(float(test_pc) >= float(target) - 1e-12)
                        if test_pc is not None and target is not None
                        else None
                    ),
                    "calibration_error": (
                        float(test_pc) - float(target)
                        if test_pc is not None and target is not None
                        else None
                    ),
                    # These remain the pure-threshold validation operating point.
                    "calibration_pair_completeness": calibration.get("pair_completeness"),
                    "calibration_pair_quality": calibration.get("pair_quality"),
                    "calibration_reduction_ratio": calibration.get("reduction_ratio"),
                    "calibration_query_coverage": calibration.get("query_coverage"),
                }
            )
    return rows


def calibrated_candidate_sizes(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, SplitImage],
    similarities: np.ndarray,
    calibration_rows: Sequence[dict[str, float | int]],
) -> list[dict[str, Any]]:
    """Per-query candidate-set sizes on an evaluation task at calibrated thresholds.

    Logged at evaluation time (into curves.json) so downstream figures never have
    to reconstruct them from the similarity cache. One row per feasible
    calibration operating point.
    """
    matrix = _task_matrix(task, image_ids, records, similarities)
    if matrix.positive_pairs == 0:
        return []
    valid = np.isfinite(matrix.similarities)
    rows: list[dict[str, Any]] = []
    for calibration in calibration_rows:
        threshold = calibration.get("threshold")
        if threshold is None:
            continue
        selected = (matrix.similarities >= float(threshold)) & valid
        sizes = selected.sum(axis=1)
        rows.append(
            {
                "calibration_target_pair_completeness": calibration.get("target_pair_completeness"),
                "threshold": float(threshold),
                "candidates_per_query": [int(size) for size in sizes],
                "query_coverage": query_coverage(sizes > 0),
            }
        )
    return rows


def summarize_target_pc(
    sims: np.ndarray, positives: np.ndarray, targets: Sequence[float]
) -> list[dict[str, float | int]]:
    """Report threshold operating points needed to reach requested pair completeness."""
    if not targets:
        return []
    pc_targets = [float(target) for target in targets]
    if any(target <= 0.0 or target > 1.0 for target in pc_targets):
        raise ValueError("evaluation.target_pc values must be in (0, 1]")
    valid = np.isfinite(sims)
    scores = sims[valid].reshape(-1)
    labels = positives[valid].reshape(-1).astype(np.int64)
    total_positive_pairs = int(labels.sum())
    possible_pairs = int(len(labels))
    if total_positive_pairs == 0 or possible_pairs == 0:
        return []
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    score_changes = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])
    group_ends = np.r_[score_changes, len(sorted_scores) - 1]
    candidate_pairs = group_ends + 1
    true_positives = np.cumsum(sorted_labels)[group_ends]
    pair_completeness = true_positives / float(total_positive_pairs)
    max_score_per_query = np.max(sims, axis=1, where=valid, initial=-np.inf)
    rows: list[dict[str, float | int]] = []
    for target in pc_targets:
        feasible = np.flatnonzero(pair_completeness >= target)
        if len(feasible) == 0:
            max_pc = float(pair_completeness[-1]) if len(pair_completeness) else 0.0
            rows.append(
                {
                    "target_pair_completeness": target,
                    "selection_status": "target_not_attained",
                    "target_attained": False,
                    "maximum_pair_completeness": max_pc,
                    "threshold": None,
                    "pair_completeness": max_pc,
                    "fallback_policy": "none",
                }
            )
            continue
        idx = int(feasible[0])
        selected = int(candidate_pairs[idx])
        tp = int(true_positives[idx])
        fp = selected - tp
        pc = float(pair_completeness[idx])
        rows.append(
            {
                "target_pair_completeness": target,
                "selection_status": "target_attained",
                "target_attained": True,
                "maximum_pair_completeness": float(pair_completeness[-1]),
                "fallback_policy": "not_needed",
                "pair_completeness": pc,
                "pair_quality": float(tp / selected) if selected else 0.0,
                "reduction_ratio": float(1.0 - selected / possible_pairs),
                "query_coverage": query_coverage(
                    max_score_per_query >= float(sorted_scores[group_ends[idx]])
                ),
                "threshold": float(sorted_scores[group_ends[idx]]),
                "candidate_pairs": selected,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": int(total_positive_pairs - tp),
            }
        )
    return rows


def build_curves(
    task_id: str,
    sims: np.ndarray,
    positives: np.ndarray,
    bins: int = 80,
) -> dict[str, Any]:
    valid = np.isfinite(sims)
    pos_scores = sims[positives & valid]
    neg_scores = sims[(~positives) & valid]
    edges = np.linspace(-1.0, 1.0, bins + 1)
    histogram = {
        "bin_edges": edges.tolist(),
        "match_hist": np.histogram(pos_scores, bins=edges)[0].astype(int).tolist(),
        "neg_hist": np.histogram(neg_scores, bins=edges)[0].astype(int).tolist(),
        "match_total": int(len(pos_scores)),
        "neg_total": int(len(neg_scores)),
    }
    curve = _pr_curve(pos_scores, neg_scores) if len(pos_scores) and len(neg_scores) else []
    return {"task_id": task_id, "histogram": histogram, "precision_recall": curve}


def _pr_curve(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    num_points: int = 2001,
) -> list[dict[str, float | int]]:
    thresholds = np.linspace(-1.0, 1.0, num_points)
    pos_sorted, neg_sorted = np.sort(pos_scores), np.sort(neg_scores)
    tp = len(pos_scores) - np.searchsorted(pos_sorted, thresholds, side="left")
    fp = len(neg_scores) - np.searchsorted(neg_sorted, thresholds, side="left")
    fn = len(pos_scores) - tp
    candidate_pairs = tp + fp
    precision = np.divide(
        tp,
        candidate_pairs,
        out=np.zeros_like(tp, dtype=np.float64),
        where=candidate_pairs > 0,
    )
    recall = tp / float(len(pos_scores))
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    possible_pairs = float(len(pos_scores) + len(neg_scores))
    rr = 1.0 - (candidate_pairs / possible_pairs)
    return [
        {
            "threshold": float(thresholds[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "pair_quality": float(precision[i]),
            "pair_completeness": float(recall[i]),
            "reduction_ratio": float(rr[i]),
            "pq": float(precision[i]),
            "pc": float(recall[i]),
            "rr": float(rr[i]),
            "f1": float(f1[i]),
            "candidate_pairs": int(candidate_pairs[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
        }
        for i in range(num_points)
    ]


def cosine_cache(
    embeddings: np.ndarray,
    device: str,
    gpu_dtype: str,
    cache_dtype: str,
    *,
    inputs_normalized: bool = True,
    fp32_matmul_precision: str = "ieee",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    if cache_dtype != "float32":
        raise ValueError(
            "FP32 similarity caching is mandatory; cache_dtype must be 'float32'"
        )
    result = cosine_similarity_matrix(
        embeddings,
        embeddings,
        device,
        gpu_dtype,
        inputs_normalized=inputs_normalized,
        fp32_matmul_precision=fp32_matmul_precision,
        return_metadata=True,
    )
    sims, metadata = result
    sims = sims.astype(np.float32, copy=False)
    metadata = {**metadata, "cache_dtype": str(sims.dtype)}
    if return_metadata:
        return sims, metadata
    return sims
