"""Per-profile retrieval reporting for hard-synth evaluations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

PROFILES = ("archival", "print", "cropped_record", "framed_photo")
_METRICS = ("pair_completeness", "pair_quality", "reduction_ratio", "query_coverage")


def _macro_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[object, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key)].append(row)
    result = []
    for value, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        macro = {key: value}
        for metric in _METRICS:
            numeric = [float(row[metric]) for row in values if row.get(metric) is not None]
            if numeric:
                macro[metric] = sum(numeric) / len(numeric)
        result.append(macro)
    return result


def hard_synth_profile_summary(
    task_results: list[dict[str, Any]],
    *,
    held_out_profile: str | None = None,
) -> dict[str, Any]:
    """Return all four profile rows plus fixed-k/threshold macro averages."""
    rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        matches = [
            task for task in task_results
            if str(task.get("task_id", "")).endswith(f"_to_{profile}")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one retrieval result for profile {profile!r}")
        task = matches[0]
        num_queries = int(task.get("num_queries", 0))
        num_positives = int(task.get("num_positive_pairs", num_queries * 2))
        if num_queries and num_positives != num_queries * 2:
            raise ValueError(f"Profile {profile!r} does not have exactly two positives/query")
        rows.append({
            "profile": profile,
            "profile_status": "held_out" if profile == held_out_profile else "seen",
            "task_id": task.get("task_id"),
            "num_queries": num_queries,
            "positives_per_query": 2,
            "fixed_k": task.get("metrics") or [],
            "calibrated_threshold": task.get("calibrated_threshold_metrics")
            or task.get("fixed_threshold_metrics") or task.get("target_pc_metrics") or [],
        })
    fixed_rows = [metric for row in rows for metric in row["fixed_k"]]
    threshold_rows = [metric for row in rows for metric in row["calibrated_threshold"]]
    threshold_key = next(
        (
            key
            for key in (
                "calibration_target_pair_completeness",
                "target_pair_completeness",
                "target_pc",
            )
            if any(metric.get(key) is not None for metric in threshold_rows)
        ),
        "threshold",
    )
    return {
        "profiles": rows,
        "macro_average": {
            "fixed_k": _macro_rows(fixed_rows, "k"),
            "calibrated_threshold": _macro_rows(threshold_rows, threshold_key),
        },
    }
