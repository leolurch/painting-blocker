"""Paper figures that compare training seeds of one otherwise fixed setup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .chart_sidecar import require_chart_sidecars, write_chart_sidecar
from .multisplit_charts import FALLBACK_COLORS, THESIS_RC_PARAMS


def _save_figure(fig: Any, output_dir: Path, stem: str, numeric_data: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for fmt in ("pdf", "png"):
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        paths.append(path)
    paths.append(write_chart_sidecar(output_dir, stem, numeric_data))
    require_chart_sidecars(paths)
    return paths


PC_AT_K = (1, 5, 10, 20, 40, 80)
CALIBRATION_RHO = 0.99


def _eval_json_paths(run_dir: Path, evaluation_name: str) -> list[Path]:
    root = run_dir / "evaluations" / evaluation_name
    return sorted(
        path
        for path in root.rglob("eval.json")
        if "rotation-seed-" not in path.parts
    )


def _partition_eval_paths(run_dir: Path, evaluation_name: str) -> list[tuple[str, Path]]:
    root = run_dir / "evaluations" / evaluation_name
    found: list[tuple[str, Path]] = []
    for partition in sorted(path for path in root.glob("rotation-seed-*") if path.is_dir()):
        matches = sorted(partition.rglob("eval.json"))
        if matches:
            found.append((partition.name.removeprefix("rotation-seed-"), matches[0]))
    return found


def _task_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    pair_completeness = {
        int(row["k"]): float(row["pair_completeness"]) for row in task["metrics"]
    }
    calibrated = {
        round(float(row["calibration_target_pair_completeness"]), 2): {
            "pair_completeness": float(row["pair_completeness"]),
            "reduction_ratio": float(row["reduction_ratio"]),
            "pair_quality": float(row["pair_quality"]),
        }
        for row in task["calibrated_threshold_metrics"]
    }
    return {
        "task_id": str(task["task_id"]),
        "pair_completeness_at_k": pair_completeness,
        "calibrated": calibrated,
    }


def _training_seed(run_dir: Path) -> int:
    marker = "trainseed-"
    name = run_dir.name
    if marker not in name:
        raise ValueError(f"Run directory name has no training seed: {run_dir.name}")
    return int(name.rsplit(marker, 1)[-1])


def collect_training_seed_comparison(
    run_dirs: list[Path],
    *,
    full_evaluations: dict[str, str],
    partition_evaluation: str | None = None,
) -> dict[str, Any]:
    """Load PC@k and calibrated metrics for each training-seed run."""
    seeds: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        seed = _training_seed(run_dir)
        full: dict[str, Any] = {}
        for label, evaluation_name in full_evaluations.items():
            matches = _eval_json_paths(run_dir, evaluation_name)
            if len(matches) != 1:
                raise ValueError(
                    f"Seed {seed} evaluation {evaluation_name} has {len(matches)} eval.json files"
                )
            full[label] = _task_metrics(matches[0])
        partitions: dict[str, Any] = {}
        if partition_evaluation is not None:
            for split_seed, path in _partition_eval_paths(run_dir, partition_evaluation):
                partitions[split_seed] = _task_metrics(path)
        seeds.append(
            {
                "training_seed": seed,
                "run_dir": str(run_dir),
                "full": full,
                "partitions": partitions,
            }
        )
    seeds.sort(key=lambda row: int(row["training_seed"]))
    if len({row["training_seed"] for row in seeds}) != len(seeds):
        raise ValueError("Training seeds must be unique")
    return {"seeds": seeds}


def _seed_color(index: int) -> str:
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def _pc_series(record: dict[str, Any], label: str) -> list[float]:
    values = record["full"][label]["pair_completeness_at_k"]
    missing = [k for k in PC_AT_K if k not in values]
    if missing:
        raise ValueError(f"Seed {record['training_seed']} {label} is missing k={missing}")
    return [float(values[k]) for k in PC_AT_K]


def render_pc_at_k_by_training_seed(
    comparison: dict[str, Any],
    output_dir: Path,
    *,
    panels: tuple[tuple[str, str], ...] = (
        ("syn", "SYN medium"),
        ("wikidata_full", "Wikidata"),
    ),
) -> list[Path]:
    """Absolute PC@k and the deviation from the three-seed mean."""
    import matplotlib.pyplot as plt

    seeds = comparison["seeds"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(THESIS_RC_PARAMS):
        fig, axes = plt.subplots(
            2,
            len(panels),
            figsize=(5.75, 4.6),
            sharex="col",
            squeeze=False,
        )
        sidecar_panels: list[dict[str, Any]] = []
        for column, (label, title) in enumerate(panels):
            series = np.array([_pc_series(record, label) for record in seeds], dtype=float)
            mean = series.mean(axis=0)
            delta = (series - mean) * 100.0
            top, bottom = axes[0, column], axes[1, column]
            for index, record in enumerate(seeds):
                color = _seed_color(index)
                seed = int(record["training_seed"])
                top.plot(
                    PC_AT_K,
                    series[index],
                    color=color,
                    marker="o",
                    label=f"seed {seed}",
                )
                bottom.plot(
                    PC_AT_K,
                    delta[index],
                    color=color,
                    marker="o",
                    label=f"seed {seed}",
                )
            top.set_title(title)
            top.set_ylim(0.0, 1.02)
            top.set_ylabel("Pair completeness")
            bottom.axhline(0.0, color="#666666", linewidth=0.6)
            bottom.set_xscale("log")
            bottom.set_xticks(list(PC_AT_K))
            bottom.set_xticklabels([str(k) for k in PC_AT_K])
            bottom.set_xlabel("Candidates per query, $k$")
            bottom.set_ylabel(r"$\Delta$PC (pp)")
            span = float(np.max(np.abs(delta)))
            limit = max(1.0, np.ceil(span * 2.0) / 2.0)
            bottom.set_ylim(-limit, limit)
            if column == 0:
                top.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
            sidecar_panels.append(
                {
                    "label": label,
                    "title": title,
                    "k": list(PC_AT_K),
                    "training_seeds": [int(record["training_seed"]) for record in seeds],
                    "pair_completeness": series.tolist(),
                    "delta_from_mean_percentage_points": delta.tolist(),
                }
            )
        stem = "training_seed_pc_at_k"
        paths = _save_figure(fig, output_dir, stem, {"figure": stem, "panels": sidecar_panels})
        plt.close(fig)
    return paths


def _partition_matrix(
    comparison: dict[str, Any],
    *,
    metric: str,
) -> tuple[list[str], np.ndarray]:
    seeds = comparison["seeds"]
    split_ids = sorted(
        {split for record in seeds for split in record["partitions"]},
        key=lambda value: int(value),
    )
    if not split_ids:
        raise ValueError("No evaluation partitions were loaded")
    matrix = np.zeros((len(split_ids), len(seeds)), dtype=float)
    for column, record in enumerate(seeds):
        for row, split_id in enumerate(split_ids):
            partition = record["partitions"].get(split_id)
            if partition is None:
                raise ValueError(
                    f"Seed {record['training_seed']} is missing partition {split_id}"
                )
            if metric == "pc_at_10":
                matrix[row, column] = float(partition["pair_completeness_at_k"][10])
            elif metric == "calibrated_pc_rho_0.99":
                matrix[row, column] = float(
                    partition["calibrated"][CALIBRATION_RHO]["pair_completeness"]
                )
            else:
                raise ValueError(f"Unsupported partition metric {metric}")
    return split_ids, matrix


def render_partition_training_seed_span(
    comparison: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Within-split training-seed span against the spread across evaluation splits."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        ("pc_at_10", r"PC@$10$"),
        ("calibrated_pc_rho_0.99", r"PC at $\rho=0.99$"),
    )
    seeds = comparison["seeds"]
    with plt.rc_context(THESIS_RC_PARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.4), sharey=False)
        sidecar: dict[str, Any] = {"figure": "training_seed_vs_split", "panels": []}
        for axis, (metric, ylabel) in zip(axes, specs):
            split_ids, matrix = _partition_matrix(comparison, metric=metric)
            order = np.argsort(matrix.mean(axis=1))
            ordered = matrix[order]
            x = np.arange(len(split_ids))
            for column, record in enumerate(seeds):
                axis.plot(
                    x,
                    ordered[:, column],
                    color=_seed_color(column),
                    marker="o",
                    linestyle="none",
                    label=f"seed {int(record['training_seed'])}",
                    markersize=3.5,
                )
            for row in range(ordered.shape[0]):
                axis.plot(
                    [row, row],
                    [float(ordered[row].min()), float(ordered[row].max())],
                    color="#666666",
                    linewidth=0.7,
                    zorder=0,
                )
            axis.set_xticks(x)
            axis.set_xticklabels(
                [split_ids[int(index)] for index in order],
                rotation=60,
                ha="right",
                fontsize=7,
            )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Wikidata evaluation split")
            axis.set_ylim(
                max(0.0, float(ordered.min()) - 0.02),
                min(1.015, float(ordered.max()) + 0.012),
            )
            within = ordered.max(axis=1) - ordered.min(axis=1)
            sidecar["panels"].append(
                {
                    "metric": metric,
                    "split_seeds_ordered_by_mean": [split_ids[int(index)] for index in order],
                    "training_seeds": [int(record["training_seed"]) for record in seeds],
                    "values": ordered.tolist(),
                    "within_split_range": within.tolist(),
                    "median_within_split_range": float(np.median(within)),
                    "across_split_mean_range": float(
                        ordered.mean(axis=1).max() - ordered.mean(axis=1).min()
                    ),
                }
            )
        axes[1].legend(frameon=False, loc="lower right")
        stem = "training_seed_vs_eval_split"
        paths = _save_figure(fig, output_dir, stem, sidecar)
        plt.close(fig)
    return paths


def write_hplus_training_seed_charts(run_root: Path, output_dir: Path) -> dict[str, Any]:
    """Build the H+ 4-epoch seed-comparison figures from completed run directories."""
    run_dirs = sorted(
        path
        for path in run_root.glob("*2572903*trainseed-*")
        if path.is_dir()
    )
    comparison = collect_training_seed_comparison(
        run_dirs,
        full_evaluations={
            "syn": (
                "eval_museum_deduplication_hard_synth_v1_original_queries_"
                "class_preserving_val_test_50_50_medium_v1_calibrated_v1_"
                "static_finetuned_checkpoints"
            ),
            "wikidata_full": "eval_hard_synth_test_calibrated_wikidata_1_5_full_v1",
        },
        partition_evaluation="eval_hard_synth_checkpoint_wikidata_1_5_calibrated_v1_frozen",
    )
    paths = render_pc_at_k_by_training_seed(comparison, output_dir)
    paths.extend(render_partition_training_seed_span(comparison, output_dir))
    return {"status": "ok", "paths": [str(path) for path in paths]}
