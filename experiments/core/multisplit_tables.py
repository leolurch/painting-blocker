"""LaTeX tables for descriptive multisplit evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .latex_table_output import write_latex_table_with_values
from .latex_table_rendering import latex_escape, latex_label

ROW_END = " " + r"\\"


def _fmt_summary(row: dict[str, Any], metric: str, digits: int = 3) -> str:
    mean = row.get(f"{metric}_mean")
    sd = row.get(f"{metric}_sample_sd_split")
    lower = row.get(f"{metric}_min")
    upper = row.get(f"{metric}_max")
    if None in (mean, sd, lower, upper):
        return "--"
    return (
        f"${float(mean):.{digits}f} \\pm {float(sd):.{digits}f}$ "
        f"[{float(lower):.{digits}f}, {float(upper):.{digits}f}]"
    )


def _table_prelude(caption: str, label: str, alignment: str) -> list[str]:
    return [
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{latex_label(label)}}}",
        "    \\resizebox{\\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{alignment}}}",
        "        \\toprule",
    ]


def _finish(lines: list[str]) -> str:
    lines.extend(["        \\bottomrule", "    \\end{tabular}%", "    }", "\\end{table}"])
    return "\n".join(lines) + "\n"


def _single_table_setting(
    config: dict[str, Any], key: str, *, legacy_key: str | None = None
) -> Any:
    settings = ((config.get("charts") or {}).get("multisplit_tables") or {})
    value = settings.get(key)
    if value is None and legacy_key is not None:
        value = settings.get(legacy_key)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"multisplit_tables.{key} must define exactly one value")
        return value[0]
    return value


def _fixed_k(config: dict[str, Any]) -> int | None:
    value = _single_table_setting(config, "fixed_k")
    return int(value) if value is not None else None


def _calibration_target(config: dict[str, Any]) -> float | None:
    value = _single_table_setting(
        config, "calibration_target", legacy_key="calibration_targets"
    )
    return float(value) if value is not None else None


def _attainment_tolerance(config: dict[str, Any]) -> float:
    aggregation = ((config.get("multisplit") or {}).get("aggregation") or {})
    return float(aggregation.get("attainment_tolerance", 1e-12))


def _at_calibration_target(row: dict[str, Any], target: float) -> bool:
    value = row.get("calibration_target_pair_completeness")
    return value is not None and abs(float(value) - target) <= 1e-12


def _model_short_labels(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(entry["analysis_model_id"]): str(
            (entry.get("presentation") or {}).get("short_label")
            or (entry.get("presentation") or {}).get("label")
            or entry["analysis_model_id"]
        )
        for entry in config.get("models") or []
        if (
            isinstance(entry, dict)
            and entry.get("analysis_model_id") is not None
            and entry.get("include") is True
        )
    }


def _row_model_label(row: dict[str, Any], labels: dict[str, str]) -> str:
    model_id = str(row.get("configuration_id") or row.get("analysis_model_id") or "")
    return labels.get(model_id, str(row.get("display_name") or model_id))


def _extreme(rows: list[dict[str, Any]], key: str, *, highest: bool) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values) if highest else min(values)


def _winner_number(value: float, winner: float | None, digits: int) -> str:
    formatted = f"{value:.{digits}f}"
    if winner is not None and abs(value - winner) <= 1e-12:
        return f"\\mathbf{{{formatted}}}"
    return formatted


def _winner_mean_sd(
    row: dict[str, Any],
    mean_key: str,
    sd_key: str,
    *,
    digits: int,
    mean_winner: float | None,
    sd_winner: float | None,
) -> str:
    if row.get(mean_key) is None or row.get(sd_key) is None:
        return "--"
    mean = _winner_number(float(row[mean_key]), mean_winner, digits)
    sd = _winner_number(float(row[sd_key]), sd_winner, digits)
    return f"${mean} \\pm {sd}$"


def _test_attainment_counts(
    row: dict[str, Any], target: float, tolerance: float
) -> tuple[int, int]:
    split_values = row.get("pair_completeness_split_values") or []
    values = [
        float(value["mean_over_training_seeds"])
        for value in split_values
        if value.get("mean_over_training_seeds") is not None
    ]
    if values:
        return sum(value >= target - tolerance for value in values), len(values)
    return int(row.get("target_attainment_count", 0)), int(
        row["num_split_realizations"]
    )


def _test_attainment_cell(
    attained: int, total: int, winner: float
) -> str:
    value = f"{attained}/{total}"
    rate = attained / total if total else 0.0
    return f"\\textbf{{{attained}}}/{total}" if abs(rate - winner) <= 1e-12 else value


def _fixed_table(artifact: dict[str, Any], config: dict[str, Any]) -> str:
    configured_k = _fixed_k(config)
    rows = [row for row in artifact.get("summary_rows") or [] if row.get("selection_mode") == "top_k"]
    if rows and configured_k is None:
        raise ValueError("multisplit_tables.fixed_k is required for fixed-k rows")
    rows = [row for row in rows if int(row["k"]) == configured_k]
    if not rows:
        return ""
    rr_values = {
        tuple(row.get(f"reduction_ratio_{suffix}") for suffix in ("mean", "sample_sd_split", "min", "max"))
        for row in rows
    }
    if len(rr_values) != 1:
        raise ValueError(
            "Fixed-k reduction ratio is not model-independent and cannot be moved to the caption"
        )
    n = len(artifact["expected_split_seeds"])
    rr_summary = _fmt_summary(rows[0], "reduction_ratio", 4)
    caption = (
        f"Fixed-candidate-budget performance at $k={configured_k}$ over {n} predefined "
        "overlapping class-disjoint split realizations. Each split is weighted equally. Values "
        "are mean $\\pm$ sample SD ($ddof=1$) [observed minimum, maximum]. Observed ranges are "
        f"not confidence intervals. RR is model-independent at this budget: {rr_summary}. "
        "PC is the primary fixed-$k$ criterion; PQ@$k$ and F-scores are deterministic functions "
        "of the retained hits for a common population and budget and are not independent evidence."
    )
    lines = _table_prelude(caption, f"tab:{config['analysis_id']}-multisplit-fixed", "lrr")
    lines.extend([
        "        Model & PC mean $\\pm$ SD [range] & PQ mean $\\pm$ SD [range]" + ROW_END,
        "        \\midrule",
    ])
    for row in sorted(rows, key=lambda value: (str(value.get("display_name")), int(value["k"]))):
        lines.append(
            "        " + " & ".join([
                latex_escape(str(row.get("display_name") or row["configuration_id"])),
                _fmt_summary(row, "pair_completeness"),
                _fmt_summary(row, "pair_quality"),
            ]) + ROW_END
        )
    return _finish(lines)


def _calibration_table(artifact: dict[str, Any], config: dict[str, Any]) -> str:
    target = _calibration_target(config)
    rows = [row for row in artifact.get("summary_rows") or [] if row.get("selection_mode") == "calibrated_threshold"]
    if rows and target is None:
        raise ValueError(
            "multisplit_tables.calibration_target is required for calibrated rows"
        )
    rows = [row for row in rows if _at_calibration_target(row, target)]
    labels = _model_short_labels(config)
    rows = [
        row for row in rows
        if str(row.get("configuration_id") or row.get("analysis_model_id")) in labels
    ]
    if not rows:
        return ""
    n = len(artifact["expected_split_seeds"])
    default_caption = (
        f"Validation-calibrated performance over {n} predefined overlapping class-disjoint splits. "
        "Boldface marks the highest mean values for test PC and RR and the lowest "
        "corresponding $\\sigma$ values and candidate-pair mean."
    )
    table_settings = ((config.get("charts") or {}).get("multisplit_tables") or {})
    caption = str(table_settings.get("calibration_caption") or default_caption)
    tolerance = _attainment_tolerance(config)
    attainment_counts = {
        id(row): _test_attainment_counts(row, target, tolerance)
        for row in rows
    }
    attainment_winner = max(
        attained / total if total else 0.0
        for attained, total in attainment_counts.values()
    )
    winners = {
        "pc_mean": _extreme(rows, "pair_completeness_mean", highest=True),
        "pc_sd": _extreme(rows, "pair_completeness_sample_sd_split", highest=False),
        "rr_mean": _extreme(rows, "reduction_ratio_mean", highest=True),
        "rr_sd": _extreme(rows, "reduction_ratio_sample_sd_split", highest=False),
        "candidate_pairs": _extreme(rows, "candidate_pairs_mean", highest=False),
    }
    lines = _table_prelude(
        caption,
        f"tab:{config['analysis_id']}-multisplit-calibrated",
        "lrrrr",
    )
    lines.extend([
        f"        \\multicolumn{{5}}{{c}}{{Calibration target $\\rho={target:.3f}$}}" + ROW_END,
        "        \\midrule",
        "        Model & $\\mathrm{PC}_{\\mathrm{test}}$ mean $\\pm\\,\\sigma$"
        + " & $\\mathrm{RR}$ mean $\\pm\\,\\sigma$ & $|C|$ mean & "
        + f"$\\mathrm{{PC}}_{{\\mathrm{{test}}}} \\ge {target - tolerance:.3f}$"
        + ROW_END,
        "        \\midrule",
    ])
    for row in sorted(
        rows,
        key=lambda value: (
            _row_model_label(value, labels),
            float(value.get("calibration_target_pair_completeness") or 0.0),
        ),
    ):
        candidate_pairs = float(row["candidate_pairs_mean"])
        lines.append(
            "        " + " & ".join([
                latex_escape(_row_model_label(row, labels)),
                _winner_mean_sd(
                    row,
                    "pair_completeness_mean",
                    "pair_completeness_sample_sd_split",
                    digits=3,
                    mean_winner=winners["pc_mean"],
                    sd_winner=winners["pc_sd"],
                ),
                _winner_mean_sd(
                    row,
                    "reduction_ratio_mean",
                    "reduction_ratio_sample_sd_split",
                    digits=4,
                    mean_winner=winners["rr_mean"],
                    sd_winner=winners["rr_sd"],
                ),
                "$"
                + _winner_number(candidate_pairs, winners["candidate_pairs"], 1)
                + "$",
                _test_attainment_cell(
                    *attainment_counts[id(row)], attainment_winner
                ),
            ]) + ROW_END
        )
    return _finish(lines)


def _calibrated_top_k_table(artifact: dict[str, Any], config: dict[str, Any]) -> str:
    target = _calibration_target(config)
    rows = [row for row in artifact.get("summary_rows") or [] if row.get("selection_mode") == "calibrated_top_k"]
    if rows and target is None:
        raise ValueError(
            "multisplit_tables.calibration_target is required for calibrated rows"
        )
    rows = [row for row in rows if _at_calibration_target(row, target)]
    labels = _model_short_labels(config)
    rows = [
        row for row in rows
        if str(row.get("configuration_id") or row.get("analysis_model_id")) in labels
    ]
    if not rows:
        return ""
    caption = (
        "Validation-selected candidate budgets. For each split and training repeat, the smallest "
        "predeclared $k$ satisfying pooled validation PC was frozen and applied unchanged to test. "
        "Failure to satisfy the validation constraint and test target attainment are reported "
        "separately. Split summaries are equal-weighted and descriptive; ranges are not confidence intervals."
    )
    lines = _table_prelude(caption, f"tab:{config['analysis_id']}-multisplit-calibrated-k", "lrrrrrrr")
    lines.extend([
        "        Model & Selected $k$ mean $\\pm$ SD & Val attained & Test PC & Test attained & Test PQ & Test RR & Candidate pairs mean" + ROW_END,
        "        \\midrule",
    ])
    for row in sorted(
        rows,
        key=lambda value: (
            _row_model_label(value, labels),
            float(value.get("calibration_target_pair_completeness") or 0.0),
        ),
    ):
        selected_k = (
            f"${float(row['selected_k_mean']):.1f} \\pm {float(row['selected_k_sample_sd_split']):.1f}$"
            if row.get("selected_k_mean") is not None else "target not attained"
        )
        lines.append(
            "        " + " & ".join([
                latex_escape(_row_model_label(row, labels)),
                selected_k,
                f"{row.get('validation_selection_attainment_count', 0)}/{row['num_split_realizations']}",
                _fmt_summary(row, "pair_completeness"),
                f"{row.get('target_attainment_count', 0)}/{row['num_split_realizations']}",
                _fmt_summary(row, "pair_quality"),
                _fmt_summary(row, "reduction_ratio", 4),
                f"{float(row['candidate_pairs_mean']):.1f}" if row.get("candidate_pairs_mean") is not None else "--",
            ]) + ROW_END
        )
    return _finish(lines)


def _syn_test_comparison_table(
    artifact: dict[str, Any],
    config: dict[str, Any],
    synthetic_pc_by_model: dict[str, float],
) -> str:
    table_settings = ((config.get("charts") or {}).get("multisplit_tables") or {})
    settings = table_settings.get("syn_test_comparison") or {}
    if not bool(settings.get("enabled", False)):
        return ""

    model_ids = [str(value) for value in settings.get("models") or []]
    if not model_ids:
        raise ValueError("multisplit_tables.syn_test_comparison.models is required")
    labels = _model_short_labels(config)
    unavailable = [model_id for model_id in model_ids if model_id not in labels]
    if unavailable:
        raise ValueError(
            "Comparison-table models must exist and have include: true: "
            + ", ".join(unavailable)
        )

    target = float(settings.get("target_pc", 0.99))
    task = settings.get("task")
    calibrated_rows = [
        row for row in artifact.get("summary_rows") or []
        if row.get("selection_mode") == "calibrated_threshold"
        and _at_calibration_target(row, target)
        and (task is None or str(row.get("task_id")) == str(task))
    ]
    rows_by_model = {
        str(row.get("configuration_id") or row.get("analysis_model_id")): row
        for row in calibrated_rows
    }
    missing_synthetic = [
        model_id for model_id in model_ids if model_id not in synthetic_pc_by_model
    ]
    missing_test = [model_id for model_id in model_ids if model_id not in rows_by_model]
    if missing_synthetic or missing_test:
        details = []
        if missing_synthetic:
            details.append("missing SYN PC: " + ", ".join(missing_synthetic))
        if missing_test:
            details.append("missing calibrated test row: " + ", ".join(missing_test))
        raise ValueError("Comparison table is incomplete (" + "; ".join(details) + ")")

    synthetic_k = int(settings.get("synthetic_pc_at_k", 10))
    synthetic_winner = max(round(synthetic_pc_by_model[model_id], 3) for model_id in model_ids)
    test_pc_winner = max(
        round(float(rows_by_model[model_id]["pair_completeness_mean"]), 3)
        for model_id in model_ids
    )
    test_rr_winner = max(
        round(float(rows_by_model[model_id]["reduction_ratio_mean"]), 4)
        for model_id in model_ids
    )

    def format_winner(value: float, digits: int, winner: float) -> str:
        formatted = f"{value:.{digits}f}"
        return f"\\textbf{{{formatted}}}" if round(value, digits) == winner else formatted

    target_label = f"{target:.2f}".lstrip("0")
    lines = [
        "\\begin{tabular}{lrrr}",
        "    \\toprule",
        f"    Model & $\\mathrm{{PC}}_{{\\mathrm{{test}}}}$ at $k={synthetic_k}$"
        + f" & $\\mathrm{{PC}}_{{\\mathrm{{test}}}}$ at $\\rho_{{\\mathrm{{val}}}}={target_label}$"
        + f" & $\\mathrm{{RR}}_{{\\mathrm{{test}}}}$ at $\\rho_{{\\mathrm{{val}}}}={target_label}$"
        + ROW_END,
        "    \\midrule",
    ]
    for model_id in model_ids:
        row = rows_by_model[model_id]
        lines.append(
            "    "
            + " & ".join([
                latex_escape(labels[model_id]),
                format_winner(
                    synthetic_pc_by_model[model_id], 3, synthetic_winner
                ),
                format_winner(
                    float(row["pair_completeness_mean"]), 3, test_pc_winner
                ),
                format_winner(
                    float(row["reduction_ratio_mean"]), 4, test_rr_winner
                ),
            ])
            + ROW_END
        )
    lines.extend(["    \\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _cross_dataset_metric_bundle(
    artifact: dict[str, Any],
    config: dict[str, Any],
    synthetic_pc_by_model: dict[str, float],
    synthetic_calibrated_pc_by_model: dict[str, float] | None = None,
) -> dict[str, Any]:
    settings = (
        ((config.get("charts") or {}).get("multisplit_tables") or {}).get(
            "syn_test_comparison"
        )
        or {}
    )
    model_ids = [
        str(value)
        for value in settings.get("metric_bundle_models")
        or settings.get("models")
        or []
    ]
    target = float(settings.get("target_pc", 0.99))
    targets = [
        float(value) for value in settings.get("metric_bundle_targets") or [target]
    ]
    if target not in targets:
        targets.append(target)
    targets = list(dict.fromkeys(targets))
    task = settings.get("task")
    rows_by_target: dict[float, dict[str, dict[str, Any]]] = {}
    for calibration_target in targets:
        calibrated_rows = [
            row
            for row in artifact.get("summary_rows") or []
            if row.get("selection_mode") == "calibrated_threshold"
            and _at_calibration_target(row, calibration_target)
            and (task is None or str(row.get("task_id")) == str(task))
        ]
        rows_by_target[calibration_target] = {
            str(row.get("configuration_id") or row.get("analysis_model_id")): row
            for row in calibrated_rows
        }
        missing_wik = [
            model_id
            for model_id in model_ids
            if model_id not in rows_by_target[calibration_target]
        ]
        if missing_wik:
            raise ValueError(
                f"Cross-dataset metric bundle lacks WIK multisplit rows at "
                f"rho={calibration_target:g}: " + ", ".join(missing_wik)
            )
    rows_by_model = rows_by_target[target]
    synthetic_calibrated_pc_by_model = synthetic_calibrated_pc_by_model or {}
    return {
        "schema_version": 1,
        "artifact_type": "cross_dataset_table_metrics",
        "dataset_id": artifact.get("dataset_id"),
        "wik_task_id": task,
        "wik_calibration_target_pair_completeness": target,
        "wik_calibration_targets": targets,
        "wik_aggregation": "equal_split_mean",
        "num_split_realizations": len(artifact.get("expected_split_seeds") or []),
        "synthetic_fixed_k": int(settings.get("synthetic_pc_at_k", 10)),
        "synthetic_calibration_target_pair_completeness": target,
        "models": [
            {
                "model_id": model_id,
                "syn_test_pc_at_k": synthetic_pc_by_model.get(model_id),
                "syn_test_pc_at_calibrated_threshold": (
                    synthetic_calibrated_pc_by_model.get(model_id)
                ),
                "wik_test_pc_mean": float(
                    rows_by_model[model_id]["pair_completeness_mean"]
                ),
                "wik_test_pc_sample_sd": float(
                    rows_by_model[model_id]["pair_completeness_sample_sd_split"]
                ),
                "wik_test_pq_mean": float(
                    rows_by_model[model_id]["pair_quality_mean"]
                ),
                "wik_test_pq_sample_sd": float(
                    rows_by_model[model_id]["pair_quality_sample_sd_split"]
                ),
                "wik_test_rr_mean": float(
                    rows_by_model[model_id]["reduction_ratio_mean"]
                ),
                "wik_test_rr_sample_sd": float(
                    rows_by_model[model_id]["reduction_ratio_sample_sd_split"]
                ),
                "wik_calibrated_metrics_by_target": {
                    f"{calibration_target:g}": {
                        "wik_test_pc_mean": float(
                            rows_by_target[calibration_target][model_id][
                                "pair_completeness_mean"
                            ]
                        ),
                        "wik_test_pc_sample_sd": float(
                            rows_by_target[calibration_target][model_id][
                                "pair_completeness_sample_sd_split"
                            ]
                        ),
                        "wik_test_pq_mean": float(
                            rows_by_target[calibration_target][model_id][
                                "pair_quality_mean"
                            ]
                        ),
                        "wik_test_pq_sample_sd": float(
                            rows_by_target[calibration_target][model_id][
                                "pair_quality_sample_sd_split"
                            ]
                        ),
                        "wik_test_rr_mean": float(
                            rows_by_target[calibration_target][model_id][
                                "reduction_ratio_mean"
                            ]
                        ),
                        "wik_test_rr_sample_sd": float(
                            rows_by_target[calibration_target][model_id][
                                "reduction_ratio_sample_sd_split"
                            ]
                        ),
                    }
                    for calibration_target in targets
                },
            }
            for model_id in model_ids
        ],
    }


def _apply_synthetic_model_id_aliases(
    values: dict[str, float], settings: dict[str, Any]
) -> dict[str, float]:
    """Expose sidecar metrics under the corresponding multisplit model IDs."""
    configured = settings.get("synthetic_model_id_aliases") or {}
    if not isinstance(configured, dict):
        raise ValueError(
            "synthetic_model_id_aliases must map multisplit model IDs to "
            "synthetic-sidecar model IDs"
        )
    resolved = dict(values)
    for model_id, source_model_id in configured.items():
        model_id = str(model_id)
        source_model_id = str(source_model_id)
        if source_model_id not in values:
            raise ValueError(
                "Synthetic model alias source is absent from the sidecar: "
                f"{model_id} -> {source_model_id}"
            )
        if model_id in values and values[model_id] != values[source_model_id]:
            raise ValueError(
                "Synthetic model alias conflicts with a direct sidecar value: "
                f"{model_id} -> {source_model_id}"
            )
        resolved[model_id] = values[source_model_id]
    return resolved


def _load_synthetic_calibrated_pc_sidecar(
    output_dir: Path, settings: dict[str, Any]
) -> dict[str, float]:
    configured = settings.get("synthetic_calibrated_source")
    if not configured:
        raise ValueError(
            "multisplit_tables.syn_test_comparison.synthetic_calibrated_source is required"
        )
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = (output_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic calibrated-PC sidecar not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_target = float(settings.get("target_pc", 0.99))
    observed_target = payload.get("calibration_target_pair_completeness")
    if observed_target is None or abs(float(observed_target) - expected_target) > 1e-12:
        raise ValueError(
            "Synthetic calibrated-PC sidecar target does not match configured "
            f"target {expected_target:g}: {observed_target!r}"
        )
    values = {
        str(row["model_id"]): float(row["pc"])
        for row in payload.get("bars") or []
        if row.get("pc") is not None
    }
    return _apply_synthetic_model_id_aliases(values, settings)


def _load_synthetic_pc_sidecar(
    output_dir: Path, settings: dict[str, Any]
) -> dict[str, float]:
    configured = settings.get("synthetic_pc_source")
    if not configured:
        raise ValueError(
            "multisplit_tables.syn_test_comparison.synthetic_pc_source is required"
        )
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = (output_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic PC sidecar not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_k = int(settings.get("synthetic_pc_at_k", 10))
    if int(payload.get("candidate_budget", -1)) != expected_k:
        raise ValueError(
            f"Synthetic PC sidecar budget {payload.get('candidate_budget')} does not "
            f"match configured k={expected_k}"
        )
    values = {
        str(row["model_id"]): float(row["pair_completeness"])
        for row in payload.get("bars") or []
    }
    return _apply_synthetic_model_id_aliases(values, settings)


def _paired_table(artifact: dict[str, Any], config: dict[str, Any]) -> str:
    configured_k = _fixed_k(config)
    target = _calibration_target(config)
    rows = [
        row for row in artifact.get("paired_comparisons") or []
        if (
            row.get("calibration_target_pair_completeness") is not None
            and target is not None
            and _at_calibration_target(row, target)
        )
        or (
            row.get("calibration_target_pair_completeness") is None
            and configured_k is not None
            and int(row.get("k") or -1) == configured_k
        )
    ]
    if not rows:
        return ""
    n = len(artifact["expected_split_seeds"])
    caption = (
        f"Paired differences on the same {n} predefined split realizations. Differences are joined "
        "by exact split identity. Values are mean $\\pm$ sample SD [observed range]; W--T--L counts "
        "are descriptive consistency summaries, not significance tests. Splits overlap and observed "
        "ranges are not confidence intervals."
    )
    lines = _table_prelude(caption, f"tab:{config['analysis_id']}-multisplit-paired", "llrrr")
    lines.extend([
        "        Comparison & Operating point & $\\Delta$PC mean $\\pm$ SD [range]; W--T--L & $\\Delta$PQ mean $\\pm$ SD [range]; W--T--L & $\\Delta$RR mean $\\pm$ SD [range]; W--T--L" + ROW_END,
        "        \\midrule",
    ])
    for row in rows:
        target = row.get("calibration_target_pair_completeness")
        operating = f"$\\rho={float(target):.3f}$" if target is not None else f"$k={row.get('k')}$"
        cells = [latex_escape(str(row["comparison_id"])), operating]
        for metric, digits in (("pair_completeness", 3), ("pair_quality", 3), ("reduction_ratio", 4)):
            mean = row.get(f"{metric}_difference_mean")
            if mean is None:
                cells.append("--")
                continue
            sd = float(row[f"{metric}_difference_sample_sd_split"])
            lower = float(row[f"{metric}_difference_min"])
            upper = float(row[f"{metric}_difference_max"])
            wins = int(row.get(f"{metric}_wins", 0))
            ties = int(row.get(f"{metric}_ties", 0))
            losses = int(row["num_split_realizations"]) - wins - ties
            cells.append(
                f"${float(mean):+.{digits}f} \\pm {sd:.{digits}f}$ "
                f"[{lower:+.{digits}f}, {upper:+.{digits}f}]; {wins}--{ties}--{losses}"
            )
        lines.append("        " + " & ".join(cells) + ROW_END)
    return _finish(lines)


def _denominator_table(artifact: dict[str, Any], config: dict[str, Any]) -> str:
    populations = list(artifact.get("split_populations") or [])
    if not populations:
        return ""
    caption = (
        "Per-split evaluation denominators. Counts are retained for audit and are never pooled "
        "across overlapping split realizations."
    )
    lines = _table_prelude(caption, f"tab:{config['analysis_id']}-multisplit-denominators", "llrrrrrr")
    lines.extend([
        "        Seed & Task & Queries & Candidates & Positive pairs & Cartesian pairs & Eligible pairs & Excluded pairs" + ROW_END,
        "        \\midrule",
    ])
    for row in populations:
        lines.append(
            "        " + " & ".join([
                str(row["split_seed"]),
                latex_escape(str(row["task_id"])),
                str(row.get("num_queries") or 0),
                str(row.get("num_candidates") or 0),
                str(row.get("num_positive_pairs") or 0),
                str(row.get("num_cartesian_pairs") or 0),
                str(row.get("num_possible_pairs") or 0),
                str(row.get("num_excluded_pairs") or 0),
            ]) + ROW_END
        )
    return _finish(lines)


def _profile_table(artifact: dict[str, Any], config: dict[str, Any]) -> str:
    target = _calibration_target(config)
    rows = [
        row for row in artifact.get("profile_diagnostic_summaries") or []
        if row.get("selection_mode") == "calibrated_threshold"
    ]
    if rows and target is None:
        raise ValueError(
            "multisplit_tables.calibration_target is required for calibrated rows"
        )
    rows = [row for row in rows if _at_calibration_target(row, target)]
    if not rows:
        return ""
    caption = (
        "Profile-specific diagnostic performance at the one globally validation-selected pooled "
        "operating point. Profile rows do not tune independent thresholds and are not part of the "
        "formal calibration objective. Values are equal-weighted split means $\\pm$ descriptive "
        "sample SD; materially weaker profiles must be interpreted explicitly."
    )
    lines = _table_prelude(caption, f"tab:{config['analysis_id']}-profile-diagnostics", "lllrrrr")
    lines.extend([
        "        Model & Profile & $\\rho$ & PC mean $\\pm$ SD & PQ mean & RR mean & Candidate pairs mean" + ROW_END,
        "        \\midrule",
    ])
    labels = {
        str(entry["analysis_model_id"]): str((entry.get("presentation") or {}).get("label") or entry["analysis_model_id"])
        for entry in config.get("models") or [] if isinstance(entry, dict)
    }
    for row in sorted(rows, key=lambda value: (labels.get(str(value["analysis_model_id"]), ""), str(value["profile"]), float(value.get("calibration_target_pair_completeness") or 0.0))):
        model = str(row["analysis_model_id"])
        lines.append(
            "        " + " & ".join([
                latex_escape(labels.get(model, model)),
                latex_escape(str(row["profile"])),
                f"{float(row['calibration_target_pair_completeness']):.3f}",
                f"${float(row['pair_completeness_mean']):.3f} \\pm {float(row['pair_completeness_sample_sd_split']):.3f}$",
                f"{float(row.get('pair_quality_mean', 0.0)):.3f}",
                f"{float(row.get('reduction_ratio_mean', 0.0)):.4f}",
                f"{float(row.get('candidate_pairs_mean', 0.0)):.1f}",
            ]) + ROW_END
        )
    return _finish(lines)


def write_multisplit_tables(
    artifact: dict[str, Any], config: dict[str, Any], output_dir: Path
) -> list[Path]:
    settings = ((config.get("charts") or {}).get("multisplit_tables") or {})
    if not bool(settings.get("enabled", False)):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "multisplit_fixed_k.tex": _fixed_table(artifact, config),
        "multisplit_calibration.tex": _calibration_table(artifact, config),
        "multisplit_calibrated_top_k.tex": _calibrated_top_k_table(artifact, config),
        "multisplit_paired.tex": _paired_table(artifact, config),
        "multisplit_denominators.tex": _denominator_table(artifact, config),
        "multisplit_profile_diagnostics.tex": _profile_table(artifact, config),
    }
    comparison_settings = settings.get("syn_test_comparison") or {}
    metric_bundle: dict[str, Any] | None = None
    if bool(comparison_settings.get("enabled", False)):
        synthetic_pc = _load_synthetic_pc_sidecar(output_dir, comparison_settings)
        synthetic_calibrated_pc = _load_synthetic_calibrated_pc_sidecar(
            output_dir, comparison_settings
        )
        contents["multisplit_syn_test_comparison.tex"] = _syn_test_comparison_table(
            artifact, config, synthetic_pc
        )
        metric_bundle = _cross_dataset_metric_bundle(
            artifact, config, synthetic_pc, synthetic_calibrated_pc
        )
    paths: list[Path] = []
    for name, content in contents.items():
        if not content:
            continue
        path = output_dir / name
        paths.extend(write_latex_table_with_values(path, content))
    if metric_bundle is not None:
        bundle_path = output_dir / "multisplit_cross_dataset_metrics.json"
        atomic_write_json(bundle_path, metric_bundle)
        paths.append(bundle_path)
    return paths
