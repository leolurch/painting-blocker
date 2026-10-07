"""Reference-style blocking LaTeX tables for one run.

Kept separate from ``latex_tables`` on purpose: these are additive outputs in
the reference format (``\\best``/``\\secondbest`` column marking, PQ in
percent, RR as comments). The existing fixed_k/calibrated tables and their
files are not touched.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .aggregate_runs import aggregate_paths
from .latex_table_output import LATEX_TABLE_DEPENDENCIES, write_latex_table_with_values
from .latex_table_rendering import latex_escape
from .latex_tables import (
    _dataset_label,
    _filter_rows,
    _label_prefix,
    _load_run_json,
    _model_labels,
    _ordered_models,
    _table_context,
)

MACRO_HEADER = LATEX_TABLE_DEPENDENCIES
INLINE_TABLE_HEADER = (
    MACRO_HEADER
    + "% Inline-table dependencies:\n"
    + "%   \\usepackage{float}\n"
    + "%   \\usepackage{adjustbox}\n"
)
ROW_END = " " + r"\\"


def write_run_blocking_tables(
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    task_id: str | None = None,
) -> dict[str, Path]:
    """Write the reference-style blocking tables for one completed run directory."""
    run_path = Path(run_dir).expanduser()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    out_dir = Path(output_dir).expanduser() if output_dir is not None else run_path / "latex_tables"
    rows = aggregate_paths([run_path])
    run_meta = _load_run_json(run_path)
    model_ids = _ordered_models(run_meta, rows)
    model_labels = _model_labels(run_path, model_ids, run_meta, rows)
    contents = {
        "blocking_fixed_k": render_blocking_fixed_k_table(rows, run_meta, model_ids, model_labels, task_id),
        "blocking_calibrated": render_blocking_calibrated_table(rows, run_meta, model_ids, model_labels, task_id),
    }
    paths: dict[str, Path] = {}
    for key, content in contents.items():
        if not content:
            continue
        path = out_dir / f"{key}.tex"
        written = write_latex_table_with_values(path, MACRO_HEADER + content)
        paths[key] = path
        for index, values_path in enumerate(written[1:], start=1):
            suffix = "values" if len(written) == 2 else f"values_{index}"
            paths[f"{key}_{suffix}"] = values_path
    return paths


def render_blocking_fixed_k_table(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
    model_ids: list[str],
    model_labels: dict[str, str],
    task_id: str | None = None,
    ks: list[int] | tuple[int, ...] | None = None,
    model_groups: dict[str, str] | None = None,
    model_group_order: list[str] | tuple[str, ...] | None = None,
    model_order_within_groups: dict[str, list[str] | tuple[str, ...]] | None = None,
    pinned_models: list[str] | tuple[str, ...] | None = None,
    baseline_comparison_model: str | None = None,
    baseline_comparison_group: str | None = None,
    baseline_comparison_position: str = "first",
    baseline_comparison_separator: str | None = None,
    alphabetical_within_groups: bool = False,
    separate_model_groups: bool = False,
    highlight_best_per_group: bool = False,
    comparison_metrics: dict[str, dict[str, Any]] | None = None,
    performance_columns: list[str | dict[str, Any]] | tuple[str | dict[str, Any], ...] | None = None,
    performance_values: dict[str, dict[str, float | None]] | None = None,
    performance_placeholder: str = "-",
    performance_group_heading: str = "Performance requirements",
    vertical_group_rules: bool = False,
    comparison_synthetic_k: int = 80,
    comparison_wik_target: float = 0.99,
    comparison_wik_targets: list[float] | tuple[float, ...] | None = None,
    show_comparison_sd: bool = True,
) -> str:
    """Fixed-budget table: PC@k and PQ@k as fractions, best/second-best marked.

    RR@k is appended as LaTeX comments because for fixed k it is identical across
    models within a task; the stored reduction_ratio is reported rather than the
    idealized ``1 - k/|T|``, so excluded pairs and dropped queries stay accounted.
    """
    filtered = _filter_rows(rows, "top_k", task_id)
    if not filtered:
        return ""
    available_ks = sorted(
        {int(row["k"]) for row in filtered if row.get("k") is not None}
    )
    selected_ks = _selected_fixed_ks(available_ks, ks)
    filtered = [row for row in filtered if int(row.get("k") or -1) in selected_ks]
    context = _table_context(rows, filtered, model_ids, model_labels, task_id)
    include_task = context["include_task"]
    if baseline_comparison_model is not None and baseline_comparison_model not in model_ids:
        raise ValueError(
            "blocking_tables.baseline_comparison_model is not a displayed model: "
            f"{baseline_comparison_model}"
        )
    if baseline_comparison_position not in {"first", "last"}:
        raise ValueError(
            "blocking_tables.baseline_comparison_position must be 'first' or 'last'"
        )
    if baseline_comparison_separator not in {None, "dashed"}:
        raise ValueError(
            "blocking_tables.baseline_comparison_separator must be 'dashed' or null"
        )
    effective_groups = dict(model_groups or {})
    if baseline_comparison_model is not None and baseline_comparison_group is not None:
        effective_groups[baseline_comparison_model] = baseline_comparison_group
    effective_pins = list(pinned_models or [])
    trailing_models: list[str] = []
    if baseline_comparison_model is not None:
        if baseline_comparison_position == "first":
            if baseline_comparison_model not in effective_pins:
                effective_pins.insert(0, baseline_comparison_model)
        else:
            trailing_models.append(baseline_comparison_model)
    row_specs = _ordered_group_row_specs(
        context["row_specs"],
        effective_groups,
        model_group_order,
        model_order_within_groups,
        effective_pins,
        trailing_models,
        alphabetical_within_groups,
    )
    row_groups = {
        row_key: effective_groups.get(row_key[0], "ungrouped")
        for row_key, _label in row_specs
    }
    values: dict[tuple[tuple[str, str | None], int, str], float] = {}
    for row in filtered:
        row_key = (str(row.get("model_id")), str(row.get("task_id")) if include_task else None)
        for metric in ("pair_completeness", "pair_quality"):
            value = row.get(metric)
            if value is not None:
                values[(row_key, int(row["k"]), metric)] = float(value)
    if highlight_best_per_group:
        group_marks = _group_best_marks(
            row_specs, row_groups, selected_ks, values
        )
        global_marks = _global_best_marks(row_specs, selected_ks, values)
        marks = {
            key: ("best" if key in global_marks else "second")
            for key in group_marks
        }
    else:
        marks = _best_second_marks(row_specs, selected_ks, values)
    comparison_metrics = comparison_metrics or {}
    performance_values = performance_values or {}
    configured_performance_specs = _performance_column_specs(performance_columns)
    configured_performance_columns = [
        str(spec["label"]) for spec in configured_performance_specs
    ]
    if not performance_placeholder:
        raise ValueError("blocking_tables.performance_placeholder must not be empty")
    has_comparison = bool(comparison_metrics)
    selected_comparison_targets = [
        float(value)
        for value in comparison_wik_targets or [comparison_wik_target]
    ]
    if len(selected_comparison_targets) != len(set(selected_comparison_targets)):
        raise ValueError("blocking comparison calibration targets must be unique")
    comparison_display_values: dict[str, dict[str, float | None]] = {}
    comparison_specs: list[dict[str, Any]] = [
        {"key": "syn_test_pc_at_k", "objective": "max"}
    ]
    for target in selected_comparison_targets:
        prefix = f"rho{str(target).replace('.', 'p')}"
        comparison_specs.extend([
            {"key": f"{prefix}_pc_mean", "objective": "max"},
            *(
                [{"key": f"{prefix}_pc_sd", "objective": "min"}]
                if show_comparison_sd
                else []
            ),
            {"key": f"{prefix}_rr_mean", "objective": "max"},
            *(
                [{"key": f"{prefix}_rr_sd", "objective": "min"}]
                if show_comparison_sd
                else []
            ),
            {"key": f"{prefix}_pq_mean", "objective": "max"},
        ])
        target_key = f"{target:g}"
        for model_id in model_ids:
            source = comparison_metrics.get(model_id, {})
            target_metrics = (
                source.get("wik_calibrated_metrics_by_target") or {}
            ).get(target_key)
            if target_metrics is None and math.isclose(
                target, comparison_wik_target, rel_tol=1e-12, abs_tol=1e-12
            ):
                target_metrics = source
            target_metrics = target_metrics or {}
            destination = comparison_display_values.setdefault(model_id, {})
            destination["syn_test_pc_at_k"] = source.get("syn_test_pc_at_k")
            destination.update({
                f"{prefix}_pc_mean": target_metrics.get("wik_test_pc_mean"),
                f"{prefix}_pc_sd": target_metrics.get("wik_test_pc_sample_sd"),
                f"{prefix}_rr_mean": target_metrics.get("wik_test_rr_mean"),
                f"{prefix}_rr_sd": target_metrics.get("wik_test_rr_sample_sd"),
                f"{prefix}_pq_mean": target_metrics.get("wik_test_pq_mean"),
            })
    comparison_marks: dict[tuple[tuple[str, str | None], str], str] = {}
    performance_marks: dict[tuple[tuple[str, str | None], str], str] = {}
    if highlight_best_per_group and comparison_metrics:
        group_comparison_marks = _performance_group_extreme_marks(
            row_specs, row_groups, comparison_display_values, comparison_specs
        )
        global_comparison_marks = _performance_global_extreme_marks(
            row_specs, comparison_display_values, comparison_specs
        )
        comparison_marks = {
            key: ("best" if key in global_comparison_marks else "second")
            for key in group_comparison_marks
        }
    if highlight_best_per_group and performance_values:
        group_performance_marks = _performance_group_extreme_marks(
            row_specs, row_groups, performance_values, configured_performance_specs
        )
        global_performance_marks = _performance_global_extreme_marks(
            row_specs, performance_values, configured_performance_specs
        )
        performance_marks = {
            key: ("best" if key in global_performance_marks else "second")
            for key in group_performance_marks
        }

    n_k = len(selected_ks)
    dataset = latex_escape(_dataset_label(run_meta))
    fixed_metrics = (
        ("pair_completeness",)
        if has_comparison
        else ("pair_completeness", "pair_quality")
    )
    performance_column_count = (
        len(configured_performance_columns) if has_comparison else 0
    )
    # Merged layout: SYN PC + WIK PC@k + calibrated PC/RR/PQ per target.
    extra_columns = (
        1 + 3 * len(selected_comparison_targets) + performance_column_count
        if has_comparison
        else 0
    )
    numeric_columns = len(fixed_metrics) * n_k + extra_columns
    total_columns = 1 + numeric_columns
    if has_comparison:
        performance_sentence = (
            " We report peak VRAM in GiB ($\\overline{\\mathrm{VRAM}}$)."
            if performance_column_count
            else ""
        )
        highlight_sentence = (
            " Global best values per column are \\best{bold}; the best value "
            "within each remaining model group is \\secondbest{underlined}."
            if highlight_best_per_group
            else " Best per column is \\best{bold}; second best is "
            "\\secondbest{underlined}."
        )
        target_set = ",".join(
            f"{target:.2f}".lstrip("0") for target in selected_comparison_targets
        )
        if show_comparison_sd:
            calibration_sentence = (
                " Calibrated $PC$ and $RR$ are reported as mean $\\pm$ sample SD "
                f"across splits at $\\rho\\in\\{{{target_set}\\}}$; calibrated "
                "$PQ$ is reported as a mean. Lowest values are best for SD"
                + (
                    " and $\\overline{\\mathrm{VRAM}}$"
                    if performance_column_count
                    else ""
                )
                + "; highest values are best otherwise."
            )
        else:
            calibration_sentence = (
                " Calibrated $PC$, $RR$, and $PQ$ are reported as means across "
                f"splits at $\\rho\\in\\{{{target_set}\\}}$."
                + (
                    " Lowest values are best for $\\overline{\\mathrm{VRAM}}$;"
                    " highest values are best otherwise."
                    if performance_column_count
                    else " Highest values are best."
                )
            )
        caption = (
            "Blocking performance across \\emph{WIK} and \\emph{SYN}, with and "
            "without $\\rho$-calibration."
            + performance_sentence
            + highlight_sentence
            + calibration_sentence
        )
    else:
        caption = (
            f"Fixed-$k$ blocking performance on {dataset}: pair completeness "
            "(PC@$k$) and pairs quality (PQ@$k$) per candidate budget $k$."
        )
    if has_comparison and vertical_group_rules:
        tabular_groups = [
            "l",
            "r",
            "r" * n_k,
            *(["r" * 3] * len(selected_comparison_targets)),
        ]
        if performance_column_count:
            tabular_groups.append("r" * performance_column_count)
        tabular_spec = "|".join(tabular_groups)
    else:
        tabular_spec = "l" + "r" * numeric_columns
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{tab:{_label_prefix(run_meta, task_id)}-blocking-fixed-k}}",
        "    \\resizebox{\\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{tabular_spec}}}",
        "        \\toprule",
    ]
    if has_comparison:
        syn_column = 2
        wik_pc_start = 3
        wik_pc_end = wik_pc_start + n_k - 1
        separated_alignment = "c|" if vertical_group_rules else "c"
        final_alignment = "c"
        top_header = (
            f"        \\multirow{{2}}{{*}}{{Model}}"
            f" & \\multicolumn{{1}}{{{separated_alignment}}}"
            "{$PC^{sel}_{\\emph{SYN}}$}"
            f" & \\multicolumn{{{n_k}}}{{{separated_alignment}}}"
            "{$PC^{full}_{\\emph{WIK}}$}"
        )
        cmidrules = (
            f"        \\cmidrule(lr){{{syn_column}-{syn_column}}}"
            f" \\cmidrule(lr){{{wik_pc_start}-{wik_pc_end}}}"
        )
        next_column = wik_pc_end + 1
        for index, target in enumerate(selected_comparison_targets):
            calibrated_start = next_column
            calibrated_end = calibrated_start + 2
            target_is_last_group = (
                index == len(selected_comparison_targets) - 1
                and performance_column_count == 0
            )
            alignment = final_alignment if target_is_last_group else separated_alignment
            target_label = f"{target:.2f}".lstrip("0")
            top_header += (
                f" & \\multicolumn{{3}}{{{alignment}}}"
                f"{{$\\rho^{{cal}}_{{\\emph{{WIK}}}}={target_label}$}}"
            )
            cmidrules += (
                f" \\cmidrule(lr){{{calibrated_start}-{calibrated_end}}}"
            )
            next_column = calibrated_end + 1
        if performance_column_count:
            performance_start = next_column
            performance_end = performance_start + performance_column_count - 1
            top_header += (
                f" & \\multicolumn{{{performance_column_count}}}"
                f"{{{final_alignment}}}{{{performance_group_heading}}}"
            )
            cmidrules += (
                f" \\cmidrule(lr){{{performance_start}-{performance_end}}}"
            )
        top_header += ROW_END
        target_subheaders = [
            "$PC^{test}_{\\emph{WIK}}$",
            "$RR^{test}_{\\emph{WIK}}$",
            "$PQ^{test}_{\\emph{WIK}}$",
        ]
        subheaders = [
            f"$k={comparison_synthetic_k}$",
            *[f"$k={k}$" for k in selected_ks],
            *(target_subheaders * len(selected_comparison_targets)),
            *configured_performance_columns,
        ]
        lines.extend([
            top_header,
            cmidrules,
            "        & " + " & ".join(subheaders) + ROW_END,
        ])
    else:
        lines.extend([
            f"        \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{{{n_k}}}{{c}}{{PC@$k$}}"
            f" & \\multicolumn{{{n_k}}}{{c}}{{PQ@$k$}}{ROW_END}",
            f"        \\cmidrule(lr){{2-{1 + n_k}}} \\cmidrule(lr){{{2 + n_k}-{1 + 2 * n_k}}}",
            "        & " + " & ".join([f"$k={k}$" for k in selected_ks] * 2) + ROW_END,
        ])
    lines.append("        \\midrule")
    previous_group: str | None = None
    for row_key, label_text in row_specs:
        group = row_groups[row_key]
        if (
            row_key[0] == baseline_comparison_model
            and baseline_comparison_separator == "dashed"
            and previous_group is not None
        ):
            lines.append(_full_width_dashed_rule(total_columns))
        if separate_model_groups and previous_group is not None and group != previous_group:
            lines.append("        \\midrule")
        previous_group = group
        cells = []
        comparison = comparison_display_values.get(row_key[0], {})
        if has_comparison:
            cells.append(
                _format_comparison_cell(
                    comparison.get("syn_test_pc_at_k"),
                    comparison_marks.get((row_key, "syn_test_pc_at_k")),
                )
            )
        for metric in fixed_metrics:
            for k in selected_ks:
                value = values.get((row_key, k, metric))
                if value is None or not math.isfinite(value):
                    cells.append("--")
                    continue
                text = _fmt_without_leading_zero(value)
                mark = marks.get((row_key, k, metric))
                if mark == "best":
                    text = f"\\best{{{text}}}"
                elif mark == "second":
                    text = f"\\secondbest{{{text}}}"
                cells.append(text)
        if has_comparison:
            for target in selected_comparison_targets:
                prefix = f"rho{str(target).replace('.', 'p')}"
                pc_cell = (
                    _format_comparison_mean_sd_cell(
                        comparison.get(f"{prefix}_pc_mean"),
                        comparison.get(f"{prefix}_pc_sd"),
                        comparison_marks.get((row_key, f"{prefix}_pc_mean")),
                        comparison_marks.get((row_key, f"{prefix}_pc_sd")),
                    )
                    if show_comparison_sd
                    else _format_comparison_cell(
                        comparison.get(f"{prefix}_pc_mean"),
                        comparison_marks.get((row_key, f"{prefix}_pc_mean")),
                    )
                )
                rr_cell = (
                    _format_comparison_mean_sd_cell(
                        comparison.get(f"{prefix}_rr_mean"),
                        comparison.get(f"{prefix}_rr_sd"),
                        comparison_marks.get((row_key, f"{prefix}_rr_mean")),
                        comparison_marks.get((row_key, f"{prefix}_rr_sd")),
                    )
                    if show_comparison_sd
                    else _format_comparison_cell(
                        comparison.get(f"{prefix}_rr_mean"),
                        comparison_marks.get((row_key, f"{prefix}_rr_mean")),
                    )
                )
                cells.extend([
                    pc_cell,
                    rr_cell,
                    _format_comparison_cell(
                        comparison.get(f"{prefix}_pq_mean"),
                        comparison_marks.get((row_key, f"{prefix}_pq_mean")),
                    ),
                ])
            model_performance = performance_values.get(row_key[0], {})
            for spec in configured_performance_specs:
                cells.append(
                    _format_performance_cell(
                        model_performance.get(str(spec["key"])),
                        spec,
                        performance_marks.get((row_key, str(spec["key"]))),
                        performance_placeholder,
                    )
                )
        lines.append(
            f"        {latex_escape(label_text)} & " + " & ".join(cells) + ROW_END
        )
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table}",
    ])
    lines.extend(_rr_comment_lines(filtered, selected_ks, include_task))
    return "\n".join(lines) + "\n"


def render_blocking_performance_table(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
    model_ids: list[str],
    model_labels: dict[str, str],
    task_id: str | None,
    *,
    performance_columns: list[dict[str, Any]],
    performance_values: dict[str, dict[str, float | None]],
    performance_placeholder: str = "-",
    performance_group_heading: str = "B200, $B=1$",
    model_groups: dict[str, str] | None = None,
    model_group_order: list[str] | tuple[str, ...] | None = None,
    model_order_within_groups: dict[str, list[str] | tuple[str, ...]] | None = None,
    baseline_comparison_model: str | None = None,
    baseline_comparison_group: str | None = None,
    baseline_comparison_position: str = "first",
    baseline_comparison_separator: str | None = None,
    alphabetical_within_groups: bool = False,
    separate_model_groups: bool = False,
    highlight_best_per_group: bool = False,
    vertical_group_rules: bool = False,
) -> str:
    """Render a standalone single-image embedding performance table."""
    column_specs = _performance_column_specs(performance_columns)
    if not column_specs:
        return ""
    filtered = _filter_rows(rows, "top_k", task_id)
    if not filtered:
        return ""
    context = _table_context(rows, filtered, model_ids, model_labels, task_id)
    effective_groups = dict(model_groups or {})
    if baseline_comparison_model is not None and baseline_comparison_group is not None:
        effective_groups[baseline_comparison_model] = baseline_comparison_group
    pins: list[str] = []
    trailing: list[str] = []
    if baseline_comparison_model is not None:
        if baseline_comparison_position == "first":
            pins.append(baseline_comparison_model)
        elif baseline_comparison_position == "last":
            trailing.append(baseline_comparison_model)
        else:
            raise ValueError(
                "blocking_tables.baseline_comparison_position must be 'first' or 'last'"
            )
    row_specs = _ordered_group_row_specs(
        context["row_specs"],
        effective_groups,
        model_group_order,
        model_order_within_groups,
        pins,
        trailing,
        alphabetical_within_groups,
    )
    row_groups = {
        row_key: effective_groups.get(row_key[0], "ungrouped")
        for row_key, _label in row_specs
    }
    marks: dict[tuple[tuple[str, str | None], str], str] = {}
    if highlight_best_per_group:
        group_marks = _performance_group_extreme_marks(
            row_specs, row_groups, performance_values, column_specs
        )
        global_marks = _performance_global_extreme_marks(
            row_specs, performance_values, column_specs
        )
        marks = {
            key: ("best" if key in global_marks else "second")
            for key in group_marks
        }
    total_columns = 1 + len(column_specs)
    tabular_spec = (
        "l|" + "r" * len(column_specs)
        if vertical_group_rules
        else "l" + "r" * len(column_specs)
    )
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\caption{Single-image embedding performance on an NVIDIA B200 at "
        "batch size $B=1$. We report pooled end-to-end p50 and p90 latency, "
        "sequential throughput in images per second, and peak VRAM in GiB "
        "($\\mathrm{VRAM}_{\\max}$). Global best values per column are "
        "\\best{bold}; the best value within each remaining model group is "
        "\\secondbest{underlined}. Lowest values are best for latency and "
        "$\\mathrm{VRAM}_{\\max}$; highest values are best otherwise.}",
        f"    \\label{{tab:{_label_prefix(run_meta, task_id)}-blocking-performance}}",
        "    \\begin{adjustbox}{max width=\\textwidth}",
        f"    \\begin{{tabular}}{{{tabular_spec}}}",
        "        \\toprule",
        f"        \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{{{len(column_specs)}}}{{c}}{{{performance_group_heading}}}{ROW_END}",
        f"        \\cmidrule(lr){{2-{total_columns}}}",
        "        & " + " & ".join(str(spec["label"]) for spec in column_specs) + ROW_END,
        "        \\midrule",
    ]
    previous_group: str | None = None
    for row_key, label_text in row_specs:
        group = row_groups[row_key]
        if (
            row_key[0] == baseline_comparison_model
            and baseline_comparison_separator == "dashed"
            and previous_group is not None
        ):
            lines.append(_full_width_dashed_rule(total_columns))
        if separate_model_groups and previous_group is not None and group != previous_group:
            lines.append("        \\midrule")
        previous_group = group
        model_values = performance_values.get(row_key[0], {})
        cells = [
            _format_performance_cell(
                model_values.get(str(spec["key"])),
                spec,
                marks.get((row_key, str(spec["key"]))),
                performance_placeholder,
            )
            for spec in column_specs
        ]
        lines.append(
            f"        {latex_escape(label_text)} & " + " & ".join(cells) + ROW_END
        )
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def render_blocking_calibration_targets_table(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
    model_ids: list[str],
    model_labels: dict[str, str],
    task_id: str | None,
    *,
    targets: list[float] | tuple[float, ...],
    comparison_metrics: dict[str, dict[str, Any]],
    model_groups: dict[str, str] | None = None,
    model_group_order: list[str] | tuple[str, ...] | None = None,
    model_order_within_groups: dict[str, list[str] | tuple[str, ...]] | None = None,
    baseline_comparison_model: str | None = None,
    baseline_comparison_group: str | None = None,
    baseline_comparison_position: str = "first",
    baseline_comparison_separator: str | None = None,
    alphabetical_within_groups: bool = False,
    separate_model_groups: bool = False,
    highlight_best_per_group: bool = False,
    vertical_group_rules: bool = False,
) -> str:
    """Render an additional side-by-side comparison of calibrated targets."""
    selected_targets = [float(target) for target in targets]
    if not selected_targets:
        return ""
    if len(selected_targets) != len(set(selected_targets)):
        raise ValueError("blocking calibration comparison targets must be unique")
    filtered = _filter_rows(rows, "top_k", task_id)
    if not filtered:
        return ""
    context = _table_context(rows, filtered, model_ids, model_labels, task_id)
    effective_groups = dict(model_groups or {})
    if baseline_comparison_model is not None and baseline_comparison_group is not None:
        effective_groups[baseline_comparison_model] = baseline_comparison_group
    pins: list[str] = []
    trailing: list[str] = []
    if baseline_comparison_model is not None:
        if baseline_comparison_position == "first":
            pins.append(baseline_comparison_model)
        elif baseline_comparison_position == "last":
            trailing.append(baseline_comparison_model)
        else:
            raise ValueError(
                "blocking_tables.baseline_comparison_position must be 'first' or 'last'"
            )
    row_specs = _ordered_group_row_specs(
        context["row_specs"],
        effective_groups,
        model_group_order,
        model_order_within_groups,
        pins,
        trailing,
        alphabetical_within_groups,
    )
    row_groups = {
        row_key: effective_groups.get(row_key[0], "ungrouped")
        for row_key, _label in row_specs
    }
    flat_values: dict[str, dict[str, float | None]] = {}
    metric_specs: list[dict[str, Any]] = []
    for target in selected_targets:
        prefix = f"rho{str(target).replace('.', 'p')}"
        metric_specs.extend([
            {"key": f"{prefix}_pc_mean", "objective": "max"},
            {"key": f"{prefix}_pc_sd", "objective": "min"},
            {"key": f"{prefix}_rr_mean", "objective": "max"},
            {"key": f"{prefix}_rr_sd", "objective": "min"},
            {"key": f"{prefix}_pq_mean", "objective": "max"},
            {"key": f"{prefix}_pq_sd", "objective": "min"},
        ])
        target_key = f"{target:g}"
        for model_id in model_ids:
            source = comparison_metrics.get(model_id, {})
            target_metrics = (
                source.get("wik_calibrated_metrics_by_target") or {}
            ).get(target_key, {})
            destination = flat_values.setdefault(model_id, {})
            destination.update({
                f"{prefix}_pc_mean": target_metrics.get("wik_test_pc_mean"),
                f"{prefix}_pc_sd": target_metrics.get("wik_test_pc_sample_sd"),
                f"{prefix}_rr_mean": target_metrics.get("wik_test_rr_mean"),
                f"{prefix}_rr_sd": target_metrics.get("wik_test_rr_sample_sd"),
                f"{prefix}_pq_mean": target_metrics.get("wik_test_pq_mean"),
                f"{prefix}_pq_sd": target_metrics.get("wik_test_pq_sample_sd"),
            })
    marks: dict[tuple[tuple[str, str | None], str], str] = {}
    if highlight_best_per_group:
        group_marks = _performance_group_extreme_marks(
            row_specs, row_groups, flat_values, metric_specs
        )
        global_marks = _performance_global_extreme_marks(
            row_specs, flat_values, metric_specs
        )
        marks = {
            key: ("best" if key in global_marks else "second")
            for key in group_marks
        }

    total_columns = 1 + 3 * len(selected_targets)
    tabular_spec = (
        "|".join(["l", *(["rrr"] * len(selected_targets))])
        if vertical_group_rules
        else "l" + "r" * (total_columns - 1)
    )
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\caption{Blocking performance on \\emph{WIK} with "
        "$\\rho$-calibration at $\\rho\\in\\{.95,.99\\}$. Global best values per "
        "column are \\best{bold}; the best value within each remaining model "
        "group is \\secondbest{underlined}. Calibrated $PC$, $RR$, and $PQ$ are "
        "reported as mean $\\pm$ sample SD across splits. Lowest values are best "
        "for SD; highest values are best otherwise.}",
        f"    \\label{{tab:{_label_prefix(run_meta, task_id)}-blocking-calibration-targets}}",
        "    \\begin{adjustbox}{max width=\\textwidth}",
        f"    \\begin{{tabular}}{{{tabular_spec}}}",
        "        \\toprule",
    ]
    top_header = "        \\multirow{2}{*}{Model}"
    cmidrules: list[str] = []
    first_column = 2
    for index, target in enumerate(selected_targets):
        start = first_column + 3 * index
        end = start + 2
        alignment = "c|" if vertical_group_rules and index < len(selected_targets) - 1 else "c"
        target_label = f"{target:.2f}".lstrip("0")
        top_header += (
            f" & \\multicolumn{{3}}{{{alignment}}}"
            f"{{$\\rho^{{cal}}_{{\\emph{{WIK}}}}={target_label}$}}"
        )
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.extend([
        top_header + ROW_END,
        "        " + " ".join(cmidrules),
        "        & " + " & ".join(
            [
                "$PC^{test}_{\\emph{WIK}}$",
                "$RR^{test}_{\\emph{WIK}}$",
                "$PQ^{test}_{\\emph{WIK}}$",
            ] * len(selected_targets)
        ) + ROW_END,
        "        \\midrule",
    ])
    previous_group: str | None = None
    for row_key, label_text in row_specs:
        group = row_groups[row_key]
        if (
            row_key[0] == baseline_comparison_model
            and baseline_comparison_separator == "dashed"
            and previous_group is not None
        ):
            lines.append(_full_width_dashed_rule(total_columns))
        if separate_model_groups and previous_group is not None and group != previous_group:
            lines.append("        \\midrule")
        previous_group = group
        values = flat_values.get(row_key[0], {})
        cells: list[str] = []
        for target in selected_targets:
            prefix = f"rho{str(target).replace('.', 'p')}"
            cells.extend([
                _format_comparison_mean_sd_cell(
                    values.get(f"{prefix}_pc_mean"),
                    values.get(f"{prefix}_pc_sd"),
                    marks.get((row_key, f"{prefix}_pc_mean")),
                    marks.get((row_key, f"{prefix}_pc_sd")),
                ),
                _format_comparison_mean_sd_cell(
                    values.get(f"{prefix}_rr_mean"),
                    values.get(f"{prefix}_rr_sd"),
                    marks.get((row_key, f"{prefix}_rr_mean")),
                    marks.get((row_key, f"{prefix}_rr_sd")),
                ),
                _format_comparison_mean_sd_cell(
                    values.get(f"{prefix}_pq_mean"),
                    values.get(f"{prefix}_pq_sd"),
                    marks.get((row_key, f"{prefix}_pq_mean")),
                    marks.get((row_key, f"{prefix}_pq_sd")),
                ),
            ])
        lines.append(
            f"        {latex_escape(label_text)} & " + " & ".join(cells) + ROW_END
        )
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def _full_width_dashed_rule(total_columns: int) -> str:
    """Return a package-free dashed rule spanning the complete tabular width."""
    return (
        "        \\noalign{\\vskip-.6ex}\n"
        f"        \\multicolumn{{{total_columns}}}{{@{{}}c@{{}}}}{{"
        "\\leaders\\hbox{\\rule[.5ex]{3pt}{.3pt}\\hspace{2pt}}"
        "\\hfill\\kern0pt} \\\\[-.8ex]"
    )


def _fmt_without_leading_zero(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-." + text[3:]
    return text


def _performance_column_specs(
    columns: list[str | dict[str, Any]] | tuple[str | dict[str, Any], ...] | None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for column in columns or []:
        if isinstance(column, str):
            spec: dict[str, Any] = {
                "key": column,
                "label": column,
                "objective": "max",
                "decimals": 1,
                "leading_zero": False,
            }
        elif isinstance(column, dict):
            spec = dict(column)
        else:
            raise ValueError("blocking_tables.performance_columns entries must be strings or mappings")
        key = str(spec.get("key") or "").strip()
        label = str(spec.get("label") or "").strip()
        objective = str(spec.get("objective") or "max")
        decimals = int(spec.get("decimals", 1))
        if not key or not label:
            raise ValueError("blocking_tables performance columns require nonblank key and label")
        if objective not in {"min", "max"}:
            raise ValueError("blocking_tables performance objective must be 'min' or 'max'")
        if decimals < 0:
            raise ValueError("blocking_tables performance decimals must be nonnegative")
        specs.append({
            **spec,
            "key": key,
            "label": label,
            "objective": objective,
            "decimals": decimals,
            "leading_zero": bool(spec.get("leading_zero", False)),
        })
    keys = [str(spec["key"]) for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("blocking_tables performance column keys must be unique")
    return specs


def _format_performance_cell(
    value: float | None,
    spec: dict[str, Any],
    mark: str | None,
    placeholder: str,
) -> str:
    if value is None or not math.isfinite(float(value)):
        return placeholder
    text = f"{float(value):.{int(spec['decimals'])}f}"
    if not bool(spec.get("leading_zero", False)):
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
    if mark == "best":
        return f"\\best{{{text}}}"
    if mark == "second":
        return f"\\secondbest{{{text}}}"
    return text


def _performance_global_extreme_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    values: dict[str, dict[str, float | None]],
    specs: list[dict[str, Any]],
) -> set[tuple[tuple[str, str | None], str]]:
    marked: set[tuple[tuple[str, str | None], str]] = set()
    for spec in specs:
        key = str(spec["key"])
        present = [
            float(values[row_key[0]][key])
            for row_key, _label in row_specs
            if values.get(row_key[0], {}).get(key) is not None
            and math.isfinite(float(values[row_key[0]][key]))
        ]
        if not present:
            continue
        extreme = min(present) if spec["objective"] == "min" else max(present)
        for row_key, _label in row_specs:
            value = values.get(row_key[0], {}).get(key)
            if value is not None and math.isclose(
                float(value), extreme, rel_tol=1e-12, abs_tol=1e-12
            ):
                marked.add((row_key, key))
    return marked


def _performance_group_extreme_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    row_groups: dict[tuple[str, str | None], str],
    values: dict[str, dict[str, float | None]],
    specs: list[dict[str, Any]],
) -> set[tuple[tuple[str, str | None], str]]:
    marked: set[tuple[tuple[str, str | None], str]] = set()
    groups = list(dict.fromkeys(row_groups[row_key] for row_key, _label in row_specs))
    for group in groups:
        group_rows = [
            row_key for row_key, _label in row_specs if row_groups[row_key] == group
        ]
        for spec in specs:
            key = str(spec["key"])
            present = [
                float(values[row_key[0]][key])
                for row_key in group_rows
                if values.get(row_key[0], {}).get(key) is not None
                and math.isfinite(float(values[row_key[0]][key]))
            ]
            if not present:
                continue
            extreme = min(present) if spec["objective"] == "min" else max(present)
            for row_key in group_rows:
                value = values.get(row_key[0], {}).get(key)
                if value is not None and math.isclose(
                    float(value), extreme, rel_tol=1e-12, abs_tol=1e-12
                ):
                    marked.add((row_key, key))
    return marked


def _format_comparison_cell(value: float | None, mark: str | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    text = _fmt_without_leading_zero(float(value))
    if mark == "best":
        return f"\\best{{{text}}}"
    if mark == "second":
        return f"\\secondbest{{{text}}}"
    return text


def _format_comparison_mean_sd_cell(
    mean: float | None,
    sample_sd: float | None,
    mean_mark: str | None,
    sample_sd_mark: str | None,
) -> str:
    mean_text = _format_comparison_cell(mean, mean_mark)
    if mean_text == "--":
        return mean_text
    if sample_sd is None or not math.isfinite(float(sample_sd)):
        return mean_text + " $\\pm$ --"
    return mean_text + " $\\pm$ " + _format_comparison_cell(
        sample_sd, sample_sd_mark
    )


def _comparison_global_best_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    metrics: dict[str, dict[str, float | None]],
) -> set[tuple[tuple[str, str | None], str]]:
    marked: set[tuple[tuple[str, str | None], str]] = set()
    for metric_name in (
        "syn_test_pc_at_k",
        "wik_test_pc_mean",
        "wik_test_pq_mean",
        "wik_test_rr_mean",
    ):
        present = [
            float(metrics[row_key[0]][metric_name])
            for row_key, _label in row_specs
            if metrics.get(row_key[0], {}).get(metric_name) is not None
            and math.isfinite(float(metrics[row_key[0]][metric_name]))
        ]
        if not present:
            continue
        best = max(present)
        for row_key, _label in row_specs:
            value = metrics.get(row_key[0], {}).get(metric_name)
            if value is not None and math.isclose(
                float(value), best, rel_tol=1e-12, abs_tol=1e-12
            ):
                marked.add((row_key, metric_name))
    return marked


def _comparison_group_best_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    row_groups: dict[tuple[str, str | None], str],
    metrics: dict[str, dict[str, float | None]],
) -> set[tuple[tuple[str, str | None], str]]:
    marked: set[tuple[tuple[str, str | None], str]] = set()
    metric_names = (
        "syn_test_pc_at_k",
        "wik_test_pc_mean",
        "wik_test_pq_mean",
        "wik_test_rr_mean",
    )
    groups = list(dict.fromkeys(row_groups[row_key] for row_key, _label in row_specs))
    for group in groups:
        group_rows = [
            row_key for row_key, _label in row_specs if row_groups[row_key] == group
        ]
        for metric_name in metric_names:
            present = [
                float(metrics[row_key[0]][metric_name])
                for row_key in group_rows
                if metrics.get(row_key[0], {}).get(metric_name) is not None
                and math.isfinite(float(metrics[row_key[0]][metric_name]))
            ]
            if not present:
                continue
            best = max(present)
            for row_key in group_rows:
                value = metrics.get(row_key[0], {}).get(metric_name)
                if value is not None and math.isclose(
                    float(value), best, rel_tol=1e-12, abs_tol=1e-12
                ):
                    marked.add((row_key, metric_name))
    return marked


def _comparison_global_lowest_sd_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    metrics: dict[str, dict[str, float | None]],
) -> set[tuple[tuple[str, str | None], str]]:
    marked: set[tuple[tuple[str, str | None], str]] = set()
    for metric_name in (
        "wik_test_pc_sample_sd",
        "wik_test_rr_sample_sd",
    ):
        present = [
            float(metrics[row_key[0]][metric_name])
            for row_key, _label in row_specs
            if metrics.get(row_key[0], {}).get(metric_name) is not None
            and math.isfinite(float(metrics[row_key[0]][metric_name]))
        ]
        if not present:
            continue
        lowest = min(present)
        for row_key, _label in row_specs:
            value = metrics.get(row_key[0], {}).get(metric_name)
            if value is not None and math.isclose(
                float(value), lowest, rel_tol=1e-12, abs_tol=1e-12
            ):
                marked.add((row_key, metric_name))
    return marked


def _comparison_group_lowest_sd_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    row_groups: dict[tuple[str, str | None], str],
    metrics: dict[str, dict[str, float | None]],
) -> set[tuple[tuple[str, str | None], str]]:
    marked: set[tuple[tuple[str, str | None], str]] = set()
    metric_names = (
        "wik_test_pc_sample_sd",
        "wik_test_rr_sample_sd",
    )
    groups = list(dict.fromkeys(row_groups[row_key] for row_key, _label in row_specs))
    for group in groups:
        group_rows = [
            row_key for row_key, _label in row_specs if row_groups[row_key] == group
        ]
        for metric_name in metric_names:
            present = [
                float(metrics[row_key[0]][metric_name])
                for row_key in group_rows
                if metrics.get(row_key[0], {}).get(metric_name) is not None
                and math.isfinite(float(metrics[row_key[0]][metric_name]))
            ]
            if not present:
                continue
            lowest = min(present)
            for row_key in group_rows:
                value = metrics.get(row_key[0], {}).get(metric_name)
                if value is not None and math.isclose(
                    float(value), lowest, rel_tol=1e-12, abs_tol=1e-12
                ):
                    marked.add((row_key, metric_name))
    return marked


def _ordered_group_row_specs(
    row_specs: list[tuple[tuple[str, str | None], str]],
    model_groups: dict[str, str],
    model_group_order: list[str] | tuple[str, ...] | None,
    model_order_within_groups: dict[str, list[str] | tuple[str, ...]] | None,
    pinned_models: list[str] | tuple[str, ...] | None,
    trailing_models: list[str] | tuple[str, ...] | None,
    alphabetical_within_groups: bool,
) -> list[tuple[tuple[str, str | None], str]]:
    if (
        not model_groups
        and not model_group_order
        and not model_order_within_groups
        and not pinned_models
        and not trailing_models
    ):
        return row_specs
    groups = list(model_group_order or [])
    if len(groups) != len(set(groups)):
        raise ValueError("blocking_tables.model_group_order must not contain duplicates")
    group_rank = {group: index for index, group in enumerate(groups)}
    configured_within_group = model_order_within_groups or {}
    within_group_ranks: dict[str, dict[str, int]] = {}
    displayed_model_ids = {row_key[0] for row_key, _label in row_specs}
    for group, configured_ids in configured_within_group.items():
        ordered_ids = [str(model_id) for model_id in configured_ids]
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError(
                f"blocking_tables.model_order_within_groups.{group} contains duplicates"
            )
        unknown = [model_id for model_id in ordered_ids if model_id not in displayed_model_ids]
        if unknown:
            raise ValueError(
                f"blocking_tables.model_order_within_groups.{group} contains "
                "undisplayed models: " + ", ".join(unknown)
            )
        wrong_group = [
            model_id
            for model_id in ordered_ids
            if model_groups.get(model_id, "ungrouped") != group
        ]
        if wrong_group:
            raise ValueError(
                f"blocking_tables.model_order_within_groups.{group} contains "
                "models assigned to another group: " + ", ".join(wrong_group)
            )
        within_group_ranks[str(group)] = {
            model_id: index for index, model_id in enumerate(ordered_ids)
        }
    pins = list(pinned_models or [])
    if len(pins) != len(set(pins)):
        raise ValueError("blocking_tables.pinned_models must not contain duplicates")
    pin_rank = {model_id: index for index, model_id in enumerate(pins)}
    trailing = list(trailing_models or [])
    if len(trailing) != len(set(trailing)):
        raise ValueError("blocking_tables trailing models must not contain duplicates")
    overlap = set(pins) & set(trailing)
    if overlap:
        raise ValueError(
            "blocking_tables models cannot be both pinned first and placed last: "
            + ", ".join(sorted(overlap))
        )
    trailing_rank = {
        model_id: index for index, model_id in enumerate(trailing)
    }
    original_rank = {row_key: index for index, (row_key, _label) in enumerate(row_specs)}

    def sort_key(item: tuple[tuple[str, str | None], str]) -> tuple[Any, ...]:
        row_key, label = item
        model_id = row_key[0]
        group = model_groups.get(model_id, "ungrouped")
        group_key = (
            (0, group_rank[group])
            if group in group_rank
            else (1, group.casefold())
        )
        if model_id in pin_rank:
            within_group = (0, pin_rank[model_id])
        elif model_id in trailing_rank:
            within_group = (2, trailing_rank[model_id])
        elif model_id in within_group_ranks.get(group, {}):
            within_group = (1, 0, within_group_ranks[group][model_id])
        elif alphabetical_within_groups:
            within_group = (1, 1, label.casefold(), model_id)
        else:
            within_group = (1, 1, original_rank[row_key])
        return (*group_key, *within_group)

    return sorted(row_specs, key=sort_key)


def _global_best_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    ks: list[int],
    values: dict[tuple[tuple[str, str | None], int, str], float],
) -> set[tuple[tuple[str, str | None], int, str]]:
    marked: set[tuple[tuple[str, str | None], int, str]] = set()
    for metric in ("pair_completeness", "pair_quality"):
        for k in ks:
            present = [
                values[(row_key, k, metric)]
                for row_key, _label in row_specs
                if (row_key, k, metric) in values
                and math.isfinite(values[(row_key, k, metric)])
            ]
            if not present:
                continue
            best = max(present)
            for row_key, _label in row_specs:
                value = values.get((row_key, k, metric))
                if value is not None and math.isclose(
                    value, best, rel_tol=1e-12, abs_tol=1e-12
                ):
                    marked.add((row_key, k, metric))
    return marked


def _group_best_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    row_groups: dict[tuple[str, str | None], str],
    ks: list[int],
    values: dict[tuple[tuple[str, str | None], int, str], float],
) -> dict[tuple[tuple[str, str | None], int, str], str]:
    marks: dict[tuple[tuple[str, str | None], int, str], str] = {}
    groups = list(dict.fromkeys(row_groups[row_key] for row_key, _label in row_specs))
    for group in groups:
        group_rows = [
            row_key for row_key, _label in row_specs if row_groups[row_key] == group
        ]
        for metric in ("pair_completeness", "pair_quality"):
            for k in ks:
                present = [
                    values[(row_key, k, metric)]
                    for row_key in group_rows
                    if (row_key, k, metric) in values
                    and math.isfinite(values[(row_key, k, metric)])
                ]
                if not present:
                    continue
                best = max(present)
                for row_key in group_rows:
                    value = values.get((row_key, k, metric))
                    if value is not None and math.isclose(
                        value, best, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        marks[(row_key, k, metric)] = "best"
    return marks


def _selected_fixed_ks(
    available_ks: list[int], configured_ks: list[int] | tuple[int, ...] | None
) -> list[int]:
    if configured_ks is None:
        return available_ks
    if not isinstance(configured_ks, (list, tuple)) or not configured_ks:
        raise ValueError("blocking_tables.ks must be a non-empty list of positive integers")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in configured_ks
    ):
        raise ValueError("blocking_tables.ks must contain only positive integers")
    selected = list(configured_ks)
    if len(selected) != len(set(selected)):
        raise ValueError("blocking_tables.ks must not contain duplicate values")
    missing = [value for value in selected if value not in available_ks]
    if missing:
        raise ValueError(
            "blocking_tables.ks requests unavailable fixed-k values: "
            + ", ".join(str(value) for value in missing)
        )
    return selected


def render_blocking_calibrated_table(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
    model_ids: list[str],
    model_labels: dict[str, str],
    task_id: str | None = None,
) -> str:
    """Calibrated-threshold table: per model x calibration target rho, the threshold chosen on
    the calibration split and the achieved test metrics, including the transfer
    result dPC = PC_test - rho. Empty string when the run has no calibrated rows.
    """
    filtered = [row for row in rows if row.get("selection_mode") == "calibrated_threshold"]
    if task_id is not None:
        filtered = [row for row in filtered if str(row.get("task_id")) == task_id]
    if not filtered:
        return ""
    context = _table_context(rows, filtered, model_ids, model_labels, task_id)
    include_task = context["include_task"]
    by_row_key: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for row in filtered:
        row_key = (str(row.get("model_id")), str(row.get("task_id")) if include_task else None)
        by_row_key.setdefault(row_key, []).append(row)

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Calibrated-threshold blocking on \\emph{WIK}. Thresholds "
        "$\\hat{\\tau}$ are selected per model on the calibration split to reach "
        "target $PC$ level $\\rho$, then applied unchanged to the held-out test "
        "split. $\\Delta PC=PC^{test}_{\\emph{WIK}}-\\rho$ is the transfer gap.}",
        f"    \\label{{tab:{_label_prefix(run_meta, task_id)}-blocking-calibrated}}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{lrrrrrrr}",
        "        \\toprule",
        "        Model & $\\rho^{cal}_{\\emph{WIK}}$ & $\\hat{\\tau}$ & "
        "$PC^{test}_{\\emph{WIK}}$ & $\\Delta PC$ & "
        "$PQ^{test}_{\\emph{WIK}}$ (\\%) & $RR^{test}_{\\emph{WIK}}$ & QC"
        + ROW_END,
        "        \\midrule",
    ]
    for row_key, label_text in context["row_specs"]:
        model_rows = sorted(
            by_row_key.get(row_key, []),
            key=lambda row: float(row.get("calibration_target_pair_completeness") or 0.0),
        )
        for row in model_rows:
            rho = row.get("calibration_target_pair_completeness")
            pc = row.get("pair_completeness")
            cells = [
                latex_escape(label_text),
                _fmt(rho, "{:.3f}"),
                _fmt(row.get("threshold"), "{:.3f}"),
                _fmt(pc, "{:.3f}"),
                _fmt_signed_delta(pc, rho),
                _fmt(row.get("pair_quality"), "{:.2f}", scale=100.0),
                _fmt(row.get("reduction_ratio"), "{:.4f}"),
                _fmt(row.get("query_coverage"), "{:.3f}"),
            ]
            lines.append("        " + " & ".join(cells) + ROW_END)
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def _best_second_marks(
    row_specs: list[tuple[tuple[str, str | None], str]],
    ks: list[int],
    values: dict[tuple[tuple[str, str | None], int, str], float],
) -> dict[tuple[tuple[str, str | None], int, str], str]:
    """Mark the best and second-best distinct value per (k, metric) column."""
    marks: dict[tuple[tuple[str, str | None], int, str], str] = {}
    for metric in ("pair_completeness", "pair_quality"):
        for k in ks:
            present = sorted(
                {
                    values[(row_key, k, metric)]
                    for row_key, _label in row_specs
                    if (row_key, k, metric) in values
                    and math.isfinite(values[(row_key, k, metric)])
                },
                reverse=True,
            )
            if not present:
                continue
            for row_key, _label in row_specs:
                value = values.get((row_key, k, metric))
                if value is None:
                    continue
                if math.isclose(value, present[0], rel_tol=1e-12, abs_tol=1e-12):
                    marks[(row_key, k, metric)] = "best"
                elif len(present) > 1 and math.isclose(value, present[1], rel_tol=1e-12, abs_tol=1e-12):
                    marks[(row_key, k, metric)] = "second"
    return marks


def _rr_comment_lines(
    filtered: list[dict[str, Any]],
    ks: list[int],
    include_task: bool,
) -> list[str]:
    rr_by_key: dict[tuple[str | None, int], float] = {}
    for row in filtered:
        value = row.get("reduction_ratio")
        if value is None or row.get("k") is None:
            continue
        key = (str(row.get("task_id")) if include_task else None, int(row["k"]))
        rr_by_key.setdefault(key, float(value))
    lines = []
    for (task, k) in sorted(rr_by_key, key=lambda item: (str(item[0]), item[1])):
        suffix = f" ({task})" if task is not None else ""
        lines.append(f"% RR@{k}{suffix} = {rr_by_key[(task, k)]:.4f}  (model-independent within task)")
    return lines


def _fmt(value: Any, spec: str, scale: float = 1.0) -> str:
    if value is None:
        return "--"
    number = float(value)
    if not math.isfinite(number):
        return "--"
    return spec.format(number * scale)


def _fmt_signed_delta(pc: Any, rho: Any) -> str:
    if pc is None or rho is None:
        return "--"
    delta = float(pc) - float(rho)
    if not math.isfinite(delta):
        return "--"
    sign = "+" if delta >= 0 else "-"
    return f"${sign}{abs(delta):.3f}$"
