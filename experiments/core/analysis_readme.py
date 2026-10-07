"""Generate calculation notes beside every ``analysis-render`` chart bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import atomic_write_text
from .calibration_protocol_bars import chart_targets

README_NAME = "Readme.md"


def _enabled(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = (config.get("charts") or {}).get(name)
    return value if isinstance(value, dict) and bool(value.get("enabled", False)) else None


def _effective_enabled_chart(
    config: dict[str, Any], name: str
) -> dict[str, Any] | None:
    chart = _enabled(config, name)
    if chart is None:
        return None
    style_from = chart.get("style_from")
    if style_from is None:
        return chart
    inherited = (config.get("charts") or {}).get(str(style_from))
    return {**inherited, **chart} if isinstance(inherited, dict) else chart


def _calibration_display_notes(chart: dict[str, Any]) -> list[str]:
    include_crosses = bool(chart.get("include_specific_result_crosses", True))
    include_attainment = bool(chart.get("include_attainment_score", True))
    if include_crosses:
        result_note = (
            "- Split-level held-out PC markers are displayed; horizontal black "
            "ticks remain the equal-split PC means."
        )
    else:
        result_note = (
            "- Split-level held-out PC markers are omitted; horizontal black "
            "ticks still display the equal-split PC means."
        )
    if bool(chart.get("show_pc_mean_values", False)):
        decimals = max(0, int(chart.get("pc_mean_value_decimals", 3)))
        mean_value_note = (
            "- Each horizontal mean-PC tick is annotated immediately underneath "
            f"with its achieved mean PC rounded to {decimals} decimal places."
        )
    else:
        mean_value_note = "- Numeric mean-PC annotations are omitted."
    if include_attainment:
        attainment_note = (
            "- Attainment counts, labels, target reference, region (when enabled), "
            "and attained/missed point encoding are displayed."
        )
    else:
        attainment_note = (
            "- Attainment counts, labels, target reference, region, and "
            "attained/missed point encoding are omitted."
        )
    pc_axis_hidden = (
        chart.get("pc_axis_label") == ""
        and chart.get("pc_axis_description") == ""
    )
    rr_axis_hidden = (
        chart.get("rr_axis_label") == ""
        and chart.get("rr_axis_description") == ""
    )
    if pc_axis_hidden and rr_axis_hidden:
        axis_note = "- PC and RR axis labels and descriptions are omitted."
    else:
        axis_note = (
            "- PC and RR axis labels and descriptions are resolved from the "
            "chart configuration, with generated defaults for null or absent values."
        )
    pc_ylim = chart.get("pc_ylim")
    if isinstance(pc_ylim, (list, tuple)) and len(pc_ylim) == 2:
        limits_note = (
            "- The displayed PC axis is zoomed to "
            f"`[{float(pc_ylim[0]):g}, {float(pc_ylim[1]):g}]`; clipping changes "
            "only the view, not stored values."
        )
    else:
        limits_note = "- The PC axis limits are selected automatically."
    return [result_note, mean_value_note, attainment_note, axis_note, limits_note]


def _values(values: Any) -> str:
    return ", ".join(f"`{value:g}`" if isinstance(value, float) else f"`{value}`" for value in (values or []))


def _formats(chart: dict[str, Any], default: tuple[str, ...] = ("pdf",)) -> str:
    return ", ".join(f"`{value}`" for value in (chart.get("formats") or default))


def _task(chart: dict[str, Any]) -> str:
    return f"`{chart.get('task')}`" if chart.get("task") is not None else "the sole available task"


def _single_source_readme(config: dict[str, Any]) -> str:
    analysis_id = str(config.get("analysis_id") or "analysis")
    compatibility = config.get("compatibility") or {}
    calibration = config.get("calibration") or {}
    population = (
        f"dataset `{compatibility.get('dataset_id')}`, split `{compatibility.get('split_id')}`"
        if compatibility
        else "the task population recorded in each source artifact"
    )
    lines = [
        f"# Chart calculations: `{analysis_id}`",
        "",
        "## Scope and metric definitions",
        "",
        "- `analysis-render` reads immutable `eval.json`, `curves.json`, and `threshold_calibration.json` artifacts; it does not recompute similarities or evaluation metrics.",
        f"- Evaluation population: {population}. Selected models are accepted only when query, candidate, positive-pair, count, and evaluated-`k` signatures match exactly.",
        "- A positive pair is a query–candidate pair whose records have the same class. Self-pairs are removed when the task sets `exclude_self`; queries with no remaining positive are dropped before evaluation.",
        "- After those exclusions, let `P` be remaining positive pairs, `N` finite eligible query–candidate pairs, `S` selected pairs, and `TP` selected positive pairs.",
        "- Pair completeness: `PC = TP / P`. Pair quality: `PQ = TP / S` (`0` when `S = 0`). Reduction ratio: `RR = 1 - S / N`.",
        "- PC, PQ, and RR are pooled pair-level ratios. They are not averages of per-query precision or recall.",
        "- For embedding-based models, cosine similarity is the FP32 dot product of unit-normalized descriptors. A threshold selects every finite pair with `similarity >= threshold`.",
        "- Model inclusion, labels, colors, order, task, target values, axis limits, and output formats come from the analysis YAML. `source_manifest.json` identifies and hashes the selected artifacts.",
        "- Every graph has a same-basename `.json` sidecar containing the numeric series, bars, references, and displayed limits used for that graph; textual identifiers are included only to associate values with models or comparisons.",
        "",
        "## Threshold calibration used by calibrated charts",
        "",
        "- For each target `ρ`, all finite calibration-pair scores are sorted from high to low; equal scores remain one indivisible group.",
        "- The selected threshold is the highest score cutoff whose cumulative `PC` first satisfies `PC >= ρ`; no interpolation or tolerance is used for this selection. No threshold is fabricated when the target is infeasible.",
        "- The selected threshold is frozen and applied unchanged to the configured evaluation task. Bare `PC`, `PQ`, and `RR` fields are evaluation-population results; `calibration_*` fields are calibration-population results.",
        "- Chart labels use “validation” for the calibration population. Its actual dataset, subset, task, and split are those recorded by the calibration artifact/configuration and need not be the plotted evaluation population.",
    ]
    if calibration:
        lines.append(
            "- Configured calibration population: "
            f"dataset `{calibration.get('dataset_id')}`, subset `{calibration.get('subset')}`, "
            f"task `{calibration.get('task_id')}`, split `{calibration.get('split_id')}`."
        )
    lines += [
        "",
        "## Generated charts",
        "",
    ]

    chart = _enabled(config, "pc_at_k_curve")
    if chart:
        targets = [float(value) for value in chart.get("target_pc_lines") or []]
        reference = f" A dotted reference is drawn at the largest configured target, `ρ={max(targets):g}`." if targets else ""
        lines += [
            "### `figures/pc_at_k_<task>.*` — fixed-budget PC curve",
            "",
            f"- Task: {_task(chart)}. Formats: {_formats(chart)}.",
            "- For requested budget `k`, each retained query contributes its `k_eff = min(k, max configured k, number of candidates)` highest-scoring candidates.",
            "- `S_k = number of retained queries × k_eff`; `TP_k` counts positive pairs among those candidates; the plotted ordinate is `PC@k = TP_k / P`. Requested `k`, not `k_eff`, is the displayed x-value.",
            f"- The x-axis is logarithmic; points are joined in increasing `k`.{reference}",
            "",
        ]

    chart = _enabled(config, "pc_at_k_curve_zoomed")
    if chart:
        minimum_k_exclusive = int(chart.get("minimum_k_exclusive", 20))
        pc_limits = chart.get("pc_limits") or [0.9, 1.0]
        output_stem = str(
            chart.get("output_stem")
            or f"pc_at_k_gt{minimum_k_exclusive}_zoomed"
        )
        source_note = (
            " Chart-specific eval sources are configured for this graph; their paths "
            "and SHA-256 hashes are recorded in its JSON sidecar and "
            "`source_manifest.json`."
            if chart.get("source_model_dirs")
            else ""
        )
        lines += [
            f"### `bar_charts/{output_stem}_<task>.*` — operational fixed-budget PC zoom",
            "",
            f"- Task: {_task(chart)}. Formats: {_formats(chart)}.",
            f"- The graph retains stored PC@k points only for `k > {minimum_k_exclusive}` and displays `PC ∈ [{float(pc_limits[0]):g}, {float(pc_limits[1]):g}]`.",
            "- A model is included in the legend only when its retained curve intersects the displayed PC interval; models with no visible point or line segment are omitted from both plot and legend.",
            f"- Values are stored pooled fixed-budget results; axis filtering and clipping do not recompute metrics.{source_note}",
            "",
        ]

    chart = _enabled(config, "pc_rr_tradeoff")
    if chart:
        targets = [float(value) for value in chart.get("target_pc_lines") or []]
        reference = f" The horizontal reference is the largest configured target, `ρ={max(targets):g}`." if targets else ""
        lines += [
            "### `figures/pc_rr_tradeoff_<task>.*` — threshold-sweep PC–RR trade-off",
            "",
            f"- Task: {_task(chart)}. Formats: {_formats(chart)}.",
            "- `curves.json` stores 2,001 cutoffs uniformly spaced from `-1` through `1`; each cutoff supplies `PC(τ)` and `RR(τ)` from pairs with similarity `>= τ`.",
            f"- Each line plots stored `(RR(τ), PC(τ))` points in threshold order; no interpolation, averaging, or operating-point selection is performed.{reference}",
            "- The renderer displays only `RR ∈ [0.9, 1.0005]` and `PC ∈ [0.5, 1.01]`; clipping does not alter stored values.",
            "",
        ]

    chart = _enabled(config, "pc_at_k_star_bars")
    if chart:
        preferred = int(chart.get("preferred_k", 25))
        lines += [
            "### `figures/pc<k*>_bars_<task>.*` — one fixed-budget bar per model",
            "",
            f"- Task: {_task(chart)}. Formats: {_formats(chart)}. Preferred budget: `{preferred}`.",
            "- `k*` is the preferred budget when present in any selected model; otherwise it is the largest evaluated `k <= 30`, or the smallest evaluated `k` if none is `<= 30`.",
            "- A model is shown only if it has a finite stored `PC@k*`. Bar height is exactly that pooled `TP_k* / P` value.",
            "- Bars are ordered by descending PC; equal values are ordered by analysis model storage key. The y-axis starts at `max(0, smallest bar - 0.15)` and ends at `1`; this truncation changes only the view.",
            "",
        ]

    chart = _enabled(config, "calibrated_threshold_bars")
    if chart:
        lines += [
            "### `figures/pc_calibrated_rho<ρ>_bars_<task>.*` — calibration-to-evaluation transfer",
            "",
            f"- Task: {_task(chart)}. Targets: {_values(chart.get('calibration_targets'))}. Formats: {_formats(chart)}.",
            "- One file is emitted per available configured `ρ` (the filename replaces the decimal point with `p`). The solid left-axis bar is evaluation `PC = TP / P` at each model's frozen calibration-selected threshold.",
            "- The adjacent hatched right-axis bar is evaluation `RR = 1 - S / N` at the same threshold; it is not calibration RR.",
            "- The dotted line is the requested calibration target `ρ`, not an evaluation result. Bars are ordered by descending evaluation PC, then storage key; the PC axis starts at `max(0, minimum PC - 0.15)` and the RR axis spans `[0, 1]`.",
            "",
        ]

    chart = _effective_enabled_chart(config, "calibration_protocol_bars")
    if chart:
        protocols = ", ".join(f"`{value}`" for value in (chart.get("protocols") or []))
        lines += [
            "### `figures/calibration_protocols/<protocol>/` — zero-margin protocol bars",
            "",
            f"- Protocols: {protocols}. Task: {_task(chart)}. Targets: {_values(chart_targets(chart))}. Formats: {_formats(chart)}.",
            "- Each protocol directory contains the same bar-chart filenames as the threshold charts in `figures/`: `pc_calibrated_rho<ρ>_bars_<task>.*` and, when calibration RR is stored, `validation_rr_rho<ρ>_bars_<task>.*`.",
            "- `calibrated_k` reads `calibrated_top_k_metrics`: the smallest validation top-k that reaches `ρ`, applied unchanged on the evaluation task. No extra candidates are added.",
            "- `union` reads only the cell of `calibrated_union.json` whose K margin and threshold margin are both zero. A test candidate is kept when it is inside that calibrated top-k or its cosine is at least the calibrated threshold. Validation RR is not stored for this cell, so that bar chart is omitted.",
            "- Safety-margin sweeps are not rendered.",
            "",
        ]

    chart = _enabled(config, "validation_rr_bars")
    if chart:
        lines += [
            "### `figures/validation_rr_rho<ρ>_bars_<task>.*` — calibration-population RR",
            "",
            f"- Task association: {_task(chart)}. Targets: {_values(chart.get('calibration_targets'))}. Formats: {_formats(chart)}.",
            "- One file is emitted per available configured `ρ` (the filename replaces the decimal point with `p`). Bar height is stored `calibration_reduction_ratio = 1 - S_cal / N_cal` at the threshold selected to attain calibration PC target `ρ`.",
            "- This chart reports calibration-population RR only; it does not show evaluation RR. Bars are ordered by descending calibration RR, then storage key.",
            "",
        ]

    chart = _enabled(config, "ranked_pc_at_k_bars")
    if chart:
        orientation = str(chart.get("orientation", "horizontal"))
        value_decimals = int(chart.get("value_decimals", 3))
        value_position = "above" if orientation == "vertical" else "beside"
        lines += [
            "### `bar_charts/ranked_pc_at_<k>_<task>.*` — fixed-budget rankings",
            "",
            f"- Task: {_task(chart)}. Budgets: {_values(chart.get('ks'))}. Formats: {_formats(chart, ('svg',))}.",
            f"- One chart is emitted per configured `k`; {orientation} bar extent and the value printed {value_position} each bar represent stored pooled `PC@k = TP_k / P`.",
            f"- Models are sorted by descending PC. Bar geometry uses the stored float; visible annotations use {value_decimals} decimal places. `ranked_pc_summary.csv` and each JSON sidecar retain the unformatted PC value.",
            "",
        ]

    chart = _enabled(config, "similarity_distributions")
    if chart:
        lines += [
            "### `figures/similarity_distributions_<task>.*` — match/non-match score densities",
            "",
            f"- Task: {_task(chart)}. Formats: {_formats(chart)}.",
            "- Evaluation-time cosine scores are counted in 80 equal-width bins over `[-1, 1]`, separately for positive (“match”) and negative (“non-match”) eligible pairs.",
            "- Each class is normalized independently: bin density is `count / (total count in that class's plotted bins × bin width)`, so each filled histogram has unit area regardless of class imbalance.",
            "- The dotted line is the feasible calibration threshold with the largest recorded target `ρ`; it is not estimated from the displayed histogram. The displayed similarity range is `[-0.2, 1.0]`.",
            "",
        ]

    chart = _enabled(config, "candidate_sizes")
    if chart:
        lines += [
            "### `figures/candidate_sizes_rho<ρ>_<task>.*` — candidates per query",
            "",
            f"- Task: {_task(chart)}. Targets: {_values(chart.get('calibration_targets'))}. Formats: {_formats(chart)}.",
            "- One file is emitted per available configured `ρ` (the filename replaces the decimal point with `p`). For each retained evaluation query, stored size is `|C_τ(q)| = count(candidate similarities >= the model's frozen threshold τ)`.",
            "- Before plotting on a logarithmic x-axis, only zero sizes are replaced by `0.5`; positive counts are unchanged. Coverage remains the unfloored fraction of queries with size `> 0` but is not drawn.",
            "- The box uses the floored per-query values: center is the median, box is Q1–Q3, whiskers extend to the most extreme values within `1.5 × IQR`, and outliers are hidden.",
            "",
        ]

    for name, stem in (("pq_pc_curve", "pq_pc"), ("pq_pc_curve_zoomed", "pq_pc_zoomed")):
        chart = _enabled(config, name)
        if not chart:
            continue
        pc_limits = chart.get("pc_limits", [0.0, 1.0])
        pq_limits = chart.get("pq_limits", [0.0, 1.0])
        lines += [
            f"### `figures/{stem}_<task>.*` — threshold-sweep PQ–PC curve",
            "",
            f"- Task: {_task(chart)}. Formats: {_formats(chart)}. Display limits: `PC={pc_limits}`, `PQ={pq_limits}`.",
            "- Every point comes from one of the same 2,001 fixed threshold cutoffs: x is `PC(τ) = TP(τ) / P`; y is `PQ(τ) = TP(τ) / S(τ)`.",
            "- Points are sorted by PC before being joined. Axis limits only crop the view; the zoomed and unzoomed charts use identical stored points.",
            "",
        ]

    lines += [
        "## Interpretation limits",
        "",
        "- Curves connect discrete evaluated operating points; connecting segments are visual guides, not evaluated intermediate values.",
        "- A calibration target is an objective on the calibration population. Whether evaluation PC attains that target is an observed transfer result.",
        "- Missing required artifacts follow each chart's configured missing-data policy; consult `analysis_result.json`, `source_manifest.json`, and `INCOMPLETE.txt` when present.",
        "",
    ]
    return "\n".join(lines)


def _multisplit_readme(config: dict[str, Any]) -> str:
    analysis_id = str(config.get("analysis_id") or "analysis")
    split_block = config.get("multisplit") or {}
    compatibility = config.get("compatibility") or {}
    seeds = split_block.get("expected_split_seeds") or []
    has_groups = bool(config.get("model_groups"))
    lines = [
        f"# Chart calculations: `{analysis_id}`",
        "",
        "## Scope and common calculations",
        "",
        "- `analysis-render` reads stored evaluation artifacts for predefined split realizations; it does not create splits, recompute similarities, recalibrate thresholds, or pool pair counts across splits.",
        f"- Dataset: `{compatibility.get('dataset_id')}`. Evaluated split seeds ({len(seeds)}): {_values(seeds)}. Each split is one equally weighted descriptive unit; these are overlapping partitions of the same eligible union, not independent samples.",
        "- Within one source evaluation, `PC = TP / P`, `PQ = TP / S`, and `RR = 1 - S / N`, where `P` is eligible positive pairs, `N` eligible finite pairs, `S` selected pairs, and `TP` selected positive pairs.",
        "- If a model has training repeats, each metric is first averaged arithmetically across repeats within the same split. The charts then use those split-level means.",
        "- Across `n` splits, `mean = (1/n) Σ x_s`; observed range is `[min_s x_s, max_s x_s]`; sample SD in tabular artifacts is `sqrt(Σ(x_s-mean)^2/(n-1))`.",
        "- Split ranges are literal observed minima and maxima, not confidence intervals. Charts are descriptive; no significance or population-level inference is claimed.",
        "- Model/task filters, labels, colors, markers, budgets, target values, and formats come from the analysis YAML. Exact source files and hashes are in `source_manifest.json`.",
        "- Every graph has a same-basename `.json` sidecar containing its numeric series, split points, ranges, references, displayed limits, and deterministic jitter coordinates; textual identifiers only associate values with models or comparisons.",
        "",
    ]
    if has_groups:
        lines += [
            "## All-model and canonical-model chart variants",
            "",
            "- Standard direct model-series charts are emitted once with all chart-selected models and once with filename suffix `_canonical_models`. Calibrated top-k charts are emitted once per configured target; threshold-plus-top-k-floor charts are emitted once per configured `(ρ, minimum k)` combination.",
            "- The suffixed chart keeps the predeclared `canonical` member of each configured model group and keeps every ungrouped model.",
            "- Canonical membership is fixed before chart rendering and never selected from a displayed validation or test metric. The same representative is therefore used in every chart.",
            "",
        ]

    lines += ["## Generated charts", ""]

    chart = _enabled(config, "multisplit_pc_at_k_band")
    if chart:
        metric = str(chart.get("metric") or "pair_completeness")
        lines += [
            "### `figures/multisplit_pc_at_k_band[...].*` — split-mean fixed-budget curve",
            "",
            f"- Task: {_task(chart)}. Metric: `{metric}`. Formats: {_formats(chart)}.",
            "- At each evaluated `k`, the line value is the equal-split arithmetic mean of split-level metric values; for PC, each split value is its own pooled `TP_k / P`.",
            "- The shaded band at each `k` spans the smallest through largest split-level value. It is not `mean ± SD` and not a confidence interval.",
            "- The x-axis is logarithmic; line segments only connect evaluated budgets.",
            "",
        ]

    chart = _enabled(config, "multisplit_strip")
    if chart:
        k_value = chart.get("k", 40)
        budget_text = "every evaluated budget" if k_value == "all" else _values(k_value if isinstance(k_value, list) else [k_value])
        lines += [
            "### `figures/multisplit_strip_pc_at_<k>[...].*` — split-level values",
            "",
            f"- Task: {_task(chart)}. Budgets: {budget_text}. Formats: {_formats(chart)}.",
            "- Each point is one split-level PC value after any within-split repeat averaging. The short black horizontal segment is their equal-weight arithmetic mean.",
            "- Horizontal offsets are cosmetic draws from `Uniform(-0.13, 0.13)` using the configured deterministic jitter seed; they encode no quantity.",
            "- Models are ordered by ascending displayed mean PC.",
            "",
        ]

    chart = _enabled(config, "multisplit_paired_difference")
    if chart:
        metrics = chart.get("metrics")
        if metrics is None:
            metrics = [chart.get("metric") or "pair_completeness"]
        selection_mode = str(chart.get("selection_mode") or "top_k")
        if selection_mode == "top_k":
            operating_point = f"fixed budget `k={int(chart.get('k', 40))}`"
        else:
            operating_point = (
                f"`{selection_mode}` with calibration target "
                f"`ρ={float(chart.get('target_pc')):g}`"
            )
            if selection_mode == "calibrated_threshold_top_k_floor":
                operating_point += (
                    f" and minimum `k={int(chart.get('minimum_top_k'))}`"
                )
        tie_tolerance = float(
            ((config.get("multisplit") or {}).get("aggregation") or {}).get(
                "tie_tolerance", 1e-12
            )
        )
        lines += [
            "### `figures/multisplit_paired_<metric>_<operation>.*` — same-split paired differences",
            "",
            f"- Task: {_task(chart)}. Operating point: {operating_point}. Metrics: {_values([str(value) for value in metrics])}. Formats: {_formats(chart)}.",
            "- Training repeats are averaged within each predefined split before computing `Δ = metric(minuend) - metric(subtrahend)` on matching split identities.",
            "- The horizontal segment is the arithmetic mean of same-split differences; the zero line marks equality.",
            f"- A win is `Δ > {tie_tolerance:g}`, a tie is `|Δ| <= {tie_tolerance:g}`, and a loss is `Δ < -{tie_tolerance:g}`; W–T–L counts are descriptive, not a significance test.",
            "- Fixed-k RR differences are intentionally rejected because RR is model-independent at a common fixed candidate budget.",
            "",
        ]

    chart = _enabled(config, "multisplit_heatmap")
    if chart:
        k = int(chart.get("k", 40))
        lines += [
            "### `figures/multisplit_heatmap_pc_at_<k>[...].*` — model-centered split deviations",
            "",
            f"- Task: {_task(chart)}. Budget: `{k}`. Formats: {_formats(chart)}.",
            "- For model `m` and split `s`, cell value is `PC@k(m,s) - mean_s PC@k(m,s)`; every row therefore has arithmetic mean zero apart from floating-point roundoff.",
            "- The diverging color scale is symmetric around zero, with limits `± max absolute cell value` over the displayed matrix.",
            "- Values compare a model with its own split mean; colors do not encode absolute PC or direct between-model differences.",
            "",
        ]

    chart = _effective_enabled_chart(config, "multisplit_calibration_transfer")
    if chart:
        target = float(chart.get("target_pc", 0.99))
        lines += [
            "### `figures/multisplit_calibration_transfer_rho<ρ>[...].*` — validation-to-test transfer",
            "",
            f"- Task: {_task(chart)}. Target: `ρ={target:g}`. Formats: {_formats(chart)}.",
            "- In every split/repeat, the threshold is selected on that split's calibration/validation partition to reach `ρ`, then applied unchanged to its test partition.",
            "- Test PC uses the left vertical axis; horizontal black ticks are equal-split means. Translucent bars use the right vertical axis and show mean test `RR = 1 - S_test/N_test`.",
            *_calibration_display_notes(chart),
            "- Models are ordered by ascending mean test PC. When split markers are enabled, horizontal jitter is cosmetic and deterministically seeded; test attainment never changes threshold selection.",
            "",
        ]

    chart = _effective_enabled_chart(config, "calibration_protocol_bars")
    if chart:
        protocols = ", ".join(f"`{value}`" for value in (chart.get("protocols") or []))
        lines += [
            "### `figures/calibration_protocols/<protocol>/multisplit_calibration_transfer_rho<ρ>.*` — zero-margin protocol transfer",
            "",
            f"- Protocols: {protocols}. Task: {_task(chart)}. Targets: {_values(chart_targets(chart))}. Formats: {_formats(chart)}.",
            "- Each protocol directory repeats the threshold transfer filenames: `multisplit_calibration_transfer_rho<ρ>.*` and, when model groups are configured, the `_canonical_models` copy.",
            "- `calibrated_k` uses the validation-selected top-k with no added candidates. `union` uses the stored union of that top-k and the calibrated threshold, again with both margins at zero.",
            "- The chart geometry matches the threshold transfer chart. Safety-margin sweeps are not rendered.",
            "",
        ]

    chart = _effective_enabled_chart(config, "multisplit_calibrated_top_k_transfer")
    if chart:
        targets = [float(value) for value in chart.get("target_pc") or []]
        lines += [
            "### `figures/multisplit_calibrated_top_k_transfer_rho<ρ>.*` — validation-selected window transfer",
            "",
            f"- Task: {_task(chart)}. Calibration targets: {_values(targets)}. Formats: {_formats(chart)}.",
            "- One chart is emitted for each target. In every split, validation searches the predeclared integer window sizes and selects the smallest `k` attaining pooled validation PC target `ρ`; that `k` is frozen and applied unchanged to held-out test.",
            "- Equal-split mean held-out PC uses the left axis; translucent bars show equal-split mean held-out RR on the right axis. The JSON sidecar additionally records selected-k mean, sample SD, observed range, and each split's selected k, PC, PQ, and RR.",
            *_calibration_display_notes(chart),
            "- Validation selection attainment and held-out target attainment remain recorded in numeric artifacts even when diagram attainment elements are omitted. If the target is unattainable by the largest predeclared k, the run explicitly marks failure and evaluates that least-restrictive k so the split remains visible; the fallback never counts as validation attainment. No threshold is calibrated in this operating mode.",
            "",
        ]

    chart = _effective_enabled_chart(config, "multisplit_threshold_top_k_floor_transfer")
    if chart:
        targets = [float(value) for value in chart.get("target_pc") or []]
        floors = [int(value) for value in chart.get("minimum_top_k") or []]
        lines += [
            "### `figures/multisplit_threshold_top_k_floor_transfer_rho<ρ>_k<k>.*` — threshold-plus-floor transfer",
            "",
            f"- Task: {_task(chart)}. Calibration targets: {_values(targets)}. Minimum top-k floors: {_values(floors)}. Formats: {_formats(chart)}.",
            "- One chart is emitted for every configured `(ρ, minimum k)` Cartesian-product combination.",
            "- In every split, the pure threshold is selected on validation at target `ρ` and frozen. Test candidates satisfy `similarity >= threshold OR per-query rank <= minimum k`; the floor is not jointly recalibrated.",
            "- Equal-split mean test PC uses the left axis. Translucent bars show equal-split mean test RR on the right axis; exact PC, PQ, RR, and per-split values are retained in the JSON sidecar.",
            *_calibration_display_notes(chart),
            "- Charts are descriptive over the predefined splits.",
            "",
        ]

    chart = _enabled(config, "multisplit_similarity_distributions")
    if chart:
        target = chart.get("target_pc")
        lines += [
            "### `figures/multisplit_similarity_distributions_<task>[...].*` — equal-split score densities",
            "",
            f"- Task: {_task(chart)}. Calibration target for threshold overlay: `{target}`. Formats: {_formats(chart)}.",
            "- In each source evaluation and bin, density is `count / (class total × bin width)`, separately for match and non-match scores; each source class density has unit area.",
            "- Training-repeat densities are averaged equally within split; resulting split densities are averaged equally for the filled steps. Pair counts are never pooled.",
            "- Light envelopes are the bin-wise minimum and maximum across split-level mean densities, not confidence intervals and not one jointly observed distribution.",
            "- The dotted line is the equal-split mean calibration-selected threshold; the gray span is its observed split-level minimum–maximum range.",
            "",
        ]

    lines += [
        "## Interpretation and audit files",
        "",
        "- `raw_metrics.csv` contains source rows; `split_level_metrics.csv` contains within-split repeat means; `multisplit_summary.csv/json` contains equal-split summaries used by charts.",
        "- `split_metric_denominators.csv` preserves per-split denominators and candidate counts. It should be used when population sizes differ across split realizations.",
        "- `analysis_result.json` lists emitted files and warnings. `source_manifest.json` records immutable provenance. Missing or filtered data are never replaced by invented values.",
        "",
    ]
    return "\n".join(lines)


def render_analysis_readme(config: dict[str, Any]) -> str:
    """Return concise calculation documentation for this configured chart bundle."""
    if str(config.get("mode") or "single_source") == "multisplit":
        return _multisplit_readme(config)
    return _single_source_readme(config)


def write_analysis_readme(config: dict[str, Any], output_dir: Path | str) -> Path:
    """Write ``Readme.md`` into one analysis output root."""
    path = Path(output_dir) / README_NAME
    atomic_write_text(path, render_analysis_readme(config))
    return path
