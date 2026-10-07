"""Blocking result figures (Figures 1-5) rendered from run-dir artifacts.

Separate module from the manual chart CLIs (``plot_pq_pc``, ``plot_ranked_pc_bars``).
Every figure reads only artifacts written at evaluation time (eval.json,
curves.json, threshold_calibration.json, run.json, resolved_config.yml);
figures never recompute metrics or touch similarity caches.

Figure catalogue (single dataset per run; cross-dataset variants out of scope):
  1. PC@k curves, one line per model               -> pc_at_k_{task}
  2. PC-RR trade-off threshold sweep               -> pc_rr_tradeoff_{task}
  3. PC@k* bar chart, one bar per model            -> pc{k*}_bars_{task}
  4. Match vs non-match similarity histograms      -> similarity_distributions_{task}
  5. Candidate-set sizes at calibrated thresholds  -> candidate_sizes_rho{rho}_{task}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .plot_ranked_pc_bars import _safe_name

DEFAULT_FIGURE_FORMATS: tuple[str, ...] = ("pdf",)

# Keep colors stable across figures; assigned by model order when the
# experiment config does not pin a per-model color.
FALLBACK_PALETTE = (
    "#999999", "#6baed6", "#3182bd", "#9e6ebd", "#e6a23c",
    "#98d594", "#2ca02c", "#fc9272", "#ef3b2c", "#a50f15",
)
RC_PARAMS = {
    "font.family": "serif",
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}
MATCH_COLOR = "#a50f15"
NON_MATCH_COLOR = "#9ecae1"


class FigureGenerationUnavailable(RuntimeError):
    """Raised when figures cannot be rendered in this environment (no matplotlib)."""


def _import_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise FigureGenerationUnavailable(
            "matplotlib is not installed; run figures were skipped"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    storage_key: str
    label: str
    color: str
    linestyle: str
    group: str | None
    is_baseline: bool
    model_dir: Path


@dataclass(frozen=True)
class RunFigureContext:
    run_dir: Path
    run_meta: dict[str, Any]
    config: dict[str, Any]
    models: list[ModelSpec]
    task_ids: list[str]
    top_k: list[int]
    calibration_target_pc: list[float]


def load_run_context(run_dir: Path | str) -> RunFigureContext:
    run_path = Path(run_dir).expanduser()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    run_meta = _load_json(run_path / "run.json") or {"run_id": run_path.name}
    config = _load_config(run_path)
    includes = {
        str(entry.get("model_id")): entry
        for entry in ((config.get("models") or {}).get("include") or [])
        if isinstance(entry, dict) and entry.get("model_id")
    }
    models: list[ModelSpec] = []
    for index, entry in enumerate(run_meta.get("model_runs") or []):
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            continue
        model_id = str(entry.get("model_id") or "")
        storage_key = str(entry.get("model_storage_key") or "")
        if not model_id or not storage_key:
            continue
        include = includes.get(model_id, {})
        label = str(
            include.get("display_name") or include.get("label") or include.get("name")
            or entry.get("display_name") or model_id
        )
        color = include.get("color")
        group = include.get("group")
        models.append(
            ModelSpec(
                model_id=model_id,
                storage_key=storage_key,
                label=label,
                color=str(color) if color else FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)],
                linestyle="--" if group == "frozen" else "-",
                group=str(group) if group else None,
                is_baseline=str(entry.get("kind") or "") == "analytical_baseline",
                model_dir=run_path / "models" / storage_key,
            )
        )
    evaluation = dict(config.get("evaluation") or {})
    calibration = dict(evaluation.get("threshold_calibration") or {})
    task_ids: list[str] = []
    for spec in models:
        for task in (_load_json(spec.model_dir / "eval.json") or {}).get("tasks", []):
            tid = str(task.get("task_id") or "")
            if tid and tid not in task_ids:
                task_ids.append(tid)
    return RunFigureContext(
        run_dir=run_path,
        run_meta=run_meta,
        config=config,
        models=models,
        task_ids=task_ids,
        top_k=[int(k) for k in evaluation.get("top_k") or []],
        calibration_target_pc=[float(v) for v in calibration.get("target_pc") or []],
    )


def load_model_artifacts(spec: ModelSpec) -> dict[str, Any]:
    return {
        "eval": _load_json(spec.model_dir / "eval.json") or {},
        "curves": _load_json(spec.model_dir / "curves.json") or {},
        "calibration": _load_json(spec.model_dir / "threshold_calibration.json") or {},
    }


# ---------------------------------------------------------------- data layer --


def fig1_data(models: list[ModelSpec], artifacts: dict[str, dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
    """PC@k per model from eval.json metrics rows (the configured k grid)."""
    entries = []
    for spec in models:
        task = _eval_task(artifacts[spec.storage_key], task_id)
        rows = sorted(
            (row for row in (task or {}).get("metrics", []) if row.get("k") is not None),
            key=lambda row: int(row["k"]),
        )
        ks = [int(row["k"]) for row in rows]
        pc = [_finite_or_none(row.get("pair_completeness")) for row in rows]
        if ks and all(value is not None for value in pc):
            entries.append({"spec": spec, "ks": ks, "pc": pc})
    return entries


def fig2_data(models: list[ModelSpec], artifacts: dict[str, dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
    """PC(tau) vs RR(tau) per model from the stored threshold sweep in curves.json."""
    entries = []
    for spec in models:
        task = _curves_task(artifacts[spec.storage_key], task_id)
        rows = (task or {}).get("precision_recall") or []
        rr = [_finite_or_none(row.get("rr", row.get("reduction_ratio"))) for row in rows]
        pc = [_finite_or_none(row.get("pc", row.get("pair_completeness"))) for row in rows]
        points = [(r, p) for r, p in zip(rr, pc) if r is not None and p is not None]
        if points:
            entries.append(
                {
                    "spec": spec,
                    "rr": [r for r, _p in points],
                    "pc": [p for _r, p in points],
                }
            )
    return entries


def fig3_data(
    models: list[ModelSpec],
    artifacts: dict[str, dict[str, Any]],
    task_id: str,
    preferred_k: int = 25,
) -> tuple[int | None, list[dict[str, Any]]]:
    """One PC@k* bar per model. k* = preferred_k when evaluated, else the
    largest evaluated k <= 30, else the smallest evaluated k."""
    per_model = fig1_data(models, artifacts, task_id)
    all_ks = sorted({k for entry in per_model for k in entry["ks"]})
    if not all_ks:
        return None, []
    if preferred_k in all_ks:
        k_star = preferred_k
    else:
        small = [k for k in all_ks if k <= 30]
        k_star = max(small) if small else min(all_ks)
    bars = [
        {"spec": entry["spec"], "pc": entry["pc"][entry["ks"].index(k_star)]}
        for entry in per_model
        if k_star in entry["ks"]
    ]
    # Put the strongest result first (leftmost), independent of presentation
    # order. The storage key makes ties deterministic across renderings.
    bars.sort(key=lambda entry: (-entry["pc"], entry["spec"].storage_key))
    return k_star, bars


def calibrated_threshold_bars_data(
    models: list[ModelSpec],
    artifacts: dict[str, dict[str, Any]],
    task_id: str,
) -> dict[float, list[dict[str, Any]]]:
    """Achieved test PC at thresholds selected on validation, grouped by target PC."""
    by_rho: dict[float, list[dict[str, Any]]] = {}
    for spec in models:
        task = _eval_task(artifacts[spec.storage_key], task_id)
        for row in (task or {}).get("calibrated_threshold_metrics") or []:
            rho = _finite_or_none(row.get("calibration_target_pair_completeness"))
            pc = _finite_or_none(row.get("pair_completeness"))
            rr = _finite_or_none(row.get("reduction_ratio"))
            threshold = _finite_or_none(row.get("threshold"))
            if rho is None or pc is None or rr is None or threshold is None:
                continue
            by_rho.setdefault(rho, []).append(
                {
                    "spec": spec,
                    "pc": pc,
                    "rr": rr,
                    "threshold": threshold,
                    "calibration_pc": _finite_or_none(row.get("calibration_pair_completeness")),
                }
            )
    for bars in by_rho.values():
        bars.sort(key=lambda entry: (-entry["pc"], entry["spec"].storage_key))
    return by_rho


def validation_rr_bars_data(
    models: list[ModelSpec],
    artifacts: dict[str, dict[str, Any]],
    task_id: str,
) -> dict[float, list[dict[str, Any]]]:
    """Validation RR at each configured PC calibration target, grouped by target."""
    by_rho: dict[float, list[dict[str, Any]]] = {}
    for spec in models:
        task = _eval_task(artifacts[spec.storage_key], task_id)
        for row in (task or {}).get("calibrated_threshold_metrics") or []:
            rho = _finite_or_none(row.get("calibration_target_pair_completeness"))
            rr = _finite_or_none(row.get("calibration_reduction_ratio"))
            if rho is None or rr is None:
                continue
            by_rho.setdefault(rho, []).append(
                {
                    "spec": spec,
                    "rr": rr,
                    "calibration_pc": _finite_or_none(row.get("calibration_pair_completeness")),
                }
            )
    for bars in by_rho.values():
        bars.sort(key=lambda entry: (-entry["rr"], entry["spec"].storage_key))
    return by_rho


def fig4_data(models: list[ModelSpec], artifacts: dict[str, dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
    """Density-normalized match/non-match similarity histograms per model."""
    entries = []
    for spec in models:
        task = _curves_task(artifacts[spec.storage_key], task_id)
        hist = (task or {}).get("histogram") or {}
        edges = np.asarray(hist.get("bin_edges") or [], dtype=np.float64)
        match = np.asarray(hist.get("match_hist") or [], dtype=np.float64)
        neg = np.asarray(hist.get("neg_hist") or [], dtype=np.float64)
        if edges.size < 2 or match.size != edges.size - 1 or neg.size != edges.size - 1:
            continue
        if match.sum() <= 0 or neg.sum() <= 0:
            continue
        widths = np.diff(edges)
        entries.append(
            {
                "spec": spec,
                "bin_edges": edges,
                "match_density": match / (match.sum() * widths),
                "neg_density": neg / (neg.sum() * widths),
                "threshold": _calibrated_threshold(artifacts[spec.storage_key]),
            }
        )
    return entries


def fig5_data(
    models: list[ModelSpec],
    artifacts: dict[str, dict[str, Any]],
    task_id: str,
) -> dict[float, list[dict[str, Any]]]:
    """Per calibration target rho: candidate-set sizes per query, per model.

    Sizes come from the eval-time ``calibrated_candidate_sizes`` block in
    curves.json; zeros are floored to 0.5 so log-scaled axes keep uncovered
    queries visible (their share is reported via query_coverage instead).
    """
    by_rho: dict[float, list[dict[str, Any]]] = {}
    for spec in models:
        task = _curves_task(artifacts[spec.storage_key], task_id)
        for row in (task or {}).get("calibrated_candidate_sizes") or []:
            rho = _finite_or_none(row.get("calibration_target_pair_completeness"))
            sizes = row.get("candidates_per_query")
            if rho is None or not sizes:
                continue
            by_rho.setdefault(rho, []).append(
                {
                    "spec": spec,
                    "threshold": _finite_or_none(row.get("threshold")),
                    "sizes": np.maximum(np.asarray(sizes, dtype=np.float64), 0.5),
                    "query_coverage": _finite_or_none(row.get("query_coverage")),
                }
            )
    return by_rho


def _eval_task(artifacts: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for task in artifacts["eval"].get("tasks", []):
        if str(task.get("task_id")) == task_id:
            return task
    return None


def _curves_task(artifacts: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for task in artifacts["curves"].get("tasks", []):
        if str(task.get("task_id")) == task_id:
            return task
    return None


def _calibrated_threshold(artifacts: dict[str, Any]) -> float | None:
    """Threshold of the highest-target feasible calibration operating point."""
    rows = [
        row
        for row in artifacts["calibration"].get("operating_points", [])
        if _finite_or_none(row.get("threshold")) is not None
        and _finite_or_none(row.get("target_pair_completeness")) is not None
    ]
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["target_pair_completeness"]))
    return float(best["threshold"])


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(run_path: Path) -> dict[str, Any]:
    path = run_path / "resolved_config.yml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _frozen_separator_index(specs: list[ModelSpec]) -> int | None:
    """Index boundary between a leading frozen block and the finetuned block."""
    groups = [spec.group for spec in specs]
    n_frozen = sum(1 for group in groups if group == "frozen")
    n_finetuned = sum(1 for group in groups if group == "finetuned")
    if not n_frozen or not n_finetuned:
        return None
    if all(group == "frozen" for group in groups[:n_frozen]) and "frozen" not in groups[n_frozen:]:
        return n_frozen
    return None


# ------------------------------------------------------------- render layer --


def write_run_figures(
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    task_id: str | None = None,
    formats: tuple[str, ...] = DEFAULT_FIGURE_FORMATS,
) -> dict[str, Path]:
    """Render all applicable blocking figures for one completed run directory."""
    matplotlib, plt = _import_matplotlib()
    context = load_run_context(run_dir)
    out_dir = Path(output_dir).expanduser() if output_dir is not None else context.run_dir / "figures"
    task_ids = [task_id] if task_id is not None else context.task_ids
    if task_id is not None and task_id not in context.task_ids:
        raise ValueError(f"task_id {task_id!r} not found in run {context.run_dir}")
    artifacts = {spec.storage_key: load_model_artifacts(spec) for spec in context.models}
    rho_line = max(context.calibration_target_pc) if context.calibration_target_pc else None
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    with matplotlib.rc_context(RC_PARAMS):
        for tid in task_ids:
            safe = _safe_name(tid)
            entries = fig1_data(context.models, artifacts, tid)
            if entries:
                fig = _render_fig1(plt, entries, rho_line)
                written.update(_save(fig, plt, out_dir, f"pc_at_k_{safe}", formats))
            entries = fig2_data(context.models, artifacts, tid)
            if entries:
                fig = _render_fig2(plt, entries, rho_line)
                written.update(_save(fig, plt, out_dir, f"pc_rr_tradeoff_{safe}", formats))
            k_star, bars = fig3_data(context.models, artifacts, tid)
            if bars:
                fig = _render_fig3(plt, k_star, bars)
                written.update(_save(fig, plt, out_dir, f"pc{k_star}_bars_{safe}", formats))
            entries = fig4_data(context.models, artifacts, tid)
            if entries:
                fig = _render_fig4(plt, entries)
                written.update(_save(fig, plt, out_dir, f"similarity_distributions_{safe}", formats))
            for rho, dists in sorted(fig5_data(context.models, artifacts, tid).items()):
                fig = _render_fig5(plt, rho, dists)
                rho_tag = f"{rho:g}".replace(".", "p")
                written.update(_save(fig, plt, out_dir, f"candidate_sizes_rho{rho_tag}_{safe}", formats))
    return written


def _save(fig, plt, out_dir: Path, stem: str, formats: tuple[str, ...]) -> dict[str, Path]:
    written = {}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        written[f"{stem}.{fmt}"] = path
    plt.close(fig)
    return written


def _render_fig1(plt, entries: list[dict[str, Any]], rho_line: float | None):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for entry in entries:
        spec = entry["spec"]
        ax.plot(
            entry["ks"], entry["pc"], marker="o", ms=3, lw=1.3,
            color=spec.color, ls=spec.linestyle, label=spec.label,
        )
    ax.set_xscale("log")
    ticks = sorted({k for entry in entries for k in entry["ks"]})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(k) for k in ticks])
    ax.set_xlabel(r"Candidate budget $k$")
    ax.set_ylabel(r"Pairs completeness  PC@$k$")
    if rho_line is not None:
        ax.axhline(rho_line, color="black", lw=0.7, ls=":", alpha=0.6)
        ax.text(ticks[0] * 1.05, rho_line + 0.004, rf"target $\rho={rho_line:g}$", fontsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    _legend(ax, len(entries), "lower right")
    return fig


def zoomed_pc_at_k_data(
    entries: list[dict[str, Any]],
    minimum_k_exclusive: int,
    pc_limits: tuple[float, float],
) -> list[dict[str, Any]]:
    """Keep k values in the zoom and omit models whose curves are invisible."""
    lower, upper = pc_limits
    visible_entries: list[dict[str, Any]] = []
    for entry in entries:
        points = [
            (int(k), float(pc))
            for k, pc in zip(entry["ks"], entry["pc"])
            if int(k) > minimum_k_exclusive
        ]
        if not points:
            continue
        pc_values = [pc for _k, pc in points]
        point_visible = any(lower <= pc <= upper for pc in pc_values)
        segment_visible = any(
            max(left, right) >= lower and min(left, right) <= upper
            for left, right in zip(pc_values, pc_values[1:])
        )
        if not point_visible and not segment_visible:
            continue
        visible_entries.append(
            {
                **entry,
                "ks": [k for k, _pc in points],
                "pc": pc_values,
            }
        )
    return visible_entries


def _render_zoomed_pc_at_k(
    plt,
    entries: list[dict[str, Any]],
    rho_line: float | None,
    pc_limits: tuple[float, float],
    labels: dict[str, str] | None = None,
    xscale: str = "log",
):
    """Render an operational PC@k zoom with a visibility-matched legend."""
    fig, ax = plt.subplots(figsize=(5.75, 3.1))
    label_overrides = labels or {}
    for entry in entries:
        spec = entry["spec"]
        ax.plot(
            entry["ks"],
            entry["pc"],
            marker="o",
            ms=4,
            lw=1.3,
            color=spec.color,
            ls=spec.linestyle,
            label=label_overrides.get(spec.storage_key, spec.label),
        )
    if xscale not in {"linear", "log"}:
        raise ValueError("zoomed PC@k xscale must be 'linear' or 'log'")
    ax.set_xscale(xscale)
    ticks = sorted({k for entry in entries for k in entry["ks"]})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(k) for k in ticks])
    from matplotlib.ticker import NullFormatter

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(*pc_limits)
    ax.set_xlabel(r"Candidate budget $k$")
    ax.set_ylabel(r"Pairs completeness ($PC@k$)")
    if rho_line is not None and pc_limits[0] <= rho_line <= pc_limits[1]:
        ax.axhline(rho_line, color="black", lw=0.7, ls=":", alpha=0.6)
        ax.text(
            ticks[0] * 1.01,
            rho_line + 0.0015,
            rf"target $\rho={rho_line:g}$",
            fontsize=7,
        )
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.legend(
        ncol=min(4, len(entries)),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        handlelength=1.8,
    )
    fig.subplots_adjust(bottom=0.32)
    return fig


def _legend(ax, n_entries: int, inside_loc: str) -> None:
    """Legend inside for few models; outside the axes for crowded runs."""
    if n_entries <= 10:
        ax.legend(ncol=2, frameon=False, loc=inside_loc, handlelength=1.8)
        return
    ax.legend(
        ncol=1 + n_entries // 25,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6,
        handlelength=1.8,
    )


def _render_fig2(plt, entries: list[dict[str, Any]], rho_line: float | None):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for entry in entries:
        spec = entry["spec"]
        ax.plot(entry["rr"], entry["pc"], lw=1.3, color=spec.color, ls=spec.linestyle, label=spec.label)
    ax.set_xlabel("Reduction Ratio  RR")
    ax.set_ylabel("Pairs completeness  PC")
    ax.set_xlim(0.9, 1.0005)  # zoom into the operationally relevant region
    ax.set_ylim(0.5, 1.01)
    if rho_line is not None:
        ax.axhline(rho_line, color="black", lw=0.7, ls=":", alpha=0.6)
    ax.grid(lw=0.3, alpha=0.4)
    _legend(ax, len(entries), "lower left")
    return fig


def _render_fig3(plt, k_star: int | None, bars: list[dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(bars) + 1.0), 3.0))
    x = np.arange(len(bars))
    values = [entry["pc"] for entry in bars]
    ax.bar(x, values, 0.65, color=[entry["spec"].color for entry in bars], edgecolor="white", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([entry["spec"].label for entry in bars], rotation=30, ha="right")
    ax.set_ylabel(f"PC@{k_star}")
    # Differences live near the top; start the axis just below the worst value.
    ax.set_ylim(max(0.0, min(values) - 0.15), 1.0)
    separator = _frozen_separator_index([entry["spec"] for entry in bars])
    if separator is not None:
        ax.axvline(separator - 0.5, color="black", lw=0.6, ls=":", alpha=0.7)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    return fig


def _render_calibrated_threshold_bars(
    plt,
    rho: float,
    bars: list[dict[str, Any]],
    *,
    pc_ylabel: str = "Test PC at validation-selected threshold",
):
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(bars) + 1.0), 3.0))
    rr_ax = ax.twinx()
    x = np.arange(len(bars))
    pc_values = [entry["pc"] for entry in bars]
    rr_values = [entry["rr"] for entry in bars]
    colors = [entry["spec"].color for entry in bars]
    width = 0.36
    pc_bars = ax.bar(
        x - width / 2,
        pc_values,
        width,
        color=colors,
        edgecolor="white",
        lw=0.4,
        label="Test PC (left axis)",
        zorder=2,
    )
    rr_bars = rr_ax.bar(
        x + width / 2,
        rr_values,
        width,
        color=colors,
        edgecolor=colors,
        alpha=0.35,
        hatch="//",
        lw=0.5,
        label="Test RR (right axis)",
        zorder=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [entry["spec"].label for entry in bars], rotation=30, ha="right"
    )
    ax.set_ylabel(pc_ylabel)
    rr_ax.set_ylabel("Test Reduction Ratio (RR)")
    # RC_PARAMS hides right spines globally; this chart explicitly needs one.
    rr_ax.spines["right"].set_visible(True)
    ax.set_ylim(max(0.0, min(pc_values) - 0.15), 1.0)
    rr_ax.set_ylim(0.0, 1.0)
    # Keep the PC bars and calibration reference in front of the twin-axis bars.
    ax.set_zorder(rr_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    # Preserve the original validation calibration objective as a reference.
    ax.axhline(rho, color="black", lw=0.8, ls=":", alpha=0.8, zorder=3)
    ax.annotate(
        rf"validation target $\rho={rho:g}$",
        xy=(0.01, rho),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7,
        va="bottom",
    )
    ax.legend(
        [pc_bars, rr_bars],
        ["Test PC (left axis)", "Test RR (right axis)"],
        loc="lower right",
        frameon=False,
        ncol=2,
        fontsize=7,
    )
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    return fig


def _render_validation_rr_bars(
    plt,
    rho: float,
    bars: list[dict[str, Any]],
    *,
    show_values: bool = False,
    value_decimals: int = 4,
):
    if value_decimals < 0:
        raise ValueError("validation RR bar-label decimals must be non-negative")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(bars) + 1.0), 3.0))
    x = np.arange(len(bars))
    values = [entry["rr"] for entry in bars]
    container = ax.bar(
        x,
        values,
        0.65,
        color=[entry["spec"].color for entry in bars],
        edgecolor="white",
        lw=0.4,
    )
    if show_values:
        ax.bar_label(
            container,
            labels=[f"{value:.{value_decimals}f}" for value in values],
            label_type="center",
            rotation=90,
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [entry["spec"].label for entry in bars], rotation=30, ha="right"
    )
    ax.set_ylabel(rf"Validation RR at PC target $\rho={rho:g}$")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    return fig


def _render_fig4(plt, entries: list[dict[str, Any]]):
    ncols = min(3, len(entries))
    nrows = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols + 0.8, 2.1 * nrows + 0.5),
        sharex=True, sharey=True, squeeze=False,
    )
    flat = axes.reshape(-1)
    for ax, entry in zip(flat, entries):
        edges = entry["bin_edges"]
        ax.stairs(entry["neg_density"], edges, fill=True, alpha=0.55, color=NON_MATCH_COLOR, label="non-match")
        ax.stairs(entry["match_density"], edges, fill=True, alpha=0.55, color=MATCH_COLOR, label="match")
        if entry["threshold"] is not None:
            ax.axvline(entry["threshold"], color="black", lw=0.7, ls=":", alpha=0.7)
        ax.set_title(entry["spec"].label, fontsize=9)
        ax.set_xlim(-0.2, 1.0)
    for ax in flat[len(entries):]:
        ax.set_visible(False)
    for index in range(len(entries)):
        if index % ncols == 0:
            flat[index].set_ylabel("density")
        if index >= len(entries) - ncols:
            flat[index].set_xlabel("cosine similarity")
    flat[len(entries) - 1].legend(frameon=False)
    return fig


def _render_fig5(plt, rho: float, dists: list[dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(5.2, max(2.0, 0.45 * len(dists) + 1.2)))
    boxplot = ax.boxplot(
        [entry["sizes"] for entry in dists],
        orientation="horizontal",
        tick_labels=[entry["spec"].label for entry in dists],
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black"),
    )
    for patch, entry in zip(boxplot["boxes"], dists):
        patch.set_facecolor(entry["spec"].color)
        patch.set_alpha(0.7)
    ax.set_xscale("log")
    ax.set_xlabel(
        rf"candidates per query $|C_\tau(q)|$  (calibrated $\hat{{\tau}}_{{{rho:g}}}$)"
    )
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    return fig
