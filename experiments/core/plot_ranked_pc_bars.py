"""Rank models by fixed-k pair completeness and write bar charts.

The input is an experiment run directory (or an aggregate_metrics.csv). For each
requested k, rows with selection_mode=top_k are sorted descending by
pair_completeness and plotted as a horizontal bar chart.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional at import time, required only for model styles
    yaml = None  # type: ignore[assignment]


DEFAULT_BAR_COLOR = "#4C78A8"
VERTICAL_BAR_RC_PARAMS = {
    "font.family": "serif",
    "font.serif": (
        "P052", "TeX Gyre Pagella", "Palatino", "STIXGeneral", "DejaVu Serif"
    ),
    "mathtext.fontset": "custom",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "font.size": 9.3,
    "axes.labelsize": 9.3,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.pad_inches": 0.02,
}


def _csv_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        return path / "aggregate_metrics.csv"
    return path


def _safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _load_model_styles(run_dir: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    """Load per-model labels and bar colors from a run's resolved experiment config."""
    if run_dir is None or yaml is None:
        return {}, {}
    config_path = run_dir / "resolved_config.yml"
    if not config_path.is_file():
        return {}, {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    labels: dict[str, str] = {}
    colors: dict[str, str] = {}
    for model in ((data.get("models") or {}).get("include") or []):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("model_id") or "")
        if not model_id:
            continue
        label = str(model.get("display_name") or model.get("label") or model.get("name") or "")
        color = model.get("color")
        if label:
            labels[model_id] = label
        if isinstance(color, str) and color.strip():
            colors[model_id] = color.strip()
    return labels, colors


def _short_label(label: str, max_chars: int = 72) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "…"


def load_rows(input_path: Path, task_id: str | None, ks: set[int]) -> tuple[list[dict[str, Any]], Path | None]:
    csv_path = _csv_path(input_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"aggregate_metrics.csv not found: {csv_path}")
    run_dir = csv_path.parent if csv_path.name == "aggregate_metrics.csv" else None
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("selection_mode") != "top_k":
                continue
            if task_id and row.get("task_id") != task_id:
                continue
            try:
                k = int(row.get("k") or "")
            except ValueError:
                continue
            if k not in ks:
                continue
            pc = _safe_float(row.get("pair_completeness"))
            if pc is None:
                continue
            row = dict(row)
            row["k"] = k
            row["pair_completeness"] = pc
            rows.append(row)
    if not rows:
        raise ValueError(f"No top_k pair-completeness rows found for ks={sorted(ks)} task={task_id!r}")
    return rows, run_dir


def write_summary_csv(rows: list[dict[str, Any]], labels: dict[str, str], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "k", "task_id", "model_id", "display_name", "pair_completeness", "pair_quality", "candidate_pairs"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task_id") or ""), int(row["k"]))].append(row)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (_task, _k), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            group = sorted(group, key=lambda row: float(row["pair_completeness"]), reverse=True)
            for rank, row in enumerate(group, start=1):
                model_id = str(row.get("model_id") or "")
                writer.writerow(
                    {
                        "rank": rank,
                        "k": row["k"],
                        "task_id": row.get("task_id"),
                        "model_id": model_id,
                        "display_name": labels.get(model_id, model_id),
                        "pair_completeness": row.get("pair_completeness"),
                        "pair_quality": row.get("pair_quality"),
                        "candidate_pairs": row.get("candidate_pairs"),
                    }
                )
    return output


def plot_ranked_bars(
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    output_dir: Path,
    title_prefix: str | None = None,
    colors: dict[str, str] | None = None,
    formats: tuple[str, ...] = ("svg",),
    style: dict[str, Any] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task_id") or "task"), int(row["k"]))].append(row)

    model_colors = colors or {}
    chart_style = style or {}
    orientation = str(chart_style.get("orientation", "horizontal"))
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("ranked bar orientation must be 'horizontal' or 'vertical'")
    try:
        return _plot_ranked_bars_matplotlib(
            grouped,
            labels,
            model_colors,
            output_dir,
            title_prefix,
            formats,
            chart_style,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        if set(formats) != {"svg"} or orientation != "horizontal":
            raise RuntimeError(
                "Non-SVG or vertical ranked bars require matplotlib"
            ) from exc
        return _plot_ranked_bars_svg(
            grouped, labels, model_colors, output_dir, title_prefix
        )


def _plot_ranked_bars_matplotlib(
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    labels: dict[str, str],
    colors: dict[str, str],
    output_dir: Path,
    title_prefix: str | None = None,
    formats: tuple[str, ...] = ("svg",),
    style: dict[str, Any] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_style = style or {}
    orientation = str(chart_style.get("orientation", "horizontal"))
    rc_params = VERTICAL_BAR_RC_PARAMS if orientation == "vertical" else {}
    outputs: list[Path] = []
    with matplotlib.rc_context(rc_params):
        for (task, k), group in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            ranked = sorted(
                group,
                key=lambda row: float(row["pair_completeness"]),
                reverse=True,
            )
            values = [float(row["pair_completeness"]) for row in ranked]
            model_ids = [str(row.get("model_id") or "") for row in ranked]
            model_labels = [
                _short_label(labels.get(model_id, model_id))
                for model_id in model_ids
            ]
            bar_colors = [
                colors.get(model_id, DEFAULT_BAR_COLOR) for model_id in model_ids
            ]
            if orientation == "vertical":
                fig, ax = _render_vertical_ranked_bars(
                    plt,
                    k,
                    task,
                    values,
                    model_ids,
                    model_labels,
                    bar_colors,
                    title_prefix,
                    chart_style,
                )
                save_options = {"bbox_inches": "tight", "pad_inches": 0.02}
            else:
                height = max(4.5, 0.34 * len(ranked) + 1.6)
                fig, ax = plt.subplots(figsize=(11, height))
                y_pos = list(range(len(ranked)))
                ax.barh(y_pos, values, color=bar_colors)
                ax.set_yticks(y_pos, labels=model_labels, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel(f"Pairs completeness PC@{k}")
                ax.set_title(
                    title_prefix or f"Model ranking by PC@{k} ({task})"
                )
                ax.grid(axis="x", alpha=0.25)
                for y, value in zip(y_pos, values):
                    ax.text(
                        min(value + 0.006, 0.995),
                        y,
                        f"{value:.3f}",
                        va="center",
                        fontsize=8,
                    )
                fig.tight_layout()
                save_options = {}
            for fmt in formats:
                output = output_dir / f"ranked_pc_at_{k}_{_safe_name(task)}.{fmt}"
                fig.savefig(output, format=fmt, **save_options)
                outputs.append(output)
            plt.close(fig)
    return outputs


def _render_vertical_ranked_bars(
    plt: Any,
    k: int,
    task: str,
    values: list[float],
    model_ids: list[str],
    model_labels: list[str],
    bar_colors: list[str],
    title_prefix: str | None,
    style: dict[str, Any],
) -> tuple[Any, Any]:
    """Render compact compact vertical PC bars with numeric annotations."""
    figure_width = float(style.get("figure_width", 5.75))
    figure_height = float(style.get("figure_height", 3.1))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    x_pos = list(range(len(values)))
    bar_width = float(style.get("bar_width", 0.72))
    from matplotlib.colors import to_rgba

    bar_alpha = min(1.0, max(0.0, float(style.get("bar_alpha", 1.0))))
    bars = ax.bar(
        x_pos,
        values,
        width=bar_width,
        color=[to_rgba(color, bar_alpha) for color in bar_colors],
        linewidth=0,
        zorder=2,
    )

    hatches = style.get("model_hatches") or {}
    if not isinstance(hatches, dict):
        raise ValueError("ranked_pc_at_k_bars.model_hatches must be a mapping")
    hatch_linewidth = max(0.0, float(style.get("hatch_linewidth", 0.45)))
    from matplotlib.patches import Rectangle

    for model_id, bar, color in zip(model_ids, bars, bar_colors):
        hatch = hatches.get(model_id)
        if hatch:
            # Keep the translucent fill and opaque hatch in separate patches,
            # as in the multisplit calibration-transfer figure.
            ax.add_patch(
                Rectangle(
                    (bar.get_x(), bar.get_y()),
                    bar.get_width(),
                    bar.get_height(),
                    facecolor="none",
                    edgecolor=color,
                    linewidth=hatch_linewidth,
                    hatch=str(hatch),
                    zorder=3,
                )
            )

    limits = style.get("pc_limits") or [0.0, 1.02]
    if not isinstance(limits, (list, tuple)) or len(limits) != 2:
        raise ValueError("ranked_pc_at_k_bars.pc_limits must contain two numbers")
    lower, upper = (float(value) for value in limits)
    if lower >= upper:
        raise ValueError("ranked_pc_at_k_bars.pc_limits must be increasing")
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylabel(str(style.get("axis_label") or rf"Pairs completeness ($PC@{k}$)"))
    if bool(style.get("show_title", False)):
        ax.set_title(title_prefix or f"Model ranking by PC@{k} ({task})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.75, zorder=0)
    ax.tick_params(axis="x", length=0)

    label_layout = str(style.get("model_label_layout", "rotated"))
    ax.set_xticks(x_pos)
    if label_layout == "alternating_guides":
        ax.set_xticklabels([])
        row_offsets = style.get("model_label_row_offsets") or [-18.0, -35.0]
        if not isinstance(row_offsets, (list, tuple)) or len(row_offsets) != 2:
            raise ValueError(
                "ranked_pc_at_k_bars.model_label_row_offsets must contain two numbers"
            )
        label_fontsize = float(style.get("model_label_fontsize", 8.0))
        guide_color = str(style.get("model_label_guide_color", "#777777"))
        guide_linewidth = max(
            0.0, float(style.get("model_label_guide_linewidth", 0.45))
        )
        for index, label in enumerate(model_labels):
            offset = float(row_offsets[index % 2])
            ax.annotate(
                label,
                xy=(index, 0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=label_fontsize,
                annotation_clip=False,
            )
            ax.annotate(
                "",
                xy=(index, 0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(index, offset + 5.0),
                textcoords=("data", "offset points"),
                arrowprops={
                    "arrowstyle": "-",
                    "color": guide_color,
                    "linewidth": guide_linewidth,
                },
                annotation_clip=False,
            )
        fig.subplots_adjust(bottom=float(style.get("bottom_margin", 0.23)))
    else:
        ax.set_xticklabels(
            model_labels,
            rotation=float(style.get("model_label_rotation", 30.0)),
            ha="right",
        )
        fig.subplots_adjust(bottom=float(style.get("bottom_margin", 0.25)))

    if bool(style.get("show_values", True)):
        decimals = max(0, int(style.get("value_decimals", 6)))
        value_pad = float(style.get("value_pad", 0.008))
        value_fontsize = float(style.get("value_fontsize", 7.3))
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + value_pad,
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                zorder=4,
            )
    return fig, ax


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _plot_ranked_bars_svg(
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    labels: dict[str, str],
    colors: dict[str, str],
    output_dir: Path,
    title_prefix: str | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    for (task, k), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ranked = sorted(group, key=lambda row: float(row["pair_completeness"]), reverse=True)
        output = output_dir / f"ranked_pc_at_{k}_{_safe_name(task)}.svg"
        _write_svg_bar_chart(ranked, labels, colors, output, k, task, title_prefix)
        outputs.append(output)
    return outputs


def _write_svg_bar_chart(
    ranked: list[dict[str, Any]],
    labels: dict[str, str],
    colors: dict[str, str],
    output: Path,
    k: int,
    task: str,
    title_prefix: str | None,
) -> None:
    margin_left = 430
    margin_right = 90
    margin_top = 54
    row_h = 24
    bar_h = 16
    width = 1100
    height = max(180, margin_top + row_h * len(ranked) + 52)
    plot_w = width - margin_left - margin_right
    title = title_prefix or f"Model ranking by PC@{k} ({task})"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:18px;font-weight:700}.axis{font-size:12px;fill:#333}.label{font-size:11px;fill:#222}.value{font-size:11px;fill:#222}.grid{stroke:#ddd;stroke-width:1}</style>",
        f'<text class="title" x="{width / 2:.1f}" y="28" text-anchor="middle">{html.escape(title)}</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = margin_left + tick * plot_w
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top - 8}" x2="{x:.1f}" y2="{height - 42}"/>')
        parts.append(f'<text class="axis" x="{x:.1f}" y="{height - 24}" text-anchor="middle">{tick:.2f}</text>')
    parts.append(f'<text class="axis" x="{margin_left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle">Pairs completeness PC@{k}</text>')
    for i, row in enumerate(ranked, start=1):
        value = float(row["pair_completeness"])
        model_id = str(row.get("model_id") or "")
        label = _short_label(labels.get(model_id, model_id), 78)
        color = html.escape(colors.get(model_id, DEFAULT_BAR_COLOR), quote=True)
        y = margin_top + (i - 1) * row_h
        bar_w = max(0.0, min(1.0, value)) * plot_w
        parts.append(f'<text class="label" x="8" y="{y + 13}" text-anchor="start">{i:02d}. {html.escape(label)}</text>')
        parts.append(f'<rect class="bar" fill="{color}" x="{margin_left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="2"/>')
        parts.append(f'<text class="value" x="{margin_left + bar_w + 6:.1f}" y="{y + 13}">{value:.3f}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.core.plot_ranked_pc_bars")
    parser.add_argument("--run", "--input", dest="input", required=True, type=Path, help="Run dir or aggregate_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run>/bar_charts")
    parser.add_argument("--task", default=None, help="Optional task_id filter")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--title-prefix", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, run_dir = load_rows(args.input, args.task, set(args.ks))
    labels, colors = _load_model_styles(run_dir)
    # A cross-run aggregate has no single resolved_config.yml. Preserve labels
    # flattened into aggregate_metrics.csv by the aggregator.
    for row in rows:
        model_id = str(row.get("model_id") or "")
        display_name = str(row.get("display_name") or "")
        if model_id and display_name:
            labels.setdefault(model_id, display_name)
    output_dir = args.output_dir or ((run_dir or _csv_path(args.input).parent) / "bar_charts")
    summary = write_summary_csv(rows, labels, output_dir / "ranked_pc_summary.csv")
    charts = plot_ranked_bars(rows, labels, output_dir, args.title_prefix, colors)
    print(json.dumps({"status": "ok", "summary": str(summary), "charts": [str(path) for path in charts]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
