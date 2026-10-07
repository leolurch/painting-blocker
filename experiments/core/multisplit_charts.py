"""Scientifically conservative charts for multisplit descriptive analyses."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_text
from .calibration_protocol_bars import (
    SELECTION_MODE,
    CalibrationProtocolBarsError,
    chart_targets,
    parse_protocol_chart,
    protocol_directory,
)
from .chart_sidecar import require_chart_sidecars, write_chart_sidecar
from .plot_ranked_pc_bars import _safe_name


FALLBACK_COLORS = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
    "#D55E00", "#000000", "#F0E442",
)
FALLBACK_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
MATCH_COLOR = "#a50f15"
NON_MATCH_COLOR = "#9ecae1"
MODEL_GROUP_CANONICAL_SUFFIX = "_canonical_models"
THESIS_FIGURE_WIDTH = 5.75
THESIS_RC_PARAMS: dict[str, Any] = {
    "figure.figsize": (THESIS_FIGURE_WIDTH, 3.1),
    "figure.constrained_layout.use": True,
    "font.family": "serif",
    "font.serif": ("P052", "TeX Gyre Pagella", "Palatino", "STIXGeneral", "DejaVu Serif"),
    "mathtext.fontset": "custom",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "font.size": 10.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.pad_inches": 0.02,
}


def _with_chart_style(function: Any) -> Any:
    """Scope chart rcParams to one render invocation."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            import matplotlib
        except ModuleNotFoundError:
            return function(*args, **kwargs)
        with matplotlib.rc_context(THESIS_RC_PARAMS):
            return function(*args, **kwargs)

    return wrapped


def _enabled(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = (config.get("charts") or {}).get(name)
    return value if isinstance(value, dict) and bool(value.get("enabled", False)) else None


def _formats(chart: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(str(value) for value in chart.get("formats") or ["pdf"])
    invalid = sorted(set(values) - {"pdf", "svg", "png"})
    if invalid:
        raise ValueError(f"Unsupported multisplit chart formats: {', '.join(invalid)}")
    return values


def _model_label_fontsize(
    plt: Any,
    chart: dict[str, Any],
    base_fontsize: float | None = None,
) -> float:
    base = (
        float(plt.rcParams["font.size"])
        if base_fontsize is None
        else float(base_fontsize)
    )
    scale = max(0.0, float(chart.get("model_label_font_scale", 1.0)))
    return base * scale


def _axis_ylabel(
    chart: dict[str, Any],
    metric: str,
    default_label: str,
    default_description: str = "",
    *,
    description_separator: str = "\n",
) -> str:
    """Resolve a configurable axis label while retaining legacy overrides."""
    label_key = f"{metric}_axis_label"
    description_key = f"{metric}_axis_description"
    legacy_key = f"{metric}_ylabel"
    if label_key not in chart and description_key not in chart and legacy_key in chart:
        legacy_value = chart[legacy_key]
        return "" if legacy_value is None else str(legacy_value)

    configured_label = chart.get(label_key)
    configured_description = chart.get(description_key)
    label = default_label if configured_label is None else str(configured_label)
    description = (
        default_description
        if configured_description is None
        else str(configured_description)
    )
    if label and description:
        return f"{label}{description_separator}{description}"
    return label or description


def _calibration_display_metadata(
    fig: Any, chart: dict[str, Any]
) -> dict[str, Any]:
    """Record the effective presentation switches beside numeric chart data."""
    return {
        "include_attainment_score": bool(
            chart.get("include_attainment_score", True)
        ),
        "include_specific_result_crosses": bool(
            chart.get("include_specific_result_crosses", True)
        ),
        "show_pc_mean_values": bool(chart.get("show_pc_mean_values", False)),
        "pc_mean_value_decimals": max(
            0, int(chart.get("pc_mean_value_decimals", 3))
        ),
        "pc_axis_text": str(fig.axes[0].get_ylabel()),
        "rr_axis_text": str(fig.axes[1].get_ylabel()),
    }


def _save(
    fig: Any,
    plt: Any,
    output: Path,
    stem: str,
    chart: dict[str, Any],
    numeric_data: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    for fmt in _formats(chart):
        path = output / f"{stem}.{fmt}"
        savefig_options: dict[str, Any] = {}
        if bool(chart.get("trim_outer_padding", False)):
            savefig_options.update(bbox_inches="tight", pad_inches=0)
        fig.savefig(path, format=fmt, **savefig_options)
        paths.append(path)
    plt.close(fig)
    sidecar = write_chart_sidecar(output, stem, numeric_data)
    paths.append(sidecar)
    return paths


def _models(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [entry for entry in config.get("models") or [] if isinstance(entry, dict) and entry.get("include")]
    return sorted(entries, key=lambda row: (int((row.get("presentation") or {}).get("order", 10**9)), str(row["analysis_model_id"])))


def _styles(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(_models(config)):
        presentation = entry.get("presentation") or {}
        result[str(entry["analysis_model_id"])] = {
            "label": str(
                presentation.get("short_label")
                or presentation.get("label")
                or entry["analysis_model_id"]
            ),
            "color": str(presentation.get("color") or FALLBACK_COLORS[index % len(FALLBACK_COLORS)]),
            "marker": str(presentation.get("marker") or FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)]),
            "linestyle": str(presentation.get("linestyle") or ("--" if presentation.get("group") == "frozen" else "-")),
            "group": str(presentation.get("group") or "ungrouped"),
        }
    return result


def _chart_model_ids(chart: dict[str, Any]) -> set[str] | None:
    configured = chart.get("models")
    if configured is None:
        return None
    return {str(value) for value in configured}


def _filter_chart_models(
    rows: list[dict[str, Any]], chart: dict[str, Any]
) -> list[dict[str, Any]]:
    model_ids = _chart_model_ids(chart)
    if model_ids is None:
        return rows
    return [
        row for row in rows
        if str(row.get("configuration_id")) in model_ids
    ]


def _task_rows(artifact: dict[str, Any], chart: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(artifact.get("summary_rows") or [])
    task = chart.get("task")
    if task is not None:
        rows = [row for row in rows if str(row.get("task_id")) == str(task)]
    return _filter_chart_models(rows, chart)


def _model_group_members(config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    groups = config.get("model_groups") or {}
    return [
        (str(group_id), [str(value) for value in settings.get("members") or []])
        for group_id, settings in groups.items()
        if isinstance(settings, dict)
    ]


def _canonical_model_group_members(config: dict[str, Any]) -> dict[str, str]:
    """Return the predeclared representative of every model group.

    Canonical representatives are configuration choices, never selected from
    displayed test metrics. ``canonical`` is required by config validation.
    """
    groups = config.get("model_groups") or {}
    return {
        str(group_id): str(settings["canonical"])
        for group_id, settings in groups.items()
        if isinstance(settings, dict) and settings.get("members")
    }


def _canonical_filtered_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    additional_models: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    canonical = _canonical_model_group_members(config)
    excluded = {
        member
        for group_id, members in _model_group_members(config)
        for member in members
        if member != canonical[group_id]
    }
    excluded -= {str(value) for value in additional_models or []}
    return [
        row for row in rows
        if str(row.get("configuration_id")) not in excluded
    ]


def _canonical_filtered_chart_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    chart: dict[str, Any],
) -> list[dict[str, Any]]:
    return _canonical_filtered_rows(
        rows,
        config,
        chart.get("canonical_additional_models"),
    )


def _model_group_chart_modes(config: dict[str, Any]) -> tuple[tuple[str, bool], ...]:
    """Return all-model and predeclared-canonical display modes."""
    modes: tuple[tuple[str, bool], ...] = (("", False),)
    if _model_group_members(config):
        modes += ((MODEL_GROUP_CANONICAL_SUFFIX, True),)
    return modes


def _chart_k_values(chart: dict[str, Any], rows: list[dict[str, Any]]) -> list[int]:
    configured = chart.get("k", 40)
    if configured == "all":
        return sorted({int(row["k"]) for row in rows if row.get("k") is not None})
    if isinstance(configured, list):
        return list(dict.fromkeys(int(value) for value in configured))
    return [int(configured)]


def _split_values(row: dict[str, Any], metric: str) -> list[tuple[int, float]]:
    return [
        (int(value["split_seed"]), float(value["mean_over_training_seeds"]))
        for value in row.get(f"{metric}_split_values") or []
    ]


def _similarity_distribution_entries(
    artifact: dict[str, Any],
    config: dict[str, Any],
    chart: dict[str, Any],
    *,
    select_canonical_models: bool = False,
) -> list[dict[str, Any]]:
    task = chart.get("task")
    entries = _filter_chart_models(
        [
            row for row in artifact.get("similarity_distributions") or []
            if task is None or str(row.get("task_id")) == str(task)
        ],
        chart,
    )
    return (
        _canonical_filtered_chart_rows(entries, config, chart)
        if select_canonical_models
        else entries
    )


def _similarity_thresholds(
    artifact: dict[str, Any], chart: dict[str, Any]
) -> dict[str, dict[str, float]]:
    target = chart.get("target_pc")
    if target is None:
        return {}
    result: dict[str, dict[str, float]] = {}
    for row in _task_rows(artifact, chart):
        if row.get("selection_mode") != "calibrated_threshold":
            continue
        observed_target = row.get("calibration_target_pair_completeness")
        if observed_target is None or abs(float(observed_target) - float(target)) > 1e-12:
            continue
        if any(row.get(key) is None for key in ("threshold_mean", "threshold_min", "threshold_max")):
            continue
        result[str(row["configuration_id"])] = {
            "mean": float(row["threshold_mean"]),
            "min": float(row["threshold_min"]),
            "max": float(row["threshold_max"]),
        }
    return result


def _render_similarity_distributions(
    plt: Any,
    entries: list[dict[str, Any]],
    styles: dict[str, dict[str, str]],
    thresholds: dict[str, dict[str, float]],
    chart: dict[str, Any],
) -> Any:
    ncols = min(5, len(entries))
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(THESIS_FIGURE_WIDTH, 0.6 * max(3.0, 1.75 * nrows + 0.45)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = axes.reshape(-1)
    show_range = bool(chart.get("show_split_range", True))
    for ax, entry in zip(flat, entries):
        model_id = str(entry["configuration_id"])
        edges = np.asarray(entry["bin_edges"], dtype=float)
        match_mean = np.asarray(entry["match_density_mean"], dtype=float)
        non_match_mean = np.asarray(entry["non_match_density_mean"], dtype=float)
        if show_range:
            match_low = np.asarray(entry["match_density_min"], dtype=float)
            match_high = np.asarray(entry["match_density_max"], dtype=float)
            non_match_low = np.asarray(entry["non_match_density_min"], dtype=float)
            non_match_high = np.asarray(entry["non_match_density_max"], dtype=float)
            ax.fill_between(
                edges,
                np.r_[non_match_low, non_match_low[-1]],
                np.r_[non_match_high, non_match_high[-1]],
                step="post",
                color=NON_MATCH_COLOR,
                alpha=0.18,
                linewidth=0,
            )
            ax.fill_between(
                edges,
                np.r_[match_low, match_low[-1]],
                np.r_[match_high, match_high[-1]],
                step="post",
                color=MATCH_COLOR,
                alpha=0.12,
                linewidth=0,
            )
        ax.stairs(
            non_match_mean,
            edges,
            fill=True,
            alpha=0.45,
            color=NON_MATCH_COLOR,
            label="non-match mean",
        )
        ax.stairs(
            match_mean,
            edges,
            fill=True,
            alpha=0.45,
            color=MATCH_COLOR,
            label="match mean",
        )
        threshold = thresholds.get(model_id)
        if threshold is not None:
            ax.axvspan(
                threshold["min"], threshold["max"], color="black", alpha=0.08, linewidth=0
            )
            ax.axvline(
                threshold["mean"], color="black", lw=0.8, ls=":", alpha=0.8,
                label="mean calibrated threshold",
            )
        ax.set_title(
            styles[model_id]["label"],
            fontsize=_model_label_fontsize(plt, chart, 10.0),
        )
        xlim = chart.get("xlim") or [-0.2, 1.0]
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    for ax in flat[len(entries):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=len(labels),
        frameon=False,
        fontsize=10,
    )
    return fig


def _render_calibration_transfer(
    plt: Any,
    rows: list[dict[str, Any]],
    styles: dict[str, dict[str, str]],
    chart: dict[str, Any],
    target: float,
    attainment_tolerance: float = 1e-12,
) -> Any:
    """Render test PC vertically with contextual mean test RR bars."""
    rows = sorted(
        rows, key=lambda row: float(row.get("pair_completeness_mean", 0.0))
    )
    rng = np.random.default_rng(int(chart.get("jitter_seed", 0)))
    fig, ax = plt.subplots(
        figsize=(THESIS_FIGURE_WIDTH, float(chart.get("figure_height", 3.8)))
    )
    ax_rr = ax.twinx()
    include_attainment_score = bool(
        chart.get("include_attainment_score", True)
    )
    include_specific_result_crosses = bool(
        chart.get("include_specific_result_crosses", True)
    )
    configured_colors = {
        str(model_id): str(color)
        for model_id, color in (chart.get("model_colors") or {}).items()
    }

    def display_color(model_id: str) -> str:
        return configured_colors.get(model_id, styles[model_id]["color"])

    rr_alpha = min(0.40, max(0.0, float(chart.get("rr_bar_alpha", 0.30))))
    from matplotlib import patheffects
    from matplotlib.colors import to_rgba
    from matplotlib.markers import MarkerStyle

    bar_colors = [
        display_color(str(row["configuration_id"])) for row in rows
    ]
    rr_bars = ax_rr.bar(
        range(len(rows)),
        [float(row["reduction_ratio_mean"]) for row in rows],
        width=0.72,
        color=[to_rgba(color, rr_alpha) for color in bar_colors],
        linewidth=0,
        zorder=1,
    )
    hatch_config = chart.get("rr_bar_hatches") or {}
    if not isinstance(hatch_config, dict):
        raise ValueError("rr_bar_hatches must be a mapping")
    group_hatches = hatch_config.get("groups") or {}
    model_hatches = hatch_config.get("models") or {}
    if not isinstance(group_hatches, dict) or not isinstance(model_hatches, dict):
        raise ValueError("rr_bar_hatches.groups and .models must be mappings")
    from matplotlib.patches import Rectangle

    for row, bar, color in zip(rows, rr_bars, bar_colors):
        model_id = str(row["configuration_id"])
        hatch = model_hatches.get(
            model_id,
            group_hatches.get(styles[model_id]["group"]),
        )
        if hatch:
            # Keep the translucent fill and opaque hatch in separate patches.
            # This also preserves hatch strokes in vector PDF renderers, where
            # translucent pattern fills are not consistently displayed.
            hatch_overlay = Rectangle(
                (bar.get_x(), bar.get_y()),
                bar.get_width(),
                bar.get_height(),
                facecolor="none",
                edgecolor=color,
                linewidth=max(
                    0.0,
                    float(chart.get("rr_bar_hatch_edge_linewidth", 0.45)),
                ),
                hatch=str(hatch),
                zorder=2,
            )
            ax_rr.add_patch(hatch_overlay)
    ax_rr.set_ylim(0.0, 1.0)
    dataset_label = chart.get("dataset_label")
    show_axis_label_details = bool(chart.get("show_axis_label_details", True))
    if dataset_label:
        default_rr_axis_label = rf"$RR^{{test}}_{{\mathit{{{dataset_label}}}}}$"
        default_rr_axis_description = (
            "(filled bars)" if show_axis_label_details else ""
        )
    else:
        default_rr_axis_label = "Test reduction ratio (RR)"
        default_rr_axis_description = ""
    ax_rr.set_ylabel(
        _axis_ylabel(
            chart,
            "rr",
            default_rr_axis_label,
            default_rr_axis_description,
            description_separator=" ",
        )
    )
    ax_rr.grid(False)

    ax.set_zorder(ax_rr.get_zorder() + 1)
    ax.patch.set_visible(False)
    attainment_threshold = max(0.0, target - attainment_tolerance)
    all_pc_values: list[float] = []
    for index, row in enumerate(rows):
        model_id = str(row["configuration_id"])
        style = styles[model_id]
        color = display_color(model_id)
        point_marker = str(chart.get("pc_marker") or style["marker"])
        values = np.asarray(
            [value for _seed, value in _split_values(row, "pair_completeness")],
            dtype=float,
        )
        all_pc_values.extend(values.tolist())
        x = index + rng.uniform(-0.13, 0.13, len(values))
        marker_size = max(0.0, float(chart.get("pc_marker_size", 20.0)))
        marker_linewidth = max(
            0.0, float(chart.get("pc_marker_linewidth", 1.0))
        )
        marker_halo_linewidth = max(
            0.0, float(chart.get("pc_marker_halo_linewidth", 0.0))
        )
        point_collections: list[Any] = []
        if include_specific_result_crosses:
            if include_attainment_score:
                attained = values >= attainment_threshold
                attained_points = ax.scatter(
                    x[attained],
                    values[attained],
                    color=color,
                    marker=point_marker,
                    s=marker_size,
                    linewidth=marker_linewidth,
                    alpha=0.95,
                    zorder=5,
                )
                if MarkerStyle(point_marker).is_filled():
                    missed_marker_colors = {
                        "facecolors": "none",
                        "edgecolors": color,
                    }
                else:
                    missed_marker_colors = {"color": color}
                missed_points = ax.scatter(
                    x[~attained],
                    values[~attained],
                    marker=point_marker,
                    s=marker_size,
                    linewidth=marker_linewidth,
                    zorder=5,
                    **missed_marker_colors,
                )
                point_collections = [attained_points, missed_points]
            else:
                point_collections = [
                    ax.scatter(
                        x,
                        values,
                        color=color,
                        marker=point_marker,
                        s=marker_size,
                        linewidth=marker_linewidth,
                        alpha=0.95,
                        zorder=5,
                    )
                ]
        if (
            point_collections
            and marker_halo_linewidth > marker_linewidth
        ):
            marker_effects = [
                patheffects.Stroke(
                    linewidth=marker_halo_linewidth,
                    foreground=str(chart.get("pc_marker_halo_color") or "white"),
                ),
                patheffects.Normal(),
            ]
            for points in point_collections:
                points.set_path_effects(marker_effects)
        pc_mean = float(row["pair_completeness_mean"])
        ax.hlines(
            pc_mean,
            index - 0.28,
            index + 0.28,
            color="black",
            lw=1.4,
            zorder=6,
        )
        if bool(chart.get("show_pc_mean_values", False)):
            decimals = max(0, int(chart.get("pc_mean_value_decimals", 3)))
            ax.annotate(
                f"{pc_mean:.{decimals}f}",
                xy=(index, pc_mean),
                xytext=(
                    0.0,
                    -abs(float(chart.get("pc_mean_value_offset_points", 2.5))),
                ),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=float(chart.get("pc_mean_value_fontsize", 7.0)),
                color=str(chart.get("pc_mean_value_color") or "black"),
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": min(
                        1.0,
                        max(
                            0.0,
                            float(chart.get("pc_mean_value_background_alpha", 0.8)),
                        ),
                    ),
                    "pad": 0.1,
                },
                annotation_clip=True,
                zorder=7,
            )

    if include_attainment_score:
        attainment_axis = ax.secondary_xaxis("top")
        attainment_axis.set_xticks(
            range(len(rows)),
            [
                f"{row.get('target_attainment_count', 0)}/{row['num_split_realizations']}"
                for row in rows
            ],
        )
        attainment_axis.tick_params(
            axis="x",
            length=0,
            pad=float(chart.get("attainment_score_tickpad", 0.5)),
        )
        attainment_axis.spines["top"].set_visible(False)
        if bool(chart.get("show_attainment_axis_label", False)):
            threshold_text = f"{attainment_threshold:.3f}"
            attainment_dataset = str(dataset_label or "test")
            attainment_axis.set_xlabel(
                "Attainment score: "
                rf"$PC^{{test}}_{{\mathit{{{attainment_dataset}}}}} "
                rf"\geq {threshold_text}$",
                labelpad=float(chart.get("attainment_axis_labelpad", 4.5)),
            )

    configured_ylim = chart.get("pc_ylim")
    if configured_ylim is not None:
        ax.set_ylim(float(configured_ylim[0]), float(configured_ylim[1]))
    else:
        minimum = min(all_pc_values, default=target)
        ax.set_ylim(max(0.0, min(0.90, minimum - 0.005)), 1.002)
    if include_attainment_score:
        ax.axhline(target, color="black", ls=":", lw=0.8, zorder=4)
    if include_attainment_score and bool(chart.get("show_attainment_region", False)):
        region_top = float(ax.get_ylim()[1])
        attainment_region = Rectangle(
            (0.0, attainment_threshold),
            1.0,
            max(0.0, region_top - attainment_threshold),
            transform=ax.get_yaxis_transform(),
            facecolor=str(chart.get("attainment_region_color") or "#DCEFD8"),
            edgecolor="none",
            alpha=min(
                1.0,
                max(0.0, float(chart.get("attainment_region_alpha", 0.55))),
            ),
            zorder=0,
        )
        attainment_region.set_gid("attainment-threshold-region")
        # Attach the region to the RR axis so it is drawn beneath RR bars;
        # PC points and mean indicators remain on the foreground PC axis.
        ax_rr.add_patch(attainment_region)
    configured_label_layout = chart.get("model_label_layout")
    model_label_layout = str(
        configured_label_layout
        or ("legend" if bool(chart.get("show_model_legend", False)) else "rotated")
    )
    if model_label_layout not in {"legend", "rotated", "alternating_guides"}:
        raise ValueError(
            "model_label_layout must be 'legend', 'rotated', or "
            "'alternating_guides'"
        )
    if model_label_layout == "alternating_guides":
        ax.set_xticks([])
        row_offsets = [
            float(value)
            for value in chart.get("model_label_row_offsets") or [-0.03, -0.145]
        ]
        if len(row_offsets) != 2 or not all(value < 0.0 for value in row_offsets):
            raise ValueError(
                "model_label_row_offsets must contain exactly two negative values"
            )
        guide_gap = max(0.0, float(chart.get("model_label_guide_gap", 0.015)))
        from matplotlib.transforms import ScaledTranslation

        transform = ax.get_xaxis_transform()
        extra_row_spacing_points = max(
            0.0,
            float(chart.get("model_label_row_spacing_increment_points", 0.5)),
        )
        first_label_row = str(chart.get("model_label_first_row") or "top")
        if first_label_row not in {"top", "bottom"}:
            raise ValueError("model_label_first_row must be 'top' or 'bottom'")
        label_fontsize = _model_label_fontsize(
            plt,
            chart,
            float(chart.get("model_label_fontsize", plt.rcParams["font.size"])),
        )
        for index, row in enumerate(rows):
            label_row_index = index % 2
            if first_label_row == "bottom":
                label_row_index = 1 - label_row_index
            label_y = row_offsets[label_row_index]
            ax.plot(
                [index, index],
                [0.0, label_y + guide_gap],
                color="black",
                lw=float(chart.get("model_label_guide_linewidth", 0.55)),
                solid_capstyle="butt",
                transform=transform,
                clip_on=False,
                zorder=7,
            )
            label_transform = transform
            if label_row_index == 1 and extra_row_spacing_points > 0.0:
                label_transform = transform + ScaledTranslation(
                    0.0,
                    -extra_row_spacing_points / 72.0,
                    fig.dpi_scale_trans,
                )
            ax.annotate(
                styles[str(row["configuration_id"])]["label"],
                (index, label_y),
                xycoords=label_transform,
                ha="center",
                va="top",
                fontsize=label_fontsize,
                annotation_clip=False,
            )
    elif model_label_layout == "legend":
        from matplotlib.patches import Patch

        ax.set_xticks([])
        legend_group_order = [
            str(value) for value in chart.get("legend_group_order") or []
        ]
        if len(legend_group_order) != len(set(legend_group_order)):
            raise ValueError("legend_group_order must not contain duplicates")
        group_rank = {
            group: index for index, group in enumerate(legend_group_order)
        }
        legend_rows = sorted(
            enumerate(rows),
            key=lambda item: (
                group_rank.get(
                    styles[str(item[1]["configuration_id"])]["group"],
                    len(group_rank),
                ),
                item[0],
            ),
        )
        legend_handles = [
            Patch(
                facecolor=display_color(str(row["configuration_id"])),
                edgecolor="none",
                label=styles[str(row["configuration_id"])]["label"],
            )
            for _index, row in legend_rows
        ]
        legend_ncols = min(4, len(legend_handles))
        legend_nrows = int(np.ceil(len(legend_handles) / legend_ncols))
        # Matplotlib fills legends column-first. Reorder handles so the visible
        # rows retain the left-to-right model order used in the chart.
        legend_handles = [
            legend_handles[row * legend_ncols + column]
            for column in range(legend_ncols)
            for row in range(legend_nrows)
            if row * legend_ncols + column < len(legend_handles)
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            fontsize=_model_label_fontsize(plt, chart),
            bbox_to_anchor=(0.5, -0.01),
            ncols=legend_ncols,
            frameon=False,
            borderaxespad=0,
            borderpad=0,
            columnspacing=0.8,
            handlelength=1.0,
            handletextpad=0.35,
            labelspacing=0.25,
        )
    else:  # rotated
        ax.set_xticks(
            range(len(rows)),
            [styles[str(row["configuration_id"])]["label"] for row in rows],
            rotation=35,
            ha="right",
        )
        ax.tick_params(
            axis="x", labelsize=_model_label_fontsize(plt, chart)
        )
    if dataset_label:
        default_pc_axis_label = (
            rf"$PC^{{test}}_{{\mathit{{{dataset_label}}}}}$ at "
            rf"$\rho_{{cal}}={target:g}$"
        )
        default_pc_axis_description = (
            "(points and mean bar)" if show_axis_label_details else ""
        )
    else:
        default_pc_axis_label = (
            "Test PC at validation-selected threshold\n"
            f"($\\rho={target:g}$)"
        )
        default_pc_axis_description = ""
    ax.set_ylabel(
        _axis_ylabel(
            chart,
            "pc",
            str(chart.get("_default_pc_axis_label") or default_pc_axis_label),
            str(
                chart.get(
                    "_default_pc_axis_description",
                    default_pc_axis_description,
                )
            ),
        )
    )
    if bool(chart.get("show_explanatory_heading", True)):
        heading_parts: list[str] = []
        if include_specific_result_crosses:
            heading_parts.append(
                "filled = attained; open = missed"
                if include_attainment_score
                else "points = split-level test PC"
            )
        heading_parts.append("black ticks = mean test PC")
        if include_attainment_score:
            heading_parts.append("top = attained/total")
        heading_parts.append("bars = mean test RR")
        ax.set_title("; ".join(heading_parts), pad=22)
    ax.grid(axis="y", alpha=0.3)
    return fig


def _emit_calibration_transfer(
    plt: Any,
    config: dict[str, Any],
    styles: dict[str, dict[str, str]],
    chart: dict[str, Any],
    base_rows: list[dict[str, Any]],
    target: float,
    attainment_tolerance: float,
    output_dir: Path,
) -> list[Path]:
    """Write the threshold transfer bar chart, including its canonical-model copy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, select_canonical in _model_group_chart_modes(config):
        rows = (
            _canonical_filtered_chart_rows(base_rows, config, chart)
            if select_canonical
            else list(base_rows)
        )
        fig = _render_calibration_transfer(
            plt,
            rows,
            styles,
            chart,
            target,
            attainment_tolerance,
        )
        plotted_rows = sorted(
            rows, key=lambda row: float(row.get("pair_completeness_mean", 0.0))
        )
        rng = np.random.default_rng(int(chart.get("jitter_seed", 0)))
        chart_rows: list[dict[str, Any]] = []
        for index, row in enumerate(plotted_rows):
            pc_by_seed = dict(_split_values(row, "pair_completeness"))
            rr_by_seed = dict(_split_values(row, "reduction_ratio"))
            x = index + rng.uniform(-0.13, 0.13, len(pc_by_seed))
            chart_rows.append({
                "model_id": str(row["configuration_id"]),
                "pair_completeness_mean": float(row["pair_completeness_mean"]),
                "reduction_ratio_mean": float(row["reduction_ratio_mean"]),
                "target_attainment_count": int(row.get("target_attainment_count", 0)),
                "target_attainment_tolerance": float(
                    row.get("target_attainment_tolerance", 1e-12)
                ),
                "target_attainment_threshold": float(
                    row.get("target_attainment_threshold", target)
                ),
                "num_split_realizations": int(row["num_split_realizations"]),
                "points": [
                    {
                        "split_seed": seed,
                        "pair_completeness": pc_by_seed[seed],
                        "reduction_ratio": rr_by_seed[seed],
                        "x": x_value,
                    }
                    for seed, x_value in zip(pc_by_seed, x)
                ],
            })
        pc_limits = tuple(float(value) for value in fig.axes[0].get_ylim())
        rr_limits = tuple(float(value) for value in fig.axes[1].get_ylim())
        tag = f"{target:g}".replace(".", "p")
        written.extend(_save(
            fig,
            plt,
            output_dir,
            f"multisplit_calibration_transfer_rho{tag}{suffix}",
            chart,
            {
                "calibration_target_pair_completeness": target,
                "target_attainment_tolerance": attainment_tolerance,
                "target_attainment_threshold": max(
                    0.0, target - attainment_tolerance
                ),
                "show_attainment_region": bool(
                    chart.get("show_attainment_region", False)
                ),
                "attainment_region_color": str(
                    chart.get("attainment_region_color") or "#DCEFD8"
                ),
                "attainment_region_alpha": min(
                    1.0,
                    max(
                        0.0,
                        float(chart.get("attainment_region_alpha", 0.55)),
                    ),
                ),
                "pair_completeness_limits": pc_limits,
                "reduction_ratio_limits": rr_limits,
                "reduction_ratio_bar_alpha": min(
                    0.40, max(0.0, float(chart.get("rr_bar_alpha", 0.30)))
                ),
                "pair_completeness_marker": str(chart.get("pc_marker") or "per-model"),
                "layout": "vertical_pc_with_rr_bars",
                **_calibration_display_metadata(fig, chart),
                "model_label_layout": str(
                    chart.get("model_label_layout")
                    or (
                        "legend"
                        if bool(chart.get("show_model_legend", False))
                        else "rotated"
                    )
                ),
                "series": chart_rows,
            },
        ))
    return written


@_with_chart_style
def write_multisplit_charts(
    artifact: dict[str, Any], config: dict[str, Any], output_dir: Path
) -> tuple[list[Path], list[str]]:
    try:
        import matplotlib
    except ModuleNotFoundError:
        return [], ["matplotlib is not installed; multisplit charts were skipped"]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    styles = _styles(config)
    paths: list[Path] = []
    warnings: list[str] = []

    chart = _enabled(config, "multisplit_pc_at_k_band")
    if chart:
        metric = str(chart.get("metric") or "pair_completeness")
        base_rows = [row for row in _task_rows(artifact, chart) if row.get("selection_mode") == "top_k"]
        for suffix, select_canonical in _model_group_chart_modes(config):
            rows = (
                _canonical_filtered_chart_rows(base_rows, config, chart)
                if select_canonical
                else base_rows
            )
            fig, ax = plt.subplots(figsize=(THESIS_FIGURE_WIDTH, 3.15))
            chart_series: list[dict[str, Any]] = []
            for model_id in styles:
                selected = sorted(
                    [row for row in rows if str(row["configuration_id"]) == model_id],
                    key=lambda row: int(row["k"]),
                )
                if not selected:
                    continue
                style = styles[model_id]
                ks = [int(row["k"]) for row in selected]
                means = [float(row[f"{metric}_mean"]) for row in selected]
                lower = [float(row[f"{metric}_min"]) for row in selected]
                upper = [float(row[f"{metric}_max"]) for row in selected]
                chart_series.append({
                    "model_id": model_id,
                    "candidate_budget": ks,
                    "mean": means,
                    "minimum": lower,
                    "maximum": upper,
                })
                ax.plot(ks, means, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], ms=3.5, lw=1.3, label=style["label"])
                ax.fill_between(ks, lower, upper, color=style["color"], alpha=0.15, linewidth=0)
            ax.set_xscale("log")
            configured_ks = sorted({int(row["k"]) for row in rows})
            ax.set_xticks(configured_ks, [str(value) for value in configured_ks])
            ax.set_xlabel("Candidate budget k")
            ax.set_ylabel("Mean PC@k; band = observed split range")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(
                frameon=False,
                fontsize=_model_label_fontsize(plt, chart, 10.0),
                ncol=2,
            )
            paths.extend(_save(
                fig,
                plt,
                output_dir,
                f"multisplit_pc_at_k_band{suffix}",
                chart,
                {"series": chart_series},
            ))

    chart = _enabled(config, "multisplit_strip")
    if chart:
        metric = str(chart.get("metric") or "pair_completeness")
        top_k_rows = [
            row for row in _task_rows(artifact, chart)
            if row.get("selection_mode") == "top_k"
        ]
        for k in _chart_k_values(chart, top_k_rows):
            base_rows = [row for row in top_k_rows if int(row.get("k") or -1) == k]
            for suffix, select_canonical in _model_group_chart_modes(config):
                rows = (
                    _canonical_filtered_chart_rows(base_rows, config, chart)
                    if select_canonical
                    else list(base_rows)
                )
                rows.sort(key=lambda row: float(row.get(f"{metric}_mean", 0.0)))
                rng = np.random.default_rng(int(chart.get("jitter_seed", 0)))
                fig, ax = plt.subplots(figsize=(THESIS_FIGURE_WIDTH, 3.1))
                chart_rows: list[dict[str, Any]] = []
                for index, row in enumerate(rows):
                    model_id = str(row["configuration_id"])
                    style = styles[model_id]
                    values = [value for _seed, value in _split_values(row, metric)]
                    x = index + rng.uniform(-0.13, 0.13, len(values))
                    chart_rows.append({
                        "model_id": model_id,
                        "mean": float(row[f"{metric}_mean"]),
                        "points": [
                            {"split_seed": seed, "value": value, "x": x_value}
                            for (seed, value), x_value in zip(_split_values(row, metric), x)
                        ],
                    })
                    ax.scatter(x, values, color=style["color"], marker=style["marker"], s=18, alpha=0.75)
                    ax.hlines(float(row[f"{metric}_mean"]), index - 0.28, index + 0.28, color="black", lw=1.4)
                ax.set_xticks(range(len(rows)), [styles[str(row["configuration_id"])]["label"] for row in rows], rotation=30, ha="right")
                ax.tick_params(
                    axis="x", labelsize=_model_label_fontsize(plt, chart)
                )
                ax.set_ylabel(f"PC@{k} per split (horizontal tick = mean)")
                ax.grid(axis="y", alpha=0.3)
                paths.extend(_save(
                    fig,
                    plt,
                    output_dir,
                    f"multisplit_strip_pc_at_{k}{suffix}",
                    chart,
                    {"candidate_budget": k, "series": chart_rows},
                ))

    chart = _enabled(config, "multisplit_paired_difference")
    if chart:
        metrics = chart.get("metrics")
        if metrics is None:
            metrics = [chart.get("metric") or "pair_completeness"]
        metrics = [str(value) for value in metrics]
        selection_mode = str(chart.get("selection_mode") or "top_k")
        task = str(chart["task"])
        rows = [
            row for row in artifact.get("paired_comparisons") or []
            if row.get("selection_mode") == selection_mode
            and str(row.get("task_id") or "") == task
        ]
        operation_label: str
        operation_stem: str
        operation_payload: dict[str, Any]
        if selection_mode == "top_k":
            k = int(chart.get("k", 40))
            rows = [row for row in rows if int(row.get("k") or -1) == k]
            operation_label = f"at fixed k={k}"
            operation_stem = f"at_k{k}"
            operation_payload = {"candidate_budget": k}
        else:
            target_pc = float(chart["target_pc"])
            rows = [
                row for row in rows
                if abs(
                    float(row.get("calibration_target_pair_completeness") or -1.0)
                    - target_pc
                )
                <= 1e-12
            ]
            target_tag = f"{target_pc:g}".replace(".", "p")
            mode_tag = {
                "calibrated_threshold": "threshold",
                "calibrated_top_k": "top_k",
                "calibrated_threshold_top_k_floor": "threshold_top_k_floor",
            }[selection_mode]
            operation_label = (
                f"at validation-selected {mode_tag.replace('_', ' ')} "
                f"(ρ={target_pc:g})"
            )
            operation_stem = f"{mode_tag}_rho{target_tag}"
            operation_payload = {
                "calibration_target_pair_completeness": target_pc,
            }
            if selection_mode == "calibrated_threshold_top_k_floor":
                minimum_top_k = int(chart["minimum_top_k"])
                rows = [
                    row for row in rows
                    if int(row.get("minimum_top_k") or -1) == minimum_top_k
                ]
                operation_label += f", minimum k={minimum_top_k}"
                operation_stem += f"_min_k{minimum_top_k}"
                operation_payload["minimum_top_k"] = minimum_top_k
        configured = {str(value) for value in chart.get("comparisons") or []}
        if configured:
            rows = [row for row in rows if str(row["comparison_id"]) in configured]
        chart_model_ids = _chart_model_ids(chart)
        if chart_model_ids is not None:
            rows = [
                row for row in rows
                if str(row.get("minuend")) in chart_model_ids
                and str(row.get("subtrahend")) in chart_model_ids
            ]
        if not rows:
            warnings.append(
                f"No paired comparisons matched {selection_mode} {operation_label}"
            )
        metric_labels = {
            "pair_completeness": ("PC", "pc"),
            "pair_quality": ("PQ", "pq"),
            "reduction_ratio": ("RR", "rr"),
        }
        for metric in metrics:
            if not rows:
                continue
            metric_label, metric_tag = metric_labels[metric]
            rng = np.random.default_rng(int(chart.get("jitter_seed", 0)))
            fig, ax = plt.subplots(figsize=(THESIS_FIGURE_WIDTH, 3.1))
            chart_rows: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                split_values = row.get(f"{metric}_difference_split_values") or []
                values = [float(value["difference"]) for value in split_values]
                x = index + rng.uniform(-0.13, 0.13, len(values))
                color = FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
                marker = FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)]
                ax.scatter(
                    x,
                    values,
                    color=color,
                    marker=marker,
                    s=18,
                    alpha=0.75,
                )
                ax.hlines(
                    float(row[f"{metric}_difference_mean"]),
                    index - 0.28,
                    index + 0.28,
                    color="black",
                    lw=1.4,
                )
                wins = int(row.get(f"{metric}_wins", 0))
                ties = int(row.get(f"{metric}_ties", 0))
                losses = int(row["num_split_realizations"]) - wins - ties
                chart_rows.append(
                    {
                        "comparison_id": str(row["comparison_id"]),
                        "minuend": str(row["minuend"]),
                        "subtrahend": str(row["subtrahend"]),
                        "difference_mean": float(
                            row[f"{metric}_difference_mean"]
                        ),
                        "difference_sample_sd_split": float(
                            row[f"{metric}_difference_sample_sd_split"]
                        ),
                        "difference_min": float(row[f"{metric}_difference_min"]),
                        "difference_max": float(row[f"{metric}_difference_max"]),
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "tie_tolerance": float(row.get("tie_tolerance", 1e-12)),
                        "points": [
                            {
                                "split_seed": int(value["split_seed"]),
                                "difference": float(value["difference"]),
                                "x": x_value,
                            }
                            for value, x_value in zip(split_values, x)
                        ],
                    }
                )
                ax.annotate(
                    f"{wins}–{ties}–{losses}",
                    (index, 1.01),
                    xycoords=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            ax.axhline(0.0, color="black", ls=":", lw=0.8)
            ax.set_xticks(
                range(len(rows)),
                [str(row["comparison_id"]) for row in rows],
                rotation=30,
                ha="right",
            )
            ax.tick_params(
                axis="x", labelsize=_model_label_fontsize(plt, chart)
            )
            ax.set_ylabel(f"Paired same-split Δ{metric_label} {operation_label}")
            ax.set_title(
                "wins–ties–losses are descriptive", fontsize=10, loc="left"
            )
            ax.grid(axis="y", alpha=0.3)
            paths.extend(
                _save(
                    fig,
                    plt,
                    output_dir,
                    f"multisplit_paired_{metric_tag}_{operation_stem}",
                    chart,
                    {
                        "metric": metric,
                        "selection_mode": selection_mode,
                        "task_id": task,
                        **operation_payload,
                        "zero_reference": 0.0,
                        "aggregation_order": (
                            "mean_over_training_seeds_within_split_then_"
                            "paired_difference_then_equal_split_summary"
                        ),
                        "comparisons": chart_rows,
                    },
                )
            )

    chart = _enabled(config, "multisplit_heatmap")
    if chart:
        metric = str(chart.get("metric") or "pair_completeness")
        k = int(chart.get("k", 40))
        rows = [
            row for row in _task_rows(artifact, chart)
            if row.get("selection_mode") == "top_k" and int(row.get("k") or -1) == k
        ]
        base_rows = rows
        for suffix, select_canonical in _model_group_chart_modes(config):
            rows = (
                _canonical_filtered_chart_rows(base_rows, config, chart)
                if select_canonical
                else base_rows
            )
            rows_by_model = {str(row["configuration_id"]): row for row in rows}
            model_ids = [value for value in styles if value in rows_by_model]
            split_seeds = [int(value) for value in artifact["expected_split_seeds"]]
            matrix = np.asarray([
                [dict(_split_values(rows_by_model[model_id], metric))[seed] for seed in split_seeds]
                for model_id in model_ids
            ], dtype=float)
            centered = matrix - matrix.mean(axis=1, keepdims=True)
            limit = float(np.max(np.abs(centered))) if centered.size else 1.0
            if limit == 0.0:
                limit = 1.0
            fig, ax = plt.subplots(
                figsize=(THESIS_FIGURE_WIDTH, max(2.8, 0.27 * len(model_ids) + 1.0))
            )
            image = ax.imshow(centered, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_xticks(range(len(split_seeds)), [str(seed) for seed in split_seeds])
            ax.set_yticks(range(len(model_ids)), [styles[value]["label"] for value in model_ids])
            ax.tick_params(
                axis="y", labelsize=_model_label_fontsize(plt, chart)
            )
            ax.set_xlabel("Predefined split seed")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
            colorbar.set_label(f"PC@{k} − model split mean")
            paths.extend(_save(
                fig,
                plt,
                output_dir,
                f"multisplit_heatmap_pc_at_{k}{suffix}",
                chart,
                {
                    "candidate_budget": k,
                    "split_seeds": split_seeds,
                    "color_limits": [-limit, limit],
                    "rows": [
                        {
                            "model_id": model_id,
                            "mean": float(matrix[index].mean()),
                            "values": matrix[index],
                            "centered_values": centered[index],
                        }
                        for index, model_id in enumerate(model_ids)
                    ],
                },
            ))

    chart = _enabled(config, "multisplit_calibration_transfer")
    if chart:
        target = float(chart.get("target_pc", 0.99))
        rows = [
            row for row in _task_rows(artifact, chart)
            if row.get("selection_mode") == "calibrated_threshold"
            and abs(float(row.get("calibration_target_pair_completeness") or -1.0) - target) <= 1e-12
        ]
        base_rows = rows
        attainment_tolerance = float(
            (((config.get("multisplit") or {}).get("aggregation") or {}).get(
                "attainment_tolerance", 1e-12
            ))
        )
        paths.extend(
            _emit_calibration_transfer(
                plt,
                config,
                styles,
                chart,
                base_rows,
                target,
                attainment_tolerance,
                output_dir,
            )
        )

    chart = _enabled(config, "multisplit_calibrated_top_k_transfer")
    if chart:
        style_from = chart.get("style_from")
        if style_from is not None:
            inherited = (config.get("charts") or {}).get(str(style_from))
            if not isinstance(inherited, dict):
                raise ValueError(
                    "multisplit_calibrated_top_k_transfer.style_from must name "
                    "another chart mapping"
                )
            chart = {**inherited, **chart}
        attainment_tolerance = float(
            (((config.get("multisplit") or {}).get("aggregation") or {}).get(
                "attainment_tolerance", 1e-12
            ))
        )
        all_rows = [
            row for row in _task_rows(artifact, chart)
            if row.get("selection_mode") == "calibrated_top_k"
        ]
        for target in [float(value) for value in chart.get("target_pc") or []]:
            rows = [
                row for row in all_rows
                if abs(
                    float(
                        row.get("calibration_target_pair_completeness") or -1.0
                    )
                    - target
                )
                <= 1e-12
            ]
            if not rows:
                warnings.append(
                    "No validation-calibrated top-k rows matched target PC "
                    f"{target:g}"
                )
                continue
            chart_for_target = {
                **chart,
                "_default_pc_axis_label": (
                    rf"$PC^{{test}}$ at validation-selected $k$ "
                    rf"($\rho_{{cal}}={target:g}$)"
                ),
                "_default_pc_axis_description": "",
            }
            fig = _render_calibration_transfer(
                plt,
                rows,
                styles,
                chart_for_target,
                target,
                attainment_tolerance,
            )
            plotted_rows = sorted(
                rows,
                key=lambda row: float(row.get("pair_completeness_mean", 0.0)),
            )
            rng = np.random.default_rng(int(chart.get("jitter_seed", 0)))
            chart_rows: list[dict[str, Any]] = []
            for index, row in enumerate(plotted_rows):
                pc_by_seed = dict(_split_values(row, "pair_completeness"))
                pq_by_seed = dict(_split_values(row, "pair_quality"))
                rr_by_seed = dict(_split_values(row, "reduction_ratio"))
                selected_k_by_seed = dict(_split_values(row, "selected_k"))
                selection_by_seed = dict(_split_values(row, "selection_attained"))
                x = index + rng.uniform(-0.13, 0.13, len(pc_by_seed))
                chart_rows.append(
                    {
                        "model_id": str(row["configuration_id"]),
                        "selected_k_mean": float(row["selected_k_mean"]),
                        "selected_k_sample_sd_split": float(
                            row["selected_k_sample_sd_split"]
                        ),
                        "selected_k_min": int(row["selected_k_min"]),
                        "selected_k_max": int(row["selected_k_max"]),
                        "validation_selection_attainment_count": int(
                            row.get("validation_selection_attainment_count", 0)
                        ),
                        "fallback_split_count": int(
                            row["num_split_realizations"]
                            - row.get("validation_selection_attainment_count", 0)
                        ),
                        "pair_completeness_mean": float(
                            row["pair_completeness_mean"]
                        ),
                        "pair_quality_mean": float(row["pair_quality_mean"]),
                        "reduction_ratio_mean": float(
                            row["reduction_ratio_mean"]
                        ),
                        "target_attainment_count": int(
                            row.get("target_attainment_count", 0)
                        ),
                        "num_split_realizations": int(
                            row["num_split_realizations"]
                        ),
                        "points": [
                            {
                                "split_seed": seed,
                                "selected_k": int(selected_k_by_seed[seed]),
                                "validation_target_attained": bool(
                                    selection_by_seed[seed] >= 0.5
                                ),
                                "used_least_restrictive_fallback": bool(
                                    selection_by_seed[seed] < 0.5
                                ),
                                "pair_completeness": pc_by_seed[seed],
                                "pair_quality": pq_by_seed[seed],
                                "reduction_ratio": rr_by_seed[seed],
                                "x": x_value,
                            }
                            for seed, x_value in zip(pc_by_seed, x)
                        ],
                    }
                )
            target_tag = f"{target:g}".replace(".", "p")
            paths.extend(
                _save(
                    fig,
                    plt,
                    output_dir,
                    f"multisplit_calibrated_top_k_transfer_rho{target_tag}",
                    chart,
                    {
                        "selection_mode": "calibrated_top_k",
                        "calibration_rule": (
                            "smallest_integer_k_attaining_validation_target_pc"
                        ),
                        "calibration_target_pair_completeness": target,
                        "target_attainment_tolerance": attainment_tolerance,
                        "target_attainment_threshold": max(
                            0.0, target - attainment_tolerance
                        ),
                        "pair_completeness_limits": [
                            float(value) for value in fig.axes[0].get_ylim()
                        ],
                        "reduction_ratio_limits": [
                            float(value) for value in fig.axes[1].get_ylim()
                        ],
                        "layout": "vertical_pc_with_rr_bars",
                        **_calibration_display_metadata(fig, chart_for_target),
                        "series": chart_rows,
                    },
                )
            )

    chart = _enabled(config, "multisplit_threshold_top_k_floor_transfer")
    if chart:
        style_from = chart.get("style_from")
        if style_from is not None:
            inherited = (config.get("charts") or {}).get(str(style_from))
            if not isinstance(inherited, dict):
                raise ValueError(
                    "multisplit_threshold_top_k_floor_transfer.style_from must "
                    "name another chart mapping"
                )
            chart = {**inherited, **chart}
        attainment_tolerance = float(
            (((config.get("multisplit") or {}).get("aggregation") or {}).get(
                "attainment_tolerance", 1e-12
            ))
        )
        all_rows = [
            row for row in _task_rows(artifact, chart)
            if row.get("selection_mode")
            == "calibrated_threshold_top_k_floor"
        ]
        for target in [float(value) for value in chart.get("target_pc") or []]:
            for minimum_top_k in [
                int(value) for value in chart.get("minimum_top_k") or []
            ]:
                rows = [
                    row for row in all_rows
                    if abs(
                        float(
                            row.get("calibration_target_pair_completeness")
                            or -1.0
                        )
                        - target
                    )
                    <= 1e-12
                    and int(row.get("minimum_top_k") or -1) == minimum_top_k
                ]
                if not rows:
                    warnings.append(
                        "No calibrated-threshold + top-k-floor rows matched "
                        f"target PC {target:g}, minimum k {minimum_top_k}"
                    )
                    continue
                chart_for_combination = {
                    **chart,
                    "_default_pc_axis_label": (
                        rf"$PC^{{test}}$ at $\rho_{{cal}}={target:g}$"
                        "\n"
                        rf"threshold $\cup$ top-${minimum_top_k} floor"
                    ),
                    "_default_pc_axis_description": "",
                }
                fig = _render_calibration_transfer(
                    plt,
                    rows,
                    styles,
                    chart_for_combination,
                    target,
                    attainment_tolerance,
                )
                plotted_rows = sorted(
                    rows,
                    key=lambda row: float(
                        row.get("pair_completeness_mean", 0.0)
                    ),
                )
                rng = np.random.default_rng(int(chart.get("jitter_seed", 0)))
                chart_rows: list[dict[str, Any]] = []
                for index, row in enumerate(plotted_rows):
                    pc_by_seed = dict(_split_values(row, "pair_completeness"))
                    pq_by_seed = dict(_split_values(row, "pair_quality"))
                    rr_by_seed = dict(_split_values(row, "reduction_ratio"))
                    x = index + rng.uniform(-0.13, 0.13, len(pc_by_seed))
                    chart_rows.append(
                        {
                            "model_id": str(row["configuration_id"]),
                            "pair_completeness_mean": float(
                                row["pair_completeness_mean"]
                            ),
                            "pair_quality_mean": float(
                                row["pair_quality_mean"]
                            ),
                            "reduction_ratio_mean": float(
                                row["reduction_ratio_mean"]
                            ),
                            "target_attainment_count": int(
                                row.get("target_attainment_count", 0)
                            ),
                            "num_split_realizations": int(
                                row["num_split_realizations"]
                            ),
                            "points": [
                                {
                                    "split_seed": seed,
                                    "pair_completeness": pc_by_seed[seed],
                                    "pair_quality": pq_by_seed[seed],
                                    "reduction_ratio": rr_by_seed[seed],
                                    "x": x_value,
                                }
                                for seed, x_value in zip(pc_by_seed, x)
                            ],
                        }
                    )
                target_tag = f"{target:g}".replace(".", "p")
                stem = (
                    "multisplit_threshold_top_k_floor_transfer_"
                    f"rho{target_tag}_k{minimum_top_k}"
                )
                paths.extend(
                    _save(
                        fig,
                        plt,
                        output_dir,
                        stem,
                        chart,
                        {
                            "selection_mode": (
                                "calibrated_threshold_top_k_floor"
                            ),
                            "candidate_rule": "threshold_or_top_k",
                            "calibration_target_pair_completeness": target,
                            "minimum_top_k": minimum_top_k,
                            "target_attainment_tolerance": attainment_tolerance,
                            "target_attainment_threshold": max(
                                0.0, target - attainment_tolerance
                            ),
                            "pair_completeness_limits": [
                                float(value) for value in fig.axes[0].get_ylim()
                            ],
                            "reduction_ratio_limits": [
                                float(value) for value in fig.axes[1].get_ylim()
                            ],
                            "layout": "vertical_pc_with_rr_bars",
                            **_calibration_display_metadata(
                                fig, chart_for_combination
                            ),
                            "series": chart_rows,
                        },
                    )
                )

    chart = _enabled(config, "multisplit_similarity_distributions")
    if chart:
        for suffix, select_canonical in _model_group_chart_modes(config):
            entries = _similarity_distribution_entries(
                artifact,
                config,
                chart,
                select_canonical_models=select_canonical,
            )
            if not entries:
                warnings.append(
                    "No stored similarity histograms matched the multisplit "
                    f"similarity-distribution chart{suffix}"
                )
            else:
                thresholds = _similarity_thresholds(artifact, chart)
                fig = _render_similarity_distributions(plt, entries, styles, thresholds, chart)
                task_tag = _safe_name(str(chart.get("task") or entries[0].get("task_id") or "task"))
                chart_series: list[dict[str, Any]] = []
                for entry in entries:
                    model_id = str(entry["configuration_id"])
                    series = {
                        "model_id": model_id,
                        "bin_edges": entry["bin_edges"],
                        "match_density_mean": entry["match_density_mean"],
                        "non_match_density_mean": entry["non_match_density_mean"],
                    }
                    if bool(chart.get("show_split_range", True)):
                        series.update({
                            "match_density_min": entry["match_density_min"],
                            "match_density_max": entry["match_density_max"],
                            "non_match_density_min": entry["non_match_density_min"],
                            "non_match_density_max": entry["non_match_density_max"],
                        })
                    if model_id in thresholds:
                        series["threshold_mean"] = thresholds[model_id]["mean"]
                        series["threshold_min"] = thresholds[model_id]["min"]
                        series["threshold_max"] = thresholds[model_id]["max"]
                    chart_series.append(series)
                xlim = chart.get("xlim") or [-0.2, 1.0]
                paths.extend(
                    _save(
                        fig,
                        plt,
                        output_dir,
                        f"multisplit_similarity_distributions_{task_tag}{suffix}",
                        chart,
                        {
                            "similarity_limits": [float(xlim[0]), float(xlim[1])],
                            "series": chart_series,
                        },
                    )
                )

    try:
        protocol_chart = parse_protocol_chart(config)
    except CalibrationProtocolBarsError as exc:
        raise ValueError(str(exc)) from exc
    if protocol_chart is not None:
        attainment_tolerance = float(
            (((config.get("multisplit") or {}).get("aggregation") or {}).get(
                "attainment_tolerance", 1e-12
            ))
        )
        targets = chart_targets(protocol_chart)
        if not targets:
            warnings.append("calibration_protocol_bars has no calibration target")
        for protocol in protocol_chart["protocols"]:
            mode = SELECTION_MODE[protocol]
            for target in targets:
                rows = [
                    row for row in _task_rows(artifact, protocol_chart)
                    if row.get("selection_mode") == mode
                    and abs(
                        float(row.get("calibration_target_pair_completeness") or -1.0)
                        - target
                    )
                    <= 1e-12
                ]
                if not rows:
                    warnings.append(
                        "No zero-margin "
                        f"{protocol} rows matched target PC {target:g}"
                    )
                    continue
                paths.extend(
                    _emit_calibration_transfer(
                        plt,
                        config,
                        styles,
                        protocol_chart,
                        rows,
                        target,
                        attainment_tolerance,
                        protocol_directory(output_dir, protocol),
                    )
                )

    require_chart_sidecars(paths)

    if paths:
        n = len(artifact.get("expected_split_seeds") or [])
        dataset = str(artifact.get("dataset_id") or "dataset")
        tasks = ", ".join(sorted({str(row.get("task_id")) for row in artifact.get("summary_rows") or []}))
        frozen = bool(((config.get("multisplit") or {}).get("protocol") or {}).get("frozen_before_evaluation"))
        if _model_group_members(config):
            grouping_caption = (
                "Standard direct model-series charts are emitted once with all configured models and once "
                f"with the `{MODEL_GROUP_CANONICAL_SUFFIX}` filename suffix. Calibrated top-k charts "
                "are emitted once per target and threshold-plus-floor charts once per configured "
                "(target, floor) combination. In each suffixed "
                "version, each configured model group shows its predeclared canonical "
                "member plus any chart-specific `canonical_additional_models`. Canonical "
                "membership and explicit additions are fixed in the analysis config and do not "
                "depend on any displayed validation or test result.\n\n"
            )
        else:
            grouping_caption = "No model groups were configured; each chart is emitted once.\n\n"
        calibration_chart = (
            (config.get("charts") or {}).get("multisplit_calibration_transfer")
            or {}
        )
        attainment_tolerance = float(
            (((config.get("multisplit") or {}).get("aggregation") or {}).get(
                "attainment_tolerance", 1e-12
            ))
        )
        include_attainment_score = bool(
            calibration_chart.get("include_attainment_score", True)
        )
        include_specific_result_crosses = bool(
            calibration_chart.get("include_specific_result_crosses", True)
        )
        if not include_specific_result_crosses:
            calibration_point_description = (
                "Split-level result markers are omitted; horizontal black ticks "
                "still show equal-split mean test PC."
            )
        elif not include_attainment_score:
            calibration_point_description = (
                "Points are split-level test PC values without attainment encoding."
            )
        elif str(calibration_chart.get("pc_marker") or "") == "x":
            calibration_point_description = (
                "Crosses are split-level test PC values."
            )
        else:
            calibration_point_description = (
                "Filled PC points satisfy the configured attainment rule and "
                "open points do not."
            )
        if include_attainment_score:
            attainment_description = (
                "Top annotations report the number satisfying "
                "$PC_{test} \\geq \\rho_{cal} - \\epsilon$ out of all splits, "
                f"with configured shortfall tolerance "
                f"$\\epsilon={attainment_tolerance:g}$. The dotted line marks "
                "the nominal calibration target. "
            )
            if bool(calibration_chart.get("show_attainment_region", False)):
                attainment_description += (
                    "The colored region spans from the dynamically computed "
                    "attainment threshold to the top of the PC axis. "
                )
        else:
            attainment_description = (
                "Attainment scores, labels, target reference, region, and "
                "attained/missed marker encoding are omitted. "
            )
        captions = (
            "# Multisplit figure captions\n\n"
            f"All figures use {n} predefined overlapping class-disjoint validation–test split "
            f"realizations of {dataset} for task(s) {tasks}; models are {'frozen' if frozen else 'selected independently per training repeat'}. "
            "Splits are weighted equally after averaging training repeats within split. Sample SD uses ddof=1. "
            "Results are descriptive across the evaluated splits; observed ranges are not confidence intervals.\n\n"
            f"{grouping_caption}"
            "- **PC@k band:** line is equal-weighted split mean; band is the literal observed min–max split range, not a CI.\n"
            "- **Strip plot:** points are split-level values; jitter is cosmetic and deterministically seeded; horizontal ticks are means.\n"
            "- **Paired difference:** differences are joined within identical split hashes; W–T–L is descriptive and is not a significance test.\n"
            "- **Split heatmap:** each cell is the split value minus that model's mean; vertical structure describes shared split difficulty.\n"
            "- **Calibration transfer:** each operating point is selected independently on that split/repeat's validation partition, frozen, and applied unchanged to test. Test PC is shown on the left vertical axis; horizontal black ticks are equal-split means. "
            f"{calibration_point_description} {attainment_description}"
            "Translucent bars use the right vertical axis and show mean test RR. Test attainment is an outcome, not a selection requirement.\n"
            "- **Calibrated top-k transfer:** one chart is emitted per calibration target. Validation selects the smallest predeclared integer k attaining target PC; k is frozen and applied unchanged to test. Unattained validation targets use an explicitly marked least-restrictive fallback so no split disappears. Sidecars report selected-k split statistics together with test PC, PQ, and RR.\n"
            "- **Threshold-plus-top-k floor transfer:** one chart is emitted per configured calibration target and minimum floor. The pure validation-calibrated threshold is frozen before taking its union with the held-out per-query top-k floor. PC points and mean RR bars use the same descriptive split aggregation as calibration transfer.\n"
            "- **Similarity distributions:** match and non-match histograms are normalized separately to unit-area densities in every source evaluation. Training repeats are averaged within split, then splits are weighted equally. Filled steps are split means; light envelopes are observed bin-wise split ranges, not confidence intervals. The dotted threshold is the equal-split mean validation-selected threshold and its gray band is the observed split range.\n"
        )
        caption_path = output_dir / "multisplit_figure_captions.md"
        atomic_write_text(caption_path, captions)
        paths.append(caption_path)
    return paths, warnings
