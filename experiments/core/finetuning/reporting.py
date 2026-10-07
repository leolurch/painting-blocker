"""CSV and SVG reporting for finetuning validation histories."""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Any


def write_training_reports(
    history: list[dict[str, Any]],
    output_dir: Path,
    *,
    references: list[dict[str, Any]] | None = None,
    best_epoch: int | None = None,
    last_epoch: int | None = None,
    best_validation_set: str | None = None,
    checkpoint_selection: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write per-epoch validation metrics CSV and a loss/PC@k SVG chart.

    ``references`` holds frozen-backbone reference entries (see
    ``reference_evaluation.evaluate_references``). They are rendered as a
    ``frozen_backbone`` stage that sits before epoch 1 and never participates in
    checkpoint selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    references = list(references or [])
    last = int(last_epoch) if last_epoch is not None else _last_epoch(history)
    csv_path = output_dir / "validation_metrics.csv"
    svg_path = output_dir / "training_metrics.svg"
    _write_validation_metrics_csv(
        csv_path,
        history,
        references=references,
        best_epoch=best_epoch,
        last_epoch=last,
        best_validation_set=best_validation_set,
        checkpoint_selection=checkpoint_selection,
    )
    _write_training_metrics_svg(
        svg_path,
        history,
        references=references,
        best_epoch=best_epoch,
        last_epoch=last,
        best_validation_set=best_validation_set,
        checkpoint_selection=checkpoint_selection,
    )
    return {"validation_metrics_csv": str(csv_path), "training_metrics_svg": str(svg_path)}


def _last_epoch(history: list[dict[str, Any]]) -> int | None:
    epochs = [int(row["epoch"]) for row in history if row.get("epoch") is not None]
    return max(epochs) if epochs else None


def _top_k_values(rows: list[dict[str, Any]]) -> list[int]:
    values: set[int] = set()
    for row in rows:
        for _name, _is_primary, metrics in _validation_sets(row):
            for metric in _recall_rows(metrics):
                if metric.get("k") is not None:
                    values.add(int(metric["k"]))
    return sorted(values)


def _validation_sets(row: dict[str, Any]) -> list[tuple[str, bool, dict[str, Any]]]:
    validation = dict(row.get("validation") or {})
    primary = str(validation.get("primary") or "")
    sets = validation.get("sets") or {}
    if not isinstance(sets, dict) or not sets:
        return []
    return [(str(name), str(name) == primary, dict(metrics)) for name, metrics in sets.items() if isinstance(metrics, dict)]


def _recall_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("recall_at_k") or metrics.get("metrics") or []
    return [dict(item) for item in rows if isinstance(item, dict)]


def _metric_by_k(metrics: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(metric["k"]): metric for metric in _recall_rows(metrics) if metric.get("k") is not None}


def _calibrated(metrics: dict[str, Any]) -> dict[str, Any]:
    return dict(metrics.get("calibrated_threshold") or {})


def _metrics_csv_row(
    *,
    stage: str,
    reference_id: str,
    model_id: str,
    epoch: Any,
    validation_name: str,
    is_primary: bool,
    is_selection_set: bool,
    selection_metric: str,
    train_loss: str,
    markers: list[str],
    metrics: dict[str, Any],
    top_k: list[int],
) -> dict[str, Any]:
    calibrated = _calibrated(metrics)
    csv_row: dict[str, Any] = {
        "stage": stage,
        "reference_id": reference_id,
        "model_id": model_id,
        "epoch": epoch,
        "validation_set": validation_name,
        "is_primary_validation": "true" if is_primary else "false",
        "is_checkpoint_selection_validation": "true" if is_selection_set else "false",
        "checkpoint_selection_metric": selection_metric,
        "train_loss": train_loss,
        "checkpoint_marker": ";".join(markers),
        "is_best_checkpoint": "true" if "best" in markers else "false",
        "is_last_checkpoint": "true" if "last" in markers else "false",
        "threshold": _csv_float(calibrated.get("threshold")),
        "target_pair_completeness": _csv_float(calibrated.get("target_pair_completeness")),
        "calibrated_pair_completeness": _csv_float(calibrated.get("pair_completeness")),
        "calibrated_pair_quality": _csv_float(calibrated.get("pair_quality")),
        "calibrated_reduction_ratio": _csv_float(calibrated.get("reduction_ratio")),
        "calibrated_query_coverage": _csv_float(calibrated.get("query_coverage")),
        "calibrated_candidate_pairs": calibrated.get("candidate_pairs", ""),
        "calibrated_possible_pairs": calibrated.get("possible_pairs", ""),
    }
    by_k = _metric_by_k(metrics)
    for k in top_k:
        metric = by_k.get(k, {})
        csv_row[f"pc_at_{k}"] = _csv_float(metric.get("pair_completeness"))
        csv_row[f"pq_at_{k}"] = _csv_float(metric.get("pair_quality"))
        csv_row[f"qc_at_{k}"] = _csv_float(metric.get("query_coverage"))
    return csv_row


def _write_validation_metrics_csv(
    path: Path,
    history: list[dict[str, Any]],
    *,
    references: list[dict[str, Any]] | None = None,
    best_epoch: int | None,
    last_epoch: int | None,
    best_validation_set: str | None,
    checkpoint_selection: dict[str, Any] | None,
) -> None:
    references = list(references or [])
    top_k = _top_k_values(list(history) + references)
    selection_metric = _checkpoint_selection_metric_label(checkpoint_selection)
    fieldnames = [
        "stage",
        "reference_id",
        "model_id",
        "epoch",
        "validation_set",
        "is_primary_validation",
        "is_checkpoint_selection_validation",
        "checkpoint_selection_metric",
        "train_loss",
        "checkpoint_marker",
        "is_best_checkpoint",
        "is_last_checkpoint",
        "threshold",
        "target_pair_completeness",
        "calibrated_pair_completeness",
        "calibrated_pair_quality",
        "calibrated_reduction_ratio",
        "calibrated_query_coverage",
        "calibrated_candidate_pairs",
        "calibrated_possible_pairs",
    ]
    for k in top_k:
        fieldnames.extend([f"pc_at_{k}", f"pq_at_{k}", f"qc_at_{k}"])

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        # Frozen references sit before epoch 1 and never carry an epoch number.
        for reference in references:
            reference_id = str(reference.get("id") or "frozen_backbone")
            model_id = str(reference.get("model_id") or "")
            for validation_name, is_primary, validation_metrics in _validation_sets(reference):
                writer.writerow(
                    _metrics_csv_row(
                        stage="frozen_backbone",
                        reference_id=reference_id,
                        model_id=model_id,
                        epoch="",
                        validation_name=validation_name,
                        is_primary=is_primary,
                        is_selection_set=False,
                        selection_metric=selection_metric,
                        train_loss="",
                        markers=["frozen_backbone"],
                        metrics=validation_metrics,
                        top_k=top_k,
                    )
                )
        for row in history:
            epoch = int(row["epoch"])
            for validation_name, is_primary, validation_metrics in _validation_sets(row):
                markers: list[str] = []
                selected_for_best = best_validation_set is None or validation_name == best_validation_set
                if best_epoch is not None and epoch == int(best_epoch) and selected_for_best:
                    markers.append("best")
                if last_epoch is not None and epoch == int(last_epoch):
                    markers.append("last")
                writer.writerow(
                    _metrics_csv_row(
                        stage="epoch",
                        reference_id="",
                        model_id="",
                        epoch=epoch,
                        validation_name=validation_name,
                        is_primary=is_primary,
                        is_selection_set=validation_name == best_validation_set,
                        selection_metric=selection_metric,
                        train_loss=_csv_float(row.get("train_loss")),
                        markers=markers,
                        metrics=validation_metrics,
                        top_k=top_k,
                    )
                )
    tmp.replace(path)


def _validation_set_names(history: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in history:
        for name, _is_primary, _metrics in _validation_sets(row):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _validation_metrics_for(row: dict[str, Any], name: str) -> dict[str, Any]:
    for set_name, _is_primary, metrics in _validation_sets(row):
        if set_name == name:
            return metrics
    return {}


def _line_styles(names: list[str]) -> dict[str, str]:
    patterns = ["", "3 4", "7 4", "9 3 2 3", "2 3", "12 4 3 4"]
    return {name: patterns[idx % len(patterns)] for idx, name in enumerate(names)}


def _checkpoint_selection_metric_label(selection: dict[str, Any] | None) -> str:
    if not selection:
        return ""
    metric = str(selection.get("metric") or "")
    if metric == "pc_at_k":
        return f"{selection.get('validation_set', '')} PC@{selection.get('k', 10)}"
    if metric == "mean_precision_at_k":
        k_values = ",".join(str(k) for k in selection.get("k", []))
        return (
            f"{selection.get('validation_set', '')} mean precision@k={k_values} "
            f"tie<={float(selection.get('precision_tie', 0.01)):.10g} then mean RR, earliest epoch"
        )
    if metric == "mean_pair_completeness_at_k":
        k_values = ",".join(str(k) for k in selection.get("k", []))
        return (
            f"{selection.get('validation_set', '')} mean PC@k={k_values} "
            f"tie<={float(selection.get('pc_tie', 0.005)):.10g}, "
            f"RR@PC>={float(selection.get('target_pc', 0.95)):.10g} "
            f"tie<={float(selection.get('rr_tie', 0.01)):.10g}, "
            "then lower LoRA lr, lower head lr, earliest epoch"
        )
    if metric == "calibrated_threshold":
        return f"{selection.get('validation_set', '')} calibrated threshold"
    if metric == "reduction_ratio_at_target_pc":
        target = float(selection.get("target_pc", 0.99))
        return f"{selection.get('validation_set', '')} RR@PC>={target:.10g}"
    if metric == "calibration_transfer":
        target = float(selection.get("target_pc", 0.99))
        minimum = float(selection.get("minimum_selection_pc", 0.98))
        return (
            f"{selection.get('calibration_validation_set', '')} PC>={target:.10g} -> "
            f"{selection.get('selection_validation_set', '')} PC>{minimum:.10g}, RR"
        )
    if metric == "weighted_mean_pc_at_k_and_transfer":
        k_values = ",".join(str(k) for k in selection.get("fixed_k", []))
        return (
            f"weighted mean PC@k={k_values} + "
            f"PC/RR@rho={float(selection.get('target_pc', 0.99)):.10g} transfer"
        )
    return metric


def _csv_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.10g}"


def _reference_pc(references: list[dict[str, Any]], name: str, k: int) -> list[tuple[str, float]]:
    """Return ``(label, PC@k)`` for each reference that evaluated ``name``."""
    out: list[tuple[str, float]] = []
    for reference in references:
        metrics = _validation_metrics_for(reference, name)
        by_k = _metric_by_k(metrics)
        if k in by_k:
            value = _as_float(by_k[k].get("pair_completeness"))
            if value is not None:
                out.append((str(reference.get("label") or reference.get("id") or "frozen"), value))
    return out


def _build_x_ticks(
    epoch_range: tuple[float, float], epochs: list[int], frozen_x: float | None
) -> list[tuple[float, str]]:
    ticks: list[tuple[float, str]] = []
    if frozen_x is not None:
        ticks.append((float(frozen_x), "Frozen\nbackbone"))
    ints = sorted({int(e) for e in epochs})
    if ints:
        if len(ints) > 12:
            step = math.ceil(len(ints) / 10)
            sampled = ints[::step]
            if ints[-1] not in sampled:
                sampled.append(ints[-1])
            ints = sampled
        ticks.extend((float(e), str(e)) for e in ints)
    else:
        low, high = epoch_range
        for idx in range(5):
            frac = idx / 4
            value = low + frac * (high - low)
            ticks.append((value, f"{value:.3g}"))
    return ticks


def _write_training_metrics_svg(
    path: Path,
    history: list[dict[str, Any]],
    *,
    references: list[dict[str, Any]] | None = None,
    best_epoch: int | None,
    last_epoch: int | None,
    best_validation_set: str | None,
    checkpoint_selection: dict[str, Any] | None,
) -> None:
    references = list(references or [])
    has_references = bool(references)
    width, height = 1120, 500
    margin = {"left": 64, "right": 24, "top": 48, "bottom": 118}
    gap = 80
    panel_w = (width - margin["left"] - margin["right"] - gap) / 2
    panel_h = height - margin["top"] - margin["bottom"]
    loss_panel = (margin["left"], margin["top"], panel_w, panel_h)
    pc_panel = (margin["left"] + panel_w + gap, margin["top"], panel_w, panel_h)

    epochs = [int(row["epoch"]) for row in history if row.get("epoch") is not None]
    losses = [_as_float(row.get("train_loss")) for row in history]
    top_k = _top_k_values(list(history) + references)
    validation_names = _validation_set_names(list(history) + references)
    pc_series = {
        (name, k): [
            _as_float(_metric_by_k(_validation_metrics_for(row, name)).get(k, {}).get("pair_completeness"))
            for row in history
        ]
        for name in validation_names
        for k in top_k
    }
    min_epoch = min(epochs) if epochs else 1
    max_epoch = max(epochs) if epochs else 1
    frozen_x = float(min_epoch - 1) if has_references else None
    if min_epoch == max_epoch and not has_references:
        min_epoch -= 0.5
        max_epoch += 0.5
    domain_lo = frozen_x if frozen_x is not None else min_epoch
    epoch_range = (domain_lo, float(max_epoch))
    x_ticks = _build_x_ticks(epoch_range, epochs, frozen_x)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]
    selection_label = _checkpoint_selection_metric_label(checkpoint_selection)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Finetuning loss and PC at k by epoch, with frozen backbone reference">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#111827}.title{font-size:16px;font-weight:700}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2.2}.connector{fill:none;stroke-width:2;stroke-dasharray:4 3}.marker{stroke-dasharray:5 4;stroke-width:1.5}.legend{font-size:11px}</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        '<text x="64" y="26" class="title">Finetuning validation metrics by epoch</text>',
    ]
    if selection_label:
        parts.append(
            f'<text x="430" y="26">checkpoint selection: {html.escape(selection_label)}</text>'
        )
    parts.extend(_panel_axes(loss_panel, "train loss", epoch_range=epoch_range, y_range=_range_for(losses), x_ticks=x_ticks))
    parts.extend(_panel_axes(pc_panel, "PC@k", epoch_range=epoch_range, y_range=(0.0, 1.0), x_ticks=x_ticks))

    loss_range = _range_for(losses)
    loss_points = _polyline_points(epochs, losses, loss_panel, epoch_range, loss_range)
    if loss_points:
        parts.append(f'<polyline class="line" points="{loss_points}" stroke="#111827"/>')
        parts.append(_legend_item(loss_panel[0] + 14, loss_panel[1] + 18, "#111827", "loss"))

    k_colors = {k: colors[idx % len(colors)] for idx, k in enumerate(top_k)}
    line_styles = _line_styles(validation_names)
    legend_x = pc_panel[0] + 14
    legend_y = pc_panel[1] + pc_panel[3] + 42
    legend_idx = 0
    for name in validation_names:
        dash = line_styles[name]
        for k in top_k:
            color = k_colors[k]
            series = pc_series[(name, k)]
            points = _polyline_points(epochs, series, pc_panel, epoch_range, (0.0, 1.0))
            if points:
                dash_attr = f' stroke-dasharray="{html.escape(dash)}"' if dash else ""
                parts.append(f'<polyline class="line" points="{points}" stroke="{color}"{dash_attr}/>')
                parts.append(_legend_item(legend_x + (legend_idx % 3) * 150, legend_y + (legend_idx // 3) * 18, color, f"{name} PC@{k}", dash=dash))
                legend_idx += 1
            if has_references and frozen_x is not None:
                first_epoch = next((e for e, v in zip(epochs, series) if v is not None), None)
                first_value = next((v for v in series if v is not None), None)
                fx = _scale_x(frozen_x, pc_panel, epoch_range)
                for _label, frozen_value in _reference_pc(references, name, k):
                    fy = _scale_y(frozen_value, pc_panel, (0.0, 1.0))
                    if first_epoch is not None and first_value is not None:
                        ex = _scale_x(first_epoch, pc_panel, epoch_range)
                        ey = _scale_y(first_value, pc_panel, (0.0, 1.0))
                        parts.append(f'<polyline class="connector" points="{fx:.2f},{fy:.2f} {ex:.2f},{ey:.2f}" stroke="{color}"/>')
                    parts.append(f'<circle cx="{fx:.2f}" cy="{fy:.2f}" r="4.5" fill="{color}" stroke="#111827" stroke-width="1.2"/>')
    if has_references:
        marker_x = legend_x + (legend_idx % 3) * 150
        marker_y = legend_y + (legend_idx // 3) * 18
        parts.append(
            f'<g><circle cx="{marker_x + 6:.2f}" cy="{marker_y:.2f}" r="4.5" fill="#111827" stroke="#111827" stroke-width="1.2"/>'
            f'<text class="legend" x="{marker_x + 18:.2f}" y="{marker_y + 4:.2f}">Frozen backbone (before epoch 1)</text></g>'
        )

    best_label = f"best {best_validation_set}" if best_validation_set else "best"
    for label, epoch, color in ((best_label, best_epoch, "#d97706"), ("last", last_epoch, "#6b7280")):
        if epoch is None:
            continue
        for panel in (loss_panel, pc_panel):
            x = _scale_x(int(epoch), panel, epoch_range)
            y, h = panel[1], panel[3]
            parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{y + h:.2f}" stroke="{color}" class="marker"/>')
            parts.append(f'<text x="{x + 4:.2f}" y="{y + h - 6:.2f}" fill="{color}">{html.escape(label)} e{int(epoch)}</text>')

    parts.append("</svg>")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(parts) + "\n", encoding="utf-8")
    tmp.replace(path)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _range_for(values: list[float | None]) -> tuple[float, float]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return (0.0, 1.0)
    low, high = min(finite), max(finite)
    if low == high:
        pad = abs(low) * 0.1 or 1.0
        return (low - pad, high + pad)
    pad = (high - low) * 0.08
    return (low - pad, high + pad)


def _panel_axes(
    panel: tuple[float, float, float, float],
    title: str,
    *,
    epoch_range: tuple[float, float],
    y_range: tuple[float, float],
    x_ticks: list[tuple[float, str]],
) -> list[str]:
    x, y, w, h = panel
    parts = [
        f'<text x="{x:.2f}" y="{y - 14:.2f}" class="title">{html.escape(title)}</text>',
        f'<line x1="{x:.2f}" y1="{y + h:.2f}" x2="{x + w:.2f}" y2="{y + h:.2f}" class="axis"/>',
        f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{y + h:.2f}" class="axis"/>',
    ]
    for idx in range(6):
        frac = idx / 5
        yy = y + h - frac * h
        value = y_range[0] + frac * (y_range[1] - y_range[0])
        parts.append(f'<line x1="{x:.2f}" y1="{yy:.2f}" x2="{x + w:.2f}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text x="{x - 8:.2f}" y="{yy + 4:.2f}" text-anchor="end">{value:.3g}</text>')
    for value, label in x_ticks:
        xx = _scale_x(value, panel, epoch_range)
        for line_idx, line in enumerate(str(label).split("\n")):
            parts.append(
                f'<text x="{xx:.2f}" y="{y + h + 22 + line_idx * 13:.2f}" text-anchor="middle">{html.escape(line)}</text>'
            )
    parts.append(f'<text x="{x + w / 2:.2f}" y="{y + h + 58:.2f}" text-anchor="middle">epoch</text>')
    return parts


def _scale_x(epoch: float, panel: tuple[float, float, float, float], epoch_range: tuple[float, float]) -> float:
    x, _y, w, _h = panel
    low, high = epoch_range
    return x + (float(epoch) - low) / max(1e-12, high - low) * w


def _scale_y(value: float, panel: tuple[float, float, float, float], y_range: tuple[float, float]) -> float:
    _x, y, _w, h = panel
    low, high = y_range
    return y + h - (float(value) - low) / max(1e-12, high - low) * h


def _polyline_points(
    epochs: list[int],
    values: list[float | None],
    panel: tuple[float, float, float, float],
    epoch_range: tuple[float, float],
    y_range: tuple[float, float],
) -> str:
    points = []
    for epoch, value in zip(epochs, values):
        if value is None:
            continue
        points.append(f"{_scale_x(epoch, panel, epoch_range):.2f},{_scale_y(value, panel, y_range):.2f}")
    return " ".join(points)


def _legend_item(x: float, y: float, color: str, label: str, *, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{html.escape(dash)}"' if dash else ""
    return (
        f'<g><line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 18:.2f}" y2="{y:.2f}" '
        f'stroke="{color}" stroke-width="2.4"{dash_attr}/><text class="legend" x="{x + 24:.2f}" y="{y + 4:.2f}">{html.escape(label)}</text></g>'
    )
