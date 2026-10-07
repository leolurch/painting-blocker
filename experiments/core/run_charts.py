"""Default chart bundle for one completed run directory.

Renders, best-effort and stage-isolated, everything a validation run should
ship with: the reference-style blocking LaTeX tables, the blocking figures
(Figures 1-5), the ranked PC bar charts, and the PQ-PC curve. A failing stage
(e.g. matplotlib missing) is recorded in ``charts_warning.json`` instead of
failing the run; the remaining stages still execute.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .run_figures import (
    DEFAULT_FIGURE_FORMATS,
    load_run_context,
    write_run_figures,
)
from .plot_ranked_pc_bars import _load_model_styles, _safe_name, load_rows, plot_ranked_bars, write_summary_csv


def write_run_charts(
    run_dir: Path | str,
    task_id: str | None = None,
    formats: tuple[str, ...] = DEFAULT_FIGURE_FORMATS,
) -> dict[str, Any]:
    """Render tables + all charts for a run. Never raises for a single failing stage."""
    run_path = Path(run_dir).expanduser()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    paths: dict[str, str] = {}
    warnings: list[dict[str, str]] = []

    def _stage(name: str, fn) -> None:
        try:
            result = fn()
        except Exception as exc:  # stage isolation: record and continue
            warnings.append({"stage": name, "warning": f"{type(exc).__name__}: {exc}"})
        else:
            for key, path in (result or {}).items():
                paths[f"{name}:{key}"] = str(path)

    _stage("blocking_tables", lambda: _blocking_tables_stage(run_path, task_id))
    _stage("figures", lambda: write_run_figures(run_path, task_id=task_id, formats=formats))
    _stage("ranked_pc_bars", lambda: _ranked_bars_stage(run_path, task_id))
    _stage("pq_pc", lambda: _pq_pc_stage(run_path, task_id, formats))

    if warnings:
        atomic_write_json(run_path / "charts_warning.json", {"warnings": warnings})
    return {"paths": paths, "warnings": warnings}


def _blocking_tables_stage(run_path: Path, task_id: str | None) -> dict[str, Path]:
    from .blocking_tables import write_run_blocking_tables

    return write_run_blocking_tables(run_path, task_id=task_id)


def _ranked_bars_stage(run_path: Path, task_id: str | None) -> dict[str, Path]:
    ks = _available_ks(run_path / "aggregate_metrics.csv")
    if not ks:
        return {}
    rows, _detected = load_rows(run_path, task_id, ks)
    labels, colors = _load_model_styles(run_path)
    out_dir = run_path / "bar_charts"
    summary = write_summary_csv(rows, labels, out_dir / "ranked_pc_summary.csv")
    charts = plot_ranked_bars(rows, labels, out_dir, None, colors)
    return {"summary": summary, **{path.name: path for path in charts}}


def _pq_pc_stage(run_path: Path, task_id: str | None, formats: tuple[str, ...]) -> dict[str, Path]:
    from .plot_pq_pc import discover_curve_files, plot_pq_pc

    curve_files = discover_curve_files([run_path])
    if not curve_files:
        return {}
    context = load_run_context(run_path)
    task_ids = [task_id] if task_id is not None else context.task_ids
    fmt = formats[0] if formats else "pdf"
    out_dir = run_path / "figures"
    written: dict[str, Path] = {}
    for tid in task_ids:
        output = out_dir / f"pq_pc_{_safe_name(tid)}.{fmt}"
        plot_pq_pc(curve_files, tid, output)
        written[output.name] = output
    return written


def _available_ks(csv_path: Path) -> set[int]:
    if not csv_path.is_file():
        return set()
    ks: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("selection_mode") != "top_k":
                continue
            try:
                ks.add(int(row.get("k") or ""))
            except ValueError:
                continue
    return ks
