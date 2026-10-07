"""Render LaTeX summary tables from blocking experiment eval artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .aggregate_runs import aggregate_paths
from .latex_table_output import write_latex_table_with_values
from .latex_table_rendering import METRICS, latex_escape, render_metric_table, target_label


def write_run_latex_tables(
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    task_id: str | None = None,
) -> dict[str, Path]:
    """Write fixed-k and target-PC LaTeX tables for one completed run directory."""
    run_path = Path(run_dir).expanduser()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    out_dir = Path(output_dir).expanduser() if output_dir is not None else run_path / "latex_tables"
    rows = aggregate_paths([run_path])
    run_meta = _load_run_json(run_path)
    model_ids = _ordered_models(run_meta, rows)
    model_labels = _model_labels(run_path, model_ids, run_meta, rows)
    fixed_k = render_fixed_k_table(rows, run_meta, model_ids, model_labels, task_id)
    calibrated = render_calibrated_threshold_table(rows, run_meta, model_ids, model_labels, task_id)
    contents = {"fixed_k": fixed_k}
    if calibrated:
        contents["calibrated_threshold"] = calibrated
        combined = f"{fixed_k}\n\n{calibrated}"
    else:
        combined = fixed_k
    contents["combined"] = combined
    paths = {
        "fixed_k": out_dir / "fixed_k.tex",
        "combined": out_dir / "tables.tex",
    }
    if calibrated:
        paths["calibrated_threshold"] = out_dir / "calibrated_threshold.tex"
    value_paths: dict[str, Path] = {}
    for key, path in list(paths.items()):
        written = write_latex_table_with_values(path, contents[key])
        for index, values_path in enumerate(written[1:], start=1):
            suffix = "values" if len(written) == 2 else f"values_{index}"
            value_paths[f"{key}_{suffix}"] = values_path
    paths.update(value_paths)
    return paths


def render_fixed_k_table(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
    model_ids: list[str],
    model_labels: dict[str, str],
    task_id: str | None = None,
) -> str:
    filtered = _filter_rows(rows, "top_k", task_id)
    groups = [(key, f"$k={key}$") for key in sorted({int(row["k"]) for row in filtered if row.get("k") is not None})]
    context = _table_context(rows, filtered, model_ids, model_labels, task_id)
    dataset = latex_escape(_dataset_label(run_meta))
    caption = [
        f"Baseline retrieval performance for fixed retrieval windows on {dataset}.",
        "For fixed $k$, RR is identical within each task across models because each query",
        "contributes the same number of candidate pairs. Bold values mark the best result in each column.",
    ]
    return render_metric_table(
        caption,
        f"tab:{_label_prefix(run_meta, task_id)}-fixed-k",
        groups,
        context["row_specs"],
        _metric_values(filtered, "k", context["include_task"]),
        "fixed_k",
    )


def render_calibrated_threshold_table(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
    model_ids: list[str],
    model_labels: dict[str, str],
    task_id: str | None = None,
) -> str:
    """Render the validation-calibrated threshold table.

    Thresholds are chosen per model on the calibration (validation) subset to hit
    the stated calibration PC target, then applied unchanged to the held-out test
    subset. The reported RR/PC/PQ are the achieved test-set results, grouped by the
    requested calibration target (never relabeled as the achieved test PC).
    Returns an empty string when a run has no calibrated results (e.g. legacy runs).
    """
    filtered = [row for row in rows if row.get("selection_mode") == "calibrated_threshold"]
    if task_id is not None:
        filtered = [row for row in filtered if str(row.get("task_id")) == task_id]
    if not filtered:
        return ""
    groups = [
        (key, target_label(key))
        for key in sorted(
            {
                float(row["calibration_target_pair_completeness"])
                for row in filtered
                if row.get("calibration_target_pair_completeness") is not None
            }
        )
    ]
    context = _table_context(rows, filtered, model_ids, model_labels, task_id)
    dataset = latex_escape(_dataset_label(run_meta))
    caption = [
        "Similarity thresholds were selected independently for each model on the validation subset",
        f"to achieve the stated calibration PC target and then applied unchanged to the held-out test subset on {dataset}.",
        "Reported PC, PQ, and RR are test-set results. Bold values mark the best result in each column.",
    ]
    return render_metric_table(
        caption,
        f"tab:{_label_prefix(run_meta, task_id)}-calibrated-threshold",
        groups,
        context["row_specs"],
        _metric_values(filtered, "calibration_target_pair_completeness", context["include_task"]),
        "target_pc",
    )


def _filter_rows(rows: list[dict[str, Any]], selection_mode: str, task_id: str | None) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row.get("selection_mode") == selection_mode]
    if task_id is None:
        return filtered
    selected = [row for row in filtered if str(row.get("task_id")) == task_id]
    if not selected:
        raise ValueError(f"No {selection_mode} rows found for task_id={task_id!r}")
    return selected


def _table_context(
    all_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    model_ids: list[str],
    model_labels: dict[str, str],
    task_id: str | None,
) -> dict[str, Any]:
    task_source = table_rows or all_rows
    task_ids = [task_id] if task_id is not None else _ordered_unique(
        str(row.get("task_id")) for row in task_source if row.get("task_id") is not None
    )
    include_task = len(task_ids) > 1
    row_specs: list[tuple[tuple[str, str | None], str]] = []
    for model_id in model_ids:
        if include_task:
            for row_task in task_ids:
                label = f"{model_labels.get(model_id, model_id)} ({row_task})"
                row_specs.append(((model_id, row_task), label))
        else:
            row_specs.append(((model_id, None), model_labels.get(model_id, model_id)))
    return {"include_task": include_task, "row_specs": row_specs}


def _metric_values(
    rows: list[dict[str, Any]],
    group_field: str,
    include_task: bool,
) -> dict[tuple[tuple[str, str | None], float | int, str], float]:
    values: dict[tuple[tuple[str, str | None], float | int, str], float] = {}
    for row in rows:
        row_key = (str(row.get("model_id")), str(row.get("task_id")) if include_task else None)
        group_key = int(row[group_field]) if group_field == "k" else float(row[group_field])
        for metric_key, _label in METRICS:
            value = row.get(metric_key)
            if value is not None:
                values[(row_key, group_key, metric_key)] = float(value)
    return values


def _load_run_json(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        return {"run_id": run_dir.name}
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_models(run_meta: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    configured = [str(row.get("model_id")) for row in run_meta.get("model_runs", []) if row.get("model_id")]
    configured_set = set(configured)
    discovered = _ordered_unique(str(row.get("model_id")) for row in rows if row.get("model_id"))
    return configured + [model_id for model_id in discovered if model_id not in configured_set]


def _model_labels(
    run_dir: Path,
    model_ids: Iterable[str],
    run_meta: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    labels = {model_id: model_id for model_id in model_ids}
    revisions = _model_revisions(run_meta, rows)
    for entry in (run_meta or {}).get("model_runs", []):
        if isinstance(entry, dict) and entry.get("model_id"):
            model_id = str(entry["model_id"])
            label = entry.get("display_name") or entry.get("label") or entry.get("name")
            if label:
                labels[model_id] = str(label)
            revision = entry.get("model_revision") or entry.get("revision") or entry.get("resolved_revision")
            if revision:
                revisions[model_id] = str(revision)
    try:
        import yaml

        config = yaml.safe_load((run_dir / "resolved_config.yml").read_text(encoding="utf-8")) or {}
    except Exception:
        return _append_revision_labels(labels, revisions)
    includes = ((config.get("models") or {}).get("include") or []) if isinstance(config, dict) else []
    for entry in includes:
        if isinstance(entry, dict) and entry.get("model_id"):
            model_id = str(entry["model_id"])
            label = entry.get("display_name") or entry.get("label") or entry.get("name")
            labels[model_id] = str(label) if label else labels.get(model_id, model_id)
            revision = entry.get("model_revision") or entry.get("revision") or entry.get("resolved_revision")
            if revision:
                revisions[model_id] = str(revision)
    return _append_revision_labels(labels, revisions)


def _model_revisions(
    run_meta: dict[str, Any] | None,
    rows: list[dict[str, Any]] | None,
) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for row in rows or []:
        if row.get("model_id") and row.get("model_revision"):
            revisions[str(row["model_id"])] = str(row["model_revision"])
    for entry in (run_meta or {}).get("model_runs", []) or []:
        if isinstance(entry, dict) and entry.get("model_id"):
            revision = entry.get("model_revision") or entry.get("revision") or entry.get("resolved_revision")
            if revision:
                revisions[str(entry["model_id"])] = str(revision)
    return revisions


def _append_revision_labels(labels: dict[str, str], revisions: dict[str, str]) -> dict[str, str]:
    rendered = dict(labels)
    for model_id, revision in revisions.items():
        if model_id not in rendered or not revision:
            continue
        short = str(revision)[:12]
        if short and short not in rendered[model_id]:
            rendered[model_id] = f"{rendered[model_id]} (rev {short})"
    return rendered


def _dataset_label(run_meta: dict[str, Any]) -> str:
    return str((run_meta.get("dataset") or {}).get("dataset_id") or "the evaluation dataset")


def _label_prefix(run_meta: dict[str, Any], task_id: str | None) -> str:
    values = [str(run_meta.get("experiment_id") or "run"), str(run_meta.get("run_id") or "latest"), task_id or ""]
    raw = "-".join(value for value in values if value)
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw).strip("-")
    return "-".join(part for part in safe.split("-") if part)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
