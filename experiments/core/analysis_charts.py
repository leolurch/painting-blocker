"""Render a configured chart bundle from model artifacts across compatible runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from .analysis_config import load_analysis_config, primary_experiment_id, resolve_analysis_path
from .calibration_protocol_bars import (
    PC_AXIS_LABEL,
    CalibrationProtocolBarsError,
    chart_targets,
    parse_protocol_chart,
    protocol_bars_by_target,
    protocol_directory,
    protocol_validation_rr_by_target,
)
from .analysis_readme import write_analysis_readme
from .artifacts import atomic_write_json, atomic_write_text, sha256_file, utc_now_iso
from .blocking_tables import (
    INLINE_TABLE_HEADER,
    MACRO_HEADER,
    render_blocking_calibrated_table,
    render_blocking_calibration_targets_table,
    render_blocking_fixed_k_table,
    render_blocking_performance_table,
)
from .aggregate_runs import rows_from_eval
from .chart_sidecar import json_chart_value, require_chart_sidecars, write_chart_sidecar
from .latex_table_output import write_latex_table_with_values
from .plot_pq_pc import load_curve, plot_pq_pc
from .plot_ranked_pc_bars import plot_ranked_bars, write_summary_csv
from .run_figures import (
    FALLBACK_PALETTE,
    RC_PARAMS,
    ModelSpec,
    _import_matplotlib,
    _render_fig1,
    _render_zoomed_pc_at_k,
    _render_fig2,
    _render_fig3,
    _render_calibrated_threshold_bars,
    _render_validation_rr_bars,
    _render_fig4,
    _render_fig5,
    _safe_name,
    _save,
    fig1_data,
    zoomed_pc_at_k_data,
    fig2_data,
    fig3_data,
    calibrated_threshold_bars_data,
    validation_rr_bars_data,
    fig4_data,
    fig5_data,
    load_model_artifacts,
)


class AnalysisRenderError(RuntimeError):
    """Raised when a selected analysis source is unavailable or incompatible."""


def _enabled(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = (config.get("charts") or {}).get(name)
    return value if isinstance(value, dict) and bool(value.get("enabled", False)) else None


def _formats(chart: dict[str, Any], default: tuple[str, ...] = ("pdf",)) -> tuple[str, ...]:
    values = chart.get("formats") or list(default)
    result = tuple(str(value) for value in values)
    invalid = sorted(set(result) - {"pdf", "svg", "png"})
    if invalid:
        raise AnalysisRenderError(f"Unsupported chart format(s): {', '.join(invalid)}")
    return result


def _axis_limits(
    chart_name: str,
    chart: dict[str, Any],
    key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    values = chart.get(key, default)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise AnalysisRenderError(f"{chart_name}.{key} must contain exactly two numbers")
    lower, upper = (float(value) for value in values)
    if lower >= upper:
        raise AnalysisRenderError(f"{chart_name}.{key} must be strictly increasing")
    return lower, upper


def _task(chart: dict[str, Any], task_ids: list[str]) -> str:
    task_id = chart.get("task")
    if task_id is None and len(task_ids) == 1:
        return task_ids[0]
    if task_id is None:
        raise AnalysisRenderError("Chart task must be configured for a multi-task analysis")
    task_id = str(task_id)
    if task_id not in task_ids:
        raise AnalysisRenderError(f"Configured task {task_id!r} is absent from selected artifacts")
    return task_id


def _selected_models(
    config_path: Path,
    config: dict[str, Any],
) -> tuple[list[ModelSpec], list[dict[str, Any]]]:
    selected_entries = [
        entry for entry in config.get("models") or []
        if isinstance(entry, dict) and bool(entry.get("include", False))
    ]
    selected_entries.sort(key=lambda entry: (
        int((entry.get("presentation") or {}).get("order", 10**9)),
        str(entry.get("analysis_model_id")),
    ))
    if not selected_entries:
        raise AnalysisRenderError("Analysis selects no models (all include flags are false)")
    specs: list[ModelSpec] = []
    warnings: list[dict[str, Any]] = []
    for index, entry in enumerate(selected_entries):
        key = str(entry.get("analysis_model_id") or "")
        source = entry.get("source") or {}
        model_dir_value = source.get("selected_model_dir")
        if not key or not model_dir_value:
            message = f"Selected model has no resolved source: {key or entry!r}"
            warnings.append({
                "stage": "model_selection",
                "model": key or None,
                "warning": message,
                "policy": "skip_model",
            })
            continue
        model_dir = resolve_analysis_path(config_path, model_dir_value)
        if not (model_dir / "eval.json").is_file():
            message = f"Selected model source is unavailable: {model_dir}"
            warnings.append({
                "stage": "model_selection",
                "model": key,
                "warning": message,
                "policy": "skip_model",
            })
            continue
        presentation = entry.get("presentation") or {}
        group = presentation.get("group")
        if group not in (None, "frozen", "finetuned"):
            raise AnalysisRenderError(f"Invalid presentation.group for {key}: {group!r}")
        color = str(presentation.get("color") or FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)])
        specs.append(
            ModelSpec(
                model_id=str(entry.get("model_id") or key),
                storage_key=key,
                label=str(presentation.get("label") or entry.get("model_id") or key),
                color=color,
                linestyle="--" if group == "frozen" else "-",
                group=str(group) if group else None,
                is_baseline=str(entry.get("kind") or "") == "analytical_baseline",
                model_dir=model_dir,
            )
        )
    return specs, warnings


def _presentation_labels(
    config: dict[str, Any],
    specs: list[ModelSpec],
    *,
    use_short_labels: bool,
) -> dict[str, str]:
    labels = {spec.storage_key: spec.label for spec in specs}
    if not use_short_labels:
        return labels
    for entry in config.get("models") or []:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("analysis_model_id") or "")
        if model_id not in labels:
            continue
        presentation = entry.get("presentation") or {}
        labels[model_id] = str(
            presentation.get("short_label")
            or presentation.get("label")
            or labels[model_id]
        )
    return labels


def _pc_at_k_chart_data(
    config_path: Path,
    chart: dict[str, Any],
    specs: list[ModelSpec],
    artifacts: dict[str, dict[str, Any]],
    task_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load optional chart-specific eval sources without replacing analysis sources."""
    configured = chart.get("source_model_dirs")
    if configured is None:
        return fig1_data(specs, artifacts, task_id), []
    if not isinstance(configured, dict):
        raise AnalysisRenderError(
            "pc_at_k_curve_zoomed.source_model_dirs must be a mapping"
        )
    unknown = sorted(set(str(key) for key in configured) - {
        spec.storage_key for spec in specs
    })
    if unknown:
        raise AnalysisRenderError(
            "pc_at_k_curve_zoomed.source_model_dirs contains unselected models: "
            + ", ".join(unknown)
        )
    missing = [
        spec.storage_key for spec in specs if spec.storage_key not in configured
    ]
    if missing and bool(chart.get("require_source_model_dirs_for_all", False)):
        raise AnalysisRenderError(
            "pc_at_k_curve_zoomed.source_model_dirs is missing selected models: "
            + ", ".join(missing)
        )

    chart_artifacts = dict(artifacts)
    source_rows: list[dict[str, Any]] = []
    for spec in specs:
        configured_dir = configured.get(spec.storage_key)
        if configured_dir is None:
            continue
        model_dir = resolve_analysis_path(config_path, configured_dir)
        eval_path = model_dir / "eval.json"
        if not eval_path.is_file():
            raise AnalysisRenderError(
                f"Configured PC@k source is missing eval.json: {model_dir}"
            )
        chart_artifacts[spec.storage_key] = load_model_artifacts(
            replace(spec, model_dir=model_dir)
        )
        source_rows.append({
            "analysis_model_id": spec.storage_key,
            "model_dir": str(model_dir),
            "eval_sha256": sha256_file(eval_path),
        })
    _validate_task_compatibility(specs, chart_artifacts)
    return fig1_data(specs, chart_artifacts, task_id), source_rows


def _task_ids(specs: list[ModelSpec], artifacts: dict[str, dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for spec in specs:
        for task in artifacts[spec.storage_key]["eval"].get("tasks") or []:
            task_id = str(task.get("task_id") or "")
            if task_id and task_id not in result:
                result.append(task_id)
    return result


def _validate_task_compatibility(
    specs: list[ModelSpec], artifacts: dict[str, dict[str, Any]]
) -> None:
    """Reject mixed query/candidate populations before producing comparisons."""
    expected: dict[str, dict[str, Any]] | None = None
    expected_calibration: dict[str, Any] | None = None
    for spec in specs:
        current: dict[str, dict[str, Any]] = {}
        for task in artifacts[spec.storage_key]["eval"].get("tasks") or []:
            task_id = str(task.get("task_id") or "")
            if not task_id:
                continue
            current[task_id] = {
                "query_ids_hash": task.get("query_ids_hash"),
                "candidate_ids_hash": task.get("candidate_ids_hash"),
                "positive_pairs_hash": task.get("positive_pairs_hash"),
                "num_queries": task.get("num_queries"),
                "num_candidates": task.get("num_candidates"),
                "num_positive_pairs": task.get("num_positive_pairs"),
                "top_k": [row.get("k") for row in task.get("metrics") or []],
            }
        if expected is None:
            expected = current
        elif expected != current:
            raise AnalysisRenderError(
                f"Model {spec.storage_key!r} does not contain the same task populations/grids "
                "as the other selected models"
            )
        calibration = artifacts[spec.storage_key].get("calibration") or {}
        calibration_signature = {
            "task_id": calibration.get("task_id"),
            "query_ids_hash": calibration.get("query_ids_hash"),
            "candidate_ids_hash": calibration.get("candidate_ids_hash"),
            "positive_pairs_hash": calibration.get("positive_pairs_hash"),
            "num_queries": calibration.get("num_queries"),
            "num_candidates": calibration.get("num_candidates"),
            "num_positive_pairs": calibration.get("num_positive_pairs"),
        } if calibration else {}
        if expected_calibration is None:
            expected_calibration = calibration_signature
        elif expected_calibration != calibration_signature:
            raise AnalysisRenderError(
                f"Model {spec.storage_key!r} does not use the same calibration population "
                "as the other selected models"
            )


def _missing(
    entries: list[dict[str, Any]],
    specs: list[ModelSpec],
) -> list[str]:
    present = {entry["spec"].storage_key for entry in entries}
    return [spec.storage_key for spec in specs if spec.storage_key not in present]


def _handle_missing(
    chart_name: str,
    chart: dict[str, Any],
    missing: list[str],
    warnings: list[dict[str, Any]],
) -> bool:
    """Record missing model data. Return whether the entire chart should be skipped."""
    if not missing:
        return False
    policy = str(chart.get("missing_data") or "warn")
    message = f"{chart_name} lacks required artifacts for: {', '.join(missing)}"
    if policy == "error":
        raise AnalysisRenderError(message)
    warnings.append({"chart": chart_name, "warning": message, "policy": policy})
    if policy == "skip_chart":
        return True
    if policy not in {"warn", "skip_model"}:
        raise AnalysisRenderError(f"Unknown missing_data policy for {chart_name}: {policy!r}")
    return False


def _selected_rows(specs: list[ModelSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        eval_path = spec.model_dir / "eval.json"
        for row in rows_from_eval(eval_path, include_incomplete=True):
            normalized = dict(row)
            # Analysis identities remain unique even if historical runs reused a
            # model_id for different preprocessing variants.
            normalized["source_model_id"] = normalized.get("model_id")
            normalized["model_id"] = spec.storage_key
            normalized["display_name"] = spec.label
            rows.append(normalized)
    return rows


def _model_numeric_entries(
    entries: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [
        {
            "model_id": entry["spec"].storage_key,
            **{
                field: json_chart_value(entry[field])
                for field in fields
                if entry.get(field) is not None
            },
        }
        for entry in entries
    ]


def _save_figure(
    fig,
    plt,
    out_dir: Path,
    stem: str,
    chart: dict[str, Any],
    numeric_data: dict[str, Any],
) -> dict[str, Path]:
    written = _save(fig, plt, out_dir, stem, _formats(chart))
    sidecar = write_chart_sidecar(out_dir, stem, numeric_data)
    written[sidecar.name] = sidecar
    return written


def _analysis_table_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Build table-caption metadata without depending on one source run."""
    compatibility = config.get("compatibility") or {}
    return {
        "experiment_id": primary_experiment_id(compatibility),
        "run_id": config.get("analysis_id"),
        "dataset": {"dataset_id": compatibility.get("dataset_id")},
    }


def _blocking_comparison_metrics(
    config_path: Path,
    chart: dict[str, Any],
    model_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    settings = chart.get("comparison_metrics") or {}
    source = settings.get("source")
    if not source:
        return {}, None
    source_path = resolve_analysis_path(config_path, source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Blocking comparison metric source not found: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "cross_dataset_table_metrics":
        raise AnalysisRenderError(
            f"Unsupported blocking comparison metric artifact: {source_path}"
        )
    source_rows = {str(row["model_id"]): row for row in payload.get("models") or []}
    aliases = {
        str(target): str(source_id)
        for target, source_id in (settings.get("model_id_aliases") or {}).items()
    }
    missing_policy = str(settings.get("missing_data") or "error")
    result: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for model_id in model_ids:
        source_id = aliases.get(model_id, model_id)
        row = source_rows.get(source_id)
        if row is None:
            missing.append(model_id)
            result[model_id] = {}
            continue
        result[model_id] = {
            "syn_test_pc_at_k": row.get("syn_test_pc_at_k"),
            "syn_test_pc_at_calibrated_threshold": row.get(
                "syn_test_pc_at_calibrated_threshold"
            ),
            "wik_test_pc_mean": row.get("wik_test_pc_mean"),
            "wik_test_pc_sample_sd": row.get("wik_test_pc_sample_sd"),
            "wik_test_pq_mean": row.get("wik_test_pq_mean"),
            "wik_test_pq_sample_sd": row.get("wik_test_pq_sample_sd"),
            "wik_test_rr_mean": row.get("wik_test_rr_mean"),
            "wik_test_rr_sample_sd": row.get("wik_test_rr_sample_sd"),
            "wik_calibrated_metrics_by_target": row.get(
                "wik_calibrated_metrics_by_target"
            ) or {
                f"{float(payload.get('wik_calibration_target_pair_completeness', 0.99)):g}": {
                    key: row.get(key)
                    for key in (
                        "wik_test_pc_mean",
                        "wik_test_pc_sample_sd",
                        "wik_test_pq_mean",
                        "wik_test_pq_sample_sd",
                        "wik_test_rr_mean",
                        "wik_test_rr_sample_sd",
                    )
                }
            },
        }
    if missing and missing_policy == "error":
        raise AnalysisRenderError(
            "Blocking comparison metrics are missing models: " + ", ".join(missing)
        )
    if missing_policy not in {"error", "render_dash"}:
        raise AnalysisRenderError(
            f"Unknown blocking comparison missing_data policy: {missing_policy!r}"
        )
    return result, {"path": source_path, "payload": payload}


def _blocking_performance_metrics(
    config_path: Path,
    chart: dict[str, Any],
    specs: list[ModelSpec],
    *,
    columns_override: list[dict[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float | None]],
    dict[str, Any] | None,
]:
    settings = chart.get("performance_requirements") or {}
    configured_source_runs = settings.get("source_runs")
    if configured_source_runs is None:
        source_run = settings.get("source_run")
        configured_source_runs = [source_run] if source_run else []
    if not isinstance(configured_source_runs, (list, tuple)):
        raise AnalysisRenderError("performance_requirements.source_runs must be a list")
    source_run_values = [str(value) for value in configured_source_runs if value]
    if not source_run_values:
        return [], {}, None
    if len(source_run_values) != len(set(source_run_values)):
        raise AnalysisRenderError("performance_requirements.source_runs contains duplicates")

    source_rows: dict[str, dict[str, Any]] = {}
    source_row_runs: dict[str, Path] = {}
    sources: list[dict[str, Any]] = []
    for source_run in source_run_values:
        run_path = resolve_analysis_path(config_path, source_run)
        run_json_path = run_path / "run.json"
        summary_path = run_path / str(settings.get("summary") or "summary.json")
        if not run_json_path.is_file():
            raise FileNotFoundError(
                f"Performance run metadata not found: {run_json_path}"
            )
        run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        if run_payload.get("status") != "completed":
            raise AnalysisRenderError(
                f"Performance benchmark run is not completed: {run_path}"
            )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Performance summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance_path: Path | None = None
        if run_payload.get("provenance"):
            provenance_path = run_path / str(run_payload["provenance"])
            if not provenance_path.is_file():
                raise FileNotFoundError(
                    f"Performance provenance not found: {provenance_path}"
                )
        sources.append({
            "configured_source_run": source_run,
            "run_path": run_path,
            "run_json_path": run_json_path,
            "summary_path": summary_path,
            "provenance_path": provenance_path,
            "run_payload": run_payload,
            "derived_from_pilot": bool(summary.get("derived_from_pilot", False)),
            "warning": summary.get("warning"),
        })
        for row in summary.get("models") or []:
            source_id = str(row.get("model_id") or "")
            if not source_id:
                raise AnalysisRenderError(
                    f"Performance summary contains a model without model_id: "
                    f"{summary_path}"
                )
            if source_id in source_rows:
                raise AnalysisRenderError(
                    f"Performance model_id {source_id!r} occurs in both "
                    f"{source_row_runs[source_id]} and {run_path}"
                )
            source_rows[source_id] = row
            source_row_runs[source_id] = run_path

    configured_columns = (
        columns_override if columns_override is not None else settings.get("columns") or []
    )
    columns: list[dict[str, Any]] = []
    for configured in configured_columns:
        if not isinstance(configured, dict):
            raise AnalysisRenderError(
                "performance_requirements.columns entries must be mappings"
            )
        key = str(configured.get("key") or "").strip()
        label = str(configured.get("label") or "").strip()
        source_field = str(configured.get("source_field") or "").strip()
        if not key or not label or not source_field:
            raise AnalysisRenderError(
                "Performance columns require key, label, and source_field"
            )
        columns.append({
            "key": key,
            "label": label,
            "objective": str(configured.get("objective") or "max"),
            "decimals": int(configured.get("decimals", 1)),
            "leading_zero": bool(configured.get("leading_zero", False)),
            "source_field": source_field,
            "divisor": float(configured.get("divisor", 1.0)),
        })
    if not columns:
        raise AnalysisRenderError("Performance requirements configure no columns")
    if any(float(column["divisor"]) == 0.0 for column in columns):
        raise AnalysisRenderError("Performance column divisor must be nonzero")

    aliases = {
        str(target): str(source_id)
        for target, source_id in (settings.get("model_id_aliases") or {}).items()
    }
    missing_policy = str(settings.get("missing_data") or "error")
    if missing_policy not in {"error", "render_dash"}:
        raise AnalysisRenderError(
            f"Unknown performance_requirements missing_data policy: {missing_policy!r}"
        )
    result: dict[str, dict[str, float | None]] = {}
    model_sources: dict[str, str] = {}
    missing: list[str] = []
    for spec in specs:
        source_id = aliases.get(spec.storage_key, spec.model_id)
        model_sources[spec.storage_key] = source_id
        source_row = source_rows.get(source_id)
        if source_row is None:
            missing.append(spec.storage_key)
            result[spec.storage_key] = {}
            continue
        values: dict[str, float | None] = {}
        for column in columns:
            source_field = str(column["source_field"])
            if source_field not in source_row or source_row[source_field] is None:
                raise AnalysisRenderError(
                    f"Performance model {source_id!r} lacks {source_field!r}"
                )
            values[str(column["key"])] = (
                float(source_row[source_field]) / float(column["divisor"])
            )
        result[spec.storage_key] = values
    if missing and missing_policy == "error":
        raise AnalysisRenderError(
            "Performance summary is missing models: " + ", ".join(missing)
        )
    rendered_columns = [
        {
            "key": column["key"],
            "label": column["label"],
            "objective": column["objective"],
            "decimals": column["decimals"],
            "leading_zero": column["leading_zero"],
        }
        for column in columns
    ]
    return rendered_columns, result, {
        "sources": sources,
        "missing_models": missing,
        "model_sources": model_sources,
        "model_source_runs": {
            model_id: source_row_runs.get(source_id)
            for model_id, source_id in model_sources.items()
        },
    }


def _write_protocol_bar_directories(
    *,
    plt: Any,
    config: dict[str, Any],
    specs: list[Any],
    artifacts: dict[str, dict[str, Any]],
    figures_dir: Path,
    task_ids: list[str],
    warnings: list[dict[str, Any]],
    add: Any,
) -> None:
    """Write each new protocol's bar charts into its own directory.

    Filenames match the threshold charts in ``figures/``. Only the operating
    point with no safety margin is drawn.
    """
    try:
        chart = parse_protocol_chart(config)
    except CalibrationProtocolBarsError as exc:
        raise AnalysisRenderError(str(exc)) from exc
    if chart is None:
        return
    task_id = _task(chart, task_ids)
    configured_targets = set(chart_targets(chart))
    safe_task = _safe_name(task_id)
    for protocol in chart["protocols"]:
        by_rho = protocol_bars_by_target(specs, artifacts, task_id, protocol)
        validation_by_rho = protocol_validation_rr_by_target(
            specs, artifacts, task_id, protocol
        )
        if configured_targets:
            by_rho = {rho: value for rho, value in by_rho.items() if rho in configured_targets}
            validation_by_rho = {
                rho: value
                for rho, value in validation_by_rho.items()
                if rho in configured_targets
            }
        destination = protocol_directory(figures_dir, protocol)
        if not by_rho:
            _handle_missing(
                f"calibration_protocol_bars[{protocol}]",
                chart,
                [spec.storage_key for spec in specs],
                warnings,
            )
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for rho, data in sorted(by_rho.items()):
            if _handle_missing(
                f"calibration_protocol_bars[{protocol}][{rho:g}]",
                chart,
                _missing(data, specs),
                warnings,
            ):
                continue
            rho_tag = f"{rho:g}".replace(".", "p")
            add(
                f"calibration_protocol_bars:{protocol}",
                _save_figure(
                    _render_calibrated_threshold_bars(
                        plt,
                        rho,
                        data,
                        pc_ylabel=PC_AXIS_LABEL[protocol],
                    ),
                    plt,
                    destination,
                    f"pc_calibrated_rho{rho_tag}_bars_{safe_task}",
                    chart,
                    {
                        "protocol": protocol,
                        "safety_margins": False,
                        "calibration_target_pair_completeness": rho,
                        "pair_completeness_limits": [
                            max(0.0, min(float(entry["pc"]) for entry in data) - 0.15),
                            1.0,
                        ],
                        "reduction_ratio_limits": [0.0, 1.0],
                        "bars": _model_numeric_entries(data, ("pc", "rr")),
                    },
                ),
            )
        if protocol == "union" and not validation_by_rho:
            warnings.append(
                {
                    "chart": "calibration_protocol_bars[union]",
                    "warning": (
                        "Union stores test PC and RR at the zero-margin cell only; "
                        "validation RR bars were not written"
                    ),
                    "policy": "warn",
                }
            )
        for rho, data in sorted(validation_by_rho.items()):
            if _handle_missing(
                f"calibration_protocol_bars[{protocol}].validation_rr[{rho:g}]",
                chart,
                _missing(data, specs),
                warnings,
            ):
                continue
            rho_tag = f"{rho:g}".replace(".", "p")
            add(
                f"calibration_protocol_bars:{protocol}",
                _save_figure(
                    _render_validation_rr_bars(
                        plt,
                        rho,
                        data,
                        show_values=bool(chart.get("show_values", False)),
                        value_decimals=int(chart.get("value_decimals", 4)),
                    ),
                    plt,
                    destination,
                    f"validation_rr_rho{rho_tag}_bars_{safe_task}",
                    chart,
                    {
                        "protocol": protocol,
                        "safety_margins": False,
                        "calibration_target_pair_completeness": rho,
                        "reduction_ratio_limits": [0.0, 1.0],
                        "bars": _model_numeric_entries(data, ("rr",)),
                    },
                ),
            )


def write_analysis_charts(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    config = load_analysis_config(path)
    if str(config.get("mode") or "single_source") == "multisplit":
        from .multisplit_analysis import render_multisplit_analysis

        return render_multisplit_analysis(path, config)
    output_cfg = config.get("output") or {}
    output_root = resolve_analysis_path(path, output_cfg.get("directory") or f"../analyses/{config.get('analysis_id', path.stem)}")
    figures_dir = output_root / "figures"
    bars_dir = output_root / "bar_charts"
    tables_dir = output_root / "latex_tables"
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(output_cfg.get("clean_generated_directories", True)):
        for generated_dir in (figures_dir, bars_dir, tables_dir):
            if generated_dir.is_dir():
                shutil.rmtree(generated_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    specs, warnings = _selected_models(path, config)
    paths: dict[str, Path] = {
        "readme": write_analysis_readme(config, output_root),
    }
    incomplete_path = output_root / "INCOMPLETE.txt"
    if warnings:
        warning_lines = "\n".join(f"- {warning['warning']}" for warning in warnings)
        atomic_write_text(
            incomplete_path,
            "Analysis rendering is incomplete because selected models were skipped.\n\n"
            f"{warning_lines}\n",
        )
        paths["incomplete"] = incomplete_path
    elif incomplete_path.is_file():
        incomplete_path.unlink()

    artifacts = {spec.storage_key: load_model_artifacts(spec) for spec in specs}
    _validate_task_compatibility(specs, artifacts)
    task_ids = _task_ids(specs, artifacts)
    if not specs:
        source_manifest = {
            "schema_version": 1,
            "analysis_id": config.get("analysis_id"),
            "created_at": utc_now_iso(),
            "analysis_config": str(path),
            "analysis_config_sha256": sha256_file(path),
            "models": [],
            "outputs": {key: str(value) for key, value in paths.items()},
            "warnings": warnings,
        }
        manifest_path = output_root / "source_manifest.json"
        atomic_write_json(manifest_path, source_manifest)
        paths["manifest"] = manifest_path
        result = {
            "output_dir": str(output_root),
            "models": 0,
            "paths": {key: str(value) for key, value in paths.items()},
            "warnings": warnings,
        }
        atomic_write_json(output_root / "analysis_result.json", result)
        return result
    if not task_ids:
        raise AnalysisRenderError("Selected models contain no evaluation tasks")

    needs_standard_figures = any(
        _enabled(config, name)
        for name in (
            "pc_at_k_curve", "pc_at_k_curve_zoomed", "pc_rr_tradeoff", "pc_at_k_star_bars",
            "calibrated_threshold_bars", "validation_rr_bars",
            "calibration_protocol_bars",
            "similarity_distributions", "candidate_sizes",
        )
    )
    matplotlib = plt = None
    if needs_standard_figures:
        matplotlib, plt = _import_matplotlib()

    def add(stage: str, written: dict[str, Path]) -> None:
        for key, value in written.items():
            paths[f"{stage}:{key}"] = value

    chart_source_manifest: list[dict[str, Any]] = []
    context = matplotlib.rc_context(RC_PARAMS) if matplotlib is not None else None
    if context is not None:
        context.__enter__()
    try:
        chart = _enabled(config, "pc_at_k_curve")
        if chart:
            task_id = _task(chart, task_ids)
            data = fig1_data(specs, artifacts, task_id)
            if not _handle_missing("pc_at_k_curve", chart, _missing(data, specs), warnings) and data:
                targets = [float(value) for value in ((chart.get("target_pc_lines") or []))]
                rho = max(targets) if targets else None
                add("pc_at_k_curve", _save_figure(
                    _render_fig1(plt, data, rho), plt, figures_dir,
                    f"pc_at_k_{_safe_name(task_id)}", chart,
                    {
                        "target_pair_completeness": rho,
                        "series": _model_numeric_entries(data, ("ks", "pc")),
                    },
                ))

        chart = _enabled(config, "pc_at_k_curve_zoomed")
        if chart:
            task_id = _task(chart, task_ids)
            pc_limits = _axis_limits(
                "pc_at_k_curve_zoomed", chart, "pc_limits", (0.9, 1.0)
            )
            minimum_k_exclusive = int(chart.get("minimum_k_exclusive", 20))
            source_data, chart_sources = _pc_at_k_chart_data(
                path, chart, specs, artifacts, task_id
            )
            if chart_sources:
                chart_source_manifest.append({
                    "chart": "pc_at_k_curve_zoomed",
                    "sources": chart_sources,
                })
            data = zoomed_pc_at_k_data(
                source_data,
                minimum_k_exclusive,
                pc_limits,
            )
            if not _handle_missing(
                "pc_at_k_curve_zoomed",
                chart,
                _missing(source_data, specs),
                warnings,
            ) and data:
                targets = [
                    float(value) for value in (chart.get("target_pc_lines") or [])
                ]
                rho = max(targets) if targets else None
                display_labels = _presentation_labels(
                    config,
                    specs,
                    use_short_labels=bool(chart.get("use_short_labels", False)),
                )
                stem_prefix = str(
                    chart.get("output_stem")
                    or f"pc_at_k_gt{minimum_k_exclusive}_zoomed"
                )
                stem = f"{_safe_name(stem_prefix)}_{_safe_name(task_id)}"
                add("pc_at_k_curve_zoomed", _save_figure(
                    _render_zoomed_pc_at_k(
                        plt,
                        data,
                        rho,
                        pc_limits,
                        display_labels,
                        xscale=str(chart.get("xscale", "log")),
                    ),
                    plt,
                    bars_dir,
                    stem,
                    chart,
                    {
                        "candidate_budget_filter": {
                            "minimum_exclusive": minimum_k_exclusive,
                        },
                        "pair_completeness_limits": list(pc_limits),
                        "xscale": str(chart.get("xscale", "log")),
                        "target_pair_completeness": rho,
                        "legend_model_ids": [
                            entry["spec"].storage_key for entry in data
                        ],
                        "source_eval_artifacts": chart_sources,
                        "series": _model_numeric_entries(data, ("ks", "pc")),
                    },
                ))

        chart = _enabled(config, "pc_rr_tradeoff")
        if chart:
            task_id = _task(chart, task_ids)
            data = fig2_data(specs, artifacts, task_id)
            if not _handle_missing("pc_rr_tradeoff", chart, _missing(data, specs), warnings) and data:
                targets = [float(value) for value in ((chart.get("target_pc_lines") or []))]
                rho = max(targets) if targets else None
                add("pc_rr_tradeoff", _save_figure(
                    _render_fig2(plt, data, rho), plt, figures_dir,
                    f"pc_rr_tradeoff_{_safe_name(task_id)}", chart,
                    {
                        "target_pair_completeness": rho,
                        "reduction_ratio_limits": [0.9, 1.0005],
                        "pair_completeness_limits": [0.5, 1.01],
                        "series": _model_numeric_entries(data, ("rr", "pc")),
                    },
                ))

        chart = _enabled(config, "pc_at_k_star_bars")
        if chart:
            task_id = _task(chart, task_ids)
            k_star, data = fig3_data(
                specs, artifacts, task_id, preferred_k=int(chart.get("preferred_k", 25))
            )
            if not _handle_missing("pc_at_k_star_bars", chart, _missing(data, specs), warnings) and data:
                values = [float(entry["pc"]) for entry in data]
                add("pc_at_k_star_bars", _save_figure(
                    _render_fig3(plt, k_star, data), plt, figures_dir,
                    f"pc{k_star}_bars_{_safe_name(task_id)}", chart,
                    {
                        "candidate_budget": k_star,
                        "pair_completeness_limits": [max(0.0, min(values) - 0.15), 1.0],
                        "bars": _model_numeric_entries(data, ("pc",)),
                    },
                ))

        chart = _enabled(config, "calibrated_threshold_bars")
        if chart:
            task_id = _task(chart, task_ids)
            by_rho = calibrated_threshold_bars_data(specs, artifacts, task_id)
            configured_targets = {
                float(value) for value in chart.get("calibration_targets") or []
            }
            if configured_targets:
                by_rho = {
                    rho: value for rho, value in by_rho.items()
                    if rho in configured_targets
                }
            if not by_rho:
                _handle_missing(
                    "calibrated_threshold_bars",
                    chart,
                    [spec.storage_key for spec in specs],
                    warnings,
                )
            else:
                for rho, data in sorted(by_rho.items()):
                    if _handle_missing(
                        f"calibrated_threshold_bars[{rho:g}]",
                        chart,
                        _missing(data, specs),
                        warnings,
                    ):
                        continue
                    rho_tag = f"{rho:g}".replace(".", "p")
                    pc_values = [float(entry["pc"]) for entry in data]
                    add("calibrated_threshold_bars", _save_figure(
                        _render_calibrated_threshold_bars(plt, rho, data),
                        plt,
                        figures_dir,
                        f"pc_calibrated_rho{rho_tag}_bars_{_safe_name(task_id)}",
                        chart,
                        {
                            "calibration_target_pair_completeness": rho,
                            "pair_completeness_limits": [max(0.0, min(pc_values) - 0.15), 1.0],
                            "reduction_ratio_limits": [0.0, 1.0],
                            "bars": _model_numeric_entries(data, ("pc", "rr")),
                        },
                    ))

        chart = _enabled(config, "validation_rr_bars")
        if chart:
            task_id = _task(chart, task_ids)
            by_rho = validation_rr_bars_data(specs, artifacts, task_id)
            configured_targets = {
                float(value) for value in chart.get("calibration_targets") or []
            }
            if configured_targets:
                by_rho = {
                    rho: value for rho, value in by_rho.items()
                    if rho in configured_targets
                }
            if not by_rho:
                _handle_missing(
                    "validation_rr_bars",
                    chart,
                    [spec.storage_key for spec in specs],
                    warnings,
                )
            else:
                for rho, data in sorted(by_rho.items()):
                    if _handle_missing(
                        f"validation_rr_bars[{rho:g}]",
                        chart,
                        _missing(data, specs),
                        warnings,
                    ):
                        continue
                    rho_tag = f"{rho:g}".replace(".", "p")
                    add("validation_rr_bars", _save_figure(
                        _render_validation_rr_bars(
                            plt,
                            rho,
                            data,
                            show_values=bool(chart.get("show_values", False)),
                            value_decimals=int(chart.get("value_decimals", 4)),
                        ),
                        plt,
                        figures_dir,
                        f"validation_rr_rho{rho_tag}_bars_{_safe_name(task_id)}",
                        chart,
                        {
                            "calibration_target_pair_completeness": rho,
                            "reduction_ratio_limits": [0.0, 1.0],
                            "bars": _model_numeric_entries(data, ("rr",)),
                        },
                    ))

        _write_protocol_bar_directories(
            plt=plt,
            config=config,
            specs=specs,
            artifacts=artifacts,
            figures_dir=figures_dir,
            task_ids=task_ids,
            warnings=warnings,
            add=add,
        )

        chart = _enabled(config, "similarity_distributions")
        if chart:
            task_id = _task(chart, task_ids)
            data = fig4_data(specs, artifacts, task_id)
            if not _handle_missing("similarity_distributions", chart, _missing(data, specs), warnings) and data:
                add("similarity_distributions", _save_figure(
                    _render_fig4(plt, data), plt, figures_dir,
                    f"similarity_distributions_{_safe_name(task_id)}", chart,
                    {
                        "similarity_limits": [-0.2, 1.0],
                        "series": _model_numeric_entries(
                            data,
                            ("bin_edges", "match_density", "neg_density", "threshold"),
                        ),
                    },
                ))

        chart = _enabled(config, "candidate_sizes")
        if chart:
            task_id = _task(chart, task_ids)
            by_rho = fig5_data(specs, artifacts, task_id)
            configured_targets = {
                float(value) for value in chart.get("calibration_targets") or []
            }
            if configured_targets:
                by_rho = {rho: value for rho, value in by_rho.items() if rho in configured_targets}
            if not by_rho:
                _handle_missing("candidate_sizes", chart, [spec.storage_key for spec in specs], warnings)
            else:
                for rho, data in sorted(by_rho.items()):
                    if _handle_missing(
                        f"candidate_sizes[{rho:g}]", chart, _missing(data, specs), warnings
                    ):
                        continue
                    rho_tag = f"{rho:g}".replace(".", "p")
                    add("candidate_sizes", _save_figure(
                        _render_fig5(plt, rho, data), plt, figures_dir,
                        f"candidate_sizes_rho{rho_tag}_{_safe_name(task_id)}", chart,
                        {
                            "calibration_target_pair_completeness": rho,
                            "series": _model_numeric_entries(data, ("sizes",)),
                        },
                    ))
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    rows: list[dict[str, Any]] | None = None
    labels = {spec.storage_key: spec.label for spec in specs}
    colors = {spec.storage_key: spec.color for spec in specs}

    chart = _enabled(config, "ranked_pc_at_k_bars")
    if chart:
        rows = _selected_rows(specs)
        task_id = _task(chart, task_ids)
        ks = {int(value) for value in chart.get("ks") or []}
        ranked_labels = _presentation_labels(
            config,
            specs,
            use_short_labels=bool(chart.get("use_short_labels", False)),
        )
        pc_limits = _axis_limits(
            "ranked_pc_at_k_bars", chart, "pc_limits", (0.0, 1.0)
        )
        ranked_rows = [
            row for row in rows
            if row.get("selection_mode") == "top_k"
            and str(row.get("task_id")) == task_id
            and int(row.get("k") or -1) in ks
        ]
        if not ranked_rows:
            warnings.append({"chart": "ranked_pc_at_k_bars", "warning": "No matching fixed-k rows"})
        else:
            summary = write_summary_csv(
                ranked_rows, ranked_labels, bars_dir / "ranked_pc_summary.csv"
            )
            paths["ranked_pc_at_k_bars:summary"] = summary
            for output in plot_ranked_bars(
                ranked_rows,
                ranked_labels,
                bars_dir,
                colors=colors,
                formats=_formats(chart, ("svg",)),
                style=chart,
            ):
                paths[f"ranked_pc_at_k_bars:{output.name}"] = output
            groups = sorted({
                (str(row.get("task_id") or "task"), int(row["k"]))
                for row in ranked_rows
            })
            for group_task, group_k in groups:
                ranked = sorted(
                    [
                        row for row in ranked_rows
                        if str(row.get("task_id") or "task") == group_task
                        and int(row["k"]) == group_k
                    ],
                    key=lambda row: float(row["pair_completeness"]),
                    reverse=True,
                )
                sidecar = write_chart_sidecar(
                    bars_dir,
                    f"ranked_pc_at_{group_k}_{_safe_name(group_task)}",
                    {
                        "candidate_budget": group_k,
                        "layout": str(chart.get("orientation", "horizontal")),
                        "pair_completeness_limits": list(pc_limits),
                        "show_values": bool(chart.get("show_values", True)),
                        "value_decimals": int(chart.get("value_decimals", 3)),
                        "bars": [
                            {
                                "rank": rank,
                                "model_id": str(row.get("model_id") or ""),
                                "pair_completeness": float(row["pair_completeness"]),
                            }
                            for rank, row in enumerate(ranked, start=1)
                        ],
                    },
                )
                paths[f"ranked_pc_at_k_bars:{sidecar.name}"] = sidecar

    for chart_name, output_stem in (
        ("pq_pc_curve", "pq_pc"),
        ("pq_pc_curve_zoomed", "pq_pc_zoomed"),
    ):
        chart = _enabled(config, chart_name)
        if not chart:
            continue
        task_id = _task(chart, task_ids)
        styles = {
            (spec.model_dir / "curves.json").resolve(): {
                "label": spec.label,
                "color": spec.color,
                "linestyle": spec.linestyle,
                "model_id": spec.storage_key,
            }
            for spec in specs if (spec.model_dir / "curves.json").is_file()
        }
        missing = [spec.storage_key for spec in specs if not (spec.model_dir / "curves.json").is_file()]
        if not _handle_missing(chart_name, chart, missing, warnings) and styles:
            pc_limits = _axis_limits(chart_name, chart, "pc_limits", (0.0, 1.0))
            pq_limits = _axis_limits(chart_name, chart, "pq_limits", (0.0, 1.0))
            stem = f"{output_stem}_{_safe_name(task_id)}"
            series: list[dict[str, Any]] = []
            for curve_file, style in styles.items():
                loaded = load_curve(curve_file, task_id)
                if loaded is None:
                    continue
                _source_model_id, curve_rows = loaded
                points = sorted(
                    [
                        [
                            float(row.get("pair_completeness", row.get("recall", 0.0))),
                            float(row.get("pair_quality", row.get("precision", 0.0))),
                        ]
                        for row in curve_rows
                    ],
                    key=lambda pair: pair[0],
                )
                if points:
                    series.append({
                        "model_id": style["model_id"],
                        "pair_completeness": [point[0] for point in points],
                        "pair_quality": [point[1] for point in points],
                    })
            for fmt in _formats(chart):
                output = figures_dir / f"{stem}.{fmt}"
                plot_pq_pc(
                    list(styles),
                    task_id,
                    output,
                    styles=styles,
                    pc_limits=pc_limits,
                    pq_limits=pq_limits,
                )
                paths[f"{chart_name}:{output.name}"] = output
            sidecar = write_chart_sidecar(
                figures_dir,
                stem,
                {
                    "pair_completeness_limits": list(pc_limits),
                    "pair_quality_limits": list(pq_limits),
                    "series": series,
                },
            )
            paths[f"{chart_name}:{sidecar.name}"] = sidecar

    external_table_sources: list[dict[str, Any]] = []
    chart = _enabled(config, "blocking_tables")
    if chart:
        rows = rows if rows is not None else _selected_rows(specs)
        task_id = _task(chart, task_ids)
        run_meta = _analysis_table_metadata(config)
        model_ids = [spec.storage_key for spec in specs]
        comparison_metrics, comparison_metadata = _blocking_comparison_metrics(
            path, chart, model_ids
        )
        performance_columns, performance_values, performance_metadata = (
            _blocking_performance_metrics(path, chart, specs)
        )
        performance_settings = chart.get("performance_requirements") or {}
        additional_performance_settings = (
            performance_settings.get("additional_table") or {}
        )
        if bool(additional_performance_settings.get("enabled", False)):
            additional_performance_columns, additional_performance_values, _ = (
                _blocking_performance_metrics(
                    path,
                    chart,
                    specs,
                    columns_override=additional_performance_settings.get("columns") or [],
                )
            )
        else:
            additional_performance_columns = []
            additional_performance_values = {}
        if comparison_metadata is not None:
            comparison_path = Path(comparison_metadata["path"])
            external_table_sources.append({
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
                "artifact_type": comparison_metadata["payload"].get("artifact_type"),
            })
        if performance_metadata is not None:
            provenance_sources: list[dict[str, Any]] = []
            for source in performance_metadata["sources"]:
                run_json_path = Path(source["run_json_path"])
                summary_path = Path(source["summary_path"])
                run_json_hash = sha256_file(run_json_path)
                summary_hash = sha256_file(summary_path)
                external_table_sources.extend([
                    {
                        "path": str(run_json_path),
                        "sha256": run_json_hash,
                        "artifact_type": "single_stream_benchmark_run",
                    },
                    {
                        "path": str(summary_path),
                        "sha256": summary_hash,
                        "artifact_type": "single_stream_benchmark_summary",
                    },
                ])
                source_provenance_path = source.get("provenance_path")
                source_provenance_hash: str | None = None
                if source_provenance_path is not None:
                    source_provenance_path = Path(source_provenance_path)
                    source_provenance_hash = sha256_file(source_provenance_path)
                    external_table_sources.append({
                        "path": str(source_provenance_path),
                        "sha256": source_provenance_hash,
                        "artifact_type": "single_stream_benchmark_provenance",
                    })
                provenance_sources.append({
                    "configured_source_run": source["configured_source_run"],
                    "source_run": str(source["run_path"]),
                    "run_status": source["run_payload"].get("status"),
                    "derived_from_pilot": source["derived_from_pilot"],
                    "warning": source["warning"],
                    "run_json": str(run_json_path),
                    "run_json_sha256": run_json_hash,
                    "summary_json": str(summary_path),
                    "summary_json_sha256": summary_hash,
                    "provenance_json": (
                        str(source_provenance_path)
                        if source_provenance_path is not None
                        else None
                    ),
                    "provenance_json_sha256": source_provenance_hash,
                })
            provenance_path = tables_dir / "blocking_performance_requirements.json"
            atomic_write_json(provenance_path, {
                "schema_version": 3,
                "artifact_type": "blocking_performance_requirements",
                "sources": provenance_sources,
                "columns": performance_columns,
                "additional_table_columns": additional_performance_columns,
                "models": [
                    {
                        "analysis_model_id": spec.storage_key,
                        "benchmark_model_id": performance_metadata["model_sources"][spec.storage_key],
                        "benchmark_source_run": (
                            str(performance_metadata["model_source_runs"][spec.storage_key])
                            if performance_metadata["model_source_runs"][spec.storage_key]
                            is not None
                            else None
                        ),
                        "measurements": performance_values.get(spec.storage_key, {}),
                        "additional_table_measurements": additional_performance_values.get(
                            spec.storage_key, {}
                        ),
                        "missing": spec.storage_key in performance_metadata["missing_models"],
                    }
                    for spec in specs
                ],
            })
            paths["blocking_tables:performance_requirements"] = provenance_path
        configured_model_groups = {
            str(entry["analysis_model_id"]): str(
                (entry.get("presentation") or {}).get("group") or "ungrouped"
            )
            for entry in config.get("models") or []
            if isinstance(entry, dict) and entry.get("analysis_model_id") is not None
        }
        fixed = render_blocking_fixed_k_table(
            rows,
            run_meta,
            model_ids,
            labels,
            task_id,
            ks=chart.get("ks"),
            model_groups=configured_model_groups,
            model_group_order=chart.get("model_group_order"),
            model_order_within_groups=chart.get("model_order_within_groups"),
            pinned_models=chart.get("pinned_models"),
            baseline_comparison_model=chart.get("baseline_comparison_model"),
            baseline_comparison_group=chart.get("baseline_comparison_group"),
            baseline_comparison_position=str(
                chart.get("baseline_comparison_position") or "first"
            ),
            baseline_comparison_separator=chart.get(
                "baseline_comparison_separator"
            ),
            alphabetical_within_groups=bool(
                chart.get("alphabetical_within_groups", False)
            ),
            separate_model_groups=bool(chart.get("separate_model_groups", False)),
            highlight_best_per_group=bool(
                chart.get("highlight_best_per_group", False)
            ),
            comparison_metrics=comparison_metrics,
            performance_columns=(
                (performance_columns or chart.get("performance_columns"))
                if bool(chart.get("include_performance_in_main_table", True))
                else []
            ),
            performance_values=(
                performance_values
                if bool(chart.get("include_performance_in_main_table", True))
                else {}
            ),
            performance_placeholder=str(
                chart.get("performance_placeholder") or "-"
            ),
            performance_group_heading=str(
                chart.get("performance_group_heading") or "Performance requirements"
            ),
            vertical_group_rules=bool(chart.get("vertical_group_rules", False)),
            comparison_synthetic_k=int(
                (comparison_metadata or {}).get("payload", {}).get(
                    "synthetic_fixed_k", 80
                )
            ),
            comparison_wik_target=float(
                (comparison_metadata or {}).get("payload", {}).get(
                    "wik_calibration_target_pair_completeness", 0.99
                )
            ),
            comparison_wik_targets=chart.get("merged_calibration_targets"),
            show_comparison_sd=bool(
                chart.get("show_calibration_sd_in_main_table", True)
            ),
        )
        target_comparison_settings = chart.get("calibration_target_comparison") or {}
        calibration_targets = (
            render_blocking_calibration_targets_table(
                rows,
                run_meta,
                model_ids,
                labels,
                task_id,
                targets=target_comparison_settings.get("targets") or [0.95, 0.99],
                comparison_metrics=comparison_metrics,
                model_groups=configured_model_groups,
                model_group_order=chart.get("model_group_order"),
                model_order_within_groups=chart.get("model_order_within_groups"),
                baseline_comparison_model=chart.get("baseline_comparison_model"),
                baseline_comparison_group=chart.get("baseline_comparison_group"),
                baseline_comparison_position=str(
                    chart.get("baseline_comparison_position") or "first"
                ),
                baseline_comparison_separator=chart.get(
                    "baseline_comparison_separator"
                ),
                alphabetical_within_groups=bool(
                    chart.get("alphabetical_within_groups", False)
                ),
                separate_model_groups=bool(
                    chart.get("separate_model_groups", False)
                ),
                highlight_best_per_group=bool(
                    chart.get("highlight_best_per_group", False)
                ),
                vertical_group_rules=bool(
                    chart.get("vertical_group_rules", False)
                ),
            )
            if bool(target_comparison_settings.get("enabled", False))
            else ""
        )
        performance_table = (
            render_blocking_performance_table(
                rows,
                run_meta,
                model_ids,
                labels,
                task_id,
                performance_columns=additional_performance_columns,
                performance_values=additional_performance_values,
                performance_placeholder=str(
                    chart.get("performance_placeholder") or "-"
                ),
                performance_group_heading=str(
                    chart.get("performance_group_heading") or "B200, $B=1$"
                ),
                model_groups=configured_model_groups,
                model_group_order=chart.get("model_group_order"),
                model_order_within_groups=chart.get("model_order_within_groups"),
                baseline_comparison_model=chart.get("baseline_comparison_model"),
                baseline_comparison_group=chart.get("baseline_comparison_group"),
                baseline_comparison_position=str(
                    chart.get("baseline_comparison_position") or "first"
                ),
                baseline_comparison_separator=chart.get(
                    "baseline_comparison_separator"
                ),
                alphabetical_within_groups=bool(
                    chart.get("alphabetical_within_groups", False)
                ),
                separate_model_groups=bool(
                    chart.get("separate_model_groups", False)
                ),
                highlight_best_per_group=bool(
                    chart.get("highlight_best_per_group", False)
                ),
                vertical_group_rules=bool(
                    chart.get("vertical_group_rules", False)
                ),
            )
            if additional_performance_columns
            else ""
        )
        calibrated = render_blocking_calibrated_table(rows, run_meta, model_ids, labels, task_id)
        if fixed:
            output = tables_dir / "blocking_fixed_k.tex"
            written = write_latex_table_with_values(output, MACRO_HEADER + fixed)
            paths["blocking_tables:blocking_fixed_k"] = output
            paths["blocking_tables:blocking_fixed_k_values"] = written[1]
        if calibration_targets:
            output = tables_dir / "blocking_calibration_targets.tex"
            written = write_latex_table_with_values(
                output, INLINE_TABLE_HEADER + calibration_targets
            )
            paths["blocking_tables:blocking_calibration_targets"] = output
            paths["blocking_tables:blocking_calibration_targets_values"] = written[1]
        if performance_table:
            output = tables_dir / "blocking_performance.tex"
            written = write_latex_table_with_values(
                output, INLINE_TABLE_HEADER + performance_table
            )
            paths["blocking_tables:blocking_performance"] = output
            paths["blocking_tables:blocking_performance_values"] = written[1]
        if calibrated:
            output = tables_dir / "blocking_calibrated.tex"
            written = write_latex_table_with_values(output, MACRO_HEADER + calibrated)
            paths["blocking_tables:blocking_calibrated"] = output
            paths["blocking_tables:blocking_calibrated_values"] = written[1]

    require_chart_sidecars(paths.values())

    source_manifest = {
        "schema_version": 1,
        "analysis_id": config.get("analysis_id"),
        "created_at": utc_now_iso(),
        "analysis_config": str(path),
        "analysis_config_sha256": sha256_file(path),
        "models": [
            {
                "analysis_model_id": spec.storage_key,
                "model_id": spec.model_id,
                "label": spec.label,
                "model_dir": str(spec.model_dir),
                "eval_sha256": sha256_file(spec.model_dir / "eval.json"),
                "curves_sha256": (
                    sha256_file(spec.model_dir / "curves.json")
                    if (spec.model_dir / "curves.json").is_file() else None
                ),
            }
            for spec in specs
        ],
        "outputs": {key: str(value) for key, value in paths.items()},
        "external_table_sources": external_table_sources,
        "chart_sources": chart_source_manifest,
        "warnings": warnings,
    }
    manifest_path = output_root / "source_manifest.json"
    atomic_write_json(manifest_path, source_manifest)
    paths["manifest"] = manifest_path
    result = {
        "output_dir": str(output_root),
        "models": len(specs),
        "paths": {key: str(value) for key, value in paths.items()},
        "warnings": warnings,
    }
    atomic_write_json(output_root / "analysis_result.json", result)
    return result
