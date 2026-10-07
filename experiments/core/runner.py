"""Config-driven retrieval experiment runner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import (
    PACKAGE_DIR,
    atomic_write_json,
    atomic_write_text,
    prepare_run_dir,
    git_info,
    make_run_id,
    sha256_file,
    sha256_json,
    utc_now_iso,
)
from .config_schema import ExperimentConfig, ModelConfig, load_experiment_config, load_yaml
from .embedding_pipeline import prepare_embeddings
from .eval_pipeline import (
    calibrated_candidate_sizes,
    calibrate_task_top_k,
    cosine_cache,
    evaluate_task,
    evaluate_task_at_calibrated_thresholds,
    evaluate_task_at_calibrated_threshold_top_k_floors,
    evaluate_task_at_calibrated_top_k,
)
from .preflight import preflight_huggingface_revisions
from .random_baseline import RandomBaselineConfig, evaluate_random_task, parse_random_baseline_config
from .resource_metrics import ResourceMonitor, release_torch_cuda_memory, system_profile
from .reproducibility import dependency_versions
from .split_schema import load_split, records_from_split, resolve_retrieval_tasks, validate_split_manifest


@dataclass(frozen=True)
class EvaluationOptions:
    similarity: str
    top_k: list[int]
    target_pc: list[float]
    cache_dtype: str
    device: str
    gpu_dtype: str
    fp32_matmul_precision: str


@dataclass(frozen=True)
class CalibrationSpec:
    """Resolved validation-calibration configuration for one experiment run.

    The calibration task's labels are used once, per model, to select a
    similarity threshold that is then applied unchanged to the held-out
    evaluation tasks.
    """

    method: str
    calibration_id: str
    target_pc: list[float]
    task: dict[str, Any]


@dataclass(frozen=True)
class TopKCalibrationSpec:
    """Validation-only selection of the smallest predeclared qualifying k."""

    method: str
    calibration_id: str
    target_pc: list[float]
    task: dict[str, Any]
    on_unattained: str


class ModelEvaluationError(RuntimeError):
    """Raised after a run records one or more failed model evaluations."""

    def __init__(self, run_dir: Path, failures: list[dict[str, Any]]):
        self.run_dir = run_dir
        self.failures = failures
        detail = "; ".join(
            f"{failure['model_id']}: {failure.get('error', 'unknown error')}"
            for failure in failures
        )
        super().__init__(
            f"{len(failures)} model evaluation(s) failed in {run_dir}: {detail}"
        )


def run_experiment(
    experiment_path: Path | str,
    model_filter: str | None = None,
    run_id: str | None = None,
    *,
    injected_models: list[ModelConfig] | None = None,
    output_root_override: Path | str | None = None,
    parent_training: dict[str, Any] | None = None,
    split_file_override: Path | str | None = None,
) -> Path:
    """Run an evaluation experiment using its explicitly declared type.

    Ordinary configs retain the historical pipeline below. Cross-dataset
    threshold transfer is isolated in its own runner and is dispatched before
    ordinary config resolution so the existing protocol is not reinterpreted.
    ``injected_models`` is used only by explicit
    ``models.from_finetuning_checkpoint`` templates.
    """
    raw_document = load_yaml(experiment_path)
    if raw_document.get("experiment_type") == "cross_dataset_threshold_transfer":
        from .cross_dataset_threshold_transfer import run_cross_dataset_experiment

        return run_cross_dataset_experiment(
            experiment_path,
            model_filter=model_filter,
            run_id=run_id,
            injected_models=injected_models,
            output_root_override=output_root_override,
            parent_training=parent_training,
            split_file_override=split_file_override,
        )

    exp = load_experiment_config(
        experiment_path,
        injected_models=injected_models,
        split_file_override=split_file_override,
    )
    split = load_split(exp.split_file)
    validate_split_manifest(split)
    _require_image_root(exp.dataset)
    eval_raw = dict(exp.raw.get("evaluation") or {})
    tasks = resolve_retrieval_tasks(split, eval_raw.get("retrieval_tasks"))
    calibration = _calibration_spec(eval_raw, split)
    top_k_calibration = _top_k_calibration_spec(eval_raw, split)
    models = _selected_models(exp.models, model_filter)
    random_baseline = parse_random_baseline_config(
        eval_raw.get("random_baseline")
    )
    include_random_baseline = random_baseline is not None and (
        model_filter is None or model_filter == random_baseline.model_id or bool(models)
    )
    if not models and not include_random_baseline:
        raise ValueError(f"Model {model_filter!r} is not included in {exp.path}")
    if random_baseline is not None:
        _validate_random_baseline_identity(random_baseline, exp.models)
    # Validate pinned Hub revisions outside the per-model try/except below. A
    # bad revision must abort before it can be mislabeled as a skipped model.
    preflight_huggingface_revisions(exp, models)
    run_dir = prepare_run_dir(_run_dir(exp, run_id, output_root_override))
    _write_inputs(exp, split, run_dir)
    db_sha = str((split.get("dataset") or {}).get("source_db_sha256") or "")
    calibration_tasks = [
        spec.task for spec in (calibration, top_k_calibration) if spec is not None
    ]
    all_tasks = [*calibration_tasks, *tasks]
    image_ids = _union_image_ids(all_tasks)
    records = records_from_split(split)
    _validate_image_ids(image_ids, records)
    if calibration is not None:
        _write_calibration_protocol(run_dir, calibration, tasks, records)
    if top_k_calibration is not None:
        _write_calibration_protocol(
            run_dir,
            top_k_calibration,
            tasks,
            records,
            report_name="top_k_calibration_protocol.json",
        )
    model_runs = []
    failed_models = []
    if include_random_baseline and random_baseline is not None:
        baseline_dir = run_dir / "models" / random_baseline.storage_key
        baseline_dir.mkdir(parents=True, exist_ok=True)
        try:
            eval_path = _run_random_baseline(
                exp,
                tasks,
                random_baseline,
                baseline_dir,
                image_ids,
                records,
            )
            if not eval_path.is_file():
                raise FileNotFoundError(f"Model did not produce eval.json at {eval_path}")
        except Exception as exc:
            failure = _failed_model_record(
                random_baseline.model_id,
                random_baseline.storage_key,
                None,
                exc,
                display_name=random_baseline.display_name,
                kind="analytical_baseline",
            )
            failed_models.append(failure)
            model_runs.append(failure)
        else:
            model_runs.append(
                {
                    "model_id": random_baseline.model_id,
                    "model_storage_key": random_baseline.storage_key,
                    "model_revision": None,
                    "display_name": random_baseline.display_name,
                    "kind": "analytical_baseline",
                    "status": "completed",
                    "path": str(eval_path.relative_to(run_dir)),
                }
            )
    for model in models:
        model_dir = run_dir / "models" / model.storage_key
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            eval_path = _run_model(
                exp,
                tasks,
                model,
                model_dir,
                image_ids,
                records,
                db_sha,
                calibration,
                top_k_calibration,
                split,
            )
            if not eval_path.is_file():
                raise FileNotFoundError(f"Model did not produce eval.json at {eval_path}")
        except Exception as exc:
            failure = _failed_model_record(
                model.model_id,
                model.storage_key,
                model.revision,
                exc,
                display_name=_display_name(model),
            )
            failed_models.append(failure)
            model_runs.append(failure)
        else:
            completed_model = {
                "model_id": model.model_id,
                "model_storage_key": model.storage_key,
                "model_revision": model.revision,
                "status": "completed",
                "path": str(eval_path.relative_to(run_dir)),
                "resource_metrics_path": f"resource_metrics/{model.storage_key}.json",
            }
            display_name = _display_name(model)
            if display_name:
                completed_model["display_name"] = display_name
            model_runs.append(completed_model)
        finally:
            release_torch_cuda_memory(reset_compiler=True)
    if failed_models:
        _write_failed_models(run_dir, failed_models)
    run_json = _run_json(
        exp,
        split,
        run_dir,
        db_sha,
        model_runs,
        parent_training=parent_training,
    )
    atomic_write_json(run_dir / "run.json", run_json)
    try:
        from .aggregate_runs import aggregate_paths

        aggregate_paths(
            [run_dir],
            run_dir / "aggregate_metrics.csv",
            run_dir / "aggregate_metrics.json",
            # Preserve successful model output for diagnosis, while run.json's
            # failed status keeps it out of normal cross-run aggregation.
            include_incomplete=bool(failed_models),
        )
    except Exception as exc:  # aggregation should not mask model failures
        atomic_write_json(run_dir / "aggregate_warning.json", {"warning": str(exc)})
    from .latex_tables import write_run_latex_tables

    write_run_latex_tables(run_dir)
    try:
        from .run_charts import write_run_charts

        write_run_charts(run_dir)
    except Exception as exc:  # charts are best-effort and must not fail the run
        atomic_write_json(
            run_dir / "charts_warning.json",
            {"warnings": [{"stage": "charts", "warning": f"{type(exc).__name__}: {exc}"}]},
        )
    if failed_models:
        raise ModelEvaluationError(run_dir, failed_models)
    return run_dir


def _require_image_root(dataset: Any) -> None:
    if dataset.image_root is None:
        raise ValueError(f"Dataset {dataset.dataset_id!r} must configure paths.image_root")
    if not Path(dataset.image_root).is_dir():
        raise FileNotFoundError(f"Dataset image_root not found: {dataset.image_root}")


def _selected_models(models: list[Any], model_filter: str | None) -> list[Any]:
    return [model for model in models if model_filter in (None, model.model_id)]


def _display_name(model: Any) -> str | None:
    raw = getattr(model, "raw", None)
    if isinstance(raw, dict):
        for key in ("display_name", "label", "name"):
            if raw.get(key):
                return str(raw[key])
    return None


def _failed_model_record(
    model_id: str,
    storage_key: str,
    revision: str | None,
    exc: Exception,
    *,
    display_name: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    record = {
        "model_id": model_id,
        "model_storage_key": storage_key,
        "model_revision": revision,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if display_name:
        record["display_name"] = display_name
    if kind:
        record["kind"] = kind
    return record


def _write_failed_models(run_dir: Path, failed_models: list[dict[str, Any]]) -> None:
    lines = [
        "Model evaluation failures:",
        "",
    ]
    for record in failed_models:
        parts = [str(record["model_id"])]
        if record.get("model_revision"):
            parts.append(f"revision={record['model_revision']}")
        parts.append(f"storage_key={record['model_storage_key']}")
        parts.append(f"{record['error_type']}: {record['error']}")
        lines.append(" - " + " | ".join(parts))
    atomic_write_text(run_dir / "MODEL_FAILURES.txt", "\n".join(lines) + "\n")


def _run_dir(
    exp: ExperimentConfig,
    run_id: str | None,
    output_root_override: Path | str | None = None,
) -> Path:
    if output_root_override is not None:
        output = Path(output_root_override).expanduser()
        output = output if output.is_absolute() else (PACKAGE_DIR / output)
    else:
        run_cfg = exp.raw.get("run") or {}
        output = Path(run_cfg.get("output_dir") or f"runs/{exp.experiment_id}").expanduser()
        output = output if output.is_absolute() else (PACKAGE_DIR / output)
    return output / (run_id or make_run_id())


def _write_inputs(exp: ExperimentConfig, split: dict[str, Any], run_dir: Path) -> None:
    (run_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(exp.resolved, sort_keys=False),
        encoding="utf-8",
    )
    atomic_write_json(run_dir / "split.snapshot.json", split)


def _calibration_spec(eval_raw: dict[str, Any], split: dict[str, Any]) -> CalibrationSpec | None:
    raw = eval_raw.get("threshold_calibration")
    if not raw:
        return None
    task = resolve_retrieval_tasks(split, [raw["task"]])[0]
    return CalibrationSpec(
        method=str(raw.get("method") or "empirical_target_pc"),
        calibration_id=str(raw["calibration_id"]),
        target_pc=[float(value) for value in raw["target_pc"]],
        task=task,
    )


def _top_k_calibration_spec(
    eval_raw: dict[str, Any], split: dict[str, Any]
) -> TopKCalibrationSpec | None:
    raw = eval_raw.get("top_k_calibration")
    if not raw:
        return None
    task = resolve_retrieval_tasks(split, [raw["task"]])[0]
    return TopKCalibrationSpec(
        method=str(raw.get("method") or "empirical_target_pc"),
        calibration_id=str(raw["calibration_id"]),
        target_pc=[float(value) for value in raw["target_pc"]],
        task=task,
        on_unattained=str(raw.get("on_unattained") or "report_only"),
    )


def _write_calibration_protocol(
    run_dir: Path,
    calibration: CalibrationSpec | TopKCalibrationSpec,
    tasks: list[dict[str, Any]],
    records: dict[str, Any],
    *,
    report_name: str = "calibration_protocol.json",
) -> None:
    """Verify calibration/test disjointness and store an auditable protocol report.

    Hard-fails on any leakage between the calibration and evaluation populations;
    the report is written before raising so failures remain diagnosable.
    """
    cal_task = calibration.task
    cal_query = set(cal_task["query_ids"])
    cal_candidate = set(cal_task["candidate_ids"])
    cal_classes = {records[file_id].class_id for file_id in cal_query | cal_candidate}
    checks: list[dict[str, Any]] = []
    violations: list[str] = []

    def _record(name: str, ok: bool, detail: dict[str, Any]) -> None:
        checks.append({"check": name, "ok": ok, **detail})
        if not ok:
            violations.append(name)

    cal_positive_pairs = sum(
        1
        for q_id in cal_query
        for c_id in cal_candidate
        if not (cal_task.get("exclude_self") and q_id == c_id)
        and records[q_id].class_id == records[c_id].class_id
    )
    _record(
        "calibration_has_positive_pairs",
        cal_positive_pairs > 0,
        {"num_positive_pairs": cal_positive_pairs},
    )
    for task in tasks:
        query_overlap = sorted(cal_query & set(task["query_ids"]))
        candidate_overlap = sorted(cal_candidate & set(task["candidate_ids"]))
        class_overlap = sorted(
            cal_classes & {records[file_id].class_id for file_id in task["query_ids"] + task["candidate_ids"]}
        )
        _record(
            "query_ids_disjoint",
            not query_overlap,
            {"task_id": task["task_id"], "overlap": query_overlap[:10]},
        )
        _record(
            "candidate_ids_disjoint",
            not candidate_overlap,
            {"task_id": task["task_id"], "overlap": candidate_overlap[:10]},
        )
        # Class overlap is recorded for auditability but is not a hard failure:
        # same-subset directional tasks legitimately reuse the split's classes.
        checks.append(
            {
                "check": "class_ids_disjoint",
                "ok": not class_overlap,
                "task_id": task["task_id"],
                "informational": True,
                "overlap": class_overlap[:10],
            }
        )
    report = {
        "schema_version": 1,
        "calibration_id": calibration.calibration_id,
        "method": calibration.method,
        "operating_point_type": (
            "top_k" if isinstance(calibration, TopKCalibrationSpec) else "similarity_threshold"
        ),
        "calibration_task_id": cal_task["task_id"],
        "target_pc": calibration.target_pc,
        "num_calibration_queries": len(cal_query),
        "num_calibration_candidates": len(cal_candidate),
        "test_task_ids": [task["task_id"] for task in tasks],
        "checks": checks,
        "passed": not violations,
    }
    report_path = run_dir / report_name
    atomic_write_json(report_path, report)
    if violations:
        raise ValueError(
            "Threshold calibration protocol violations: "
            + ", ".join(sorted(set(violations)))
            + f" (see {report_path})"
        )


def _union_image_ids(tasks: list[dict[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for task in tasks:
        ids.update(str(value) for value in task["query_ids"])
        ids.update(str(value) for value in task["candidate_ids"])
    return sorted(ids)


def _validate_image_ids(image_ids: list[str], records: dict[str, Any]) -> None:
    missing = sorted(set(image_ids) - set(records))
    if missing:
        raise ValueError(f"Split references IDs without same_painting labels: {missing[:10]}")


def _absolute_embedding_options(exp: ExperimentConfig) -> dict[str, Any]:
    opts = dict(exp.raw.get("embedding") or {})
    if opts.get("cache_dir"):
        path = Path(opts["cache_dir"]).expanduser()
        opts["cache_dir"] = str(path if path.is_absolute() else PACKAGE_DIR / path)
    return opts


def _evaluation_options(exp: ExperimentConfig) -> EvaluationOptions:
    raw = dict(exp.raw.get("evaluation") or {})
    similarity = str(raw.get("similarity", "cosine"))
    if similarity != "cosine":
        raise ValueError("Only evaluation.similarity='cosine' is supported")
    top_k = [int(k) for k in raw.get("top_k", [1, 5, 10])]
    if not top_k or any(k <= 0 for k in top_k):
        raise ValueError("evaluation.top_k must contain positive integers")
    fp32_matmul_precision = str(raw.get("fp32_matmul_precision") or "ieee")
    if fp32_matmul_precision != "ieee":
        raise ValueError("evaluation.fp32_matmul_precision must be 'ieee'")
    gpu_dtype = str(raw.get("gpu_dtype") or "float32")
    cache_dtype = str(raw.get("cache_dtype") or "float32")
    if gpu_dtype != "float32":
        raise ValueError(
            "FP32 evaluation is mandatory; evaluation.gpu_dtype must be 'float32'"
        )
    if cache_dtype != "float32":
        raise ValueError(
            "FP32 evaluation is mandatory; evaluation.cache_dtype must be 'float32'"
        )
    return EvaluationOptions(
        similarity=similarity,
        top_k=top_k,
        target_pc=[float(value) for value in raw.get("target_pc", [])],
        cache_dtype=cache_dtype,
        device=str(raw.get("device") or "auto"),
        gpu_dtype=gpu_dtype,
        fp32_matmul_precision=fp32_matmul_precision,
    )


def _validate_random_baseline_identity(
    baseline: RandomBaselineConfig,
    models: list[Any],
) -> None:
    if baseline.model_id in {model.model_id for model in models}:
        raise ValueError(f"random_baseline model_id collides with a configured model: {baseline.model_id}")
    if baseline.storage_key in {model.storage_key for model in models}:
        raise ValueError(
            "random_baseline storage_key collides with a configured model: "
            f"{baseline.storage_key}"
        )


def _run_random_baseline(
    exp: ExperimentConfig,
    tasks: list[dict[str, Any]],
    baseline: RandomBaselineConfig,
    model_dir: Path,
    image_ids: list[str],
    records: dict[str, Any],
) -> Path:
    run_opts = dict(exp.raw.get("run") or {})
    eval_opts = _evaluation_options(exp)
    task_results, curve_results = [], []
    for task in tasks:
        result, curves = evaluate_random_task(
            task,
            records,
            eval_opts.top_k,
            bool(run_opts.get("fail_on_empty_positives", True)),
            eval_opts.target_pc,
        )
        result["artifacts"] = {"curves": "curves.json"}
        task_results.append(result)
        curve_results.append(curves)
    baseline_meta = {
        "type": "random_selection_expected",
        "analytical": True,
        "display_name": baseline.display_name,
        "selection": "uniform_without_replacement",
    }
    atomic_write_json(
        model_dir / "curves.json",
        {
            "schema_version": 1,
            "model_id": baseline.model_id,
            "model_revision": None,
            "model_storage_key": baseline.storage_key,
            "baseline": baseline_meta,
            "tasks": curve_results,
        },
    )
    eval_json = {
        "schema_version": 1,
        "model_id": baseline.model_id,
        "model_revision": None,
        "model_storage_key": baseline.storage_key,
        "baseline": baseline_meta,
        "embedding": {
            "skipped": True,
            "reason": "analytical_random_baseline",
            "num_images": len(image_ids),
            "embedding_dim": 0,
            "normalized": False,
            "embedding_sha256": None,
        },
        "evaluation": {
            "similarity": None,
            "top_k": eval_opts.top_k,
            "target_pc": eval_opts.target_pc,
            "selection": "expected_uniform_random",
        },
        "artifacts": {},
        "tasks": task_results,
        "warnings": [],
    }
    atomic_write_json(model_dir / "eval.json", eval_json)
    return model_dir / "eval.json"


def _calibrate_model(
    calibration: CalibrationSpec | None,
    image_ids: list[str],
    records: dict[str, Any],
    sims: np.ndarray,
    model_dir: Path,
) -> list[dict[str, Any]]:
    """Select model-specific thresholds on the calibration task and record them.

    Returns the calibration operating points (one per requested target PC) and
    writes ``threshold_calibration.json`` documenting that the thresholds were
    chosen on the fixed calibration population rather than on test.
    """
    if calibration is None:
        return []
    cal_result, _ = evaluate_task(
        calibration.task,
        image_ids,
        records,
        sims,
        [1],
        True,
        calibration.target_pc,
    )
    calibration_rows = cal_result["target_pc_metrics"]
    atomic_write_json(
        model_dir / "threshold_calibration.json",
        {
            "schema_version": 1,
            "method": calibration.method,
            "calibration_id": calibration.calibration_id,
            "task_id": calibration.task["task_id"],
            "query_ids_hash": cal_result["query_ids_hash"],
            "candidate_ids_hash": cal_result["candidate_ids_hash"],
            "positive_pairs_hash": cal_result["positive_pairs_hash"],
            "num_queries": cal_result["num_queries"],
            "num_candidates": cal_result["num_candidates"],
            "num_positive_pairs": cal_result["num_positive_pairs"],
            "operating_points": calibration_rows,
        },
    )
    return calibration_rows


def _calibrate_model_top_k(
    calibration: TopKCalibrationSpec | None,
    image_ids: list[str],
    records: dict[str, Any],
    sims: np.ndarray,
    model_dir: Path,
    top_k: list[int],
) -> list[dict[str, Any]]:
    if calibration is None:
        return []
    rows = calibrate_task_top_k(
        calibration.task,
        image_ids,
        records,
        sims,
        top_k,
        calibration.target_pc,
        on_unattained=calibration.on_unattained,
    )
    atomic_write_json(
        model_dir / "top_k_calibration.json",
        {
            "schema_version": 1,
            "method": calibration.method,
            "calibration_id": calibration.calibration_id,
            "task_id": calibration.task["task_id"],
            "target_pc": calibration.target_pc,
            "on_unattained": calibration.on_unattained,
            "operating_points": rows,
        },
    )
    return rows


def _profile_diagnostics(
    task: dict[str, Any],
    image_ids: list[str],
    records: dict[str, Any],
    similarities: np.ndarray,
    top_k: list[int],
    calibration_rows: list[dict[str, Any]],
    calibration_id: str | None,
    top_k_calibration_rows: list[dict[str, Any]],
    top_k_calibration_id: str | None,
    split: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate profile slices using the one globally selected operating point.

    Profiles are diagnostics only: neither thresholds nor checkpoints are chosen
    from these rows. The pooled calibration task remains the formal objective.
    """
    field = str(config.get("candidate_metadata_field") or "profile")
    images = split.get("images") or {}
    grouped: dict[str, list[str]] = {}
    for candidate_id in task["candidate_ids"]:
        raw = images.get(candidate_id)
        value = raw.get(field) if isinstance(raw, dict) else None
        if value not in (None, ""):
            grouped.setdefault(str(value), []).append(candidate_id)
    diagnostics: list[dict[str, Any]] = []
    for profile in sorted(grouped):
        profile_task = {**task, "candidate_ids": sorted(grouped[profile])}
        try:
            result, _curves = evaluate_task(
                profile_task,
                image_ids,
                records,
                similarities,
                top_k,
                False,
                [],
            )
        except (KeyError, ValueError):
            continue
        if int(result.get("num_positive_pairs") or 0) == 0:
            continue
        row: dict[str, Any] = {
            "profile": profile,
            "candidate_metadata_field": field,
            "selection_role": "diagnostic_only",
            "num_queries": result["num_queries"],
            "num_candidates": result["num_candidates"],
            "num_positive_pairs": result["num_positive_pairs"],
            "metrics": result["metrics"],
        }
        if calibration_rows and calibration_id is not None:
            row["calibrated_threshold_metrics"] = evaluate_task_at_calibrated_thresholds(
                profile_task,
                image_ids,
                records,
                similarities,
                calibration_rows,
                calibration_id,
            )
        if top_k_calibration_rows and top_k_calibration_id is not None:
            row["calibrated_top_k_metrics"] = evaluate_task_at_calibrated_top_k(
                profile_task,
                image_ids,
                records,
                similarities,
                top_k_calibration_rows,
                top_k_calibration_id,
            )
        diagnostics.append(row)
    return diagnostics


def _run_model(
    exp: ExperimentConfig,
    tasks: list[dict[str, Any]],
    model,
    model_dir: Path,
    image_ids: list[str],
    records: dict[str, Any],
    db_sha: str,
    calibration: CalibrationSpec | None = None,
    top_k_calibration: TopKCalibrationSpec | None = None,
    split: dict[str, Any] | None = None,
) -> Path:
    run_opts = dict(exp.raw.get("run") or {})
    eval_raw = dict(exp.raw.get("evaluation") or {})
    floor_raw = eval_raw.get("threshold_top_k_floor") or {}
    threshold_top_k_floors = [int(value) for value in floor_raw.get("minimum_k") or []]
    eval_opts = _evaluation_options(exp)
    emb_opts = _absolute_embedding_options(exp)
    embeddings, emb_meta = prepare_embeddings(
        exp.dataset,
        model,
        image_ids,
        records,
        model_dir,
        run_opts,
        emb_opts,
        db_sha,
    )
    task_results, curve_results, warnings = [], [], []
    eval_seconds = 0.0
    with ResourceMonitor() as retrieval_monitor:
        sim_started = time.perf_counter()
        sims, similarity_precision = cosine_cache(
            embeddings,
            eval_opts.device,
            eval_opts.gpu_dtype,
            eval_opts.cache_dtype,
            inputs_normalized=bool(emb_meta.get("normalized")),
            fp32_matmul_precision=eval_opts.fp32_matmul_precision,
            return_metadata=True,
        )
        similarity_seconds = time.perf_counter() - sim_started
        eval_started = time.perf_counter()
        calibration_rows = _calibrate_model(calibration, image_ids, records, sims, model_dir)
        top_k_calibration_rows = _calibrate_model_top_k(
            top_k_calibration, image_ids, records, sims, model_dir, eval_opts.top_k
        )
        for task in tasks:
            result, curves = evaluate_task(
                task,
                image_ids,
                records,
                sims,
                eval_opts.top_k,
                bool(run_opts.get("fail_on_empty_positives", True)),
                eval_opts.target_pc,
            )
            result["artifacts"] = {
                "similarity_cache": "similarity_cache.npy",
                "curves": "curves.json",
                "resource_metrics": "resource_metrics.json",
            }
            if calibration is not None:
                result["calibrated_threshold_metrics"] = evaluate_task_at_calibrated_thresholds(
                    task,
                    image_ids,
                    records,
                    sims,
                    calibration_rows,
                    calibration.calibration_id,
                )
                # Per-query sizes are chart data: they go into curves.json, never
                # into eval.json rows, which aggregation flattens into CSV columns.
                curves["calibrated_candidate_sizes"] = calibrated_candidate_sizes(
                    task,
                    image_ids,
                    records,
                    sims,
                    calibration_rows,
                )
                if threshold_top_k_floors:
                    result["calibrated_threshold_top_k_floor_metrics"] = (
                        evaluate_task_at_calibrated_threshold_top_k_floors(
                            task,
                            image_ids,
                            records,
                            sims,
                            calibration_rows,
                            calibration.calibration_id,
                            threshold_top_k_floors,
                        )
                    )
            if top_k_calibration is not None:
                result["calibrated_top_k_metrics"] = evaluate_task_at_calibrated_top_k(
                    task,
                    image_ids,
                    records,
                    sims,
                    top_k_calibration_rows,
                    top_k_calibration.calibration_id,
                )
            profile_config = (exp.raw.get("evaluation") or {}).get("profile_diagnostics")
            if split is not None and isinstance(profile_config, dict) and profile_config.get("enabled", False):
                result["profile_diagnostics"] = _profile_diagnostics(
                    task,
                    image_ids,
                    records,
                    sims,
                    eval_opts.top_k,
                    calibration_rows,
                    calibration.calibration_id if calibration is not None else None,
                    top_k_calibration_rows,
                    top_k_calibration.calibration_id if top_k_calibration is not None else None,
                    split,
                    profile_config,
                )
                if not result["profile_diagnostics"]:
                    message = (
                        f"Task {task['task_id']} has no candidate metadata for profile diagnostic "
                        f"field {profile_config.get('candidate_metadata_field', 'profile')!r}"
                    )
                    if profile_config.get("required", False):
                        raise ValueError(message)
                    warnings.append(message)
            task_results.append(result)
            curve_results.append(curves)
            warnings.extend(curves.get("warnings", []))
        eval_seconds = time.perf_counter() - eval_started
    retrieval_resources = retrieval_monitor.summary()
    sim_meta = _write_similarity_cache(
        model_dir,
        sims,
        image_ids,
        exp,
        model,
        emb_meta,
        run_opts,
        similarity_precision,
    )
    curves_json = {"schema_version": 1, "model_id": model.model_id, "model_revision": model.revision, "tasks": curve_results}
    atomic_write_json(model_dir / "curves.json", curves_json)
    resource_json = _resource_json(
        exp,
        model,
        embeddings,
        emb_meta,
        retrieval_resources,
        similarity_seconds,
        eval_seconds,
        task_results,
        similarity_precision,
    )
    _write_resource_metrics(model_dir, resource_json)
    eval_json = {
        "schema_version": 1,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_storage_key": model.storage_key,
        "embedding": {
            "path": "embeddings.npy",
            "metadata_path": "embeddings_metadata.json",
            "num_images": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "normalized": bool(emb_meta.get("normalized", True)),
            "embedding_sha256": emb_meta.get("embedding_sha256"),
        },
        "evaluation": {
            "similarity": eval_opts.similarity,
            "top_k": eval_opts.top_k,
            "target_pc": eval_opts.target_pc,
            "similarity_cache": sim_meta,
            **(
                {
                    "threshold_calibration": {
                        "calibration_id": calibration.calibration_id,
                        "method": calibration.method,
                        "task_id": calibration.task["task_id"],
                        "target_pc": calibration.target_pc,
                    }
                }
                if calibration is not None
                else {}
            ),
            **(
                {
                    "top_k_calibration": {
                        "calibration_id": top_k_calibration.calibration_id,
                        "method": top_k_calibration.method,
                        "task_id": top_k_calibration.task["task_id"],
                        "target_pc": top_k_calibration.target_pc,
                        "on_unattained": top_k_calibration.on_unattained,
                    }
                }
                if top_k_calibration is not None
                else {}
            ),
            **(
                {
                    "threshold_top_k_floor": {
                        "method": "frozen_calibrated_threshold_or_top_k_floor",
                        "threshold_source": "pure_calibrated_threshold",
                        "minimum_k": threshold_top_k_floors,
                    }
                }
                if threshold_top_k_floors
                else {}
            ),
        },
        "artifacts": {
            "resource_metrics": "resource_metrics.json",
            **({"threshold_calibration": "threshold_calibration.json"} if calibration is not None else {}),
            **({"top_k_calibration": "top_k_calibration.json"} if top_k_calibration is not None else {}),
        },
        "tasks": task_results,
        "warnings": warnings,
    }
    atomic_write_json(model_dir / "eval.json", eval_json)
    return model_dir / "eval.json"


def _resource_json(
    exp: ExperimentConfig,
    model,
    embeddings: np.ndarray,
    emb_meta: dict[str, Any],
    retrieval_resources: dict[str, Any],
    similarity_seconds: float,
    eval_seconds: float,
    task_results: list[dict[str, Any]],
    similarity_precision: dict[str, Any],
) -> dict[str, Any]:
    embedding_resources = dict((emb_meta.get("resources") or {}).get("embedding") or {})
    model_profile = dict((emb_meta.get("resources") or {}).get("model") or {})
    retrieval_seconds = float(retrieval_resources.get("wall_time_seconds") or 0.0)
    num_queries = int(sum(int(task.get("num_queries") or 0) for task in task_results))
    system = system_profile()
    summary = {
        "embedding_time_per_100_images_seconds": embedding_resources.get(
            "time_per_100_images_seconds"
        ),
        "retrieval_latency_per_query_seconds": (
            retrieval_seconds / num_queries if num_queries else None
        ),
        "gpu_memory_peak_used_bytes": _peak_gpu_used(embedding_resources, retrieval_resources),
        "embedding_dimensionality": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "model_size": model_profile.get("model_size_label"),
        "parameter_count": model_profile.get("parameter_count"),
        "inference_engine": model_profile.get("inference_engine"),
        "cpu_cores": system.get("cpu_cores"),
        "gpu": system.get("gpu"),
        "memory_total_bytes": system.get("memory_total_bytes"),
        "worker_threads": system.get("worker_threads"),
    }
    return {
        "schema_version": 1,
        "experiment_id": exp.experiment_id,
        "dataset_id": exp.dataset.dataset_id,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_storage_key": model.storage_key,
        "created_at": utc_now_iso(),
        "summary": summary,
        "system": system,
        "model": model_profile,
        "embedding": {
            **embedding_resources,
            "cache_reused_for_run": bool(emb_meta.get("cache_reused_for_run")),
            "embedding_dimensionality": summary["embedding_dimensionality"],
        },
        "retrieval": {
            **retrieval_resources,
            "precision": dict(similarity_precision),
            "num_queries": num_queries,
            "similarity_time_seconds": float(similarity_seconds),
            "evaluation_time_seconds": float(eval_seconds),
            "latency_per_query_seconds": summary["retrieval_latency_per_query_seconds"],
        },
    }


def _peak_gpu_used(*blocks: dict[str, Any]) -> int | None:
    values: list[int] = []
    for block in blocks:
        gpu = block.get("gpu_memory") if isinstance(block, dict) else None
        if not isinstance(gpu, dict):
            continue
        value = gpu.get("nvidia_smi_peak_used_bytes")
        if value is not None:
            values.append(int(value))
    return max(values) if values else None


def _write_resource_metrics(model_dir: Path, resource_json: dict[str, Any]) -> None:
    atomic_write_json(model_dir / "resource_metrics.json", resource_json)
    run_dir = model_dir.parents[1]
    atomic_write_json(run_dir / "resource_metrics" / f"{model_dir.name}.json", resource_json)


def _write_similarity_cache(
    model_dir: Path,
    sims: np.ndarray,
    image_ids: list[str],
    exp: ExperimentConfig,
    model,
    emb_meta: dict[str, Any],
    run_opts: dict[str, Any],
    similarity_precision: dict[str, Any],
) -> dict[str, Any]:
    path = model_dir / "similarity_cache.npy"
    if bool(run_opts.get("save_similarity_cache", True)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.save(handle, sims)
        tmp.replace(path)
    meta = {
        "path": "similarity_cache.npy" if path.exists() else None,
        "image_ids": image_ids,
        "image_ids_hash": sha256_json(image_ids),
        "embedding_sha256": emb_meta.get("embedding_sha256"),
        "similarity_function": "cosine_via_normalized_dot_product",
        "precision": dict(similarity_precision),
    }
    meta["similarity_cache_key"] = sha256_json(
        {
            "embedding_sha256": meta["embedding_sha256"],
            "image_ids_hash": meta["image_ids_hash"],
            "similarity_function": meta["similarity_function"],
            "precision": meta["precision"],
        }
    )
    atomic_write_json(model_dir / "similarity_cache_metadata.json", meta)
    return {key: value for key, value in meta.items() if key != "image_ids"}


def _run_json(
    exp: ExperimentConfig,
    split: dict[str, Any],
    run_dir: Path,
    db_sha: str,
    model_runs: list[dict[str, Any]],
    *,
    parent_training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_meta = dict(split.get("dataset") or {})
    identity = dict(dataset_meta.get("identity") or {})
    counts = {key: identity[key] for key in ("num_classes", "num_images") if key in identity}
    resolved_path = run_dir / "resolved_config.yml"
    split_path = run_dir / "split.snapshot.json"
    artifacts = {
        "aggregate_metrics_csv": "aggregate_metrics.csv",
        "aggregate_metrics_json": "aggregate_metrics.json",
        "latex_tables": "latex_tables/tables.tex",
        "latex_fixed_k_table": "latex_tables/fixed_k.tex",
    }
    if (run_dir / "latex_tables" / "calibrated_threshold.tex").is_file():
        artifacts["latex_calibrated_threshold_table"] = "latex_tables/calibrated_threshold.tex"
    if (run_dir / "calibration_protocol.json").is_file():
        artifacts["calibration_protocol"] = "calibration_protocol.json"
    if (run_dir / "top_k_calibration_protocol.json").is_file():
        artifacts["top_k_calibration_protocol"] = "top_k_calibration_protocol.json"
    has_failures = any(model_run.get("status") == "failed" for model_run in model_runs)
    if has_failures:
        artifacts["model_failures"] = "MODEL_FAILURES.txt"
    result = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "experiment_id": exp.experiment_id,
        "status": "failed" if has_failures else "completed",
        "created_at": utc_now_iso(),
        "code": {**git_info(exp.repo_root), "dependencies": dependency_versions()},
        "config": {
            "resolved_config_path": "resolved_config.yml",
            "config_sha256": sha256_file(resolved_path),
            "evaluation_protocol_sha256": sha256_json(exp.raw.get("evaluation") or {}),
        },
        "dataset": {
            "dataset_id": str(dataset_meta.get("dataset_id") or exp.dataset.dataset_id),
            "source_db_sha256": db_sha,
            "image_root": str(exp.dataset.image_root) if exp.dataset.image_root else dataset_meta.get("image_root"),
            **counts,
        },
        "split": {
            "split_id": split["split_id"],
            "split_path": "split.snapshot.json",
            "split_sha256": sha256_file(split_path),
            "split_seed": (split.get("split_strategy") or {}).get("seed"),
            "split_strategy": split.get("split_strategy"),
            "coverage_filter": split.get("coverage_filter"),
        },
        "model_runs": model_runs,
        "artifacts": artifacts,
    }
    if parent_training is not None:
        result["parent_training"] = dict(parent_training)
    return result
