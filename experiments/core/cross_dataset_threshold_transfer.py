"""Cross-dataset threshold-transfer evaluation.

This is intentionally a separate experiment type from the ordinary evaluator.
A model-specific similarity threshold is selected on one declared calibration
dataset, frozen, and then applied unchanged to retrieval tasks from a distinct
evaluation dataset.  Ordinary experiment loading and execution semantics remain
unchanged.
"""

from __future__ import annotations

import copy
import gc
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .artifacts import (
    atomic_write_json,
    prepare_run_dir,
    sha256_file,
    sha256_json,
)
from .config_schema import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    _runtime_dataset_config,
    _validate_retrieval_tasks,
    load_experiment_config,
    resolve_path,
)
from .embedding_pipeline import prepare_embeddings
from .eval_pipeline import (
    calibrated_candidate_sizes,
    cosine_cache,
    evaluate_task,
    evaluate_task_at_calibrated_thresholds,
)
from .preflight import preflight_huggingface_revisions
from .resource_metrics import ResourceMonitor, release_torch_cuda_memory
from .split_schema import (
    load_split,
    records_from_split,
    resolve_retrieval_tasks,
    validate_split_manifest,
)

EXPERIMENT_TYPE = "cross_dataset_threshold_transfer"


@dataclass(frozen=True)
class CrossDatasetCalibration:
    method: str
    calibration_id: str
    target_pc: list[float]
    dataset: DatasetConfig
    split_file: Path
    split: dict[str, Any]
    task: dict[str, Any]


@dataclass(frozen=True)
class CrossDatasetExperiment:
    experiment: ExperimentConfig
    evaluation_split: dict[str, Any]
    evaluation_tasks: list[dict[str, Any]]
    calibration: CrossDatasetCalibration


_ALLOWED_CALIBRATION_KEYS = {
    "dataset",
    "split",
    "method",
    "calibration_id",
    "target_pc",
    "task",
}


def is_cross_dataset_experiment(raw: dict[str, Any]) -> bool:
    return str(raw.get("experiment_type") or "") == EXPERIMENT_TYPE


def _require_mapping(data: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping in {source}")
    return dict(value)


def _parse_calibration(
    exp: ExperimentConfig,
    evaluation_split: dict[str, Any],
) -> CrossDatasetCalibration:
    raw = _require_mapping(exp.raw, "calibration", exp.path)
    unknown = sorted(set(raw) - _ALLOWED_CALIBRATION_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown cross-dataset calibration key(s) in {exp.path}: {', '.join(unknown)}"
        )
    method = str(raw.get("method") or "")
    if method != "empirical_target_pc":
        raise ValueError(
            "cross-dataset calibration.method must be 'empirical_target_pc' "
            f"in {exp.path}"
        )
    calibration_id = str(raw.get("calibration_id") or "").strip()
    if not calibration_id:
        raise ValueError(f"calibration.calibration_id is required in {exp.path}")
    targets_raw = raw.get("target_pc")
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError(f"calibration.target_pc must be a non-empty list in {exp.path}")
    targets = [float(value) for value in targets_raw]
    if any(value <= 0.0 or value > 1.0 for value in targets):
        raise ValueError(f"calibration.target_pc values must be in (0, 1] in {exp.path}")
    if targets != sorted(set(targets)):
        raise ValueError(
            f"calibration.target_pc must be unique and increasing in {exp.path}"
        )

    dataset_raw = _require_mapping(raw, "dataset", exp.path)
    split_raw = _require_mapping(raw, "split", exp.path)
    split_value = split_raw.get("file")
    if not split_value:
        raise ValueError(f"calibration.split.file is required in {exp.path}")
    split_file = resolve_path(str(split_value), exp.path.parent)
    dataset = _runtime_dataset_config(
        {"dataset": dataset_raw}, split_file, exp.path, exp.repo_root
    )
    split = load_split(split_file)
    validate_split_manifest(split)
    task_raw = raw.get("task")
    if not isinstance(task_raw, dict):
        raise ValueError(f"calibration.task must be a mapping in {exp.path}")
    _validate_retrieval_tasks([task_raw], exp.path)
    for selector_name in ("query", "candidates"):
        subset = str((task_raw.get(selector_name) or {}).get("subset") or "")
        if subset.strip().lower() in {"all_subsets", "all-subsets", "allsubsets", "*"}:
            raise ValueError(
                f"calibration.task.{selector_name} must select one explicit subset in {exp.path}"
            )
    task = resolve_retrieval_tasks(split, [task_raw])[0]

    evaluation_dataset_id = str((evaluation_split.get("dataset") or {}).get("dataset_id") or "")
    calibration_dataset_id = str((split.get("dataset") or {}).get("dataset_id") or "")
    if not evaluation_dataset_id or not calibration_dataset_id:
        raise ValueError("Both calibration and evaluation splits must declare dataset.dataset_id")
    if evaluation_dataset_id == calibration_dataset_id:
        raise ValueError(
            f"{EXPERIMENT_TYPE} requires distinct calibration and evaluation dataset IDs; "
            f"both are {evaluation_dataset_id!r}"
        )

    return CrossDatasetCalibration(
        method=method,
        calibration_id=calibration_id,
        target_pc=targets,
        dataset=dataset,
        split_file=split_file,
        split=split,
        task=task,
    )


def _validate_evaluation_block(exp: ExperimentConfig) -> None:
    evaluation = dict(exp.raw.get("evaluation") or {})
    forbidden = [
        key
        for key in ("threshold_calibration", "top_k_calibration")
        if evaluation.get(key) is not None
    ]
    if forbidden:
        raise ValueError(
            f"{EXPERIMENT_TYPE} owns threshold selection; evaluation must not configure: "
            + ", ".join(forbidden)
        )
    if evaluation.get("target_pc") not in (None, []):
        raise ValueError(
            f"{EXPERIMENT_TYPE} does not permit evaluation.target_pc because it would "
            "select oracle thresholds on the held-out evaluation set"
        )
    if evaluation.get("random_baseline") is not None:
        raise ValueError(
            f"{EXPERIMENT_TYPE} does not support evaluation.random_baseline because an "
            "analytical random baseline has no transferable similarity-score threshold"
        )


def _resolved_experiment(
    exp: ExperimentConfig,
    calibration: CrossDatasetCalibration,
) -> ExperimentConfig:
    resolved = copy.deepcopy(exp.resolved)
    raw_calibration = copy.deepcopy(exp.raw["calibration"])
    raw_calibration["dataset"] = {
        **dict(raw_calibration.get("dataset") or {}),
        "dataset_id": calibration.dataset.dataset_id,
        "path": str(calibration.dataset.path),
        "dataset_db": str(calibration.dataset.dataset_db),
        "image_root": (
            str(calibration.dataset.image_root)
            if calibration.dataset.image_root is not None
            else None
        ),
    }
    raw_calibration["split"] = {
        **dict(raw_calibration.get("split") or {}),
        "file": str(calibration.split_file),
    }
    resolved["calibration"] = raw_calibration
    resolved["experiment_type"] = EXPERIMENT_TYPE
    return replace(exp, resolved=resolved)


def _context_from_loaded_experiment(exp: ExperimentConfig) -> CrossDatasetExperiment:
    if not is_cross_dataset_experiment(exp.raw):
        raise ValueError(
            f"Experiment {exp.path} must set experiment_type: {EXPERIMENT_TYPE}"
        )
    _validate_evaluation_block(exp)
    evaluation_split = load_split(exp.split_file)
    validate_split_manifest(evaluation_split)
    evaluation_tasks = resolve_retrieval_tasks(
        evaluation_split, (exp.raw.get("evaluation") or {}).get("retrieval_tasks")
    )
    calibration = _parse_calibration(exp, evaluation_split)
    evaluation_task_ids = {str(task["task_id"]) for task in evaluation_tasks}
    if calibration.task["task_id"] in evaluation_task_ids:
        raise ValueError(
            f"Calibration task ID {calibration.task['task_id']!r} must differ from "
            "every evaluation task ID"
        )
    return CrossDatasetExperiment(
        experiment=_resolved_experiment(exp, calibration),
        evaluation_split=evaluation_split,
        evaluation_tasks=evaluation_tasks,
        calibration=calibration,
    )


def load_cross_dataset_experiment(
    path: Path | str,
    *,
    injected_models: list[ModelConfig] | None = None,
    allow_dynamic_models: bool = False,
    split_file_override: Path | str | None = None,
) -> CrossDatasetExperiment:
    exp = load_experiment_config(
        path,
        injected_models=injected_models,
        allow_dynamic_models=allow_dynamic_models,
        split_file_override=split_file_override,
    )
    return _context_from_loaded_experiment(exp)


def _require_image_root(dataset: DatasetConfig) -> None:
    if dataset.image_root is None:
        raise ValueError(f"Dataset {dataset.dataset_id!r} must configure an image_root")
    if not Path(dataset.image_root).is_dir():
        raise FileNotFoundError(f"Dataset image_root not found: {dataset.image_root}")


def _positive_pair_count(task: dict[str, Any], records: dict[str, Any]) -> int:
    candidate_ids = set(task["candidate_ids"])
    candidate_counts = Counter(records[file_id].class_id for file_id in candidate_ids)
    total = 0
    for query_id in task["query_ids"]:
        count = candidate_counts[records[query_id].class_id]
        if task.get("exclude_self") and query_id in candidate_ids:
            count -= 1
        total += max(count, 0)
    return total


def _source_metadata(dataset: DatasetConfig, split: dict[str, Any]) -> dict[str, Any]:
    dataset_meta = dict(split.get("dataset") or {})
    return {
        "dataset_id": str(dataset_meta.get("dataset_id") or dataset.dataset_id),
        "source_db_sha256": str(dataset_meta.get("source_db_sha256") or ""),
        "image_root": str(dataset.image_root) if dataset.image_root is not None else None,
        "split_id": split["split_id"],
        "split_seed": (split.get("split_strategy") or {}).get("seed"),
        "split_strategy": split.get("split_strategy"),
        "coverage_filter": split.get("coverage_filter"),
    }


def _preflight_context(context: CrossDatasetExperiment, *, require_roots: bool) -> dict[str, Any]:
    exp = context.experiment
    if require_roots:
        _require_image_root(exp.dataset)
        _require_image_root(context.calibration.dataset)
    evaluation_records = records_from_split(context.evaluation_split)
    calibration_records = records_from_split(context.calibration.split)
    calibration_pairs = _positive_pair_count(context.calibration.task, calibration_records)
    if calibration_pairs <= 0:
        raise ValueError("Cross-dataset calibration task has no positive pairs")
    for task in context.evaluation_tasks:
        if _positive_pair_count(task, evaluation_records) <= 0:
            raise ValueError(f"Evaluation task {task['task_id']!r} has no positive pairs")
    return {
        "experiment_id": exp.experiment_id,
        "experiment_type": EXPERIMENT_TYPE,
        "evaluation": {
            **_source_metadata(exp.dataset, context.evaluation_split),
            "num_tasks": len(context.evaluation_tasks),
            "task_counts": [
                {
                    "task_id": task["task_id"],
                    "num_queries": len(task["query_ids"]),
                    "num_candidates": len(task["candidate_ids"]),
                }
                for task in context.evaluation_tasks
            ],
        },
        "calibration": {
            **_source_metadata(context.calibration.dataset, context.calibration.split),
            "calibration_id": context.calibration.calibration_id,
            "task_id": context.calibration.task["task_id"],
            "target_pc": context.calibration.target_pc,
            "num_queries": len(context.calibration.task["query_ids"]),
            "num_candidates": len(context.calibration.task["candidate_ids"]),
            "num_positive_pairs": calibration_pairs,
        },
    }


def validate_cross_dataset_experiment(
    path: Path | str,
    *,
    allow_dynamic_models: bool = True,
    require_image_roots: bool = True,
) -> dict[str, Any]:
    context = load_cross_dataset_experiment(
        path, allow_dynamic_models=allow_dynamic_models
    )
    result = _preflight_context(context, require_roots=require_image_roots)
    result["models"] = [model.model_id for model in context.experiment.models]
    result["dynamic_checkpoint_model"] = bool(
        (context.experiment.raw.get("models") or {}).get("from_finetuning_checkpoint")
    )
    result["huggingface_revisions"] = preflight_huggingface_revisions(
        context.experiment
    )
    return result


def preflight_loaded_cross_dataset_experiment(
    exp: ExperimentConfig,
    *,
    require_image_roots: bool = True,
) -> dict[str, Any]:
    context = _context_from_loaded_experiment(exp)
    return _preflight_context(context, require_roots=require_image_roots)


def _write_inputs(context: CrossDatasetExperiment, run_dir: Path) -> None:
    (run_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(context.experiment.resolved, sort_keys=False),
        encoding="utf-8",
    )
    atomic_write_json(run_dir / "split.snapshot.json", context.evaluation_split)
    atomic_write_json(
        run_dir / "calibration_split.snapshot.json", context.calibration.split
    )


def _write_protocol(
    context: CrossDatasetExperiment,
    run_dir: Path,
    calibration_records: dict[str, Any],
) -> None:
    calibration = context.calibration
    positive_pairs = _positive_pair_count(calibration.task, calibration_records)
    report = {
        "schema_version": 1,
        "experiment_type": EXPERIMENT_TYPE,
        "method": calibration.method,
        "calibration_id": calibration.calibration_id,
        "target_pc": calibration.target_pc,
        "calibration_task_id": calibration.task["task_id"],
        "calibration_source": {
            **_source_metadata(calibration.dataset, calibration.split),
            "split_path": "calibration_split.snapshot.json",
            "split_sha256": sha256_file(run_dir / "calibration_split.snapshot.json"),
            "num_queries": len(calibration.task["query_ids"]),
            "num_candidates": len(calibration.task["candidate_ids"]),
            "num_positive_pairs": positive_pairs,
            "query_ids_hash": sha256_json(calibration.task["query_ids"]),
            "candidate_ids_hash": sha256_json(calibration.task["candidate_ids"]),
        },
        "evaluation_source": {
            **_source_metadata(context.experiment.dataset, context.evaluation_split),
            "split_path": "split.snapshot.json",
            "split_sha256": sha256_file(run_dir / "split.snapshot.json"),
            "task_ids": [task["task_id"] for task in context.evaluation_tasks],
        },
        "checks": [
            {
                "check": "calibration_has_positive_pairs",
                "ok": positive_pairs > 0,
                "num_positive_pairs": positive_pairs,
            },
            {
                "check": "distinct_dataset_namespaces",
                "ok": calibration.dataset.dataset_id
                != context.experiment.dataset.dataset_id,
                "calibration_dataset_id": calibration.dataset.dataset_id,
                "evaluation_dataset_id": context.experiment.dataset.dataset_id,
            },
            {
                "check": "threshold_transfer_is_one_way",
                "ok": True,
                "detail": "calibration labels select thresholds; evaluation labels never select thresholds",
            },
        ],
        "passed": positive_pairs > 0,
    }
    atomic_write_json(run_dir / "calibration_protocol.json", report)


def _calibrate_model(
    context: CrossDatasetExperiment,
    model: ModelConfig,
    model_dir: Path,
    calibration_records: dict[str, Any],
    calibration_image_ids: list[str],
    calibration_db_sha: str,
    run_options: dict[str, Any],
    embedding_options: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .runner import _evaluation_options

    calibration_dir = model_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    embeddings, embedding_meta = prepare_embeddings(
        context.calibration.dataset,
        model,
        calibration_image_ids,
        calibration_records,
        calibration_dir,
        run_options,
        embedding_options,
        calibration_db_sha,
    )
    eval_options = _evaluation_options(context.experiment)
    with ResourceMonitor() as monitor:
        similarity_started = time.perf_counter()
        similarities, precision = cosine_cache(
            embeddings,
            eval_options.device,
            eval_options.gpu_dtype,
            eval_options.cache_dtype,
            inputs_normalized=bool(embedding_meta.get("normalized")),
            fp32_matmul_precision=eval_options.fp32_matmul_precision,
            return_metadata=True,
        )
        similarity_seconds = time.perf_counter() - similarity_started
        evaluation_started = time.perf_counter()
        result, _ = evaluate_task(
            context.calibration.task,
            calibration_image_ids,
            calibration_records,
            similarities,
            [1],
            True,
            context.calibration.target_pc,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
    rows = list(result["target_pc_metrics"])
    atomic_write_json(
        model_dir / "threshold_calibration.json",
        {
            "schema_version": 1,
            "experiment_type": EXPERIMENT_TYPE,
            "method": context.calibration.method,
            "calibration_id": context.calibration.calibration_id,
            "task_id": context.calibration.task["task_id"],
            "source": {
                **_source_metadata(
                    context.calibration.dataset, context.calibration.split
                ),
                "split_sha256": sha256_file(
                    model_dir.parents[1] / "calibration_split.snapshot.json"
                ),
                "embedding_sha256": embedding_meta.get("embedding_sha256"),
            },
            "query_ids_hash": result["query_ids_hash"],
            "candidate_ids_hash": result["candidate_ids_hash"],
            "positive_pairs_hash": result["positive_pairs_hash"],
            "num_queries": result["num_queries"],
            "num_candidates": result["num_candidates"],
            "num_positive_pairs": result["num_positive_pairs"],
            "operating_points": rows,
        },
    )
    resource = {
        "dataset_id": context.calibration.dataset.dataset_id,
        "split_id": context.calibration.split["split_id"],
        "num_images": int(embeddings.shape[0]),
        "embedding": dict((embedding_meta.get("resources") or {}).get("embedding") or {}),
        "embedding_cache_reused_for_run": bool(
            embedding_meta.get("cache_reused_for_run")
        ),
        "embedding_sha256": embedding_meta.get("embedding_sha256"),
        "retrieval": {
            **monitor.summary(),
            "precision": precision,
            "similarity_time_seconds": similarity_seconds,
            "evaluation_time_seconds": evaluation_seconds,
            "num_queries": int(result["num_queries"]),
        },
    }
    del similarities
    del embeddings
    gc.collect()
    return rows, resource


def _run_model(
    context: CrossDatasetExperiment,
    model: ModelConfig,
    model_dir: Path,
    evaluation_records: dict[str, Any],
    evaluation_image_ids: list[str],
    evaluation_db_sha: str,
    calibration_records: dict[str, Any],
    calibration_image_ids: list[str],
    calibration_db_sha: str,
) -> Path:
    from .runner import (
        _absolute_embedding_options,
        _evaluation_options,
        _profile_diagnostics,
        _resource_json,
        _write_resource_metrics,
        _write_similarity_cache,
    )

    exp = context.experiment
    run_options = dict(exp.raw.get("run") or {})
    evaluation_options = _evaluation_options(exp)
    embedding_options = _absolute_embedding_options(exp)
    calibration_rows, calibration_resource = _calibrate_model(
        context,
        model,
        model_dir,
        calibration_records,
        calibration_image_ids,
        calibration_db_sha,
        run_options,
        embedding_options,
    )

    embeddings, embedding_meta = prepare_embeddings(
        exp.dataset,
        model,
        evaluation_image_ids,
        evaluation_records,
        model_dir,
        run_options,
        embedding_options,
        evaluation_db_sha,
    )
    task_results: list[dict[str, Any]] = []
    curve_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    with ResourceMonitor() as retrieval_monitor:
        similarity_started = time.perf_counter()
        similarities, similarity_precision = cosine_cache(
            embeddings,
            evaluation_options.device,
            evaluation_options.gpu_dtype,
            evaluation_options.cache_dtype,
            inputs_normalized=bool(embedding_meta.get("normalized")),
            fp32_matmul_precision=evaluation_options.fp32_matmul_precision,
            return_metadata=True,
        )
        similarity_seconds = time.perf_counter() - similarity_started
        evaluation_started = time.perf_counter()
        for task in context.evaluation_tasks:
            result, curves = evaluate_task(
                task,
                evaluation_image_ids,
                evaluation_records,
                similarities,
                evaluation_options.top_k,
                bool(run_options.get("fail_on_empty_positives", True)),
                [],
            )
            result["artifacts"] = {
                "similarity_cache": "similarity_cache.npy",
                "curves": "curves.json",
                "resource_metrics": "resource_metrics.json",
                "threshold_calibration": "threshold_calibration.json",
            }
            result["calibrated_threshold_metrics"] = (
                evaluate_task_at_calibrated_thresholds(
                    task,
                    evaluation_image_ids,
                    evaluation_records,
                    similarities,
                    calibration_rows,
                    context.calibration.calibration_id,
                )
            )
            curves["calibrated_candidate_sizes"] = calibrated_candidate_sizes(
                task,
                evaluation_image_ids,
                evaluation_records,
                similarities,
                calibration_rows,
            )
            profile_config = (exp.raw.get("evaluation") or {}).get(
                "profile_diagnostics"
            )
            if isinstance(profile_config, dict) and profile_config.get("enabled", False):
                result["profile_diagnostics"] = _profile_diagnostics(
                    task,
                    evaluation_image_ids,
                    evaluation_records,
                    similarities,
                    evaluation_options.top_k,
                    calibration_rows,
                    context.calibration.calibration_id,
                    [],
                    None,
                    context.evaluation_split,
                    profile_config,
                )
                if not result["profile_diagnostics"]:
                    message = (
                        f"Task {task['task_id']} has no candidate metadata for profile "
                        f"diagnostic field {profile_config.get('candidate_metadata_field', 'profile')!r}"
                    )
                    if profile_config.get("required", False):
                        raise ValueError(message)
                    warnings.append(message)
            task_results.append(result)
            curve_results.append(curves)
            warnings.extend(curves.get("warnings", []))
        evaluation_seconds = time.perf_counter() - evaluation_started
    retrieval_resources = retrieval_monitor.summary()
    similarity_meta = _write_similarity_cache(
        model_dir,
        similarities,
        evaluation_image_ids,
        exp,
        model,
        embedding_meta,
        run_options,
        similarity_precision,
    )
    atomic_write_json(
        model_dir / "curves.json",
        {
            "schema_version": 1,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "tasks": curve_results,
        },
    )
    resource_json = _resource_json(
        exp,
        model,
        embeddings,
        embedding_meta,
        retrieval_resources,
        similarity_seconds,
        evaluation_seconds,
        task_results,
        similarity_precision,
    )
    resource_json["experiment_type"] = EXPERIMENT_TYPE
    resource_json["calibration"] = calibration_resource
    _write_resource_metrics(model_dir, resource_json)
    atomic_write_json(
        model_dir / "eval.json",
        {
            "schema_version": 1,
            "experiment_type": EXPERIMENT_TYPE,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "model_storage_key": model.storage_key,
            "embedding": {
                "path": "embeddings.npy",
                "metadata_path": "embeddings_metadata.json",
                "num_images": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "normalized": bool(embedding_meta.get("normalized", True)),
                "embedding_sha256": embedding_meta.get("embedding_sha256"),
            },
            "calibration_embedding": {
                "path": "calibration/embeddings.npy",
                "metadata_path": "calibration/embeddings_metadata.json",
                "num_images": len(calibration_image_ids),
                "embedding_sha256": calibration_resource.get("embedding_sha256"),
            },
            "evaluation": {
                "similarity": evaluation_options.similarity,
                "top_k": evaluation_options.top_k,
                "target_pc": [],
                "similarity_cache": similarity_meta,
                "threshold_calibration": {
                    "transfer_type": "cross_dataset",
                    "calibration_id": context.calibration.calibration_id,
                    "method": context.calibration.method,
                    "task_id": context.calibration.task["task_id"],
                    "target_pc": context.calibration.target_pc,
                    "source_dataset_id": context.calibration.dataset.dataset_id,
                    "source_split_id": context.calibration.split["split_id"],
                },
            },
            "artifacts": {
                "resource_metrics": "resource_metrics.json",
                "threshold_calibration": "threshold_calibration.json",
                "calibration_embeddings": "calibration/embeddings.npy",
            },
            "tasks": task_results,
            "warnings": warnings,
        },
    )
    return model_dir / "eval.json"


def _finalize_run(
    context: CrossDatasetExperiment,
    run_dir: Path,
    model_runs: list[dict[str, Any]],
    failed_models: list[dict[str, Any]],
    *,
    parent_training: dict[str, Any] | None,
) -> None:
    from .aggregate_runs import aggregate_paths
    from .latex_tables import write_run_latex_tables
    from .runner import _run_json

    exp = context.experiment
    evaluation_db_sha = str(
        (context.evaluation_split.get("dataset") or {}).get("source_db_sha256")
        or ""
    )
    run_json = _run_json(
        exp,
        context.evaluation_split,
        run_dir,
        evaluation_db_sha,
        model_runs,
        parent_training=parent_training,
    )
    calibration_snapshot = run_dir / "calibration_split.snapshot.json"
    run_json["experiment_type"] = EXPERIMENT_TYPE
    run_json["config"]["evaluation_protocol_sha256"] = sha256_json(
        {
            "experiment_type": EXPERIMENT_TYPE,
            "calibration": exp.resolved.get("calibration"),
            "evaluation": exp.raw.get("evaluation") or {},
        }
    )
    run_json["calibration_source"] = {
        **_source_metadata(context.calibration.dataset, context.calibration.split),
        "split_path": "calibration_split.snapshot.json",
        "split_sha256": sha256_file(calibration_snapshot),
        "calibration_id": context.calibration.calibration_id,
        "task_id": context.calibration.task["task_id"],
        "target_pc": context.calibration.target_pc,
    }
    run_json["artifacts"]["calibration_split"] = "calibration_split.snapshot.json"
    run_json["artifacts"]["calibration_protocol"] = "calibration_protocol.json"
    atomic_write_json(run_dir / "run.json", run_json)
    try:
        aggregate_paths(
            [run_dir],
            run_dir / "aggregate_metrics.csv",
            run_dir / "aggregate_metrics.json",
            include_incomplete=bool(failed_models),
        )
    except Exception as exc:
        atomic_write_json(run_dir / "aggregate_warning.json", {"warning": str(exc)})
    write_run_latex_tables(run_dir)
    try:
        from .run_charts import write_run_charts

        write_run_charts(run_dir)
    except Exception as exc:
        atomic_write_json(
            run_dir / "charts_warning.json",
            {
                "warnings": [
                    {
                        "stage": "charts",
                        "warning": f"{type(exc).__name__}: {exc}",
                    }
                ]
            },
        )


def run_cross_dataset_experiment(
    experiment_path: Path | str,
    model_filter: str | None = None,
    run_id: str | None = None,
    *,
    injected_models: list[ModelConfig] | None = None,
    output_root_override: Path | str | None = None,
    parent_training: dict[str, Any] | None = None,
    split_file_override: Path | str | None = None,
) -> Path:
    from .runner import (
        ModelEvaluationError,
        _display_name,
        _failed_model_record,
        _run_dir,
        _selected_models,
        _write_failed_models,
    )

    context = load_cross_dataset_experiment(
        experiment_path,
        injected_models=injected_models,
        split_file_override=split_file_override,
    )
    _preflight_context(context, require_roots=True)
    exp = context.experiment
    models = _selected_models(exp.models, model_filter)
    if not models:
        raise ValueError(f"Model {model_filter!r} is not included in {exp.path}")
    preflight_huggingface_revisions(exp, models)
    run_dir = prepare_run_dir(_run_dir(exp, run_id, output_root_override))
    _write_inputs(context, run_dir)

    evaluation_records = records_from_split(context.evaluation_split)
    calibration_records = records_from_split(context.calibration.split)
    evaluation_image_ids = sorted(
        {
            file_id
            for task in context.evaluation_tasks
            for file_id in [*task["query_ids"], *task["candidate_ids"]]
        }
    )
    calibration_image_ids = sorted(
        set(context.calibration.task["query_ids"])
        | set(context.calibration.task["candidate_ids"])
    )
    _write_protocol(context, run_dir, calibration_records)
    evaluation_db_sha = str(
        (context.evaluation_split.get("dataset") or {}).get("source_db_sha256")
        or ""
    )
    calibration_db_sha = str(
        (context.calibration.split.get("dataset") or {}).get("source_db_sha256")
        or ""
    )
    model_runs: list[dict[str, Any]] = []
    failed_models: list[dict[str, Any]] = []
    for model in models:
        model_dir = run_dir / "models" / model.storage_key
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            eval_path = _run_model(
                context,
                model,
                model_dir,
                evaluation_records,
                evaluation_image_ids,
                evaluation_db_sha,
                calibration_records,
                calibration_image_ids,
                calibration_db_sha,
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
            completed = {
                "model_id": model.model_id,
                "model_storage_key": model.storage_key,
                "model_revision": model.revision,
                "status": "completed",
                "path": str(eval_path.relative_to(run_dir)),
                "resource_metrics_path": f"resource_metrics/{model.storage_key}.json",
            }
            display_name = _display_name(model)
            if display_name:
                completed["display_name"] = display_name
            model_runs.append(completed)
        finally:
            release_torch_cuda_memory(reset_compiler=True)
    if failed_models:
        _write_failed_models(run_dir, failed_models)
    _finalize_run(
        context,
        run_dir,
        model_runs,
        failed_models,
        parent_training=parent_training,
    )
    if failed_models:
        raise ModelEvaluationError(run_dir, failed_models)
    return run_dir
