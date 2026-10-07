"""Frozen-backbone reference evaluation with a shared, content-addressed cache.

A *reference* is a fixed (untrained) model — currently only the frozen backbone —
evaluated on the same validation sets as a finetuning run. Its result is a
"starting point" baseline that sits before epoch 1 in reports. References are
evaluated once and cached across every sweep task that shares the same inputs.

The reference result cache is keyed independently of the embedding cache: the
embedding cache key fingerprints the frozen descriptors; the reference result key
additionally fingerprints the split, selector, retrieval tasks, and evaluation
protocol (top_k / target_pc / cosine similarity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.core.artifacts import atomic_write_json, sha256_json
from experiments.core.cache_lock import DEFAULT_TIMEOUT_SECONDS, key_lock
from experiments.core.cache_paths import reference_evaluations_dir
from experiments.core.config_schema import ModelConfig
from experiments.core.embedding_pipeline import embedding_cache_key, get_or_create_embedding_artifact
from experiments.core.eval_pipeline import evaluate_task
from experiments.core.retrieval_metrics import cosine_similarity_matrix
from experiments.core.split_schema import (
    ids_for_multirole_selector,
    records_from_split,
    resolve_retrieval_tasks,
)

from .evaluate import _primary_threshold, _union_task_ids, evaluate_embeddings
from .validation_sets import ValidationSet, primary_validation_set

# Bump when the reference-result semantics change (metric shape, similarity,
# selection of the reported target-PC row, etc.).
REFERENCE_EVALUATOR_VERSION = 3


class ReferenceCacheMiss(RuntimeError):
    """Raised when a reference result is absent and computation is disabled."""

    def __init__(self, result_key: str) -> None:
        super().__init__(f"Reference result not cached: {result_key}")
        self.result_key = result_key


@dataclass(frozen=True)
class ReferenceModel:
    id: str
    kind: str
    label: str
    model: ModelConfig
    validation_filter: frozenset[str] | None


@dataclass(frozen=True)
class ReferenceWorkItem:
    reference: ReferenceModel
    validation_set: ValidationSet
    image_ids: list[str]
    tasks: list[dict[str, Any]] | None
    task_fingerprint: list[dict[str, Any]]
    embedding_key: str
    result_key: str
    top_k: tuple[int, ...]
    target_pc: float


def resolve_reference_models(cfg: dict[str, Any], exp) -> list[ReferenceModel]:
    """Resolve configured frozen references against the experiment's models."""
    finetune_eval = dict(cfg.get("evaluation") or {})
    raw_refs = finetune_eval.get("references")
    if not raw_refs:
        return []
    models_by_id: dict[str, ModelConfig] = {model.model_id: model for model in exp.models}
    references: list[ReferenceModel] = []
    for raw in raw_refs:
        model_id = str(raw["model_id"])
        model = models_by_id.get(model_id)
        if model is None:
            raise ValueError(f"Reference model_id {model_id!r} not found in models.include")
        filter_raw = raw.get("validation_sets")
        validation_filter = frozenset(str(name) for name in filter_raw) if filter_raw else None
        references.append(
            ReferenceModel(
                id=str(raw["id"]),
                kind=str(raw["kind"]),
                label=str(raw.get("label") or raw["id"]),
                model=model,
                validation_filter=validation_filter,
            )
        )
    return references


def _task_fingerprint(
    specs: list[dict[str, Any]], resolved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fingerprint: list[dict[str, Any]] = []
    for spec, task in zip(specs, resolved):
        fingerprint.append(
            {
                "task_id": task.get("task_id"),
                "query": spec.get("query"),
                "candidates": spec.get("candidates"),
                "query_ids_hash": sha256_json(sorted(str(x) for x in task.get("query_ids", []))),
                "candidate_ids_hash": sha256_json(sorted(str(x) for x in task.get("candidate_ids", []))),
                "positive_policy": task.get("positive_policy"),
                "exclude_self": bool(task.get("exclude_self")),
            }
        )
    return fingerprint


def _resolve_ids_and_tasks(
    validation_set: ValidationSet,
) -> tuple[list[str], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Return the sorted union of required image IDs, resolved tasks, fingerprint."""
    if validation_set.retrieval_task_specs:
        specs = [dict(task) for task in validation_set.retrieval_task_specs]
        tasks = resolve_retrieval_tasks(validation_set.split, specs)
        image_ids = _union_task_ids(tasks)
        return image_ids, tasks, _task_fingerprint(specs, tasks)
    ids = ids_for_multirole_selector(
        validation_set.split,
        str(validation_set.selector["subset"]),
        validation_set.selector["roles"],
    )
    image_ids = sorted({str(value) for value in ids})
    return image_ids, None, []


def _reference_result_key(
    embedding_key: str,
    validation_set: ValidationSet,
    task_fingerprint: list[dict[str, Any]],
    top_k: tuple[int, ...],
    target_pc: float,
) -> str:
    payload = {
        "schema_version": 1,
        "reference_evaluator_version": REFERENCE_EVALUATOR_VERSION,
        "embedding_cache_key": embedding_key,
        "dataset_id": validation_set.dataset.dataset_id,
        "dataset_db_sha256": validation_set.split_sha256,
        "split_sha256": validation_set.split_sha256,
        "selector": validation_set.selector,
        "retrieval_tasks": task_fingerprint,
        "evaluation": {
            "similarity": "cosine",
            "top_k": sorted(int(k) for k in top_k),
            "target_pc": float(target_pc),
        },
    }
    return sha256_json(payload)


def _applicable_sets(
    reference: ReferenceModel, validation_sets: list[ValidationSet]
) -> list[ValidationSet]:
    if reference.validation_filter is None:
        return list(validation_sets)
    return [vset for vset in validation_sets if vset.name in reference.validation_filter]


def resolve_reference_work_items(
    exp,
    cfg: dict[str, Any],
    validation_sets: list[ValidationSet],
    references: list[ReferenceModel],
    *,
    top_k: tuple[int, ...],
    target_pc: float,
) -> list[ReferenceWorkItem]:
    """Resolve every (reference x validation set) work item and its cache keys.

    No inference is performed here; only cache keys are computed.
    """
    items: list[ReferenceWorkItem] = []
    for reference in references:
        for validation_set in _applicable_sets(reference, validation_sets):
            image_ids, tasks, fingerprint = _resolve_ids_and_tasks(validation_set)
            embedding_key = embedding_cache_key(
                validation_set.dataset,
                validation_set.split_sha256,
                image_ids,
                reference.model,
            )
            result_key = _reference_result_key(
                embedding_key, validation_set, fingerprint, top_k, target_pc
            )
            items.append(
                ReferenceWorkItem(
                    reference=reference,
                    validation_set=validation_set,
                    image_ids=image_ids,
                    tasks=tasks,
                    task_fingerprint=fingerprint,
                    embedding_key=embedding_key,
                    result_key=result_key,
                    top_k=tuple(int(k) for k in top_k),
                    target_pc=float(target_pc),
                )
            )
    return items


def reference_result_dir(result_key: str) -> Path:
    return reference_evaluations_dir() / result_key[:2] / result_key


def _reference_result_valid(result_path: Path, result_key: str) -> bool:
    if not result_path.is_file():
        return False
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("result_key") == result_key


def reference_result_cached(result_key: str) -> bool:
    return _reference_result_valid(reference_result_dir(result_key) / "result.json", result_key)


def _evaluate_reference(work_item: ReferenceWorkItem, embeddings: np.ndarray) -> dict[str, Any]:
    validation_set = work_item.validation_set
    top_k = [int(k) for k in work_item.top_k]
    target_pc = float(work_item.target_pc)
    if work_item.tasks is not None:
        similarities = cosine_similarity_matrix(
            embeddings,
            embeddings,
            device_preference="cpu",
            gpu_dtype="float32",
            inputs_normalized=True,
            fp32_matmul_precision="ieee",
        )
        records = records_from_split(validation_set.split)
        task_results: list[dict[str, Any]] = []
        curve_results: list[dict[str, Any]] = []
        for task in work_item.tasks:
            result, curves = evaluate_task(
                task, work_item.image_ids, records, similarities, top_k, True, [target_pc]
            )
            task_results.append(result)
            curve_results.append(curves)
        primary = _primary_threshold(task_results, None)
        metrics = {
            "num_images": len(work_item.image_ids),
            "image_ids": list(work_item.image_ids),
            "target_pc": target_pc,
            "primary_task_id": task_results[0]["task_id"],
            "calibrated_threshold": primary,
            "recall_at_k": list(task_results[0].get("metrics") or []),
            "retrieval_tasks": task_results,
            "curves": curve_results,
        }
    else:
        records = records_from_split(validation_set.split)
        class_ids = [str(records[image_id].class_id) for image_id in work_item.image_ids]
        metrics = evaluate_embeddings(
            work_item.image_ids,
            class_ids,
            embeddings,
            target_pc=target_pc,
            top_k=top_k,
            inputs_normalized=True,
        )
    return {
        **metrics,
        "validation_set": validation_set.name,
        "is_primary_validation": validation_set.primary,
        "dataset_id": validation_set.dataset.dataset_id,
        "split_id": validation_set.split.get("split_id"),
        "selector": validation_set.selector,
    }


def load_or_compute_reference_result(
    work_item: ReferenceWorkItem,
    *,
    run_options: dict[str, Any],
    embedding_options: dict[str, Any],
    compute: bool = True,
    lock_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], bool]:
    """Return ``(result_payload, reused)`` for a reference work item.

    Uses "get or create" semantics: reuse a valid cached result, otherwise (when
    ``compute`` is True) generate it under a per-key lock and publish atomically.
    """
    result_dir = reference_result_dir(work_item.result_key)
    result_path = result_dir / "result.json"
    metadata_path = result_dir / "metadata.json"
    if _reference_result_valid(result_path, work_item.result_key):
        return json.loads(result_path.read_text(encoding="utf-8")), True
    if not compute:
        raise ReferenceCacheMiss(work_item.result_key)
    validation_set = work_item.validation_set
    with key_lock(work_item.result_key, timeout=lock_timeout):
        if _reference_result_valid(result_path, work_item.result_key):
            return json.loads(result_path.read_text(encoding="utf-8")), True
        artifact = get_or_create_embedding_artifact(
            validation_set.dataset,
            work_item.reference.model,
            work_item.image_ids,
            records_from_split(validation_set.split),
            run_options,
            embedding_options,
            validation_set.split_sha256,
            cache_policy="reuse_if_config_hash_matches",
            lock_timeout=lock_timeout,
        )
        embeddings = np.load(artifact.embeddings_path)
        metrics = _evaluate_reference(work_item, embeddings)
        payload = {
            "schema_version": 1,
            "reference_evaluator_version": REFERENCE_EVALUATOR_VERSION,
            "result_key": work_item.result_key,
            "embedding_cache_key": work_item.embedding_key,
            "reference": {
                "id": work_item.reference.id,
                "kind": work_item.reference.kind,
                "label": work_item.reference.label,
                "model_id": work_item.reference.model.model_id,
            },
            "validation_set": validation_set.name,
            "metrics": metrics,
        }
        result_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(result_path, payload)
        atomic_write_json(
            metadata_path,
            {
                "schema_version": 1,
                "result_key": work_item.result_key,
                "embedding_cache_key": work_item.embedding_key,
                "reference_id": work_item.reference.id,
                "model_id": work_item.reference.model.model_id,
                "validation_set": validation_set.name,
                "dataset_id": validation_set.dataset.dataset_id,
                "split_sha256": validation_set.split_sha256,
                "task_fingerprint": work_item.task_fingerprint,
                "top_k": sorted(int(k) for k in work_item.top_k),
                "target_pc": float(work_item.target_pc),
                "embedding_artifact_dir": str(artifact.artifact_dir),
            },
        )
    return payload, False


def evaluate_references(
    exp,
    cfg: dict[str, Any],
    validation_sets: list[ValidationSet],
    *,
    run_options: dict[str, Any],
    embedding_options: dict[str, Any],
    top_k: tuple[int, ...],
    target_pc: float,
    compute: bool = True,
    logger: Any = None,
    lock_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Evaluate all configured references and return history-ready entries.

    Each entry keeps its metrics under ``validation.sets`` — the same structure a
    training epoch row uses — so reporting can render the frozen result before
    epoch 1. References are completely separate from checkpoint history and never
    participate in best/last selection.
    """
    references = resolve_reference_models(cfg, exp)
    if not references:
        return []
    primary_name = primary_validation_set(validation_sets).name
    entries: list[dict[str, Any]] = []
    for reference in references:
        applicable = _applicable_sets(reference, validation_sets)
        work_items = resolve_reference_work_items(
            exp, cfg, applicable, [reference], top_k=top_k, target_pc=target_pc
        )
        sets_metrics: dict[str, Any] = {}
        sets_cache: dict[str, Any] = {}
        for work_item in work_items:
            set_name = work_item.validation_set.name
            if logger is not None:
                logger.emit(
                    "reference_start",
                    reference_id=reference.id,
                    validation_set=set_name,
                    model_id=reference.model.model_id,
                    result_key=work_item.result_key,
                )
            payload, reused = load_or_compute_reference_result(
                work_item,
                run_options=run_options,
                embedding_options=embedding_options,
                compute=compute,
                lock_timeout=lock_timeout,
            )
            sets_metrics[set_name] = payload["metrics"]
            sets_cache[set_name] = {"result_key": work_item.result_key, "reused": reused}
            if logger is not None:
                logger.emit(
                    "reference_cache_hit" if reused else "reference_cache_miss",
                    reference_id=reference.id,
                    validation_set=set_name,
                    model_id=reference.model.model_id,
                    result_key=work_item.result_key,
                )
        reference_primary = primary_name if primary_name in sets_metrics else next(iter(sets_metrics))
        entries.append(
            {
                "id": reference.id,
                "kind": reference.kind,
                "label": reference.label,
                "model_id": reference.model.model_id,
                "validation": {"primary": reference_primary, "sets": sets_metrics},
                "cache": {"sets": sets_cache},
            }
        )
        if logger is not None:
            logger.emit(
                "reference_complete",
                reference_id=reference.id,
                model_id=reference.model.model_id,
                validation_sets=list(sets_metrics),
            )
    return entries
