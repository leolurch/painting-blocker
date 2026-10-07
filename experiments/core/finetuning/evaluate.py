"""Calibrated validation/test evaluation for finetuned retrieval models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.core.descriptor_postprocessing import postprocess_descriptors
from experiments.core.eval_pipeline import (
    evaluate_task,
    evaluate_task_at_threshold,
)
from experiments.core.retrieval_metrics import query_coverage
from experiments.core.split_schema import records_from_split, resolve_retrieval_tasks

from .augmentations import build_image_transform
from .checkpointing import load_model_from_checkpoint
from .data import (
    MetricImageItem,
    MetricLearningImageDataset,
    collate_metric_batch,
    load_items_by_ids,
    load_subset_items,
)


def embed_items(
    model: torch.nn.Module,
    items: Sequence[MetricImageItem],
    *,
    transform: object,
    batch_size: int,
    device: str,
    num_workers: int = 0,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    dataset = MetricLearningImageDataset(items, transform=transform)  # type: ignore[arg-type]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_metric_batch)
    model = model.to(device)
    model.eval()
    embeddings: list[np.ndarray] = []
    image_ids: list[str] = []
    class_ids: list[str] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            values = model(pixel_values).float().cpu().numpy()
            embeddings.append(values)
            image_ids.extend(str(value) for value in batch["image_id"])
            class_ids.extend(str(value) for value in batch["class_id"])
            if progress_callback is not None:
                progress_callback(batch_index)
    if not embeddings:
        raise RuntimeError("No embeddings produced for evaluation")
    descriptors = postprocess_descriptors(np.concatenate(embeddings, axis=0))
    return image_ids, class_ids, descriptors


def _cosine_matrix(
    embeddings: np.ndarray,
    *,
    inputs_normalized: bool = False,
) -> np.ndarray:
    from experiments.core.retrieval_metrics import cosine_similarity_matrix

    return cosine_similarity_matrix(
        embeddings,
        embeddings,
        device_preference="cpu",
        gpu_dtype="float32",
        inputs_normalized=inputs_normalized,
        fp32_matmul_precision="ieee",
    )


def _label_matrix(class_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(list(class_ids))
    positives = labels[:, None] == labels[None, :]
    identity = np.eye(len(labels), dtype=bool)
    positives[identity] = False
    valid = ~identity
    return positives, valid


def select_threshold_at_pair_completeness(
    similarities: np.ndarray,
    positives: np.ndarray,
    valid: np.ndarray,
    target_pc: float,
) -> dict[str, Any]:
    if target_pc <= 0.0 or target_pc > 1.0:
        raise ValueError("target_pc must be in (0, 1]")
    scores = similarities[valid].reshape(-1)
    labels = positives[valid].reshape(-1).astype(np.int64)
    total_positive = int(labels.sum())
    possible_pairs = int(len(labels))
    if total_positive <= 0:
        raise ValueError("Evaluation subset has no positive pairs")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    score_changes = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]) if len(sorted_scores) > 1 else np.asarray([], dtype=np.int64)
    group_ends = np.r_[score_changes, len(sorted_scores) - 1]
    candidate_pairs = group_ends + 1
    true_positives = np.cumsum(sorted_labels)[group_ends]
    pair_completeness = true_positives / float(total_positive)
    feasible = np.flatnonzero(pair_completeness >= target_pc)
    if len(feasible) == 0:
        threshold = float(sorted_scores[-1])
        idx = len(group_ends) - 1
    else:
        idx = int(feasible[0])
        threshold = float(sorted_scores[group_ends[idx]])
    selected = int(candidate_pairs[idx])
    tp = int(true_positives[idx])
    fp = selected - tp
    max_score_per_query = np.max(
        similarities, axis=1, where=valid, initial=-np.inf
    )
    return {
        "threshold": threshold,
        "target_pair_completeness": float(target_pc),
        "pair_completeness": float(tp / total_positive),
        "pair_quality": float(tp / selected) if selected else 0.0,
        "reduction_ratio": float(1.0 - selected / possible_pairs),
        "query_coverage": query_coverage(max_score_per_query >= threshold),
        "candidate_pairs": selected,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": int(total_positive - tp),
        "total_positive_pairs": total_positive,
        "possible_pairs": possible_pairs,
    }


def apply_threshold(
    similarities: np.ndarray,
    positives: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    selected_mask = (similarities >= float(threshold)) & valid
    selected = int(selected_mask.sum())
    total_positive = int((positives & valid).sum())
    tp = int((selected_mask & positives).sum())
    possible_pairs = int(valid.sum())
    return {
        "threshold": float(threshold),
        "pair_completeness": float(tp / total_positive) if total_positive else 0.0,
        "pair_quality": float(tp / selected) if selected else 0.0,
        "reduction_ratio": float(1.0 - selected / possible_pairs) if possible_pairs else 0.0,
        "query_coverage": query_coverage(np.any(selected_mask, axis=1)),
        "candidate_pairs": selected,
        "true_positives": tp,
        "false_positives": int(selected - tp),
        "false_negatives": int(total_positive - tp),
        "total_positive_pairs": total_positive,
        "possible_pairs": possible_pairs,
    }


def recall_at_k(similarities: np.ndarray, positives: np.ndarray, top_k: Sequence[int]) -> list[dict[str, Any]]:
    sims = similarities.copy()
    np.fill_diagonal(sims, -np.inf)
    positives_per_query = positives.sum(axis=1)
    valid_queries = positives_per_query > 0
    if not np.any(valid_queries):
        return []
    sims = sims[valid_queries]
    pos = positives[valid_queries]
    positives_per_query = positives_per_query[valid_queries]
    max_k = min(max(int(k) for k in top_k), sims.shape[1])
    top_idx = np.argpartition(-sims, kth=max_k - 1, axis=1)[:, :max_k]
    top_scores = np.take_along_axis(sims, top_idx, axis=1)
    top_idx = np.take_along_axis(top_idx, np.argsort(-top_scores, axis=1), axis=1)
    relevant = np.take_along_axis(pos, top_idx, axis=1)
    finite_candidates_per_query = np.isfinite(sims).sum(axis=1)
    rows = []
    for k in top_k:
        k_eff = min(int(k), max_k)
        hits = relevant[:, :k_eff].sum(axis=1).astype(np.float64)
        candidate_pairs = int(len(hits) * k_eff)
        tp = int(hits.sum())
        rows.append(
            {
                "k": int(k),
                "recall": float(np.mean(hits / positives_per_query)),
                "precision": float(np.mean(hits / float(k))),
                "pair_completeness": float(tp / positives.sum()) if positives.sum() else 0.0,
                "pair_quality": float(tp / candidate_pairs) if candidate_pairs else 0.0,
                "query_coverage": query_coverage(
                    np.minimum(k_eff, finite_candidates_per_query) > 0
                ),
                "candidate_pairs": candidate_pairs,
                "true_positives": tp,
            }
        )
    return rows


def evaluate_embeddings(
    image_ids: Sequence[str],
    class_ids: Sequence[str],
    embeddings: np.ndarray,
    *,
    target_pc: float = 0.99,
    threshold: float | None = None,
    top_k: Sequence[int] = (1, 5, 10, 25, 50, 100),
    inputs_normalized: bool = False,
) -> dict[str, Any]:
    similarities = _cosine_matrix(embeddings, inputs_normalized=inputs_normalized)
    positives, valid = _label_matrix(class_ids)
    calibrated = apply_threshold(similarities, positives, valid, threshold) if threshold is not None else select_threshold_at_pair_completeness(similarities, positives, valid, target_pc)
    return {
        "num_images": len(image_ids),
        "image_ids": list(image_ids),
        "target_pc": float(target_pc),
        "calibrated_threshold": calibrated,
        "recall_at_k": recall_at_k(similarities, positives, top_k),
    }


def _task_subset(task: dict[str, Any], key: str) -> str | None:
    selector = task.get(key)
    if not isinstance(selector, dict) or "subset" not in selector:
        return None
    return str(selector["subset"])


def retrieval_task_specs_for_subset(
    experiment_config: Any,
    subset_name: str,
    *,
    selector_subset: str | None = None,
) -> list[dict[str, Any]]:
    """Return role-aware retrieval tasks configured for one finetuning subset."""
    raw = getattr(experiment_config, "raw", {}) or {}
    finetuning = dict(raw.get("finetuning") or {})
    finetune_eval = dict(finetuning.get("evaluation") or {})
    specs = finetune_eval.get("retrieval_tasks")
    if specs is None:
        specs = dict(raw.get("evaluation") or {}).get("retrieval_tasks")
    if not isinstance(specs, list) or not specs:
        return []
    aliases = {str(subset_name)}
    if selector_subset is not None:
        aliases.add(str(selector_subset))
    if subset_name == "validation":
        aliases.add("val")
    if subset_name == "val":
        aliases.add("validation")
    return [
        dict(task)
        for task in specs
        if _task_subset(task, "query") in aliases and _task_subset(task, "candidates") in aliases
    ]


def _union_task_ids(tasks: Sequence[dict[str, Any]]) -> list[str]:
    image_ids: set[str] = set()
    for task in tasks:
        image_ids.update(str(value) for value in task["query_ids"])
        image_ids.update(str(value) for value in task["candidate_ids"])
    return sorted(image_ids)


def _primary_threshold(task_results: Sequence[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    if not task_results:
        raise ValueError("No retrieval task results available for threshold selection")
    primary = task_results[0]
    if threshold is not None:
        fixed = primary.get("fixed_threshold_metrics") or []
        if not fixed:
            raise ValueError(f"Primary retrieval task {primary['task_id']!r} has no fixed-threshold metrics")
        return dict(fixed[0])
    target_rows = primary.get("target_pc_metrics") or []
    if not target_rows:
        raise ValueError(f"Primary retrieval task {primary['task_id']!r} has no target-PC metrics")
    return dict(target_rows[0])


def evaluate_retrieval_tasks(
    model: torch.nn.Module,
    dataset: Any,
    split: dict[str, Any],
    task_specs: Sequence[dict[str, Any]],
    *,
    transform: object,
    batch_size: int,
    device: str,
    num_workers: int = 0,
    target_pc: float = 0.99,
    threshold: float | None = None,
    top_k: Sequence[int] = (1, 5, 10, 25, 50, 100),
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Embed task IDs once and evaluate directional query/candidate retrieval tasks."""
    tasks = resolve_retrieval_tasks(split, [dict(task) for task in task_specs])
    image_ids_for_embedding = _union_task_ids(tasks)
    items = load_items_by_ids(dataset, split, image_ids_for_embedding, source="Finetuning retrieval tasks")
    image_ids, class_ids, embeddings = embed_items(
        model,
        items,
        transform=transform,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        progress_callback=progress_callback,
    )
    similarities = _cosine_matrix(embeddings, inputs_normalized=True)
    records = records_from_split(split)
    task_results: list[dict[str, Any]] = []
    curve_results: list[dict[str, Any]] = []
    pc_targets = [float(target_pc)]
    for task in tasks:
        result, curves = evaluate_task(
            task,
            image_ids,
            records,
            similarities,
            [int(k) for k in top_k],
            True,
            pc_targets,
        )
        if threshold is not None:
            result["fixed_threshold_metrics"] = [
                evaluate_task_at_threshold(task, image_ids, records, similarities, float(threshold))
            ]
        task_results.append(result)
        curve_results.append(curves)
    primary = _primary_threshold(task_results, threshold)
    primary_task = task_results[0]
    # Target-PC task rows historically carried the selected count but not the
    # denominator. Keep both on the calibrated operating point so checkpoint
    # selection and its artifacts can audit RR = 1 - candidates / possible.
    primary.setdefault("possible_pairs", int(primary_task.get("num_possible_pairs") or 0))
    primary.setdefault("total_positive_pairs", int(primary_task.get("num_positive_pairs") or 0))
    return {
        "num_images": len(image_ids),
        "image_ids": list(image_ids),
        "target_pc": float(target_pc),
        "primary_task_id": task_results[0]["task_id"],
        "calibrated_threshold": primary,
        "recall_at_k": list(task_results[0].get("metrics") or []),
        "retrieval_tasks": task_results,
        "curves": curve_results,
    }


def _selector_from_experiment(experiment_config: Any, name: str) -> dict[str, Any]:
    finetuning = dict((getattr(experiment_config, "raw", {}) or {}).get("finetuning") or {})
    data = finetuning.get("data")
    if not isinstance(data, dict):
        raise ValueError("finetuning.data is required to evaluate checkpoints")
    if name in data and isinstance(data[name], dict):
        return dict(data[name])
    if name in {"val", "validation"} and isinstance(data.get("validation"), list):
        validation_sets = [dict(item) for item in data["validation"] if isinstance(item, dict)]
        primary = [item for item in validation_sets if item.get("primary")]
        selected = primary[0] if primary else (validation_sets[0] if validation_sets else None)
        if selected is not None:
            return {"subset": selected["subset"], "roles": selected["roles"]}
    available = ", ".join(sorted(str(key) for key in data))
    raise ValueError(f"No finetuning.data selector for {name!r}; available: {available}")


def evaluate_checkpoint_on_subset(
    experiment_config: Any,
    split: dict[str, Any],
    checkpoint_path: Path | str,
    subset_name: str,
    *,
    threshold: float | None = None,
    target_pc: float = 0.99,
    batch_size: int = 64,
    device: str = "cpu",
    output: Path | None = None,
) -> dict[str, Any]:
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location="cpu")
    threshold_source = "argument"
    effective_threshold = threshold
    selector = _selector_from_experiment(experiment_config, subset_name)
    selector_subset = str(selector["subset"])
    is_validation = subset_name in {"val", "validation"} or selector_subset == "val"
    if effective_threshold is None and not is_validation:
        checkpoint_threshold = checkpoint.get("validation_threshold")
        if checkpoint_threshold is None:
            raise ValueError(
                "Non-validation checkpoint evaluation requires --threshold or a checkpoint validation_threshold"
            )
        effective_threshold = float(checkpoint_threshold)
        threshold_source = "checkpoint_validation_threshold"
    elif effective_threshold is None:
        threshold_source = "selected_on_subset"
    eval_transform = build_image_transform(
        {**dict(checkpoint.get("preprocessing_config") or {}), "normalization": checkpoint.get("normalization_config") or {}},
        train=False,
    )
    retrieval_specs = retrieval_task_specs_for_subset(
        experiment_config,
        subset_name,
        selector_subset=selector_subset,
    )
    if retrieval_specs:
        result = evaluate_retrieval_tasks(
            model,
            experiment_config.dataset,
            split,
            retrieval_specs,
            transform=eval_transform,
            batch_size=batch_size,
            device=device,
            target_pc=target_pc,
            threshold=effective_threshold,
        )
    else:
        items = load_subset_items(experiment_config.dataset, split, selector)
        image_ids, class_ids, embeddings = embed_items(model, items, transform=eval_transform, batch_size=batch_size, device=device)
        result = evaluate_embeddings(
            image_ids,
            class_ids,
            embeddings,
            target_pc=target_pc,
            threshold=effective_threshold,
            inputs_normalized=True,
        )
    result["checkpoint"] = str(Path(checkpoint_path).expanduser())
    result["subset"] = selector_subset
    result["selector"] = selector
    result["threshold_source"] = threshold_source
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
