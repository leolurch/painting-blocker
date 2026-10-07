"""Training loop for metric-learning image-retrieval finetuning."""

from __future__ import annotations

import math
import os
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.core.artifacts import (
    PACKAGE_DIR,
    atomic_write_json,
    git_info,
    make_run_id,
    prepare_run_dir,
    sha256_file,
)
from experiments.core.config_schema import load_experiment_config, training_repeat_seeds
from experiments.core.env import env_load
from experiments.core.reproducibility import dependency_versions, rng_state_fingerprints
from experiments.core.split_schema import load_split, validate_split_manifest

from .augmentations import build_image_transform, normalization_config, preprocessing_config
from ..checkpoint_naming import checkpoint_run_name
from .checkpointing import build_trainable_model, save_checkpoint
from .console_logging import ConsoleTrainingLogger
from .cross_era import neighbor_lists, probe_plan
from .data import (
    MetricImageItem,
    MetricLearningImageDataset,
    append_hard_historic_views,
    append_online_historic_views,
    collate_metric_batch,
    load_subset_items,
)
from .evaluate import (
    embed_items,
    evaluate_embeddings,
    evaluate_retrieval_tasks,
)
from .losses import build_loss
from .models.base import trainable_parameter_groups
from .reference_evaluation import evaluate_references
from .reporting import write_training_reports
from .samplers import (
    CrossEraGraphPKBatchSampler,
    HardNegativePKBatchSampler,
    OriginBalancedPKBatchSampler,
    PKBatchSampler,
    RoleQuotaPKBatchSampler,
    RoleStratifiedPKBatchSampler,
)
from .validation_sets import (
    ValidationSet,
    primary_validation_set as _primary_validation_set,
    resolve_experiment_path as _resolve_experiment_path,
    resolve_validation_sets as _validation_sets,
)


def _finetuning_config(exp_raw: dict[str, Any]) -> dict[str, Any]:
    return dict(exp_raw.get("finetuning") or exp_raw)


def _device(value: str | None) -> str:
    if value and value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int, *, deterministic: bool = True, warn_only: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, Torch CPU/CUDA, and configure deterministic backends."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))
    except TypeError:  # older torch without warn_only
        torch.use_deterministic_algorithms(bool(deterministic))
    return _determinism_metadata(seed, deterministic, warn_only)


def _determinism_metadata(seed: int, deterministic: bool, warn_only: bool) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_manual_seeded": True,
        "cuda_manual_seeded": bool(torch.cuda.is_available()),
        "deterministic_requested": bool(deterministic),
        "deterministic_warn_only": bool(warn_only),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)) if hasattr(torch.backends, "cudnn") else None,
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)) if hasattr(torch.backends, "cudnn") else None,
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)) if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32") else None,
        "cuda_matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)) if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul") else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _data_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _run_dir(exp, run_id: str | None) -> Path:
    ft_cfg = _finetuning_config(exp.raw)
    run_cfg = dict(exp.raw.get("run") or {})
    train_cfg = dict(ft_cfg.get("train") or {})
    output = Path(train_cfg.get("output_dir") or run_cfg.get("output_dir") or f"runs/{exp.experiment_id}_finetune").expanduser()
    output = output if output.is_absolute() else PACKAGE_DIR / output
    if run_id is None:
        generated_run_id = make_run_id()
        run_id = checkpoint_run_name(ft_cfg, generated_run_id) or generated_run_id
    return output / run_id


def _optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("name", "adamw")).lower()
    if name not in {"adam", "adamw"}:
        raise ValueError("optimizer.name must be 'adam' or 'adamw'")
    backbone, head = trainable_parameter_groups(model)
    base_lr = float(config.get("lr", config.get("base_lr", 1e-4)))
    head_lr = float(config.get("head_lr", config.get("lr_head", config.get("lr", 1e-3))))
    backbone_lr = float(config.get("backbone_lr", config.get("lr_lora", config.get("lr_backbone", base_lr))))
    weight_decay = float(config.get("weight_decay", 1e-4))
    eps = float(config.get("eps", 1e-8))
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "weight_decay": weight_decay})
    if head:
        groups.append({"params": head, "lr": head_lr, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("No trainable parameters found")
    if name == "adam":
        return torch.optim.Adam(groups, eps=eps)
    return torch.optim.AdamW(groups, weight_decay=weight_decay, eps=eps)


def _scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int):
    name = str(config.get("scheduler", "cosine"))
    if name == "step":
        step_size = int(config.get("sched_step", 6))
        gamma = float(config.get("sched_gamma", 0.1))
        if step_size < 1 or gamma <= 0:
            raise ValueError("step scheduler requires sched_step >= 1 and a positive sched_gamma")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    warmup = max(0, int(round(total_steps * float(config.get("warmup_ratio", 0.05)))))
    if name != "cosine":
        return None

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _autocast(device: str, precision: str):
    if torch.device(device).type != "cuda" or precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _grad_scaler(device: str, precision: str):
    enabled = torch.device(device).type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class _RNGReplayState:
    """Capture model-forward RNG state and replay it without advancing global RNG."""

    def __init__(self, *, capture_cuda: bool) -> None:
        self.cpu_state = torch.get_rng_state().clone()
        self.cuda_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if capture_cuda
            else None
        )

    @contextmanager
    def replay(self):
        current_cpu_state = torch.get_rng_state()
        current_cuda_states = (
            torch.cuda.get_rng_state_all() if self.cuda_states is not None else None
        )
        torch.set_rng_state(self.cpu_state)
        if self.cuda_states is not None:
            torch.cuda.set_rng_state_all(self.cuda_states)
        try:
            yield
        finally:
            torch.set_rng_state(current_cpu_state)
            if current_cuda_states is not None:
                torch.cuda.set_rng_state_all(current_cuda_states)


def _gradient_cache_config(
    train_cfg: dict[str, Any],
    sampler: Any,
    chunk_size_override: int | None = None,
) -> dict[str, Any]:
    raw = train_cfg.get("gradient_cache")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("train.gradient_cache must be a mapping")
    enabled = bool(raw.get("enabled", False))
    effective_batch_size = _sampler_batch_size(sampler)
    if effective_batch_size is None:
        if enabled:
            raise ValueError("Gradient caching requires a sampler with a fixed batch size")
        return {"enabled": False, "chunk_size": None, "effective_batch_size": None, "chunks_per_step": 1}
    if not enabled:
        return {
            "enabled": False,
            "chunk_size": effective_batch_size,
            "effective_batch_size": effective_batch_size,
            "chunks_per_step": 1,
            "verify_replay": False,
        }
    if chunk_size_override is None and "chunk_size" not in raw:
        raise ValueError("Enabled train.gradient_cache requires chunk_size")
    chunk_size = raw.get("chunk_size") if chunk_size_override is None else chunk_size_override
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("train.gradient_cache.chunk_size must be a positive integer")
    if chunk_size > effective_batch_size:
        raise ValueError("train.gradient_cache.chunk_size cannot exceed the effective sampler batch size")
    chunks_per_step = math.ceil(effective_batch_size / float(chunk_size))
    return {
        "enabled": True,
        "chunk_size": chunk_size,
        "effective_batch_size": effective_batch_size,
        "chunks_per_step": chunks_per_step,
        "verify_replay": bool(raw.get("verify_replay", True)) if enabled else False,
    }


def _gradient_cache_batch_norm_names(model: torch.nn.Module) -> list[str]:
    batch_norm = torch.nn.modules.batchnorm._BatchNorm
    return [name or "<root>" for name, module in model.named_modules() if isinstance(module, batch_norm)]


def _gradient_cache_backward(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int,
    device: str,
    precision: str,
    scaler: Any,
    verify_replay: bool = False,
    origins: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backpropagate one full embedding-batch loss using activation-sized chunks.

    The first pass caches graphless embeddings and per-chunk RNG states. The
    full loss then produces dL/dz for every embedding. Replaying each chunk with
    its original RNG state applies those cached representation gradients to the
    model parameters, exactly matching a graph-preserving chunked forward.
    """
    if pixel_values.ndim < 1 or labels.ndim != 1:
        raise ValueError("Gradient caching expects batched pixel_values and 1D labels")
    batch_size = int(pixel_values.shape[0])
    if batch_size != int(labels.shape[0]):
        raise ValueError("pixel_values and labels must have the same batch size")
    if batch_size <= 0 or chunk_size <= 0:
        raise ValueError("Gradient caching requires a non-empty batch and positive chunk_size")

    capture_cuda = torch.device(device).type == "cuda"
    cached_embeddings: list[torch.Tensor] = []
    rng_states: list[_RNGReplayState] = []
    with torch.no_grad():
        for start in range(0, batch_size, chunk_size):
            stop = min(batch_size, start + chunk_size)
            chunk = pixel_values[start:stop].to(device)
            rng_states.append(_RNGReplayState(capture_cuda=capture_cuda))
            with _autocast(device, precision):
                cached_embeddings.append(model(chunk).detach())

    labels_device = labels.to(device)
    embedding_leaf = torch.cat(cached_embeddings, dim=0).detach().requires_grad_(True)
    with _autocast(device, precision):
        loss = _call_loss(loss_fn, embedding_leaf, labels_device, origins, device, indices)
    # Scale before dL/dz is materialized. In FP16, scaling only the replay
    # gradients would be too late to prevent small representation gradients
    # from underflowing during autograd.grad.
    (embedding_grad,) = torch.autograd.grad(scaler.scale(loss), embedding_leaf)
    embedding_grad = embedding_grad.detach()

    for chunk_index, start in enumerate(range(0, batch_size, chunk_size)):
        stop = min(batch_size, start + chunk_size)
        chunk = pixel_values[start:stop].to(device)
        with rng_states[chunk_index].replay():
            with _autocast(device, precision):
                replayed_embeddings = model(chunk)
        if verify_replay and not torch.equal(
            replayed_embeddings.detach(), cached_embeddings[chunk_index]
        ):
            max_difference = float(
                (replayed_embeddings.detach().float() - cached_embeddings[chunk_index].float())
                .abs()
                .max()
                .cpu()
            )
            raise RuntimeError(
                "Gradient-cache replay did not reproduce the graphless embeddings; "
                f"chunk={chunk_index}, max_abs_difference={max_difference:.8g}"
            )
        torch.autograd.backward(
            replayed_embeddings,
            grad_tensors=embedding_grad[start:stop],
        )
    return loss.detach(), embedding_leaf.detach()


def _resolve_sampler_path(value: object, exp) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        raise ValueError("sampler.hard_negative_parquet must not be empty")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    experiment_relative = (exp.path.parent / path).resolve()
    if experiment_relative.exists():
        return experiment_relative
    return (exp.repo_root / path).resolve()


def _build_sampler(train_dataset, sampler_cfg: dict[str, Any], train_cfg: dict[str, Any], exp):
    classes_per_batch = int(sampler_cfg.get("classes_per_batch", sampler_cfg.get("p", 16)))
    images_per_class = int(sampler_cfg.get("images_per_class", sampler_cfg.get("k", 4)))
    sampler_name = str(sampler_cfg.get("type", sampler_cfg.get("name", "pk"))).lower()
    hard_names = {
        "hard_negative",
        "hard_negative_pk",
        "hard_negative_pk_batch",
        "hard_negative_pk_batch_sampler",
        "hardnegativepkbatchsampler",
    }
    pk_names = {"pk", "pk_batch", "pk_batch_sampler"}
    role_names = {
        "role_stratified_pk",
        "role_stratified_pk_batch",
        "role_stratified_pk_batch_sampler",
        "rolestratifiedpkbatchsampler",
    }
    origin_names = {
        "origin_balanced_pk",
        "origin_balanced_pk_batch",
        "originbalancedpkbatchsampler",
    }
    quota_names = {"role_quota_pk", "role_quota_pk_batch", "rolequotapkbatchsampler"}
    graph_names = {"cross_era_graph_pk", "cross_era_graph_pk_batch", "crosseragraphpkbatchsampler"}
    if sampler_name not in hard_names | pk_names | role_names | origin_names | quota_names | graph_names:
        raise ValueError(f"Unsupported sampler.type: {sampler_name!r}")
    hard_negative_path = _resolve_sampler_path(sampler_cfg.get("hard_negative_parquet"), exp)
    seed = int(train_cfg.get("seed", 42))
    batches_per_epoch = train_cfg.get("batches_per_epoch")
    if sampler_name in origin_names:
        if hard_negative_path is not None:
            raise ValueError("Origin-balanced PK sampling cannot be combined with hard-negative sampling")
        per_origin = sampler_cfg.get("per_origin")
        if not isinstance(per_origin, dict) or not per_origin:
            raise ValueError("origin_balanced_pk requires sampler.per_origin")
        return OriginBalancedPKBatchSampler(
            train_dataset.labels,
            train_dataset.roles,
            classes_per_batch,
            {str(origin): int(count) for origin, count in per_origin.items()},
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        )
    if sampler_name in quota_names:
        quotas = sampler_cfg.get("quotas")
        if not isinstance(quotas, dict) or not quotas:
            raise ValueError("role_quota_pk requires sampler.quotas")
        return RoleQuotaPKBatchSampler(
            train_dataset.labels,
            train_dataset.roles,
            classes_per_batch,
            {str(key): int(count) for key, count in quotas.items()},
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        )
    if sampler_name in graph_names:
        return CrossEraGraphPKBatchSampler(
            train_dataset.labels,
            train_dataset.roles,
            classes_per_batch,
            images_per_class,
            hard_fraction=float(sampler_cfg.get("hard_fraction", 0.5)),
            neighbors_pool=int(sampler_cfg.get("neighbors_pool", 128)),
            refresh_every_epochs=int(sampler_cfg.get("refresh_every_epochs", 1)),
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        )
    if sampler_name in role_names:
        if hard_negative_path is not None:
            raise ValueError("Role-stratified PK sampling cannot be combined with hard-negative sampling")
        anchor_role = str(sampler_cfg.get("anchor_role") or "").strip()
        positive_roles = sampler_cfg.get("positive_roles")
        if not anchor_role:
            raise ValueError("role_stratified_pk requires sampler.anchor_role")
        if isinstance(positive_roles, str) or not isinstance(positive_roles, list) or not positive_roles:
            raise ValueError("role_stratified_pk requires sampler.positive_roles as a non-empty list")
        return RoleStratifiedPKBatchSampler(
            train_dataset.class_ids,
            train_dataset.roles,
            classes_per_batch,
            images_per_class,
            anchor_role=anchor_role,
            positive_roles=[str(role) for role in positive_roles],
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        )
    if sampler_name in hard_names or hard_negative_path:
        return HardNegativePKBatchSampler(
            train_dataset.class_ids,
            classes_per_batch,
            images_per_class,
            hard_negative_parquet=hard_negative_path,
            hard_negatives_per_anchor_class=int(
                sampler_cfg.get(
                    "hard_negatives_per_anchor_class",
                    sampler_cfg.get("hard_negative_classes_per_anchor", sampler_cfg.get("h", 1)),
                )
            ),
            image_ids=[item.image_id for item in train_dataset.items],
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        )
    return PKBatchSampler(
        train_dataset.labels,
        classes_per_batch,
        images_per_class,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )


def _evaluate_model(
    model,
    items,
    transform,
    cfg: dict[str, Any],
    device: str,
    progress_callback=None,
    *,
    dataset=None,
    split: dict[str, Any] | None = None,
    retrieval_task_specs: list[dict[str, Any]] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    eval_cfg = dict(cfg.get("evaluation") or {})
    train_cfg = dict(cfg.get("train") or {})
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("eval_batch_size", 64)))
    num_workers = int(train_cfg.get("num_workers", 0))
    target_pc = float(eval_cfg.get("target_pc", 0.99))
    top_k = [int(k) for k in eval_cfg.get("top_k", [1, 5, 10, 25, 50, 100])]
    if retrieval_task_specs:
        if dataset is None or split is None:
            raise ValueError("Role-aware finetuning evaluation requires dataset and split")
        return evaluate_retrieval_tasks(
            model,
            dataset,
            split,
            retrieval_task_specs,
            transform=transform,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            target_pc=target_pc,
            threshold=threshold,
            top_k=top_k,
            progress_callback=progress_callback,
        )
    image_ids, class_ids, embeddings = embed_items(
        model,
        items,
        transform=transform,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        progress_callback=progress_callback,
    )
    return evaluate_embeddings(
        image_ids,
        class_ids,
        embeddings,
        target_pc=target_pc,
        threshold=threshold,
        top_k=top_k,
        inputs_normalized=True,
    )


def _is_better(candidate: dict[str, Any], best: dict[str, Any] | None, target_pc: float) -> bool:
    metrics = candidate["calibrated_threshold"]
    if float(metrics.get("pair_completeness", 0.0)) < target_pc:
        return False
    if best is None:
        return True
    best_metrics = best["calibrated_threshold"]
    cand_key = (
        float(metrics.get("pair_quality", 0.0)),
        float(metrics.get("reduction_ratio", 0.0)),
        -int(metrics.get("candidate_pairs", 0)),
    )
    best_key = (
        float(best_metrics.get("pair_quality", 0.0)),
        float(best_metrics.get("reduction_ratio", 0.0)),
        -int(best_metrics.get("candidate_pairs", 0)),
    )
    return cand_key > best_key


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _metric_at_k(metrics: dict[str, Any], k: int) -> dict[str, Any]:
    for row in metrics.get("recall_at_k") or metrics.get("metrics") or []:
        if isinstance(row, dict) and int(row.get("k", -1)) == int(k):
            return dict(row)
    available = [str(row.get("k")) for row in metrics.get("recall_at_k") or metrics.get("metrics") or [] if isinstance(row, dict)]
    raise ValueError(f"Checkpoint selection requested k={k}, but available k values are: {', '.join(available) or '<none>'}")


def _checkpoint_selection_config(
    eval_cfg: dict[str, Any],
    validation_sets: list[ValidationSet],
    primary_name: str,
) -> dict[str, Any]:
    raw_value = eval_cfg.get("checkpoint_selection")
    if raw_value is not None and not isinstance(raw_value, dict):
        raise ValueError("evaluation.checkpoint_selection must be a mapping")
    raw = dict(raw_value or {})
    legacy = str(eval_cfg.get("select_best_by") or "").strip().lower()
    if not raw and legacy in {"pc_at_k", "pair_completeness_at_k", "secondary_pc_at_k", "secondary_pair_completeness_at_k", "pc_at_10", "secondary_pc_at_10"}:
        raw = {
            "validation_set": eval_cfg.get("selection_validation_set") or "first_secondary",
            "metric": "pc_at_k",
            "k": eval_cfg.get("selection_k", 10),
        }
    if not raw:
        return {"validation_set": primary_name, "metric": "calibrated_threshold"}
    metric = str(raw.get("metric") or raw.get("type") or "pc_at_k").strip().lower()
    if metric in {"pair_completeness_at_k", "recall_at_k", "pc@k"}:
        metric = "pc_at_k"
    validation_name = str(raw.get("validation_set") or raw.get("set") or primary_name)
    if validation_name in {"first_secondary", "secondary"}:
        secondary = [item.name for item in validation_sets if not item.primary]
        if not secondary:
            raise ValueError("checkpoint_selection requested a secondary validation set, but none are configured")
        validation_name = secondary[0]
    available = {item.name for item in validation_sets}
    if validation_name not in available:
        raise ValueError(f"checkpoint_selection.validation_set={validation_name!r} is not configured; available: {', '.join(sorted(available))}")
    selection = {"validation_set": validation_name, "metric": metric}
    if metric == "pc_at_k":
        selection["k"] = int(raw.get("k", raw.get("top_k", 10)))
    elif metric == "weighted_mean_pc_at_k_and_transfer":
        if "target_pc" not in eval_cfg:
            raise ValueError(
                "checkpoint_selection.metric='weighted_mean_pc_at_k_and_transfer' "
                "requires evaluation.target_pc"
            )
        target_pc = float(eval_cfg["target_pc"])
        if target_pc <= 0.0 or target_pc > 1.0:
            raise ValueError("evaluation.target_pc must be in (0, 1]")
        calibration_name = str(raw.get("calibration_validation_set") or primary_name)
        selection_name = str(raw.get("selection_validation_set") or raw.get("validation_set") or "")
        if calibration_name not in available:
            raise ValueError(
                f"checkpoint_selection.calibration_validation_set={calibration_name!r} is not configured"
            )
        if not selection_name or selection_name not in available:
            raise ValueError(
                f"checkpoint_selection.selection_validation_set={selection_name!r} is not configured"
            )
        if calibration_name == selection_name:
            raise ValueError("Calibration and selection validation sets must be different")
        fixed_k = raw.get("fixed_k")
        if not isinstance(fixed_k, list) or not fixed_k:
            raise ValueError("checkpoint_selection.fixed_k must be a non-empty list")
        fixed_k = [int(k) for k in fixed_k]
        if any(k <= 0 for k in fixed_k) or len(set(fixed_k)) != len(fixed_k):
            raise ValueError("checkpoint_selection.fixed_k must contain unique positive integers")
        configured_top_k = {int(k) for k in eval_cfg.get("top_k", [])}
        missing_k = sorted(set(fixed_k) - configured_top_k)
        if missing_k:
            raise ValueError(
                "checkpoint_selection.fixed_k must be included in evaluation.top_k; missing: "
                + ", ".join(str(k) for k in missing_k)
            )
        raw_weights = raw.get("weights")
        if not isinstance(raw_weights, dict):
            raise ValueError("checkpoint_selection.weights must be a mapping")
        weight_keys = ("mean_pc_at_k", "transferred_pc", "transferred_rr")
        if set(raw_weights) != set(weight_keys):
            raise ValueError(
                "checkpoint_selection.weights must define exactly mean_pc_at_k, "
                "transferred_pc, and transferred_rr"
            )
        weights = {key: float(raw_weights[key]) for key in weight_keys}
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("checkpoint_selection weights must be finite and non-negative")
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("checkpoint_selection weights must sum to 1")
        epoch_tiebreak = str(raw.get("epoch_tiebreak", "earliest")).strip().lower()
        if epoch_tiebreak != "earliest":
            raise ValueError("checkpoint_selection.epoch_tiebreak must be 'earliest'")
        by_name = {item.name: item for item in validation_sets}
        for set_name in (calibration_name, selection_name):
            if len(by_name[set_name].retrieval_task_specs) > 1:
                raise ValueError(
                    "weighted_mean_pc_at_k_and_transfer supports at most one retrieval task "
                    f"in {set_name!r}"
                )
        selection.update(
            validation_set=selection_name,
            calibration_validation_set=calibration_name,
            selection_validation_set=selection_name,
            target_pc=target_pc,
            fixed_k=fixed_k,
            weights=weights,
            epoch_tiebreak=epoch_tiebreak,
        )
    elif metric == "reduction_ratio_at_target_pc":
        if "target_pc" not in eval_cfg:
            raise ValueError(
                "checkpoint_selection.metric='reduction_ratio_at_target_pc' requires a "
                "predeclared evaluation.target_pc in (0, 1]"
            )
        target_pc = float(eval_cfg["target_pc"])
        if target_pc <= 0.0 or target_pc > 1.0:
            raise ValueError(
                "checkpoint_selection.metric='reduction_ratio_at_target_pc' requires "
                "evaluation.target_pc in (0, 1]"
            )
        if "k" in raw or "top_k" in raw:
            raise ValueError("checkpoint_selection.k is not valid for reduction_ratio_at_target_pc")
        if "target_pc" in raw:
            raise ValueError(
                "checkpoint_selection.target_pc is not allowed; evaluation.target_pc is the single source of truth"
            )
        if len(validation_sets) != 1:
            raise ValueError(
                "checkpoint_selection.metric='reduction_ratio_at_target_pc' requires exactly one validation set"
            )
        validation_set = validation_sets[0]
        if not validation_set.primary:
            raise ValueError(
                "The sole validation set must set primary: true for reduction_ratio_at_target_pc"
            )
        if validation_name != validation_set.name:
            raise ValueError(
                "checkpoint_selection.validation_set must match the sole validation set "
                f"{validation_set.name!r}"
            )
        task_specs = validation_set.retrieval_task_specs
        if len(task_specs) > 1:
            raise ValueError(
                "reduction_ratio_at_target_pc supports one retrieval task in the sole validation set; "
                f"found {len(task_specs)}"
            )
        task_id = str(task_specs[0].get("task_id") or "") if task_specs else "all_pairs"
        if not task_id:
            raise ValueError("The validation retrieval task must define task_id")
        configured_task_id = raw.get("task_id")
        if configured_task_id is not None and str(configured_task_id) != task_id:
            raise ValueError(
                f"checkpoint_selection.task_id={configured_task_id!r} does not match the sole "
                f"validation task {task_id!r}"
            )
        selection.update(target_pc=target_pc, task_id=task_id)
    elif metric == "calibration_transfer":
        if "target_pc" not in eval_cfg:
            raise ValueError(
                "checkpoint_selection.metric='calibration_transfer' requires evaluation.target_pc"
            )
        target_pc = float(eval_cfg["target_pc"])
        if target_pc <= 0.0 or target_pc > 1.0:
            raise ValueError("evaluation.target_pc must be in (0, 1]")
        calibration_name = str(raw.get("calibration_validation_set") or primary_name)
        selection_name = str(raw.get("selection_validation_set") or raw.get("validation_set") or "")
        if calibration_name not in available:
            raise ValueError(
                f"checkpoint_selection.calibration_validation_set={calibration_name!r} is not configured"
            )
        if not selection_name or selection_name not in available:
            raise ValueError(
                f"checkpoint_selection.selection_validation_set={selection_name!r} is not configured"
            )
        if calibration_name == selection_name:
            raise ValueError("Calibration and selection validation sets must be different")
        minimum_pc = float(raw.get("minimum_selection_pc", 0.98))
        rr_margin = float(raw.get("rr_tie_margin", 0.005))
        if minimum_pc < 0.0 or minimum_pc > 1.0:
            raise ValueError("checkpoint_selection.minimum_selection_pc must be in [0, 1]")
        if rr_margin < 0.0 or rr_margin > 1.0:
            raise ValueError("checkpoint_selection.rr_tie_margin must be in [0, 1]")
        epoch_tiebreak = str(raw.get("epoch_tiebreak", "earliest")).strip().lower()
        if epoch_tiebreak != "earliest":
            raise ValueError("checkpoint_selection.epoch_tiebreak must be 'earliest'")
        by_name = {item.name: item for item in validation_sets}
        for set_name in (calibration_name, selection_name):
            if len(by_name[set_name].retrieval_task_specs) > 1:
                raise ValueError(
                    f"calibration_transfer supports at most one retrieval task in {set_name!r}"
                )
        selection.update(
            validation_set=selection_name,
            calibration_validation_set=calibration_name,
            selection_validation_set=selection_name,
            target_pc=target_pc,
            minimum_selection_pc=minimum_pc,
            rr_tie_margin=rr_margin,
            epoch_tiebreak=epoch_tiebreak,
        )
    elif metric in {"mean_precision_at_k", "mean_precision@k"}:
        raw_k = raw.get("k", raw.get("fixed_k", [5, 10, 20, 40, 80]))
        if isinstance(raw_k, int):
            raw_k = [raw_k]
        if not isinstance(raw_k, list) or not raw_k:
            raise ValueError("checkpoint_selection.k must be a non-empty list for mean_precision_at_k")
        k_values = [int(k) for k in raw_k]
        if any(k <= 0 for k in k_values) or len(set(k_values)) != len(k_values):
            raise ValueError("checkpoint_selection.k must contain unique positive integers")
        configured_top_k = {int(k) for k in eval_cfg.get("top_k", [])}
        missing_k = sorted(set(k_values) - configured_top_k)
        if missing_k:
            raise ValueError(
                "checkpoint_selection.k must be included in evaluation.top_k; missing: "
                + ", ".join(str(k) for k in missing_k)
            )
        precision_tie = float(raw.get("precision_tie", 0.01))
        if not math.isfinite(precision_tie) or precision_tie < 0.0 or precision_tie > 1.0:
            raise ValueError("checkpoint_selection.precision_tie must be in [0, 1]")
        epoch_tiebreak = str(raw.get("epoch_tiebreak", "earliest")).strip().lower()
        if epoch_tiebreak != "earliest":
            raise ValueError("checkpoint_selection.epoch_tiebreak must be 'earliest'")
        selection.update(
            metric="mean_precision_at_k",
            k=k_values,
            precision_tie=precision_tie,
            epoch_tiebreak=epoch_tiebreak,
        )
    elif metric == "mean_pair_completeness_at_k":
        raw_k = raw.get("k", raw.get("fixed_k", [5, 10, 20, 40, 60, 80]))
        if isinstance(raw_k, int):
            raw_k = [raw_k]
        if not isinstance(raw_k, list) or not raw_k:
            raise ValueError(
                "checkpoint_selection.k must be a non-empty list for mean_pair_completeness_at_k"
            )
        k_values = [int(k) for k in raw_k]
        if any(k <= 0 for k in k_values) or len(set(k_values)) != len(k_values):
            raise ValueError("checkpoint_selection.k must contain unique positive integers")
        configured_top_k = {int(k) for k in eval_cfg.get("top_k", [])}
        missing_k = sorted(set(k_values) - configured_top_k)
        if missing_k:
            raise ValueError(
                "checkpoint_selection.k must be included in evaluation.top_k; missing: "
                + ", ".join(str(k) for k in missing_k)
            )
        if "target_pc" not in eval_cfg:
            raise ValueError(
                "checkpoint_selection.metric='mean_pair_completeness_at_k' requires "
                "evaluation.target_pc in (0, 1]"
            )
        target_pc = float(eval_cfg["target_pc"])
        if target_pc <= 0.0 or target_pc > 1.0:
            raise ValueError("evaluation.target_pc must be in (0, 1]")
        pc_tie = float(raw.get("pc_tie", 0.005))
        rr_tie = float(raw.get("rr_tie", 0.01))
        if not math.isfinite(pc_tie) or pc_tie < 0.0 or pc_tie > 1.0:
            raise ValueError("checkpoint_selection.pc_tie must be in [0, 1]")
        if not math.isfinite(rr_tie) or rr_tie < 0.0 or rr_tie > 1.0:
            raise ValueError("checkpoint_selection.rr_tie must be in [0, 1]")
        epoch_tiebreak = str(raw.get("epoch_tiebreak", "earliest")).strip().lower()
        if epoch_tiebreak != "earliest":
            raise ValueError("checkpoint_selection.epoch_tiebreak must be 'earliest'")
        selection.update(
            metric="mean_pair_completeness_at_k",
            k=k_values,
            target_pc=target_pc,
            pc_tie=pc_tie,
            rr_tie=rr_tie,
            epoch_tiebreak=epoch_tiebreak,
        )
    elif metric != "calibrated_threshold":
        raise ValueError(
            "checkpoint_selection.metric must be 'pc_at_k', 'mean_precision_at_k', "
            "'mean_pair_completeness_at_k', 'calibrated_threshold', "
            "'reduction_ratio_at_target_pc', 'calibration_transfer', or "
            "'weighted_mean_pc_at_k_and_transfer'"
        )
    return selection


def _format_pc_target(value: Any) -> str:
    return f"{float(value):.10g}"


def _checkpoint_selection_label(selection: dict[str, Any]) -> str:
    if selection.get("metric") == "pc_at_k":
        return f"{selection['validation_set']} PC@{int(selection.get('k', 10))}"
    if selection.get("metric") == "mean_precision_at_k":
        k_values = ",".join(str(k) for k in selection.get("k", []))
        return (
            f"{selection['validation_set']} mean precision@k={k_values} "
            f"tie<={float(selection.get('precision_tie', 0.01)):.10g} then mean RR, earliest epoch"
        )
    if selection.get("metric") == "mean_pair_completeness_at_k":
        k_values = ",".join(str(k) for k in selection.get("k", []))
        return (
            f"{selection['validation_set']} mean PC@k={k_values} "
            f"tie<={float(selection.get('pc_tie', 0.005)):.10g}, "
            f"RR@PC>={_format_pc_target(selection.get('target_pc', 0.95))} "
            f"tie<={float(selection.get('rr_tie', 0.01)):.10g}, "
            "then lower LoRA lr, lower head lr, earliest epoch"
        )
    if selection.get("metric") == "reduction_ratio_at_target_pc":
        return (
            f"{selection['validation_set']} RR@PC>="
            f"{_format_pc_target(selection.get('target_pc', 0.99))}"
        )
    if selection.get("metric") == "calibration_transfer":
        return (
            f"{selection['calibration_validation_set']} PC>="
            f"{_format_pc_target(selection.get('target_pc', 0.99))} -> "
            f"{selection['selection_validation_set']} PC>"
            f"{_format_pc_target(selection.get('minimum_selection_pc', 0.98))}, RR"
        )
    if selection.get("metric") == "weighted_mean_pc_at_k_and_transfer":
        k_values = ",".join(str(k) for k in selection["fixed_k"])
        return (
            f"weighted {selection['selection_validation_set']} mean PC@k={k_values} + "
            f"{selection['calibration_validation_set']} PC>="
            f"{_format_pc_target(selection.get('target_pc', 0.99))} transfer PC/RR"
        )
    return f"{selection['validation_set']} calibrated threshold"


def _target_pc_operating_point(metrics: dict[str, Any], target_pc: float) -> dict[str, Any]:
    point = metrics.get("calibrated_threshold")
    if not isinstance(point, dict):
        raise ValueError("Validation metrics are missing calibrated_threshold")
    required_finite = ("threshold", "target_pair_completeness", "pair_completeness", "reduction_ratio")
    values: dict[str, float] = {}
    for key in required_finite:
        if key not in point:
            raise ValueError(f"calibrated_threshold is missing {key!r}")
        try:
            value = float(point[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calibrated_threshold.{key} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"calibrated_threshold.{key} must be a finite number")
        values[key] = value
    if not math.isclose(
        values["target_pair_completeness"], float(target_pc), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "calibrated_threshold.target_pair_completeness does not match evaluation.target_pc: "
            f"{values['target_pair_completeness']} != {target_pc}"
        )
    if values["pair_completeness"] + 1e-12 < float(target_pc):
        raise ValueError(
            "Validation threshold did not reach target pair completeness: "
            f"{values['pair_completeness']} < {target_pc}"
        )
    try:
        candidate_pairs_value = float(point["candidate_pairs"])
        possible_pairs_value = float(point["possible_pairs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "calibrated_threshold must contain integer candidate_pairs and possible_pairs"
        ) from exc
    if (
        not math.isfinite(candidate_pairs_value)
        or not math.isfinite(possible_pairs_value)
        or not candidate_pairs_value.is_integer()
        or not possible_pairs_value.is_integer()
    ):
        raise ValueError(
            "calibrated_threshold must contain integer candidate_pairs and possible_pairs"
        )
    candidate_pairs = int(candidate_pairs_value)
    possible_pairs = int(possible_pairs_value)
    if possible_pairs <= 0 or candidate_pairs < 0 or candidate_pairs > possible_pairs:
        raise ValueError(
            "calibrated_threshold candidate counts must satisfy "
            "0 <= candidate_pairs <= possible_pairs and possible_pairs > 0"
        )
    expected_rr = 1.0 - candidate_pairs / float(possible_pairs)
    if not math.isclose(values["reduction_ratio"], expected_rr, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "calibrated_threshold.reduction_ratio is inconsistent with candidate_pairs/possible_pairs"
        )
    return dict(point)


def _mean_precision_at_k_result(metrics: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    k_values = [int(k) for k in selection["k"]]
    precision_at_k: dict[str, float] = {}
    reduction_ratio_at_k: dict[str, float] = {}
    for k in k_values:
        row = _metric_at_k(metrics, k)
        if "precision" not in row or "reduction_ratio" not in row:
            raise ValueError(
                f"k={k} metrics must include precision and reduction_ratio for mean_precision_at_k"
            )
        precision_at_k[str(k)] = _finite_float(row.get("precision"), -1.0)
        reduction_ratio_at_k[str(k)] = _finite_float(row.get("reduction_ratio"), -1.0)
    values = [*precision_at_k.values(), *reduction_ratio_at_k.values()]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("precision and reduction_ratio at the selection k values must be in [0, 1]")
    return {
        "k": k_values,
        "precision_at_k": precision_at_k,
        "mean_precision": sum(precision_at_k.values()) / float(len(k_values)),
        "reduction_ratio_at_k": reduction_ratio_at_k,
        "mean_reduction_ratio": sum(reduction_ratio_at_k.values()) / float(len(k_values)),
        "precision_tie": float(selection.get("precision_tie", 0.01)),
    }


def _mean_pair_completeness_at_k_result(
    metrics: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    from experiments.core.selection_rule import PC_TIE, RR_TIE

    k_values = [int(k) for k in selection["k"]]
    completeness_at_k: dict[str, float] = {}
    for k in k_values:
        row = _metric_at_k(metrics, k)
        if "pair_completeness" not in row:
            raise ValueError(
                f"k={k} metrics must include pair_completeness for mean_pair_completeness_at_k"
            )
        completeness_at_k[str(k)] = _finite_float(row.get("pair_completeness"), -1.0)
    if any(value < 0.0 or value > 1.0 for value in completeness_at_k.values()):
        raise ValueError("pair_completeness at the selection k values must be in [0, 1]")
    point = _target_pc_operating_point(metrics, float(selection["target_pc"]))
    return {
        "k": k_values,
        "pair_completeness_at_k": completeness_at_k,
        "mean_pair_completeness": sum(completeness_at_k.values()) / float(len(k_values)),
        "target_pair_completeness": float(selection["target_pc"]),
        "reduction_ratio_at_target_pc": float(point["reduction_ratio"]),
        "pc_tie": float(selection.get("pc_tie", PC_TIE)),
        "rr_tie": float(selection.get("rr_tie", RR_TIE)),
    }


def _mean_precision_at_k_is_better(
    candidate: dict[str, Any],
    best: dict[str, Any],
    precision_tie: float,
) -> bool:
    gap = float(candidate["mean_precision"]) - float(best["mean_precision"])
    if gap > precision_tie:
        return True
    if gap < -precision_tie:
        return False
    return float(candidate["mean_reduction_ratio"]) > float(best["mean_reduction_ratio"])


def _is_better_for_selection(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
    target_pc: float,
    selection: dict[str, Any],
    *,
    candidate_epoch: int | None = None,
    incumbent_epoch: int | None = None,
) -> bool:
    metric = selection.get("metric")
    if metric == "reduction_ratio_at_target_pc":
        candidate_point = _target_pc_operating_point(candidate, target_pc)
        if best is None:
            return True
        best_point = _target_pc_operating_point(best, target_pc)
        # Strict comparison intentionally leaves the earlier checkpoint selected
        # when two epochs have exactly the same reduction ratio.
        return float(candidate_point["reduction_ratio"]) > float(best_point["reduction_ratio"])
    if metric == "mean_precision_at_k":
        candidate_result = _mean_precision_at_k_result(candidate, selection)
        if best is None:
            return True
        return _mean_precision_at_k_is_better(
            candidate_result,
            _mean_precision_at_k_result(best, selection),
            float(selection.get("precision_tie", 0.01)),
        )
    if metric == "mean_pair_completeness_at_k":
        from experiments.core.selection_rule import replaces

        if best is None:
            return True
        candidate_result = _mean_pair_completeness_at_k_result(candidate, selection)
        best_result = _mean_pair_completeness_at_k_result(best, selection)
        return replaces(
            {
                "mean_pc": candidate_result["mean_pair_completeness"],
                "rr95": candidate_result["reduction_ratio_at_target_pc"],
                "lora_lr": selection.get("lora_lr"),
                "head_lr": selection.get("head_lr"),
                "epoch": candidate_epoch,
            },
            {
                "mean_pc": best_result["mean_pair_completeness"],
                "rr95": best_result["reduction_ratio_at_target_pc"],
                "lora_lr": selection.get("lora_lr"),
                "head_lr": selection.get("head_lr"),
                "epoch": incumbent_epoch,
            },
            pc_tie=float(candidate_result["pc_tie"]),
        )
    if metric != "pc_at_k":
        return _is_better(candidate, best, target_pc)
    k = int(selection.get("k", 10))
    candidate_row = _metric_at_k(candidate, k)
    if best is None:
        return True
    best_row = _metric_at_k(best, k)
    cand_key = (
        _finite_float(candidate_row.get("pair_completeness")),
        _finite_float(candidate_row.get("pair_quality")),
        -int(candidate_row.get("candidate_pairs") or 0),
        _finite_float((candidate.get("calibrated_threshold") or {}).get("pair_quality")),
    )
    best_key = (
        _finite_float(best_row.get("pair_completeness")),
        _finite_float(best_row.get("pair_quality")),
        -int(best_row.get("candidate_pairs") or 0),
        _finite_float((best.get("calibrated_threshold") or {}).get("pair_quality")),
    )
    return cand_key > best_key


def _early_stop_patience(train_cfg: dict[str, Any]) -> int | None:
    """Read how many consecutive non-improving epochs end training.

    Absent or null keeps the configured epoch budget. A positive integer stops
    after that many evaluated epochs in a row fail to replace the selected
    checkpoint. The epoch that trips the limit is kept and evaluated.
    """
    if "early_stop_non_improving_epochs" not in train_cfg:
        return None
    raw = train_cfg.get("early_stop_non_improving_epochs")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("train.early_stop_non_improving_epochs must be a positive integer or null")
    if raw < 1:
        raise ValueError("train.early_stop_non_improving_epochs must be a positive integer or null")
    return raw


def _validation_metric_curve(metrics: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
    """Compact pair-completeness and reduction-ratio summary for one validation set."""
    rows = {int(item["k"]): item for item in metrics.get("recall_at_k") or []}
    present = [k for k in k_values if k in rows] or sorted(rows)
    precision_at_k = {str(k): float(rows[k]["precision"]) for k in present}
    completeness_at_k = {str(k): float(rows[k]["pair_completeness"]) for k in present}
    reduction_at_k = {str(k): float(rows[k]["reduction_ratio"]) for k in present}
    calibrated = dict(metrics.get("calibrated_threshold") or {})
    count = float(len(present))
    return {
        "k": present,
        "precision_at_k": precision_at_k,
        "pair_completeness_at_k": completeness_at_k,
        "reduction_ratio_at_k": reduction_at_k,
        "mean_precision": (sum(precision_at_k.values()) / count) if present else None,
        "mean_pair_completeness": (sum(completeness_at_k.values()) / count) if present else None,
        "mean_reduction_ratio": (sum(reduction_at_k.values()) / count) if present else None,
        "calibrated_reduction_ratio": calibrated.get("reduction_ratio"),
        "calibrated_pair_completeness": calibrated.get("pair_completeness"),
    }


def _next_non_improving_streak(new_best: bool, streak: int, patience: int | None) -> tuple[int, bool]:
    """Return the updated streak and whether training should stop.

    A new best resets the streak. The first epoch is a new best, so two
    following misses are required before patience 2 stops training.
    """
    if patience is None or new_best:
        return 0, False
    updated = streak + 1
    return updated, updated >= patience


def _weighted_mean_pc_at_k_and_transfer_result(
    calibration_metrics: dict[str, Any],
    selection_metrics: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    calibration_point = _target_pc_operating_point(
        calibration_metrics, float(selection["target_pc"])
    )
    selection_point = dict(selection_metrics.get("calibrated_threshold") or {})
    threshold = float(calibration_point["threshold"])
    applied_threshold = float(selection_point.get("threshold", float("nan")))
    if not math.isclose(threshold, applied_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Selection-set threshold does not match the calibration-selected threshold: "
            f"{applied_threshold} != {threshold}"
        )
    k_values = [int(k) for k in selection["fixed_k"]]
    pc_at_k = {
        str(k): _finite_float(_metric_at_k(selection_metrics, k).get("pair_completeness"), -1.0)
        for k in k_values
    }
    if any(value < 0.0 or value > 1.0 for value in pc_at_k.values()):
        raise ValueError("Fixed-k pair completeness must be finite and in [0, 1]")
    mean_pc = sum(pc_at_k.values()) / float(len(pc_at_k))
    transferred_pc = _finite_float(selection_point.get("pair_completeness"), -1.0)
    transferred_rr = _finite_float(selection_point.get("reduction_ratio"), -1.0)
    if not 0.0 <= transferred_pc <= 1.0 or not 0.0 <= transferred_rr <= 1.0:
        raise ValueError("Transferred PC and RR must be finite and in [0, 1]")
    weights = dict(selection["weights"])
    score = (
        float(weights["mean_pc_at_k"]) * mean_pc
        + float(weights["transferred_pc"]) * transferred_pc
        + float(weights["transferred_rr"]) * transferred_rr
    )
    return {
        "calibration_validation_set": selection["calibration_validation_set"],
        "selection_validation_set": selection["selection_validation_set"],
        "target_pair_completeness": float(selection["target_pc"]),
        "calibration_threshold": threshold,
        "fixed_k": k_values,
        "pc_at_k": pc_at_k,
        "mean_pc_at_k": mean_pc,
        "transferred_pair_completeness": transferred_pc,
        "transferred_reduction_ratio": transferred_rr,
        "weights": weights,
        "score": score,
        "epoch_tiebreak": selection["epoch_tiebreak"],
    }


def _calibration_transfer_result(
    calibration_metrics: dict[str, Any],
    selection_metrics: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    calibration_point = _target_pc_operating_point(
        calibration_metrics, float(selection["target_pc"])
    )
    selection_point = dict(selection_metrics.get("calibrated_threshold") or {})
    threshold = float(calibration_point["threshold"])
    applied_threshold = float(selection_point.get("threshold", float("nan")))
    if not math.isclose(threshold, applied_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Selection-set threshold does not match the calibration-selected threshold: "
            f"{applied_threshold} != {threshold}"
        )
    selection_pc = _finite_float(selection_point.get("pair_completeness"), -1.0)
    selection_rr = _finite_float(selection_point.get("reduction_ratio"), -1.0)
    minimum_pc = float(selection["minimum_selection_pc"])
    return {
        "calibration_validation_set": selection["calibration_validation_set"],
        "selection_validation_set": selection["selection_validation_set"],
        "target_pair_completeness": float(selection["target_pc"]),
        "calibration_threshold": threshold,
        "calibration_pair_completeness": float(calibration_point["pair_completeness"]),
        "selection_pair_completeness": selection_pc,
        "selection_reduction_ratio": selection_rr,
        "minimum_selection_pair_completeness": minimum_pc,
        "eligible": bool(selection_pc > minimum_pc),
        "rr_tie_margin": float(selection["rr_tie_margin"]),
        "epoch_tiebreak": selection["epoch_tiebreak"],
    }


def _select_calibration_transfer_epoch(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in candidates if bool(row.get("eligible"))]
    if not eligible:
        return None
    maximum_rr = max(float(row["selection_reduction_ratio"]) for row in eligible)
    margin = float(eligible[0]["rr_tie_margin"])
    contenders = [
        row
        for row in eligible
        if float(row["selection_reduction_ratio"]) >= maximum_rr - margin
    ]
    return min(contenders, key=lambda row: int(row["epoch"]))


def _save_trainable_snapshot(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
) -> None:
    trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {"epoch": int(epoch), "trainable_state_dict": trainable, "optimizer_state_dict": optimizer.state_dict()},
        path,
    )


def _restore_trainable_snapshot(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trainable = payload.get("trainable_state_dict")
    if not isinstance(trainable, dict):
        raise ValueError(f"Candidate snapshot has no trainable_state_dict: {path}")
    parameters = dict(model.named_parameters())
    missing = sorted(set(trainable) - set(parameters))
    if missing:
        raise ValueError(f"Candidate snapshot contains unknown parameters: {missing[:20]}")
    with torch.no_grad():
        for name, value in trainable.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["epoch"])


def _model_revision(model_cfg: dict[str, Any]) -> str | None:
    revision = model_cfg.get("revision") or model_cfg.get("backbone_revision")
    return str(revision) if revision else None


def _base_reproducibility_metadata(
    exp,
    model_cfg: dict[str, Any],
    seed_metadata: dict[str, Any],
    target_pc: float,
) -> dict[str, Any]:
    return {
        "seed": seed_metadata["seed"],
        "determinism": seed_metadata,
        "dependencies": dependency_versions(),
        "git": git_info(exp.repo_root),
        "model": {
            "type": model_cfg.get("type") or model_cfg.get("model_type"),
            "backbone": model_cfg.get("backbone") or model_cfg.get("backbone_name"),
            "revision": _model_revision(model_cfg),
        },
        "validation": {"target_pc": float(target_pc)},
        "rng_state_after_seeding": rng_state_fingerprints(torch),
    }


def _checkpoint_reproducibility_metadata(base: dict[str, Any]) -> dict[str, Any]:
    return {**base, "rng_state_at_checkpoint": rng_state_fingerprints(torch)}


def _class_count(items: list[MetricImageItem]) -> int:
    return len({item.class_id for item in items})


def _current_lr(optimizer: torch.optim.Optimizer) -> float | None:
    if not optimizer.param_groups:
        return None
    return float(max(group.get("lr", 0.0) for group in optimizer.param_groups))


def _call_loss(
    loss_fn: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    origins: torch.Tensor | None,
    device: str,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the original loss, or a variant that reads origins and dataset indices."""
    if getattr(loss_fn, "accepts_batch_context", False):
        origin_tensor = None if origins is None else origins.to(device)
        index_tensor = None if indices is None else indices.to(device)
        return loss_fn(embeddings, labels, origin_tensor, indices=index_tensor)
    if not getattr(loss_fn, "needs_origins", False) and not getattr(loss_fn, "cross_origin_only", False):
        return loss_fn(embeddings, labels)
    if origins is None:
        raise ValueError("cross-origin loss requires a historic or modern role on every training image")
    return loss_fn(embeddings, labels, origins.to(device))


def _loss_mining_stats(loss_fn: torch.nn.Module) -> dict[str, int] | None:
    stats = getattr(loss_fn, "last_stats", None)
    keys = (
        "valid_anchors",
        "positive_pairs",
        "negative_pairs",
        "mined_positive_pairs",
        "mined_negative_pairs",
    )
    if stats is None or any(not hasattr(stats, key) for key in keys):
        return None
    return {key: int(getattr(stats, key)) for key in keys}


def _mean_mining_stats(rows: list[dict[str, int]]) -> dict[str, float] | None:
    if not rows:
        return None
    means = {
        key: sum(float(row[key]) for row in rows) / float(len(rows))
        for key in rows[0]
    }
    valid_anchors = means["valid_anchors"]
    means["mined_positive_pairs_per_valid_anchor"] = (
        means["mined_positive_pairs"] / valid_anchors if valid_anchors else 0.0
    )
    means["mined_negative_pairs_per_valid_anchor"] = (
        means["mined_negative_pairs"] / valid_anchors if valid_anchors else 0.0
    )
    means["logical_steps"] = float(len(rows))
    return means


def _model_param_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _unexpected_lora_trainable_names(model: torch.nn.Module) -> list[str]:
    if getattr(model, "model_type", "") not in {
        "dinov3_lora_qv_projection",
        "siglip2_lora_qv_projection",
        "clip_lora_qv_projection",
        "resnet_hf_lora_conv_projection",
    }:
        return []
    allowed = ("lora", "projection", "dropout", *getattr(model, "backbone_trainable_prefixes", ()))
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad and not any(key in name.lower() for key in allowed)]


def _refresh_cross_era_graph(
    sampler: Any,
    dataset: MetricLearningImageDataset,
    model: torch.nn.Module,
    transform: object,
    *,
    device: str,
    batch_size: int,
    epoch: int,
    logger: ConsoleTrainingLogger,
) -> None:
    """Embed one modern and one historic view per painting and store confusers."""
    if not isinstance(sampler, CrossEraGraphPKBatchSampler):
        return
    every = sampler.refresh_every_epochs
    if every <= 0 or (epoch != 1 and (epoch - 1) % every != 0):
        return
    pairs = probe_plan(dataset.labels, dataset.roles)
    if len(pairs) < 2:
        raise RuntimeError(
            "Cross-era graph sampling needs at least two paintings with a modern and a historic view; "
            f"found {len(pairs)}"
        )
    probe_items = []
    class_ids: list[int] = []
    for class_id, modern_index, historic_index in pairs:
        probe_items.append(dataset.items[modern_index])
        probe_items.append(dataset.items[historic_index])
        class_ids.append(int(class_id))
    was_training = model.training
    try:
        _image_ids, _probe_classes, embeddings = embed_items(
            model,
            probe_items,
            transform=transform,
            batch_size=max(1, int(batch_size)),
            device=device,
            num_workers=4,
        )
    finally:
        model.train(was_training)
    modern = embeddings[0::2]
    historic = embeddings[1::2]
    neighbors, guarded = neighbor_lists(modern, historic, class_ids, pool=sampler.neighbors_pool)
    sampler.set_neighbors(neighbors)
    filled = sum(1 for ranked in neighbors.values() if ranked)
    logger.emit(
        "cross_era_graph",
        epoch=epoch,
        probe_paintings=len(class_ids),
        paintings_with_confusers=filled,
        guarded_pairs=guarded,
        pool=sampler.neighbors_pool,
    )


def _sampler_batch_size(sampler: Any) -> int | None:
    classes = getattr(sampler, "classes_per_batch", None)
    images = getattr(sampler, "images_per_class", None)
    if classes is None or images is None:
        return None
    return int(classes) * int(images)


def _sampler_summary(sampler: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": sampler.__class__.__name__,
        "classes_per_batch": int(sampler.classes_per_batch),
        "images_per_class": int(sampler.images_per_class),
        "batches_per_epoch": len(sampler),
        "num_classes": len(sampler.classes),
    }
    if isinstance(sampler, OriginBalancedPKBatchSampler):
        summary["per_origin"] = dict(sampler.per_origin)
    if isinstance(sampler, RoleQuotaPKBatchSampler):
        summary["quotas"] = dict(sampler.quotas)
    if isinstance(sampler, CrossEraGraphPKBatchSampler):
        summary["hard_fraction"] = sampler.hard_fraction
        summary["neighbors_pool"] = sampler.neighbors_pool
        summary["refresh_every_epochs"] = sampler.refresh_every_epochs
    if isinstance(sampler, RoleStratifiedPKBatchSampler):
        summary.update(
            anchor_role=sampler.anchor_role,
            positive_roles=list(sampler.positive_roles),
            role_counts=sampler.role_counts,
            ignored_role_counts=sampler.ignored_role_counts,
        )
    return summary


def train_from_experiment(
    experiment_path: Path | str,
    *,
    run_id: str | None = None,
    epochs_override: int | None = None,
    device_override: str | None = None,
    seed_override: int | None = None,
    gradcache_chunk_size_override: int | None = None,
) -> dict[str, Any]:
    # Finetuning can access gated backbones just like the frozen-evaluation
    # pipeline. Load the repository-local environment before model construction
    # so HF_TOKEN and cache settings are available in Slurm worktrees.
    env_load()
    exp = load_experiment_config(experiment_path)
    split = load_split(exp.split_file)
    validate_split_manifest(split)
    cfg = _finetuning_config(exp.raw)
    model_cfg = dict(cfg.get("model") or {})
    if not model_cfg:
        raise ValueError("Finetuning config must contain a model block")
    train_cfg = dict(cfg.get("train") or {})
    repeat_seeds = training_repeat_seeds(cfg)
    if seed_override is not None:
        if repeat_seeds and int(seed_override) not in repeat_seeds:
            raise ValueError(
                f"Training seed {int(seed_override)} is not declared in "
                f"finetuning.training_repeats.seeds={list(repeat_seeds)}"
            )
        train_cfg["seed"] = int(seed_override)
    # Keep every persisted finetuning block honest about the effective CLI
    # override rather than retaining the seed from the source YAML.
    cfg["train"] = train_cfg
    sampler_cfg = dict(cfg.get("sampler") or {})
    loss_cfg = dict(cfg.get("loss") or {})
    optimizer_cfg = dict(cfg.get("optimizer") or {})
    eval_cfg = dict(cfg.get("evaluation") or {})
    seed = int(train_cfg.get("seed", 42))
    if repeat_seeds and seed not in repeat_seeds:
        raise ValueError(
            f"Effective training seed {seed} is not declared in "
            f"finetuning.training_repeats.seeds={list(repeat_seeds)}"
        )
    configuration_id = str(cfg.get("configuration_id") or exp.experiment_id)
    training_repeats = {
        "seeds": list(repeat_seeds),
        "require_complete": bool(
            (cfg.get("training_repeats") or {}).get("require_complete", True)
        ),
    } if repeat_seeds else None
    deterministic = bool(train_cfg.get("deterministic", True))
    deterministic_warn_only = bool(train_cfg.get("deterministic_warn_only", True))
    device = _device(device_override or train_cfg.get("device"))
    run_dir = prepare_run_dir(_run_dir(exp, run_id))
    logger = ConsoleTrainingLogger.from_config(dict(cfg.get("logging") or {}), device=device)
    logger.start()
    logger.emit("setup_start", experiment=str(experiment_path), run_dir=str(run_dir), device=device)
    seed_metadata = seed_everything(seed, deterministic=deterministic, warn_only=deterministic_warn_only)
    target_pc = float(eval_cfg.get("target_pc", 0.99))
    reproducibility = _base_reproducibility_metadata(exp, model_cfg, seed_metadata, target_pc)

    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("finetuning.data is required")
    train_selector = data_cfg["train"]
    train_items = load_subset_items(exp.dataset, split, train_selector)
    if isinstance(train_selector, dict) and train_selector.get("online_historic_view"):
        train_items = append_online_historic_views(train_items)
    hard_manifest = train_selector.get("hard_historic_manifest") if isinstance(train_selector, dict) else None
    if hard_manifest:
        manifest_path = _resolve_sampler_path(hard_manifest, exp)
        if manifest_path is None:
            raise ValueError("finetuning.data.train.hard_historic_manifest must not be empty")
        train_items = append_hard_historic_views(train_items, manifest_path)
    validation_sets = _validation_sets(exp, cfg, split)
    primary_validation = _primary_validation_set(validation_sets)
    checkpoint_selection = _checkpoint_selection_config(eval_cfg, validation_sets, primary_validation.name)
    if checkpoint_selection.get("metric") == "mean_pair_completeness_at_k":
        try:
            checkpoint_selection["lora_lr"] = float(optimizer_cfg["lr_lora"])
            checkpoint_selection["head_lr"] = float(optimizer_cfg["lr_head"])
        except KeyError as exc:
            raise ValueError(
                "mean_pair_completeness_at_k requires optimizer.lr_lora and optimizer.lr_head"
            ) from exc
    primary_val_items = load_subset_items(primary_validation.dataset, primary_validation.split, primary_validation.selector)
    train_transform = build_image_transform(cfg, train=True)
    eval_transform = build_image_transform(cfg, train=False)
    train_dataset = MetricLearningImageDataset(train_items, train_transform)  # type: ignore[arg-type]
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_options: dict[str, Any] = {}
    if num_workers > 0 and "prefetch_factor" in train_cfg:
        prefetch_factor = int(train_cfg["prefetch_factor"])
        if prefetch_factor <= 0:
            raise ValueError("train.prefetch_factor must be positive")
        loader_options["prefetch_factor"] = prefetch_factor
    sampler = _build_sampler(train_dataset, sampler_cfg, train_cfg, exp)
    sampler_summary = _sampler_summary(sampler)
    gradient_cache = _gradient_cache_config(
        train_cfg,
        sampler,
        chunk_size_override=gradcache_chunk_size_override,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_metric_batch,
        worker_init_fn=_seed_worker,
        generator=_data_loader_generator(seed),
        **loader_options,
    )
    batches_per_epoch = len(loader)
    sampler_summary["gradient_cache"] = dict(gradient_cache)
    if isinstance(sampler, (RoleStratifiedPKBatchSampler, RoleQuotaPKBatchSampler, CrossEraGraphPKBatchSampler)) or gradient_cache["enabled"]:
        logger.emit("sampler", **sampler_summary)
    # Evaluate frozen references (e.g. the untrained backbone) before building the
    # trainable model so the two never co-reside in GPU memory. Reference
    # embeddings come from the shared cache; a cache hit skips model construction
    # entirely and every reference releases its CUDA memory in its finally block.
    references = evaluate_references(
        exp,
        cfg,
        validation_sets,
        run_options={**dict(exp.raw.get("run") or {}), "fail_on_missing_images": True},
        embedding_options={"batch_size": int(eval_cfg.get("batch_size", train_cfg.get("eval_batch_size", 64)))},
        top_k=tuple(int(k) for k in eval_cfg.get("top_k", [1, 5, 10, 25, 50, 100])),
        target_pc=target_pc,
        logger=logger,
        compute=True,
    )
    model = build_trainable_model(model_cfg).to(device)
    if gradient_cache["enabled"]:
        batch_norm_names = _gradient_cache_batch_norm_names(model)
        if batch_norm_names:
            raise ValueError(
                "Gradient caching does not support models with BatchNorm because graphless and "
                "replay forwards would update state twice: " + ", ".join(batch_norm_names[:20])
            )
    loss_fn = build_loss(loss_cfg)
    optimizer = _optimizer(model, optimizer_cfg)
    epochs = int(epochs_override or train_cfg.get("epochs", 10))
    if epochs <= 0:
        raise ValueError("train.epochs must be positive")
    scheduler = _scheduler(optimizer, {**train_cfg, **optimizer_cfg}, epochs * batches_per_epoch)
    epoch_scheduler = isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    precision = str(train_cfg.get("precision", "float32"))
    scaler = _grad_scaler(device, precision)
    eval_batch_size = int(eval_cfg.get("batch_size", train_cfg.get("eval_batch_size", 64)))
    validation_batches = math.ceil(len(primary_val_items) / float(eval_batch_size))
    total_steps = epochs * batches_per_epoch
    total_params, trainable_params = _model_param_counts(model)
    logger.configure(epochs=epochs, batches_per_epoch=batches_per_epoch, total_steps=total_steps)
    logger.emit(
        "train_start",
        run_dir=str(run_dir),
        device=device,
        precision=precision,
        seed=seed,
        epochs=epochs,
        train_images=len(train_items),
        train_classes=_class_count(train_items),
        val_images=len(primary_val_items),
        val_classes=_class_count(primary_val_items),
        validation_sets=[item.name for item in validation_sets],
        primary_validation_set=primary_validation.name,
        checkpoint_selection=_checkpoint_selection_label(checkpoint_selection),
        batch_size=gradient_cache["effective_batch_size"],
        physical_batch_size=gradient_cache["chunk_size"],
        gradient_cache_enabled=gradient_cache["enabled"],
        chunks_per_step=gradient_cache["chunks_per_step"],
        batches_per_epoch=batches_per_epoch,
        total_steps=total_steps,
        workers=num_workers,
        prefetch_factor=loader_options.get("prefetch_factor"),
        eval_batch_size=eval_batch_size,
        target_pc=target_pc,
    )
    trainable_ratio = float(trainable_params) / float(total_params) if total_params else 0.0
    unexpected_trainable = _unexpected_lora_trainable_names(model)
    logger.emit(
        "model",
        backbone=model_cfg.get("backbone") or model_cfg.get("backbone_name"),
        pooling=dict(model_cfg.get("pooling") or {}).get(
            "type",
            model_cfg.get("pooling_type"),
        ),
        freeze_backbone=model_cfg.get("freeze_backbone"),
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_ratio=f"{trainable_ratio:.4%}",
        lr=_current_lr(optimizer),
        weight_decay=optimizer_cfg.get("weight_decay"),
    )
    frozen_features = getattr(model, "features", None)
    backbone_still_open = False
    if bool(model_cfg.get("freeze_backbone", False)) and frozen_features is not None:
        backbone_still_open = any(parameter.requires_grad for parameter in frozen_features.parameters())
    elif frozen_features is None and bool(model_cfg.get("freeze_backbone", True)) and trainable_ratio > 0.05:
        backbone_still_open = True
    if backbone_still_open:
        logger.emit("warning", message="trainable parameter ratio exceeds 5%; check frozen backbone", trainable_ratio=f"{trainable_ratio:.4%}")
    if unexpected_trainable:
        logger.emit("warning", message="unexpected trainable LoRA parameters", parameters=unexpected_trainable[:20])
    cuda_training_device = torch.device(device) if torch.device(device).type == "cuda" else None
    if cuda_training_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_training_device)

    atomic_write_json(
        run_dir / "training_config.json",
        {
            "experiment": exp.resolved,
            "finetuning": cfg,
            "sampler_summary": sampler_summary,
            "gradient_cache": gradient_cache,
            "checkpoint_selection": checkpoint_selection,
            "reproducibility": reproducibility,
        },
    )
    checkpoint_training_config = {
        "configuration_id": configuration_id,
        "training_repeats": training_repeats,
        "model": model_cfg,
        "sampler": sampler_cfg,
        "sampler_summary": sampler_summary,
        "gradient_cache": gradient_cache,
        "loss": loss_cfg,
        "optimizer": optimizer_cfg,
        "train": train_cfg,
        "evaluation": eval_cfg,
        "checkpoint_selection": checkpoint_selection,
        "reproducibility": reproducibility,
    }
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_path = run_dir / "best_checkpoint.pt"
    last_path = run_dir / "last_checkpoint.pt"
    transfer_selection = checkpoint_selection.get("metric") in {
        "calibration_transfer",
        "weighted_mean_pc_at_k_and_transfer",
    }
    retrospective_transfer_selection = checkpoint_selection.get("metric") == "calibration_transfer"
    early_stop_patience = None if retrospective_transfer_selection else _early_stop_patience(train_cfg)
    epochs_without_new_best = 0
    early_stopped = False
    early_stop_reason: str | None = None
    transfer_candidates: list[dict[str, Any]] = []
    candidate_dir = run_dir / "checkpoint_candidates"
    if retrospective_transfer_selection:
        candidate_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    for epoch in range(1, epochs + 1):
        logger.epoch_start(epoch, epochs, _current_lr(optimizer))
        _refresh_cross_era_graph(
            sampler,
            train_dataset,
            model,
            eval_transform,
            device=device,
            batch_size=eval_batch_size,
            epoch=epoch,
            logger=logger,
        )
        sampler.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        epoch_mining_rows: list[dict[str, int]] = []
        for batch_index, batch in enumerate(loader, start=1):
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
            origins = batch.get("origins")
            indices = batch.get("indices")
            optimizer.zero_grad(set_to_none=True)
            if gradient_cache["enabled"]:
                loss, observed = _gradient_cache_backward(
                    model,
                    loss_fn,
                    pixel_values,
                    labels,
                    chunk_size=int(gradient_cache["chunk_size"]),
                    device=device,
                    precision=precision,
                    scaler=scaler,
                    verify_replay=bool(gradient_cache["verify_replay"] and global_step == 0),
                    origins=origins,
                    indices=indices,
                )
                samples_in_step = int(pixel_values.shape[0])
            else:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                with _autocast(device, precision):
                    embeddings = model(pixel_values)
                    loss = _call_loss(loss_fn, embeddings, labels, origins, device, indices)
                observed = embeddings.detach()
                scaler.scale(loss).backward()
                samples_in_step = int(pixel_values.shape[0])
            observe = getattr(loss_fn, "observe_batch", None)
            if observe is not None:
                observe(observed, labels, origins, indices)
            mining_stats = _loss_mining_stats(loss_fn)
            if mining_stats is not None:
                epoch_mining_rows.append(mining_stats)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and not epoch_scheduler:
                scheduler.step()
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            global_step += 1
            logger.update_train(
                epoch=epoch,
                batch=batch_index,
                global_step=global_step,
                loss=loss_value,
                lr=_current_lr(optimizer),
                samples=samples_in_step,
            )
        model.eval()
        validation_results: dict[str, Any] = {}
        primary_metrics: dict[str, Any] | None = None
        calibration_name = checkpoint_selection.get("calibration_validation_set")
        selection_name = checkpoint_selection.get("selection_validation_set")
        evaluation_order = list(validation_sets)
        if transfer_selection:
            by_name = {item.name: item for item in validation_sets}
            evaluation_order = [by_name[str(calibration_name)], by_name[str(selection_name)]] + [
                item for item in validation_sets if item.name not in {calibration_name, selection_name}
            ]
        calibration_threshold: float | None = None
        for validation_set in evaluation_order:
            val_items = load_subset_items(validation_set.dataset, validation_set.split, validation_set.selector)
            logger.validation_start(epoch, math.ceil(len(val_items) / float(eval_batch_size)), name=validation_set.name)
            fixed_threshold = calibration_threshold if transfer_selection and validation_set.name == selection_name else None
            val_metrics = _evaluate_model(
                model,
                val_items,
                eval_transform,
                cfg,
                device,
                progress_callback=logger.update_validation,
                dataset=validation_set.dataset,
                split=validation_set.split,
                retrieval_task_specs=validation_set.retrieval_task_specs,
                threshold=fixed_threshold,
            )
            val_metrics = {
                **val_metrics,
                "validation_set": validation_set.name,
                "is_primary_validation": validation_set.primary,
                "dataset_id": validation_set.dataset.dataset_id,
                "split_id": validation_set.split.get("split_id"),
                "selector": validation_set.selector,
                "selection_task_id": val_metrics.get("primary_task_id") or "all_pairs",
                "threshold_source": "calibration_validation_set" if fixed_threshold is not None else "selected_on_set",
            }
            validation_results[validation_set.name] = val_metrics
            if validation_set.name == calibration_name:
                calibration_threshold = float(
                    _target_pc_operating_point(val_metrics, target_pc)["threshold"]
                )
            if validation_set.primary:
                primary_metrics = val_metrics
        if primary_metrics is None:
            raise RuntimeError("Primary validation set was not evaluated")
        mean_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_mining": _mean_mining_stats(epoch_mining_rows),
            "validation": {"primary": primary_validation.name, "sets": validation_results},
        }
        history.append(row)
        if row["train_mining"] is not None:
            logger.emit("train_mining", epoch=epoch, **row["train_mining"])
        val_metrics = validation_results[str(checkpoint_selection["validation_set"])]
        selection_point = (
            _target_pc_operating_point(val_metrics, target_pc)
            if checkpoint_selection["metric"] == "reduction_ratio_at_target_pc"
            else dict(val_metrics["calibrated_threshold"])
        )
        threshold = float(calibration_threshold if transfer_selection else selection_point["threshold"])
        logger.checkpoint_start(epoch)
        save_checkpoint(
            last_path,
            model,
            optimizer=optimizer,
            epoch=epoch,
            training_config=checkpoint_training_config,
            preprocessing_config=preprocessing_config(cfg),
            normalization_config=normalization_config(cfg),
            validation_threshold=threshold,
            validation_metrics=val_metrics,
            reproducibility_metadata=_checkpoint_reproducibility_metadata(reproducibility),
        )
        logger.emit("checkpoint_saved", kind="last", epoch=epoch, path=str(last_path))
        if retrospective_transfer_selection:
            transfer_result = _calibration_transfer_result(
                validation_results[str(calibration_name)], val_metrics, checkpoint_selection
            )
            transfer_result["epoch"] = epoch
            transfer_result["snapshot_path"] = None
            if transfer_result["eligible"]:
                snapshot_path = candidate_dir / f"epoch_{epoch:03d}.pt"
                _save_trainable_snapshot(snapshot_path, model, optimizer, epoch=epoch)
                transfer_result["snapshot_path"] = str(snapshot_path)
                logger.emit("checkpoint_candidate_saved", path=str(snapshot_path), **transfer_result)
            transfer_candidates.append(transfer_result)
            row["checkpoint_selection_result"] = dict(transfer_result)
            new_best = False
        else:
            if checkpoint_selection["metric"] == "weighted_mean_pc_at_k_and_transfer":
                selection_result = _weighted_mean_pc_at_k_and_transfer_result(
                    validation_results[str(calibration_name)], val_metrics, checkpoint_selection
                )
                selection_result["epoch"] = epoch
                previous_score = (
                    _finite_float(
                        history[best_epoch - 1]["checkpoint_selection_result"].get("score"),
                        -1.0,
                    )
                    if best_epoch is not None
                    else -1.0
                )
                new_best = float(selection_result["score"]) > previous_score
                selection_result["is_new_best"] = bool(new_best)
                row["checkpoint_selection_result"] = selection_result
            else:
                new_best = _is_better_for_selection(
                    val_metrics,
                    best_metrics,
                    target_pc,
                    checkpoint_selection,
                    candidate_epoch=epoch,
                    incumbent_epoch=best_epoch,
                )
            if checkpoint_selection["metric"] == "mean_precision_at_k":
                selection_result = _mean_precision_at_k_result(val_metrics, checkpoint_selection)
                selection_result["epoch"] = epoch
                selection_result["is_new_best"] = bool(new_best)
                row["checkpoint_selection_result"] = selection_result
            if checkpoint_selection["metric"] == "mean_pair_completeness_at_k":
                selection_result = _mean_pair_completeness_at_k_result(val_metrics, checkpoint_selection)
                selection_result["epoch"] = epoch
                selection_result["is_new_best"] = bool(new_best)
                row["checkpoint_selection_result"] = selection_result
            if checkpoint_selection["metric"] == "reduction_ratio_at_target_pc":
                row["checkpoint_selection_result"] = {
                    "target_pair_completeness": float(selection_point["target_pair_completeness"]),
                    "pair_completeness": float(selection_point["pair_completeness"]),
                    "threshold": threshold,
                    "reduction_ratio": float(selection_point["reduction_ratio"]),
                    "candidate_pairs": int(selection_point["candidate_pairs"]),
                    "possible_pairs": int(selection_point["possible_pairs"]),
                    "is_new_best": bool(new_best),
                }
            if new_best:
                best_metrics = val_metrics
                best_epoch = epoch
                save_checkpoint(
                    best_path,
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    training_config=checkpoint_training_config,
                    preprocessing_config=preprocessing_config(cfg),
                    normalization_config=normalization_config(cfg),
                    validation_threshold=threshold,
                    validation_metrics=val_metrics,
                    reproducibility_metadata=_checkpoint_reproducibility_metadata(reproducibility),
                )
                logger.emit("checkpoint_saved", kind="best", epoch=epoch, path=str(best_path), reason="new_best", selection=_checkpoint_selection_label(checkpoint_selection))
        if early_stop_patience is None:
            epochs_without_new_best = 0
            stop_for_patience = False
        else:
            epochs_without_new_best, stop_for_patience = _next_non_improving_streak(
                bool(new_best), epochs_without_new_best, early_stop_patience
            )
            if stop_for_patience:
                early_stopped = True
                early_stop_reason = (
                    f"{early_stop_patience} consecutive epochs did not beat the selected checkpoint"
                )
        curve_k = [int(k) for k in checkpoint_selection.get("k") or eval_cfg.get("top_k") or []]
        epoch_curves = [
            {
                "epoch": int(item["epoch"]),
                "sets": {
                    name: _validation_metric_curve(metrics, curve_k)
                    for name, metrics in item["validation"]["sets"].items()
                },
            }
            for item in history
        ]
        atomic_write_json(
            run_dir / "training_history.json",
            {
                "schema_version": 2,
                "references": references,
                "history": history,
                "checkpoint_selection": checkpoint_selection,
                "early_stop": {
                    "non_improving_epochs": early_stop_patience,
                    "epochs_without_new_best": epochs_without_new_best,
                    "stopped": early_stopped,
                    "reason": early_stop_reason,
                    "best_epoch": best_epoch,
                },
            },
        )
        atomic_write_json(
            run_dir / "epoch_metric_curves.json",
            {
                "schema_version": 1,
                "k": curve_k,
                "epochs": epoch_curves,
            },
        )
        write_training_reports(
            history,
            run_dir,
            references=references,
            best_epoch=best_epoch,
            last_epoch=epoch,
            best_validation_set=str(checkpoint_selection["validation_set"]),
            checkpoint_selection=checkpoint_selection,
        )
        if epoch_scheduler and scheduler is not None:
            scheduler.step()
        logger.epoch_end(epoch=epoch, epochs=epochs, train_loss=mean_loss, metrics=val_metrics, new_best=new_best)
        if early_stopped:
            logger.emit(
                "early_stop",
                epoch=epoch,
                best_epoch=best_epoch,
                patience=early_stop_patience,
                reason=early_stop_reason,
            )
            break
    completed_epochs = int(history[-1]["epoch"]) if history else 0
    if retrospective_transfer_selection:
        selected_candidate = _select_calibration_transfer_epoch(transfer_candidates)
        if selected_candidate is not None:
            best_epoch = int(selected_candidate["epoch"])
            snapshot_path = Path(str(selected_candidate["snapshot_path"]))
            restored_epoch = _restore_trainable_snapshot(snapshot_path, model, optimizer)
            if restored_epoch != best_epoch:
                raise RuntimeError(f"Restored candidate epoch {restored_epoch} != selected epoch {best_epoch}")
            best_metrics = dict(
                history[best_epoch - 1]["validation"]["sets"][str(selection_name)]
            )
            best_calibration_metrics = dict(
                history[best_epoch - 1]["validation"]["sets"][str(calibration_name)]
            )
            best_threshold_value = float(
                _target_pc_operating_point(best_calibration_metrics, target_pc)["threshold"]
            )
            history[best_epoch - 1]["checkpoint_selection_result"]["is_selected"] = True
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                epoch=best_epoch,
                training_config=checkpoint_training_config,
                preprocessing_config=preprocessing_config(cfg),
                normalization_config=normalization_config(cfg),
                validation_threshold=best_threshold_value,
                validation_metrics={
                    **best_metrics,
                    "calibration_transfer": history[best_epoch - 1]["checkpoint_selection_result"],
                },
                reproducibility_metadata=_checkpoint_reproducibility_metadata(reproducibility),
            )
            logger.emit("checkpoint_saved", kind="best", epoch=best_epoch, path=str(best_path), reason="retrospective_calibration_transfer", selection=_checkpoint_selection_label(checkpoint_selection))
        for candidate in transfer_candidates:
            snapshot = candidate.get("snapshot_path")
            if snapshot:
                Path(str(snapshot)).unlink(missing_ok=True)
            candidate["snapshot_path"] = None
            history[int(candidate["epoch"]) - 1]["checkpoint_selection_result"]["snapshot_path"] = None
        candidate_dir.rmdir()
        atomic_write_json(
            run_dir / "training_history.json",
            {
                "schema_version": 2,
                "references": references,
                "history": history,
                "checkpoint_selection": checkpoint_selection,
                "calibration_transfer_candidates": transfer_candidates,
            },
        )
    best_point = (
        dict(best_metrics["calibrated_threshold"])
        if best_metrics is not None and best_metrics.get("calibrated_threshold")
        else None
    )
    best_threshold = float(best_point["threshold"]) if best_point is not None else None
    report_paths = write_training_reports(
        history,
        run_dir,
        references=references,
        best_epoch=best_epoch,
        last_epoch=completed_epochs,
        best_validation_set=str(checkpoint_selection["validation_set"]),
        checkpoint_selection=checkpoint_selection,
    )
    training_memory = (
        {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_training_device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_training_device)),
        }
        if cuda_training_device is not None
        else None
    )
    atomic_write_json(
        run_dir / "training_summary.json",
        {
            "schema_version": 2,
            "configuration_id": configuration_id,
            "training_seed": seed,
            "training_repeats": training_repeats,
            "references": references,
            "epochs": epochs,
            "epochs_completed": completed_epochs,
            "early_stop": {
                "non_improving_epochs": early_stop_patience,
                "epochs_without_new_best": epochs_without_new_best,
                "stopped": early_stopped,
                "reason": early_stop_reason,
            },
            "global_steps": global_step,
            "gradient_cache": gradient_cache,
            "training_memory": training_memory,
            "train_mining_by_epoch": [row.get("train_mining") for row in history],
            "best_validation_threshold": best_threshold,
            "best_validation_target_pair_completeness": (
                float(best_point["target_pair_completeness"])
                if best_point is not None and best_point.get("target_pair_completeness") is not None
                else None
            ),
            "best_validation_pair_completeness": (
                float(best_point["pair_completeness"]) if best_point is not None else None
            ),
            "best_validation_pair_quality": (
                float(best_point["pair_quality"])
                if best_point is not None and best_point.get("pair_quality") is not None
                else None
            ),
            "best_validation_reduction_ratio": (
                float(best_point["reduction_ratio"])
                if best_point is not None and best_point.get("reduction_ratio") is not None
                else None
            ),
            "best_validation_candidate_pairs": (
                int(best_point["candidate_pairs"])
                if best_point is not None and best_point.get("candidate_pairs") is not None
                else None
            ),
            "best_validation_possible_pairs": (
                int(best_point["possible_pairs"])
                if best_point is not None and best_point.get("possible_pairs") is not None
                else None
            ),
            "best_epoch": best_epoch,
            "training_split": {
                "split_id": split.get("split_id"),
                "split_sha256": sha256_file(exp.split_file),
                "split_seed": (split.get("split_strategy") or {}).get("seed"),
                "split_strategy": split.get("split_strategy"),
                "train_selector": (cfg.get("data") or {}).get("train"),
                "validation_sets": [
                    {
                        "name": item.get("name"),
                        "subset": item.get("subset"),
                        "roles": item.get("roles"),
                        "primary": bool(item.get("primary", False)),
                    }
                    for item in ((cfg.get("data") or {}).get("validation") or [])
                    if isinstance(item, dict)
                ],
            },
            "best_validation_set": checkpoint_selection["validation_set"],
            "best_validation_task_id": (
                best_metrics.get("selection_task_id") if best_metrics is not None else None
            ),
            "checkpoint_selection": checkpoint_selection,
            "best_checkpoint_selection_result": (
                history[best_epoch - 1].get("checkpoint_selection_result")
                if best_epoch is not None
                else None
            ),
            "calibration_transfer_candidates": transfer_candidates if retrospective_transfer_selection else None,
            "calibration_transfer_selection_status": (
                "selected" if best_epoch is not None else "no_eligible_epoch"
            ) if retrospective_transfer_selection else None,
            "reports": report_paths,
            "reproducibility": {**reproducibility, "rng_state_at_end": rng_state_fingerprints(torch)},
        },
    )
    result = {
        "status": "ok",
        "run_dir": str(run_dir),
        "configuration_id": configuration_id,
        "training_seed": seed,
        "training_repeats": training_repeats,
        "training_split": {
            "split_id": split.get("split_id"),
            "split_sha256": sha256_file(exp.split_file),
            "split_seed": (split.get("split_strategy") or {}).get("seed"),
        },
        "checkpoint_selection": checkpoint_selection,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "reports": report_paths,
        "epochs": completed_epochs,
        "epochs_planned": epochs,
        "early_stopped": early_stopped,
        "global_steps": global_step,
    }
    logger.train_done(
        status="ok",
        epochs=completed_epochs,
        global_steps=global_step,
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        run_dir=str(run_dir),
    )
    logger.stop()
    return result
