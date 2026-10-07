"""CLI for config-driven image-matching experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Support both the documented module invocation and direct execution of this
# sole top-level code file.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.core.aggregate_resources import aggregate_resource_paths
from experiments.core.aggregate_runs import aggregate_paths
from experiments.core.artifacts import PACKAGE_DIR, sha256_file
from experiments.core.winner_config_id import winner_config_id
from experiments.core.config_schema import load_dataset_config, load_experiment_config, load_yaml, resolve_post_evaluations
from experiments.core.finetuning.post_evaluation import (
    PostEvaluationError,
    checkpoint_model_config,
    preflight_post_evaluations,
    run_post_evaluations,
)
from experiments.core.latex_tables import write_run_latex_tables
from experiments.core.model_downloader import ModelDownloadError, download_experiment_models
from experiments.core.preflight import HuggingFaceRevisionError, preflight_huggingface_revisions
from experiments.core.random_baseline import parse_random_baseline_config
from experiments.core.runner import ModelEvaluationError, run_experiment
from experiments.core.split_schema import (
    ids_for_multirole_selector,
    load_split,
    materialize_split,
    resolve_retrieval_tasks,
    validate_split_manifest,
    write_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate",
        help="Validate experiment, dataset, split, and model references",
    )
    validate.add_argument("--experiment", required=True, type=Path)
    validate.set_defaults(handler=_validate)

    make = sub.add_parser("make-split", help="Materialize a static split JSON from a dataset DB")
    make.add_argument("--dataset", required=True, type=Path)
    make.add_argument("--preset", required=True)
    make.add_argument("--output", required=True, type=Path)
    make.add_argument("--seed", type=int, default=42)
    make.add_argument(
        "--train-share",
        type=float,
        default=None,
        help=(
            "Train-class share for the class_disjoint_role_v1 preset. "
            "Validation and test split the remainder equally unless --validation-share is set."
        ),
    )
    make.add_argument(
        "--validation-share",
        type=float,
        default=None,
        help=(
            "Validation-class share for the class_disjoint_role_v1 preset. "
            "Test receives the remainder; use 0 with --train-share 0 for a test-only split."
        ),
    )
    make.set_defaults(handler=_make_split)

    plan = sub.add_parser(
        "make-generation-plan",
        help="Partition classes and assets before hard-synth generation",
    )
    plan_source = plan.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--dataset", type=Path, help="Dataset YAML with dataset.db")
    plan_source.add_argument(
        "--image-dir",
        type=Path,
        help="Flat deduplicated image directory; every file is one distinct class.",
    )
    plan.add_argument("--recipe", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument(
        "--held-out-profile",
        choices=("archival", "print", "cropped_record", "framed_photo"),
        default=None,
        help="Build a leave-one-profile-out plan instead of compound OOD.",
    )
    plan.add_argument("--frames", type=Path, default=PACKAGE_DIR / "datasets/image-distorter/horizontal_frames")
    plan.add_argument("--overlays", type=Path, default=PACKAGE_DIR / "datasets/image-distorter/reflection_overlay")
    plan.add_argument("--textures", type=Path, default=PACKAGE_DIR / "datasets/image-distorter/wall_textures")
    plan.set_defaults(handler=_make_generation_plan)

    run = sub.add_parser("run", help="Run an experiment, optionally restricted to one model")
    run.add_argument("--experiment", required=True, type=Path)
    run.add_argument("--model", default=None, help="Global model_id/checkpoint ID to run")
    run.add_argument("--run-id", default=None)
    run.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for an eval template with models.from_finetuning_checkpoint=true",
    )
    run.add_argument(
        "--checkpoint-model-id",
        default=None,
        help="Optional result identity for --checkpoint (default: content-derived)",
    )
    run.add_argument(
        "--checkpoint-label",
        default=None,
        help="Optional display label for --checkpoint",
    )
    run.add_argument(
        "--configuration-id",
        default=None,
        help="Stable final-configuration identity for repeated checkpoint evaluations",
    )
    run.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Training seed lineage for --checkpoint split-sensitivity evaluations",
    )
    run.add_argument(
        "--only-download-models",
        action="store_true",
        help="Download and verify configured model weights, then exit",
    )
    run.set_defaults(handler=_run)

    rotation_validate = sub.add_parser(
        "validate-split-rotation",
        help="Validate a predefined frozen split-rotation evaluation",
    )
    rotation_validate.add_argument("--experiment", required=True, type=Path)
    rotation_validate.set_defaults(handler=_validate_split_rotation)

    rotation_run = sub.add_parser(
        "run-split-rotation",
        help="Run the ordinary evaluator over predefined static split realizations",
    )
    rotation_run.add_argument("--experiment", required=True, type=Path)
    rotation_run.add_argument("--model", default=None)
    rotation_run.add_argument("--run-id", default=None)
    rotation_run.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for an eval template with models.from_finetuning_checkpoint=true",
    )
    rotation_run.add_argument(
        "--checkpoint-model-id",
        default=None,
        help="Optional result identity for --checkpoint (default: content-derived)",
    )
    rotation_run.add_argument(
        "--checkpoint-label",
        default=None,
        help="Optional display label for --checkpoint",
    )
    rotation_run.add_argument(
        "--configuration-id",
        default=None,
        help="Stable final-configuration identity for repeated checkpoint evaluations",
    )
    rotation_run.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Training seed lineage for --checkpoint split-sensitivity evaluations",
    )
    rotation_choice = rotation_run.add_mutually_exclusive_group(required=True)
    rotation_choice.add_argument("--split-seed", type=int, default=None)
    rotation_choice.add_argument("--all-splits", action="store_true")
    rotation_run.set_defaults(handler=_run_split_rotation)

    train_cmd = sub.add_parser("train", help="Train a metric-learning projection checkpoint")
    train_cmd.add_argument("--experiment", required=True, type=Path)
    train_cmd.add_argument("--run-id", default=None)
    train_cmd.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    train_cmd.add_argument("--device", default=None, help="Override train.device")
    train_cmd.add_argument("--seed", type=int, default=None, help="Override finetuning.train.seed")
    train_cmd.add_argument(
        "--gradcache-chunk-size",
        type=int,
        default=None,
        help="Override the physical GradCache chunk. The effective PK batch stays as configured.",
    )
    train_cmd.set_defaults(handler=_train)

    met_dl = sub.add_parser(
        "download-met-benchmark",
        help="Download official Met benchmark archives (run on the cluster)",
    )
    met_dl.add_argument(
        "--root",
        type=Path,
        default=Path("data/datasets/met_benchmark").expanduser(),
    )
    met_dl.add_argument("--execute", action="store_true")
    met_dl.add_argument("--no-mini", action="store_true")
    met_dl.add_argument("--skip-full-train", action="store_true")
    met_dl.set_defaults(handler=_download_met_benchmark)

    post_eval = sub.add_parser(
        "run-post-evaluations",
        help="Run or resume linked evaluations for an existing finetuning run",
    )
    post_eval.add_argument("--experiment", required=True, type=Path)
    post_eval.add_argument("--training-run", required=True, type=Path)
    post_eval.set_defaults(handler=_run_post_evaluations)

    mine = sub.add_parser("mine-hard-negatives", help="Mine train-only hard negatives from a checkpoint")
    mine.add_argument("--experiment", required=True, type=Path)
    mine.add_argument("--checkpoint", required=True, type=Path)
    mine.add_argument("--output", type=Path, default=None)
    mine.add_argument("--subset", required=True)
    mine.add_argument("--top-m", type=int, default=50)
    mine.add_argument("--epoch", type=int, default=0)
    mine.add_argument("--batch-size", type=int, default=64)
    mine.add_argument("--device", default="auto")
    mine.set_defaults(handler=_mine_hard_negatives)

    eval_ckpt = sub.add_parser("evaluate-checkpoint", help="Evaluate a checkpoint with validation-selected or fixed threshold")
    eval_ckpt.add_argument("--experiment", required=True, type=Path)
    eval_ckpt.add_argument("--checkpoint", required=True, type=Path)
    eval_ckpt.add_argument("--subset", required=True)
    eval_ckpt.add_argument("--threshold", type=float, default=None)
    eval_ckpt.add_argument("--target-pc", type=float, default=0.99)
    eval_ckpt.add_argument("--batch-size", type=int, default=64)
    eval_ckpt.add_argument("--device", default="auto")
    eval_ckpt.add_argument("--output", type=Path, default=None)
    eval_ckpt.set_defaults(handler=_evaluate_checkpoint)

    aggregate = sub.add_parser(
        "aggregate",
        help="Aggregate eval.json files below one or more run roots",
    )
    aggregate.add_argument("--runs", nargs="*", default=[], type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--json-output", type=Path, default=None)
    aggregate.add_argument("--approved-run-manifest", "--manifest", type=Path, default=None)
    aggregate.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs without run.json or without status='completed'",
    )
    aggregate.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate (experiment_id, split_id, model_id, task_id) aggregate sources",
    )
    aggregate.set_defaults(handler=_aggregate)

    resources = sub.add_parser(
        "aggregate-resources",
        help="Aggregate resource_metrics.json files below run roots",
    )
    resources.add_argument("--runs", required=True, nargs="+", type=Path)
    resources.add_argument("--output", required=True, type=Path)
    resources.add_argument("--json-output", type=Path, default=None)
    resources.add_argument(
        "--include-cached",
        action="store_true",
        help="Include rows where embeddings were reused from cache",
    )
    resources.set_defaults(handler=_aggregate_resources)

    latex = sub.add_parser(
        "latex-tables",
        help="Render fixed-k and target-PC LaTeX tables for one run",
    )
    latex.add_argument("--run", required=True, type=Path)
    latex.add_argument("--output-dir", type=Path, default=None)
    latex.add_argument("--task", default=None, help="Optional task_id for multi-task runs")
    latex.set_defaults(handler=_latex_tables)

    figures = sub.add_parser(
        "figures",
        help="Render the default chart bundle (blocking tables + figures + bar/PQ-PC charts) for one run",
    )
    figures.add_argument("--run", required=True, type=Path)
    figures.add_argument("--task", default=None, help="Optional task_id for multi-task runs")
    figures.add_argument(
        "--format",
        action="append",
        choices=["pdf", "svg", "png"],
        default=None,
        help="Figure file format (repeatable; default: pdf)",
    )
    figures.set_defaults(handler=_figures)

    analysis_init = sub.add_parser(
        "analysis-init",
        help="Discover compatible evaluation runs and create a persistent analysis config",
    )
    analysis_init.add_argument("--reference-run", required=True, type=Path)
    analysis_init.add_argument("--runs-root", required=True, type=Path)
    analysis_init.add_argument("--output", required=True, type=Path)
    analysis_init.add_argument("--analysis-id", default=None)
    analysis_init.add_argument(
        "--exclude-new-models",
        action="store_true",
        help="Initialize newly discovered models with include=false",
    )
    analysis_init.add_argument(
        "--source-policy",
        choices=("prefer_existing", "newest_complete", "richest_then_newest"),
        default="prefer_existing",
    )
    analysis_init.set_defaults(handler=_analysis_init)

    multisplit_init = sub.add_parser(
        "analysis-init-multisplit",
        help="Create a persistent multisplit analysis config from a rotation experiment",
    )
    multisplit_init.add_argument("--experiment", required=True, type=Path)
    multisplit_init.add_argument("--reference-run", required=True, type=Path)
    multisplit_init.add_argument("--runs-root", required=True, type=Path)
    multisplit_init.add_argument("--output", required=True, type=Path)
    multisplit_init.add_argument("--analysis-id", default=None)
    multisplit_init.add_argument("--exclude-new-models", action="store_true")
    multisplit_init.add_argument(
        "--source-policy",
        choices=("prefer_existing", "newest_complete", "richest_then_newest"),
        default="prefer_existing",
    )
    multisplit_init.set_defaults(handler=_analysis_init_multisplit)

    analysis_update = sub.add_parser(
        "analysis-update",
        help="Refresh an analysis config while preserving model/chart selections",
    )
    analysis_update.add_argument("--config", required=True, type=Path)
    analysis_update.set_defaults(handler=_analysis_update)

    analysis_render = sub.add_parser(
        "analysis-render",
        help="Render configured charts from selected model artifacts across runs",
    )
    analysis_render.add_argument("--config", required=True, type=Path)
    analysis_render.set_defaults(handler=_analysis_render)

    sensitivity = sub.add_parser(
        "split-sensitivity",
        help="Aggregate and render predefined Wikidata split-sensitivity evaluations",
    )
    sensitivity.add_argument("--config", required=True, type=Path)
    sensitivity.set_defaults(handler=_split_sensitivity)

    prewarm = sub.add_parser(
        "prepare-reference-cache",
        help="Populate the shared frozen-reference evaluation cache for experiments",
    )
    prewarm.add_argument(
        "--experiment",
        action="append",
        required=True,
        type=Path,
        help="Experiment config with finetuning references (repeatable)",
    )
    prewarm.set_defaults(handler=_prepare_reference_cache)

    single_stream_prepare = sub.add_parser(
        "prepare-single-stream-benchmark-models",
        help="Download missing exact Hugging Face revisions before benchmark submission",
    )
    single_stream_prepare.add_argument("--config", required=True, type=Path)
    single_stream_prepare.set_defaults(handler=_prepare_single_stream_benchmark_models)

    single_stream_validate = sub.add_parser(
        "validate-single-stream-benchmark",
        help="Validate benchmark code, schedule, dataset, checkpoints, and model cache before submission",
    )
    single_stream_validate.add_argument("--config", required=True, type=Path)
    single_stream_validate.set_defaults(handler=_validate_single_stream_benchmark)

    single_stream_gpu = sub.add_parser(
        "validate-single-stream-gpu",
        help="Validate allocated GPU identity, BF16 support, and exact SDPA backend",
    )
    single_stream_gpu.add_argument("--config", required=True, type=Path)
    single_stream_gpu.set_defaults(handler=_validate_single_stream_gpu)

    single_stream = sub.add_parser(
        "benchmark-single-stream",
        help="Run the counterbalanced B=1 embedding-latency benchmark in one allocation",
    )
    single_stream.add_argument("--config", required=True, type=Path)
    single_stream.add_argument("--run-id", default=None)
    single_stream.add_argument(
        "--resume",
        action="store_true",
        help="Resume only if the existing run matches the frozen manifest",
    )
    single_stream.set_defaults(handler=_benchmark_single_stream)
    return parser


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    raw_document = load_yaml(args.experiment)
    if raw_document.get("experiment_type") == "cross_dataset_threshold_transfer":
        from experiments.core.cross_dataset_threshold_transfer import (
            validate_cross_dataset_experiment,
        )

        return {
            "status": "ok",
            **validate_cross_dataset_experiment(
                args.experiment,
                allow_dynamic_models=True,
                # Match ordinary `validate`: static manifests and model
                # references are checked locally; cluster image roots are
                # required by run/post-evaluation preflight.
                require_image_roots=False,
            ),
        }

    # Dynamic checkpoint eval templates can be validated before a concrete
    # checkpoint exists; normal model includes still resolve as usual.
    exp = load_experiment_config(args.experiment, allow_dynamic_models=True)
    split = load_split(exp.split_file)
    validate_split_manifest(split)
    eval_raw = dict(exp.raw.get("evaluation") or {})
    tasks = resolve_retrieval_tasks(split, eval_raw.get("retrieval_tasks")) if eval_raw else []
    finetune_selectors = _validate_finetuning_selectors(exp.raw, split)
    random_baseline = parse_random_baseline_config(
        eval_raw.get("random_baseline")
    )
    revision_checks = preflight_huggingface_revisions(exp)
    post_evaluations = preflight_post_evaluations(exp) if resolve_post_evaluations(exp) else []
    return {
        "status": "ok",
        "experiment_id": exp.experiment_id,
        "dataset": dict(split.get("dataset") or {}),
        "num_retrieval_tasks": len(tasks),
        "finetuning_selectors": finetune_selectors,
        "models": [model.model_id for model in exp.models],
        "huggingface_revisions": revision_checks,
        "random_baseline": random_baseline.model_id if random_baseline else None,
        "dynamic_checkpoint_model": bool((exp.raw.get("models") or {}).get("from_finetuning_checkpoint")),
        "post_evaluations": [
            {"experiment": str(spec.experiment), "experiment_id": linked.experiment_id, "checkpoint": spec.checkpoint}
            for spec, linked in post_evaluations
        ],
    }


def _validate_finetuning_selectors(raw: dict[str, Any], split: dict[str, Any]) -> dict[str, int]:
    finetuning = raw.get("finetuning")
    if not isinstance(finetuning, dict):
        return {}
    data = finetuning.get("data")
    if not isinstance(data, dict):
        return {}
    counts: dict[str, int] = {}
    for name, selector in data.items():
        if isinstance(selector, dict):
            counts[str(name)] = len(
                ids_for_multirole_selector(split, str(selector["subset"]), selector["roles"])
            )
    return counts


def _make_split(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset_config(args.dataset)
    split = materialize_split(
        dataset,
        args.preset,
        args.seed,
        train_share=args.train_share,
        validation_share=args.validation_share,
    )
    write_split(args.output, split)
    return {
        "status": "ok",
        "output": str(args.output),
        "split_id": split["split_id"],
    }


def _make_generation_plan(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.generation_plan import (
        make_generation_plan,
        make_image_directory_generation_plan,
        write_generation_plan,
    )

    kwargs = {
        "seed": args.seed,
        "asset_directories": {
            "frame": args.frames,
            "overlay": args.overlays,
            "texture": args.textures,
        },
        "held_out_profile": args.held_out_profile,
    }
    if args.image_dir is not None:
        plan = make_image_directory_generation_plan(args.image_dir, args.recipe, **kwargs)
    else:
        dataset = load_dataset_config(args.dataset)
        plan = make_generation_plan(dataset, args.recipe, **kwargs)
    write_generation_plan(args.output, plan)
    return {
        "status": "ok",
        "output": str(args.output),
        "plan_id": plan["plan_id"],
        "num_classes": len(plan["class_assignments"]),
        "num_assets": len(plan["asset_assignments"]),
    }


def _checkpoint_injection(
    args: argparse.Namespace,
) -> tuple[list[Any] | None, dict[str, Any] | None]:
    """Build the finetuned checkpoint model for a from_finetuning_checkpoint template."""
    checkpoint = getattr(args, "checkpoint", None)
    configuration_id = getattr(args, "configuration_id", None)
    training_seed = getattr(args, "training_seed", None)
    if checkpoint is None:
        if configuration_id is not None or training_seed is not None:
            raise ValueError("--configuration-id/--training-seed require --checkpoint")
        return None, None
    injected_models = [
        checkpoint_model_config(
            checkpoint,
            model_id=getattr(args, "checkpoint_model_id", None),
            display_name=getattr(args, "checkpoint_label", None),
        )
    ]
    if args.model is not None and args.model != injected_models[0].model_id:
        raise ValueError(
            f"--model={args.model!r} does not match injected checkpoint model_id "
            f"{injected_models[0].model_id!r}"
        )
    parent_training = None
    if configuration_id is not None or training_seed is not None:
        if configuration_id is None or training_seed is None:
            raise ValueError("--configuration-id and --training-seed must be supplied together")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        parent_training = {
            "experiment_id": str(configuration_id),
            "configuration_id": str(configuration_id),
            "training_seed": int(training_seed),
            "run_id": checkpoint_path.parent.name,
            "run_dir": str(checkpoint_path.parent),
            "checkpoint_kind": "explicit",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        identity = winner_config_id(checkpoint_path.parent)
        if identity:
            parent_training["winner_config_id"] = identity
    return injected_models, parent_training


def _run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = getattr(args, "checkpoint", None)
    if getattr(args, "only_download_models", False):
        if checkpoint is not None:
            raise ValueError("--checkpoint cannot be combined with --only-download-models")
        downloaded = download_experiment_models(args.experiment, model_filter=args.model)
        return {
            "status": "ok",
            "num_downloaded_models": len(downloaded),
            "downloaded_models": downloaded,
        }
    injected_models, parent_training = _checkpoint_injection(args)
    run_dir = run_experiment(
        args.experiment,
        model_filter=args.model,
        run_id=args.run_id,
        injected_models=injected_models,
        parent_training=parent_training,
    )
    return {"status": "ok", "run_dir": str(run_dir)}


def _validate_split_rotation(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.split_rotation_evaluation import validate_split_rotation_evaluation

    return {"status": "ok", **validate_split_rotation_evaluation(args.experiment)}


def _run_split_rotation(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.split_rotation_evaluation import run_split_rotation_evaluation

    injected_models, parent_training = _checkpoint_injection(args)
    return {
        "status": "ok",
        **run_split_rotation_evaluation(
            args.experiment,
            model_filter=args.model,
            split_seed=args.split_seed,
            all_splits=bool(args.all_splits),
            run_id=args.run_id,
            injected_models=injected_models,
            parent_training=parent_training,
        ),
    }


def _train(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.finetuning.train import train_from_experiment
    from experiments.core.resource_metrics import release_torch_cuda_memory

    # Fail on malformed/missing linked evaluation protocols before spending GPU
    # hours on training.
    training_exp = load_experiment_config(args.experiment)
    linked = preflight_post_evaluations(training_exp)
    result = train_from_experiment(
        args.experiment,
        run_id=args.run_id,
        epochs_override=args.epochs,
        device_override=args.device,
        seed_override=getattr(args, "seed", None),
        gradcache_chunk_size_override=getattr(args, "gradcache_chunk_size", None),
    )
    if linked:
        # train_from_experiment has returned, so its model/optimizer frame is no
        # longer live. Release allocator/compiler state before checkpoint reload.
        release_torch_cuda_memory(reset_compiler=True)
        evaluations = run_post_evaluations(
            training_exp,
            result["run_dir"],
            training_result=result,
        )
        result["post_evaluations"] = evaluations
    return result


def _download_met_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.datasets.met_benchmark.download_met_benchmark import download_met_benchmark

    return download_met_benchmark(
        args.root,
        include_mini=not args.no_mini,
        execute=bool(args.execute),
        skip_full_train=bool(args.skip_full_train),
    )


def _run_post_evaluations(args: argparse.Namespace) -> dict[str, Any]:
    evaluations = run_post_evaluations(args.experiment, args.training_run)
    return {
        "status": "ok",
        "training_run": str(args.training_run),
        "post_evaluations": evaluations,
    }


def _resolved_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_eval_inputs(experiment_path: Path) -> tuple[Any, dict[str, Any]]:
    exp = load_experiment_config(experiment_path)
    split = load_split(exp.split_file)
    validate_split_manifest(split)
    return exp, split


def _default_mining_output(checkpoint: Path) -> Path:
    stem = checkpoint.expanduser().resolve().stem
    return Path("outputs") / "mining" / stem / "hard_negatives.parquet"


def _mine_hard_negatives(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.finetuning.mining import mine_hard_negatives_for_checkpoint

    exp, split = _load_eval_inputs(args.experiment)
    output = args.output or (PACKAGE_DIR / _default_mining_output(args.checkpoint))
    return mine_hard_negatives_for_checkpoint(
        exp,
        split,
        args.checkpoint,
        output=output,
        subset_name=args.subset,
        top_m=args.top_m,
        epoch=args.epoch,
        batch_size=args.batch_size,
        device=_resolved_device(args.device),
    )


def _evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.finetuning.evaluate import evaluate_checkpoint_on_subset

    exp, split = _load_eval_inputs(args.experiment)
    result = evaluate_checkpoint_on_subset(
        exp,
        split,
        args.checkpoint,
        args.subset,
        threshold=args.threshold,
        target_pc=args.target_pc,
        batch_size=args.batch_size,
        device=_resolved_device(args.device),
        output=args.output,
    )
    compact = dict(result)
    compact.pop("image_ids", None)
    return {"status": "ok", **compact}


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    rows = aggregate_paths(
        args.runs,
        args.output,
        args.json_output,
        include_incomplete=args.include_incomplete,
        approved_run_manifest=args.approved_run_manifest,
        allow_duplicates=args.allow_duplicates,
    )
    return {"status": "ok", "rows": len(rows), "output": str(args.output)}


def _aggregate_resources(args: argparse.Namespace) -> dict[str, Any]:
    rows = aggregate_resource_paths(
        args.runs,
        args.output,
        args.json_output,
        include_cached=args.include_cached,
    )
    return {"status": "ok", "rows": len(rows), "output": str(args.output)}


def _latex_tables(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.blocking_tables import write_run_blocking_tables

    paths = write_run_latex_tables(args.run, args.output_dir, task_id=args.task)
    paths.update(write_run_blocking_tables(args.run, args.output_dir, task_id=args.task))
    return {"status": "ok", "outputs": {key: str(path) for key, path in paths.items()}}


def _figures(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.run_figures import DEFAULT_FIGURE_FORMATS
    from experiments.core.run_charts import write_run_charts

    formats = tuple(args.format) if args.format else DEFAULT_FIGURE_FORMATS
    result = write_run_charts(args.run, task_id=args.task, formats=formats)
    return {"status": "ok", **result}


def _analysis_init(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.analysis_config import initialize_analysis_config

    path, config, summary = initialize_analysis_config(
        args.reference_run,
        args.runs_root,
        args.output,
        analysis_id=args.analysis_id,
        new_models_include=not args.exclude_new_models,
        source_policy=args.source_policy,
    )
    return {
        "status": "ok",
        "config": str(path),
        "analysis_id": config.get("analysis_id"),
        **summary,
    }


def _analysis_init_multisplit(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.multisplit_analysis import initialize_multisplit_analysis_config

    path, config, summary = initialize_multisplit_analysis_config(
        args.experiment,
        args.reference_run,
        args.runs_root,
        args.output,
        analysis_id=args.analysis_id,
        new_models_include=not args.exclude_new_models,
        source_policy=args.source_policy,
    )
    return {
        "status": "ok",
        "config": str(path),
        "analysis_id": config.get("analysis_id"),
        **summary,
    }


def _analysis_update(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.analysis_config import update_analysis_config

    path, config, summary = update_analysis_config(args.config)
    return {
        "status": "ok",
        "config": str(path),
        "analysis_id": config.get("analysis_id"),
        **summary,
    }


def _analysis_render(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.analysis_charts import write_analysis_charts

    return {"status": "ok", **write_analysis_charts(args.config)}


def _split_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.split_sensitivity import write_split_sensitivity_analysis

    return {"status": "ok", **write_split_sensitivity_analysis(args.config)}


def _reference_embedding_options(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    return {"batch_size": int(eval_cfg.get("batch_size", 64))}


def _prepare_reference_cache(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.cache_paths import require_shared_cache_writable
    from experiments.core.finetuning.reference_evaluation import (
        load_or_compute_reference_result,
        reference_result_cached,
        resolve_reference_models,
        resolve_reference_work_items,
    )
    from experiments.core.finetuning.validation_sets import resolve_validation_sets
    from experiments.core.resource_metrics import release_torch_cuda_memory

    require_shared_cache_writable(require_env=True)
    experiments = list(args.experiment or [])
    if not experiments:
        raise ValueError("prepare-reference-cache requires at least one --experiment")

    # Resolve every reference work item, deduplicating identical result keys.
    unique: dict[str, dict[str, Any]] = {}
    for experiment_path in experiments:
        exp = load_experiment_config(experiment_path)
        split = load_split(exp.split_file)
        validate_split_manifest(split)
        cfg = dict(exp.raw.get("finetuning") or {})
        references = resolve_reference_models(cfg, exp)
        if not references:
            raise ValueError(
                f"Experiment {experiment_path} has no finetuning.evaluation.references"
            )
        validation_sets = resolve_validation_sets(exp, cfg, split)
        eval_cfg = dict(cfg.get("evaluation") or {})
        top_k = tuple(int(k) for k in eval_cfg.get("top_k", [1, 5, 10, 25, 50, 100]))
        target_pc = float(eval_cfg.get("target_pc", 0.99))
        run_options = {"fail_on_missing_images": True}
        embedding_options = _reference_embedding_options(eval_cfg)
        for work_item in resolve_reference_work_items(
            exp, cfg, validation_sets, references, top_k=top_k, target_pc=target_pc
        ):
            unique.setdefault(
                work_item.result_key,
                {
                    "work_item": work_item,
                    "run_options": run_options,
                    "embedding_options": embedding_options,
                },
            )

    results: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    for result_key, entry in unique.items():
        work_item = entry["work_item"]
        row = {
            "result_key": result_key,
            "reference_id": work_item.reference.id,
            "validation_set": work_item.validation_set.name,
            "model_id": work_item.reference.model.model_id,
        }
        if reference_result_cached(result_key):
            cache_hits += 1
            row["status"] = "cache_hit"
        else:
            load_or_compute_reference_result(
                work_item,
                run_options=entry["run_options"],
                embedding_options=entry["embedding_options"],
                compute=True,
            )
            release_torch_cuda_memory(reset_compiler=True)
            cache_misses += 1
            row["status"] = "cache_miss"
        results.append(row)

    return {
        "status": "ok",
        "experiments": len(experiments),
        "unique_reference_results": len(unique),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "results": results,
    }


def _prepare_single_stream_benchmark_models(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.single_stream_benchmark import prepare_single_stream_model_cache

    return prepare_single_stream_model_cache(args.config)


def _validate_single_stream_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.single_stream_benchmark import validate_single_stream_setup

    return validate_single_stream_setup(args.config)


def _validate_single_stream_gpu(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.single_stream_benchmark import validate_single_stream_gpu_setup

    return validate_single_stream_gpu_setup(args.config)


def _benchmark_single_stream(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.core.single_stream_benchmark import run_single_stream_benchmark

    return run_single_stream_benchmark(
        args.config,
        run_id=args.run_id,
        resume=bool(args.resume),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except ModelDownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(json.dumps({"status": "error", "downloaded_models": exc.results}, indent=2))
        return 1
    except HuggingFaceRevisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(json.dumps({"status": "error", "huggingface_revisions": exc.results}, indent=2))
        return 1
    except ModelEvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "status": "error",
                    "run_dir": str(exc.run_dir),
                    "model_failures": exc.failures,
                },
                indent=2,
            )
        )
        return 1
    except PostEvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "status": "error",
                    "training_run": str(exc.training_run_dir),
                    "post_evaluations": exc.results,
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
