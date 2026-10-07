"""Orchestrate normal retrieval evaluations after a finetuning run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.core.adapter_registry import AdapterConfig
from experiments.core.artifacts import atomic_write_json, sha256_file
from experiments.core.winner_config_id import winner_config_id
from experiments.core.config_schema import (
    ExperimentConfig,
    ModelConfig,
    PostEvaluationConfig,
    load_experiment_config,
    resolve_post_evaluations,
)
from experiments.core.runner import run_experiment
from experiments.core.split_schema import load_split, resolve_retrieval_tasks, validate_split_manifest


class PostEvaluationError(RuntimeError):
    """Raised after all configured post-evaluations have been attempted."""

    def __init__(self, training_run_dir: Path, results: list[dict[str, Any]]):
        self.training_run_dir = training_run_dir
        self.results = results
        failures = [row for row in results if row.get("status") == "failed"]
        detail = "; ".join(
            f"{row.get('experiment_id') or row.get('experiment')}: {row.get('error')}"
            for row in failures
        )
        super().__init__(f"{len(failures)} post-evaluation(s) failed: {detail}")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:160] or "evaluation"


def checkpoint_model_config(
    checkpoint: Path | str,
    *,
    model_id: str | None = None,
    display_name: str | None = None,
    checkpoint_sha256: str | None = None,
) -> ModelConfig:
    """Create an in-memory model config for a finetuning checkpoint."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Finetuning checkpoint not found: {path}")
    digest = checkpoint_sha256 or sha256_file(path)
    effective_id = model_id or f"local/checkpoint/{digest[:16]}"
    raw: dict[str, Any] = {
        "model_id": effective_id,
        "display_name": display_name or f"Finetuned checkpoint {digest[:12]}",
        "group": "finetuned",
        "adapter_kwargs": {"checkpoint_path": str(path)},
        "embedding": {
            "normalize": True,
            "dtype": "float32",
            "pooling": "default",
        },
    }
    adapter = AdapterConfig(
        adapter_name="finetuned_checkpoint_adapter",
        model_id=effective_id,
        kwargs={"checkpoint_path": str(path)},
    )
    return ModelConfig(
        path=path,
        model_id=effective_id,
        adapter=adapter,
        raw=raw,
        revision=None,
        presentation_overrides=frozenset({"display_name", "group"}),
    )


def preflight_post_evaluations(
    experiment: ExperimentConfig | Path | str,
) -> list[tuple[PostEvaluationConfig, ExperimentConfig]]:
    """Resolve and validate every linked eval before expensive training starts."""
    training_exp = (
        experiment
        if isinstance(experiment, ExperimentConfig)
        else load_experiment_config(experiment)
    )
    resolved: list[tuple[PostEvaluationConfig, ExperimentConfig]] = []
    seen_ids: set[str] = set()
    for spec in resolve_post_evaluations(training_exp):
        if not spec.experiment.is_file():
            raise FileNotFoundError(f"Post-evaluation experiment not found: {spec.experiment}")
        eval_exp = load_experiment_config(spec.experiment, allow_dynamic_models=True)
        model_block = eval_exp.raw.get("models") or {}
        if not isinstance(model_block, dict) or model_block.get("from_finetuning_checkpoint") is not True:
            raise ValueError(
                f"Post-evaluation {spec.experiment} must set "
                "models.from_finetuning_checkpoint: true"
            )
        if (eval_exp.raw.get("evaluation") or {}).get("random_baseline"):
            raise ValueError(
                f"Post-evaluation {spec.experiment} must not enable evaluation.random_baseline; "
                "evaluate shared analytical/frozen baselines once outside the finetuning grid"
            )
        if eval_exp.experiment_id in seen_ids:
            raise ValueError(
                f"Post-evaluation experiment_id {eval_exp.experiment_id!r} is linked more than once"
            )
        seen_ids.add(eval_exp.experiment_id)
        if isinstance(eval_exp.raw.get("split_rotation_evaluation"), dict):
            from experiments.core.split_rotation_evaluation import load_split_rotation_spec

            load_split_rotation_spec(spec.experiment, allow_dynamic_models=True)
        else:
            split = load_split(eval_exp.split_file)
            validate_split_manifest(split)
            eval_raw = dict(eval_exp.raw.get("evaluation") or {})
            resolve_retrieval_tasks(split, eval_raw.get("retrieval_tasks"))
            calibration = eval_raw.get("threshold_calibration")
            if isinstance(calibration, dict):
                resolve_retrieval_tasks(split, [calibration["task"]])
        if eval_exp.raw.get("experiment_type") == "cross_dataset_threshold_transfer":
            from experiments.core.cross_dataset_threshold_transfer import (
                preflight_loaded_cross_dataset_experiment,
            )

            preflight_loaded_cross_dataset_experiment(
                eval_exp,
                require_image_roots=True,
            )
        image_root = eval_exp.dataset.image_root
        if image_root is None or not Path(image_root).is_dir():
            raise FileNotFoundError(
                f"Post-evaluation dataset image_root not found: {image_root} ({spec.experiment})"
            )
        resolved.append((spec, eval_exp))
    return resolved


def _checkpoint_for_spec(
    spec: PostEvaluationConfig,
    training_run_dir: Path,
    training_result: dict[str, Any] | None,
) -> Path:
    result_key = f"{spec.checkpoint}_checkpoint"
    value = (training_result or {}).get(result_key)
    path = Path(value).expanduser() if value else training_run_dir / f"{spec.checkpoint}_checkpoint.pt"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Configured {spec.checkpoint} checkpoint does not exist for post-evaluation: {path}"
        )
    return path


def _completed_run(path: Path, checkpoint_sha256: str) -> bool:
    run_json = path / "run.json"
    if not run_json.is_file():
        return False
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
        parent = data.get("parent_training") or {}
        return (
            data.get("status") == "completed"
            and parent.get("checkpoint_sha256") == checkpoint_sha256
        )
    except (OSError, json.JSONDecodeError):
        return False


def _completed_rotation(path: Path, checkpoint_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        parent = data.get("parent_training") or {}
        return (
            data.get("status") == "completed"
            and parent.get("checkpoint_sha256") == checkpoint_sha256
        )
    except (OSError, json.JSONDecodeError):
        return False


def _rotation_run_id(
    output_root: Path,
    base: str,
    checkpoint_sha256: str,
) -> tuple[str, bool]:
    manifest = output_root / f"{base}.rotation_manifest.json"
    if _completed_rotation(manifest, checkpoint_sha256):
        return base, True
    if not manifest.exists() and not any(output_root.glob(f"{base}-seed-*")):
        return base, False
    attempt = 2
    while True:
        candidate = f"{base}_retry{attempt}"
        manifest = output_root / f"{candidate}.rotation_manifest.json"
        if _completed_rotation(manifest, checkpoint_sha256):
            return candidate, True
        if not manifest.exists() and not any(output_root.glob(f"{candidate}-seed-*")):
            return candidate, False
        attempt += 1


def _training_identity(
    training_exp: ExperimentConfig,
    run_dir: Path,
    training_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return stable configuration/seed lineage for repeated training runs."""
    result = dict(training_result or {})
    summary_path = run_dir / "training_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            summary = {}
    finetuning = training_exp.raw.get("finetuning") or {}
    configured_id = finetuning.get("configuration_id") if isinstance(finetuning, dict) else None
    seed = result.get("training_seed")
    if seed is None:
        seed = summary.get("training_seed")
    if seed is None:
        seed = (summary.get("reproducibility") or {}).get("seed")
    return {
        "configuration_id": str(
            result.get("configuration_id")
            or summary.get("configuration_id")
            or configured_id
            or training_exp.experiment_id
        ),
        "training_seed": int(seed) if seed is not None else None,
        "training_repeats": (
            result.get("training_repeats")
            or summary.get("training_repeats")
            or (
                finetuning.get("training_repeats")
                if isinstance(finetuning, dict)
                else None
            )
        ),
        "training_split": {
            **dict(summary.get("training_split") or {}),
            **dict(result.get("training_split") or {}),
        },
        "checkpoint_selection": dict(result.get("checkpoint_selection") or summary.get("checkpoint_selection") or {}),
        "best_epoch": summary.get("best_epoch"),
        "best_validation_set": summary.get("best_validation_set"),
    }


def _evaluation_run_id(
    output_root: Path,
    base: str,
    checkpoint_sha256: str,
) -> tuple[str, bool]:
    primary = output_root / base
    if _completed_run(primary, checkpoint_sha256):
        return base, True
    if not primary.exists():
        return base, False
    attempt = 2
    while (output_root / f"{base}_retry{attempt}").exists():
        if _completed_run(output_root / f"{base}_retry{attempt}", checkpoint_sha256):
            return f"{base}_retry{attempt}", True
        attempt += 1
    return f"{base}_retry{attempt}", False


def run_post_evaluations(
    experiment: ExperimentConfig | Path | str,
    training_run_dir: Path | str,
    *,
    training_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the requested checkpoint with each linked normal eval protocol."""
    training_exp = (
        experiment
        if isinstance(experiment, ExperimentConfig)
        else load_experiment_config(experiment)
    )
    linked = preflight_post_evaluations(training_exp)
    if not linked:
        return []
    run_dir = Path(training_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Finetuning run directory not found: {run_dir}")
    output_root = run_dir / "evaluations"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "post_evaluations.json"
    results: list[dict[str, Any]] = []
    training_identity = _training_identity(training_exp, run_dir, training_result)

    for spec, eval_template in linked:
        row: dict[str, Any] = {
            "experiment": str(spec.experiment),
            "experiment_id": eval_template.experiment_id,
            "checkpoint_kind": spec.checkpoint,
        }
        try:
            checkpoint = _checkpoint_for_spec(spec, run_dir, training_result)
            checkpoint_sha = sha256_file(checkpoint)
            model_id = (
                f"local/finetuned/{training_exp.experiment_id}/"
                f"{run_dir.name}/{spec.checkpoint}"
            )
            display_name = spec.label or f"{training_exp.experiment_id} ({spec.checkpoint})"
            model = checkpoint_model_config(
                checkpoint,
                model_id=model_id,
                display_name=display_name,
                checkpoint_sha256=checkpoint_sha,
            )
            base_run_id = _safe_name(eval_template.experiment_id)
            is_rotation = isinstance(eval_template.raw.get("split_rotation_evaluation"), dict)
            evaluation_root = output_root / base_run_id if is_rotation else output_root
            evaluation_root.mkdir(parents=True, exist_ok=True)
            if is_rotation:
                eval_run_id, completed = _rotation_run_id(
                    evaluation_root,
                    "rotation",
                    checkpoint_sha,
                )
                eval_run_dir = evaluation_root
            else:
                eval_run_id, completed = _evaluation_run_id(
                    evaluation_root,
                    base_run_id,
                    checkpoint_sha,
                )
                eval_run_dir = evaluation_root / eval_run_id
            row.update(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "model_id": model_id,
                    "run_dir": str(eval_run_dir),
                }
            )
            if completed:
                row["status"] = "completed"
                row["reused_completed_run"] = True
                if is_rotation:
                    row["rotation_manifest"] = str(
                        evaluation_root / f"{eval_run_id}.rotation_manifest.json"
                    )
            else:
                lineage = {
                    "experiment_id": training_exp.experiment_id,
                    "configuration_id": training_identity["configuration_id"],
                    "training_seed": training_identity["training_seed"],
                    "training_repeats": training_identity["training_repeats"],
                    "training_split": training_identity["training_split"],
                    "checkpoint_selection": training_identity["checkpoint_selection"],
                    "best_epoch": training_identity["best_epoch"],
                    "best_validation_set": training_identity["best_validation_set"],
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "checkpoint_kind": spec.checkpoint,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                }
                identity = winner_config_id(run_dir)
                if identity:
                    lineage["winner_config_id"] = identity
                if is_rotation:
                    from experiments.core.split_rotation_evaluation import (
                        run_split_rotation_evaluation,
                    )

                    produced = run_split_rotation_evaluation(
                        spec.experiment,
                        all_splits=True,
                        run_id=eval_run_id,
                        injected_models=[model],
                        output_root_override=evaluation_root,
                        parent_training=lineage,
                    )
                    row["rotation_manifest"] = produced["manifest"]
                else:
                    produced = run_experiment(
                        spec.experiment,
                        run_id=eval_run_id,
                        injected_models=[model],
                        output_root_override=evaluation_root,
                        parent_training=lineage,
                    )
                    row["run_dir"] = str(produced)
                row["status"] = "completed"
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        results.append(row)
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "training_experiment_id": training_exp.experiment_id,
                "training_run_id": run_dir.name,
                "status": "failed" if any(r.get("status") == "failed" for r in results) else "completed",
                "evaluations": results,
            },
        )

    if any(row.get("status") == "failed" for row in results):
        raise PostEvaluationError(run_dir, results)
    return results
