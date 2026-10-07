"""Explicit orchestration for evaluation across predefined static split rotations.

The ordinary runner remains a one-split operation.  This module validates a
frozen split manifest, then invokes that operation once per requested split.
It never generates a split and never changes an operating point using test data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .artifacts import PACKAGE_DIR, atomic_write_json, make_run_id, sha256_file, utc_now_iso
from .config_schema import ExperimentConfig, ModelConfig, load_experiment_config
from .runner import run_experiment
from .split_schema import load_split, records_from_split, resolve_retrieval_tasks


class SplitRotationEvaluationError(ValueError):
    """Raised when a split-rotation protocol is incomplete or unsafe."""


class SplitRotationRunError(RuntimeError):
    """Raised after retaining outputs from one or more failed split runs."""

    def __init__(self, manifest_path: Path, failures: list[dict[str, Any]]):
        self.manifest_path = manifest_path
        self.failures = failures
        super().__init__(
            f"{len(failures)} split-rotation evaluation(s) failed; see {manifest_path}"
        )


@dataclass(frozen=True)
class StaticSplitRealization:
    seed: int
    file: Path
    split_id: str
    split_sha256: str
    source_db_sha256: str | None


@dataclass(frozen=True)
class SplitRotationSpec:
    experiment: ExperimentConfig
    realizations: tuple[StaticSplitRealization, ...]
    output_dir: Path
    frozen_before_evaluation: bool


def _resolve(source: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def _class_ids(split: dict[str, Any], subset: str, role: str) -> set[int]:
    images = split["images"]
    role_ids = ((split.get("subsets") or {}).get(subset) or {}).get("roles", {}).get(role)
    if not isinstance(role_ids, list) or not role_ids:
        raise SplitRotationEvaluationError(
            f"Split {split.get('split_id')!r} requires non-empty {subset}.{role}"
        )
    return {int(images[file_id]["class_id"]) for file_id in role_ids}


def _validate_static_realizations(
    source: Path,
    experiment: ExperimentConfig,
    raw: dict[str, Any],
) -> tuple[StaticSplitRealization, ...]:
    expected = raw.get("expected_split_seeds")
    entries = raw.get("static_splits")
    if not isinstance(expected, list) or not expected:
        raise SplitRotationEvaluationError(
            "split_rotation_evaluation.expected_split_seeds must be a non-empty list"
        )
    expected_seeds = [int(value) for value in expected]
    if len(expected_seeds) != len(set(expected_seeds)):
        raise SplitRotationEvaluationError("expected split seeds contain duplicates")
    if not isinstance(entries, list) or not entries:
        raise SplitRotationEvaluationError(
            "split_rotation_evaluation.static_splits must be a non-empty list"
        )

    declared: list[int] = []
    records: list[StaticSplitRealization] = []
    source_hashes: set[str] = set()
    eligible_image_sets: list[set[str]] = []
    eligible_class_sets: list[set[int]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "seed" not in entry or not entry.get("file"):
            raise SplitRotationEvaluationError("Every static split requires seed and file")
        seed = int(entry["seed"])
        declared.append(seed)
        path = _resolve(source, entry["file"])
        split = load_split(path)
        dataset = split.get("dataset") or {}
        if str(dataset.get("dataset_id")) != experiment.dataset.dataset_id:
            raise SplitRotationEvaluationError(
                f"Static split {path} uses dataset {dataset.get('dataset_id')!r}, "
                f"expected {experiment.dataset.dataset_id!r}"
            )
        strategy = split.get("split_strategy") or {}
        ratios = strategy.get("ratios") or {}
        if strategy.get("type") != "class_disjoint_role" or int(strategy.get("seed", -1)) != seed:
            raise SplitRotationEvaluationError(
                f"Static split {path} is not class_disjoint_role with seed {seed}"
            )
        expected_ratios = {"train": 0.0, "val": 0.5, "test": 0.5}
        if any(
            not math.isclose(float(ratios.get(key, -1.0)), value, abs_tol=1e-12)
            for key, value in expected_ratios.items()
        ):
            raise SplitRotationEvaluationError(
                f"Static split {path} must use train=0, val=0.5, test=0.5"
            )
        if set(split.get("subsets") or {}) != {"val", "test"}:
            raise SplitRotationEvaluationError(
                f"Static split {path} must contain exactly val and test"
            )
        val_modern = _class_ids(split, "val", "modern")
        val_historic = _class_ids(split, "val", "historic")
        test_modern = _class_ids(split, "test", "modern")
        test_historic = _class_ids(split, "test", "historic")
        # Every query class needs historic positives. Historic-only classes are
        # distractors and do not need a modern image.
        for subset_name, modern, historic in (
            ("val", val_modern, val_historic),
            ("test", test_modern, test_historic),
        ):
            missing = modern - historic
            if missing:
                raise SplitRotationEvaluationError(
                    f"Static split {path} {subset_name} is missing historic images "
                    f"for {len(missing)} query classes"
                )
        if (val_modern | val_historic) & (test_modern | test_historic):
            raise SplitRotationEvaluationError(
                f"Static split {path} leaks classes between validation and test"
            )
        subsets = split["subsets"]
        val_ids = {str(value) for ids in subsets["val"]["roles"].values() for value in ids}
        test_ids = {str(value) for ids in subsets["test"]["roles"].values() for value in ids}
        if val_ids & test_ids:
            raise SplitRotationEvaluationError(
                f"Static split {path} leaks images between validation and test"
            )
        eligible_image_sets.append(val_ids | test_ids)
        eligible_class_sets.append(val_modern | test_modern)
        source_hash = str(dataset.get("source_db_sha256") or "")
        if source_hash:
            source_hashes.add(source_hash)
        records.append(
            StaticSplitRealization(
                seed=seed,
                file=path,
                split_id=str(split["split_id"]),
                split_sha256=sha256_file(path),
                source_db_sha256=source_hash or None,
            )
        )

    if sorted(declared) != sorted(expected_seeds):
        raise SplitRotationEvaluationError(
            f"Static split seeds {sorted(declared)} do not match expected {sorted(expected_seeds)}"
        )
    if len(source_hashes) > 1:
        raise SplitRotationEvaluationError("Static splits use different source databases")
    if any(value != eligible_image_sets[0] for value in eligible_image_sets[1:]):
        raise SplitRotationEvaluationError(
            "Static splits do not partition the same eligible image population"
        )
    if any(value != eligible_class_sets[0] for value in eligible_class_sets[1:]):
        raise SplitRotationEvaluationError(
            "Static splits do not partition the same eligible class population"
        )
    return tuple(sorted(records, key=lambda item: expected_seeds.index(item.seed)))


def load_split_rotation_spec(
    experiment_path: Path | str,
    *,
    injected_models: list[ModelConfig] | None = None,
    allow_dynamic_models: bool = False,
) -> SplitRotationSpec:
    source = Path(experiment_path).expanduser().resolve()
    raw_document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw_document, dict):
        raise SplitRotationEvaluationError(f"Experiment root must be a mapping: {source}")
    raw = raw_document.get("split_rotation_evaluation")
    if not isinstance(raw, dict):
        raise SplitRotationEvaluationError(
            f"Experiment {source} has no split_rotation_evaluation mapping"
        )
    if raw.get("schema_version", 1) != 1:
        raise SplitRotationEvaluationError("Unsupported split-rotation schema_version")
    experiment = load_experiment_config(
        source,
        injected_models=injected_models,
        allow_dynamic_models=allow_dynamic_models,
    )
    protocol = raw.get("protocol") or {}
    frozen = isinstance(protocol, dict) and protocol.get("frozen_before_evaluation") is True
    if not frozen:
        raise SplitRotationEvaluationError(
            "split_rotation_evaluation.protocol.frozen_before_evaluation: true is required"
        )
    non_static: list[str] = []
    for model in experiment.models:
        group = str(model.raw.get("group") or "")
        if group == "frozen":
            continue
        checkpoint_path = model.adapter.kwargs.get("checkpoint_path")
        is_fixed_finetuned_checkpoint = (
            group == "finetuned"
            and model.adapter.adapter_name == "finetuned_checkpoint_adapter"
            and isinstance(checkpoint_path, str)
            and bool(checkpoint_path.strip())
            and checkpoint_path != "REQUIRED_OVERRIDE_BY_EXPERIMENT"
        )
        if not is_fixed_finetuned_checkpoint:
            non_static.append(model.model_id)
    if non_static:
        raise SplitRotationEvaluationError(
            "Split rotation contains model states that are not fixed frozen models "
            "or static finetuned checkpoints: " + ", ".join(non_static)
        )
    realizations = _validate_static_realizations(source, experiment, raw)
    output_value = raw.get("output_dir") or (experiment.raw.get("run") or {}).get("output_dir")
    output = Path(output_value or f"runs/{experiment.experiment_id}_split_rotation").expanduser()
    output = output if output.is_absolute() else (PACKAGE_DIR / output).resolve()

    # Resolve every configured task on every split before any expensive work.
    evaluation = experiment.raw.get("evaluation") or {}
    calibration = evaluation.get("threshold_calibration") or {}
    for realization in realizations:
        split = load_split(realization.file)
        records_from_split(split)
        resolve_retrieval_tasks(split, evaluation.get("retrieval_tasks"))
        if calibration:
            resolve_retrieval_tasks(split, [calibration["task"]])
    return SplitRotationSpec(experiment, realizations, output, frozen)


def validate_split_rotation_evaluation(experiment_path: Path | str) -> dict[str, Any]:
    spec = load_split_rotation_spec(experiment_path)
    return {
        "experiment_id": spec.experiment.experiment_id,
        "dataset_id": spec.experiment.dataset.dataset_id,
        "output_dir": str(spec.output_dir),
        "split_realizations": [
            {
                "seed": item.seed,
                "file": str(item.file),
                "split_id": item.split_id,
                "split_sha256": item.split_sha256,
            }
            for item in spec.realizations
        ],
    }


def run_split_rotation_evaluation(
    experiment_path: Path | str,
    *,
    model_filter: str | None = None,
    split_seed: int | None = None,
    all_splits: bool = False,
    run_id: str | None = None,
    injected_models: list[ModelConfig] | None = None,
    output_root_override: Path | str | None = None,
    parent_training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = load_split_rotation_spec(experiment_path, injected_models=injected_models)
    output_dir = (
        Path(output_root_override).expanduser().resolve()
        if output_root_override is not None
        else spec.output_dir
    )
    if all_splits == (split_seed is not None):
        raise SplitRotationEvaluationError(
            "Select exactly one of all_splits=True or split_seed=<seed>"
        )
    selected = (
        list(spec.realizations)
        if all_splits
        else [item for item in spec.realizations if item.seed == int(split_seed)]
    )
    if not selected:
        raise SplitRotationEvaluationError(f"Unknown configured split seed: {split_seed}")
    base_run_id = run_id or make_run_id()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{base_run_id}.rotation_manifest.json"

    def write_manifest(status: str) -> None:
        manifest = {
            "schema_version": 1,
            "artifact_type": "split_rotation_run_manifest",
            "created_at": utc_now_iso(),
            "experiment": str(spec.experiment.path),
            "experiment_id": spec.experiment.experiment_id,
            "model_filter": model_filter,
            "expected_split_seeds": [item.seed for item in selected],
            "static_splits": [
                {
                    "seed": item.seed,
                    "file": str(item.file),
                    "split_id": item.split_id,
                    "split_sha256": item.split_sha256,
                }
                for item in selected
            ],
            "completed": completed,
            "failures": failures,
            "status": status,
            "parent_training": dict(parent_training or {}),
            "scientific_protocol": {
                "frozen_before_evaluation": True,
                "test_used_for_selection": False,
                "split_generation_during_evaluation": False,
            },
        }
        atomic_write_json(manifest_path, manifest)
        return manifest

    for realization in selected:
        selected_run_id = f"{base_run_id}-seed-{realization.seed}"
        try:
            run_dir = run_experiment(
                spec.experiment.path,
                model_filter=model_filter,
                run_id=selected_run_id,
                output_root_override=output_dir,
                split_file_override=realization.file,
                injected_models=injected_models,
                parent_training=parent_training,
            )
        except Exception as exc:
            failures.append(
                {
                    "seed": realization.seed,
                    "run_id": selected_run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            completed.append(
                {
                    "seed": realization.seed,
                    "run_id": selected_run_id,
                    "run_dir": str(run_dir),
                    "split_id": realization.split_id,
                    "split_sha256": realization.split_sha256,
                }
            )
        write_manifest("running")
    manifest = write_manifest("failed" if failures else "completed")
    if failures:
        raise SplitRotationRunError(manifest_path, failures)
    return {"manifest": str(manifest_path), **manifest}
