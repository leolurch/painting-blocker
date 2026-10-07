"""Validation-set resolution shared by finetuning training and reference prewarming.

Both the training loop (``train.py``) and the reference-cache prewarming path
(``reference_evaluation.py`` / the ``prepare-reference-cache`` CLI) must resolve
validation sets identically so that embedding and reference-result cache keys
agree. This module is the single source of truth for that resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.core.config_schema import DatasetConfig
from experiments.core.split_schema import load_split, validate_split_manifest

from .evaluate import retrieval_task_specs_for_subset


@dataclass(frozen=True)
class ValidationSet:
    name: str
    primary: bool
    dataset: DatasetConfig
    split: dict[str, Any]
    split_path: Path
    split_sha256: str
    selector: dict[str, Any]
    retrieval_task_specs: list[dict[str, Any]]


def resolve_experiment_path(value: object, exp) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    experiment_relative = (exp.path.parent / path).resolve()
    if experiment_relative.exists():
        return experiment_relative
    return (exp.repo_root / path).resolve()


def _split_sha256(split: dict[str, Any]) -> str:
    return str((split.get("dataset") or {}).get("source_db_sha256") or "")


def dataset_for_validation_set(raw: dict[str, Any], split: dict[str, Any], exp) -> DatasetConfig:
    if not raw:
        return exp.dataset
    dataset_path = resolve_experiment_path(raw["path"], exp)
    split_dataset = dict(split.get("dataset") or {})
    dataset_id = str(split_dataset.get("dataset_id") or raw.get("dataset_id") or dataset_path.name)
    image_root_raw = raw.get("image_root") or split_dataset.get("image_root")
    image_root = resolve_experiment_path(image_root_raw, exp) if image_root_raw else dataset_path / "images"
    return DatasetConfig(
        path=dataset_path,
        dataset_id=dataset_id,
        dataset_db=dataset_path / "dataset.db",
        image_root=image_root,
        identity=dict(split_dataset.get("identity") or {}),
        raw=dict(raw),
    )


def resolve_validation_sets(exp, cfg: dict[str, Any], primary_split: dict[str, Any]) -> list[ValidationSet]:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("finetuning.data is required")
    raw_sets = data_cfg.get("validation")
    if not isinstance(raw_sets, list) or not raw_sets:
        raise ValueError("finetuning.data.validation must be a non-empty list of validation-set mappings")
    primary_split_path = Path(exp.split_file)
    validation_sets: list[ValidationSet] = []
    names: set[str] = set()
    primary_count = 0
    for index, raw in enumerate(raw_sets):
        if not isinstance(raw, dict):
            raise ValueError("Each finetuning.data.validation entry must be a mapping")
        name = str(raw.get("name") or ("primary" if index == 0 else f"validation_{index + 1}"))
        if name in names:
            raise ValueError(f"Duplicate validation set name: {name!r}")
        names.add(name)
        is_primary = bool(raw.get("primary", index == 0))
        primary_count += int(is_primary)
        if raw.get("split"):
            split_path = resolve_experiment_path(dict(raw.get("split") or {})["file"], exp)
            split = load_split(split_path)
        else:
            split_path = primary_split_path
            split = primary_split
        validate_split_manifest(split)
        dataset = dataset_for_validation_set(dict(raw.get("dataset") or {}), split, exp)
        selector = {"subset": raw["subset"], "roles": raw["roles"]}
        retrieval_specs = [dict(task) for task in raw.get("retrieval_tasks") or []]
        if not retrieval_specs and is_primary:
            retrieval_specs = retrieval_task_specs_for_subset(
                exp, "validation", selector_subset=str(selector["subset"])
            )
        validation_sets.append(
            ValidationSet(
                name=name,
                primary=is_primary,
                dataset=dataset,
                split=split,
                split_path=split_path,
                split_sha256=_split_sha256(split),
                selector=selector,
                retrieval_task_specs=retrieval_specs,
            )
        )
    if primary_count != 1:
        raise ValueError("Exactly one finetuning.data.validation entry must set primary: true")
    return validation_sets


def primary_validation_set(validation_sets: list[ValidationSet]) -> ValidationSet:
    return next(item for item in validation_sets if item.primary)
