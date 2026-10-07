"""Schemas and resolvers for YAML experiment configuration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .adapter_registry import AdapterConfig, known_adapter_names
from .artifacts import sha256_json
from .db import dataset_counts
from .reproducibility import environment_provenance


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    dataset_id: str
    dataset_db: Path
    image_root: Path | None
    identity: dict[str, int]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    model_id: str
    adapter: AdapterConfig
    raw: dict[str, Any]
    revision: str | None = None
    presentation_overrides: frozenset[str] = frozenset()

    @property
    def identity_raw(self) -> dict[str, Any]:
        """Model settings that affect artifacts, excluding experiment-local styling."""
        return {
            key: value
            for key, value in self.raw.items()
            if key not in self.presentation_overrides
        }

    @property
    def storage_key(self) -> str:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_id).strip("._")
        stem = stem[:96] or "model"
        digest = sha256_json(
            {
                "model_id": self.model_id,
                "revision": self.revision,
                "adapter": self.adapter.to_dict(),
                "raw": self.identity_raw,
            }
        )[:10]
        return f"{stem}_{digest}"


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    repo_root: Path
    experiment_id: str
    dataset: DatasetConfig
    split_file: Path
    models: list[ModelConfig]
    raw: dict[str, Any]
    resolved: dict[str, Any]


@dataclass(frozen=True)
class PostEvaluationConfig:
    """One normal evaluation experiment linked to a finetuning run."""

    experiment: Path
    checkpoint: str = "best"
    label: str | None = None


def load_yaml(path: Path | str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def find_repo_root(start: Path) -> Path:
    """Find the project root using unambiguous source-layout markers.

    A plain ``candidate / 'experiments'`` check is insufficient because this
    repository intentionally has ``experiments/configs/experiments``.
    """
    resolved = start.expanduser().resolve()
    for candidate in [resolved, *resolved.parents]:
        if (
            (candidate / "experiments" / "core").is_dir()
            and (candidate / "experiments" / "cli.py").is_file()
            and (candidate / "experiments" / "configs").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root from {resolved}; expected "
        "experiments/core, experiments/cli.py, and experiments/configs"
    )


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _require(data: dict[str, Any], key: str, source: Path) -> Any:
    if key not in data:
        raise ValueError(f"Missing required key {key!r} in {source}")
    return data[key]


def load_dataset_config(path: Path | str, repo_root: Path | None = None) -> DatasetConfig:
    cfg_path = Path(path).expanduser().resolve()
    repo = repo_root or find_repo_root(cfg_path.parent)
    data = load_yaml(cfg_path)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported dataset schema_version in {cfg_path}")
    dataset_id = str(_require(data, "dataset_id", cfg_path))
    paths = _require(data, "paths", cfg_path)
    if not isinstance(paths, dict):
        raise ValueError(f"paths must be a mapping in {cfg_path}")
    dataset_db = resolve_path(_require(paths, "dataset_db", cfg_path), repo)
    image_root = paths.get("image_root")
    resolved_image_root = resolve_path(image_root, repo) if image_root else None
    identity = data.get("identity") or {}
    return DatasetConfig(cfg_path, dataset_id, dataset_db, resolved_image_root, identity, data)


def validate_dataset_config(config: DatasetConfig) -> dict[str, int]:
    if not config.dataset_db.is_file():
        raise FileNotFoundError(f"Dataset DB not found: {config.dataset_db}")
    if config.image_root is not None and not config.image_root.is_dir():
        raise FileNotFoundError(f"Dataset image_root not found: {config.image_root}")
    counts = dataset_counts(config.dataset_db)
    expected = {
        "expected_num_classes": counts["num_classes"],
        "expected_num_images": counts["num_images"],
        **config.identity,
    }
    if int(expected.get("expected_num_classes", counts["num_classes"])) != counts["num_classes"]:
        raise ValueError(f"Dataset class count mismatch for {config.dataset_id}: {counts}")
    if int(expected.get("expected_num_images", counts["num_images"])) != counts["num_images"]:
        raise ValueError(f"Dataset image count mismatch for {config.dataset_id}: {counts}")
    return counts


def _adapter_name(data: dict[str, Any], path: Path) -> str:
    adapter_name = str(_require(data, "adapter_name", path)).strip()
    if adapter_name not in known_adapter_names():
        raise ValueError(
            f"Unknown adapter_name {adapter_name!r} in {path}. Expected one of: "
            f"{', '.join(known_adapter_names())}"
        )
    return adapter_name


def _model_entries(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = load_yaml(path)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported model schema_version in {path}")
    _reject_keys(data, {"adapter", "adapter_type", "backend", "family"}, path, "model file")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"models must be a non-empty list in {path}")
    entries = [dict(model) for model in models]
    for entry in entries:
        _reject_keys(
            entry,
            {"adapter", "adapter_selector", "selector", "backend", "family", "model_name", "pretrained", "size_key"},
            path,
            f"model {entry.get('model_id', '<missing>')!r}",
        )
    return _adapter_name(data, path), entries


def _reject_keys(data: dict[str, Any], forbidden: set[str], source: Path, scope: str) -> None:
    present = sorted(set(data) & forbidden)
    if present:
        raise ValueError(f"Unsupported legacy key(s) in {scope} at {source}: {', '.join(present)}")


def _adapter_config(adapter_name: str, raw: dict[str, Any], source: Path, model_id: str) -> AdapterConfig:
    kwargs = raw.get("adapter_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError(f"Model {model_id!r} adapter_kwargs must be a mapping in {source}")
    forbidden = sorted(set(kwargs) & {"model_id", "size_key"})
    if forbidden:
        raise ValueError(
            f"Model {model_id!r} adapter_kwargs must not contain: {', '.join(forbidden)}"
        )
    revision = raw.get("revision")
    return AdapterConfig(
        adapter_name=adapter_name,
        model_id=model_id,
        kwargs=dict(kwargs),
        revision=str(revision) if revision is not None else None,
    )


def _merge_mapping_override(
    raw: dict[str, Any],
    reference: dict[str, Any],
    key: str,
    source: Path,
) -> None:
    if key not in reference:
        return
    override = reference[key]
    if not isinstance(override, dict):
        raise ValueError(f"models.include {key} override must be a mapping in {source}")
    current = raw.get(key) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Model {raw.get('model_id')!r} {key} must be a mapping before override in {source}")
    raw[key] = {**current, **override}


def _merge_reference_overrides(
    raw_model: dict[str, Any],
    reference: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    raw = dict(raw_model)
    for key in ("adapter_kwargs", "embedding", "preprocessing"):
        _merge_mapping_override(raw, reference, key, source)
    for key in ("display_name", "label", "name", "color", "revision", "group"):
        if key in reference:
            raw[key] = reference[key]
    color = raw.get("color")
    if color is not None:
        if not isinstance(color, str) or not color.strip():
            raise ValueError(f"Model {raw.get('model_id')!r} color must be a non-empty string in {source}")
        raw["color"] = color.strip()
    group = raw.get("group")
    if group is not None:
        if group not in ("frozen", "finetuned"):
            raise ValueError(
                f"Model {raw.get('model_id')!r} group must be 'frozen' or 'finetuned' in {source}"
            )
    return raw


def load_model_reference(reference: Any, base_dir: Path, source: Path | None = None) -> ModelConfig:
    if not isinstance(reference, dict):
        raise ValueError("models.include entries must be mappings with config and model_id")
    allowed = {
        "config",
        "model_id",
        "variant_id",
        "revision",
        "display_name",
        "label",
        "name",
        "color",
        "group",
        "adapter_kwargs",
        "embedding",
        "preprocessing",
    }
    unknown = sorted(set(reference) - allowed)
    if unknown:
        raise ValueError(f"Unknown model include key(s): {', '.join(unknown)}")
    cfg_source = source or base_dir
    file_part = _require(reference, "config", cfg_source)
    source_model_id = str(_require(reference, "model_id", cfg_source))
    variant_id = str(reference.get("variant_id") or source_model_id).strip()
    if not variant_id:
        raise ValueError(f"models.include variant_id must be a non-empty string in {cfg_source}")
    path = resolve_path(file_part, base_dir)
    adapter_name, entries = _model_entries(path)
    matches = [model for model in entries if str(model.get("model_id")) == source_model_id]
    if len(matches) != 1:
        raise ValueError(f"Model {source_model_id!r} not found exactly once in {path}")
    raw = _merge_reference_overrides(matches[0], reference, cfg_source)
    if variant_id != source_model_id:
        raw["source_model_id"] = source_model_id
    revision = raw.get("revision")
    presentation_overrides = frozenset(
        ({"color"} if "color" in raw else set())
        | ({"group"} if "group" in raw else set())
        | (set(reference) & {"display_name", "label", "name"})
    )
    return ModelConfig(
        path=path,
        model_id=variant_id,
        adapter=_adapter_config(adapter_name, raw, path, source_model_id),
        raw=raw,
        revision=str(revision) if revision is not None else None,
        presentation_overrides=presentation_overrides,
    )


def load_experiment_config(
    path: Path | str,
    *,
    injected_models: list[ModelConfig] | None = None,
    allow_dynamic_models: bool = False,
    split_file_override: Path | str | None = None,
) -> ExperimentConfig:
    """Load an experiment, optionally supplying its runtime checkpoint model.

    Normal experiments continue to resolve ``models.include`` exactly as before.
    Evaluation templates with ``models.from_finetuning_checkpoint: true`` are only
    runnable when ``injected_models`` is supplied; ``allow_dynamic_models`` exists
    for validation/preflight before the checkpoint has been trained.
    """
    exp_path = Path(path).expanduser().resolve()
    repo = find_repo_root(exp_path.parent)
    data = load_yaml(exp_path)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported experiment schema_version in {exp_path}")
    experiment_id = str(_require(data, "experiment_id", exp_path))
    configured_split = resolve_path(
        _require(_require(data, "split", exp_path), "file", exp_path),
        exp_path.parent,
    )
    split_file = (
        resolve_path(split_file_override, exp_path.parent)
        if split_file_override is not None
        else configured_split
    )
    dataset = _runtime_dataset_config(data, split_file, exp_path, repo)
    _validate_protocol_blocks(data, exp_path)
    model_block = _require(data, "models", exp_path) or {}
    if not isinstance(model_block, dict):
        raise ValueError(f"models must be a mapping in {exp_path}")
    dynamic_checkpoint = model_block.get("from_finetuning_checkpoint") is True
    includes = model_block.get("include")
    if dynamic_checkpoint:
        if includes not in (None, []):
            raise ValueError(
                f"models.from_finetuning_checkpoint may not be combined with models.include in {exp_path}"
            )
        if injected_models is not None:
            if not injected_models:
                raise ValueError(f"injected_models must not be empty for {exp_path}")
            models = list(injected_models)
        elif allow_dynamic_models:
            models = []
        else:
            raise ValueError(
                f"Experiment {exp_path} requires a finetuning checkpoint. Run it from a "
                "finetuning post_evaluations link or pass --checkpoint to 'experiments.cli run'."
            )
    else:
        if injected_models is not None:
            raise ValueError(
                f"Injected checkpoint models require models.from_finetuning_checkpoint: true in {exp_path}"
            )
        if not isinstance(includes, list) or not includes:
            raise ValueError(f"models.include must be a non-empty list in {exp_path}")
        models = [load_model_reference(ref, exp_path.parent, exp_path) for ref in includes]
    if len({model.model_id for model in models}) != len(models):
        raise ValueError(f"Duplicate model_id in experiment {exp_path}")
    if len({model.storage_key for model in models}) != len(models):
        raise ValueError(f"Duplicate model storage key in experiment {exp_path}")
    resolved = to_resolved_dict(repo, data, dataset, split_file, models)
    experiment = ExperimentConfig(exp_path, repo, experiment_id, dataset, split_file, models, data, resolved)
    # Validate linked-evaluation shape/path uniqueness as part of normal config
    # loading; referenced protocol contents are preflighted by the CLI before train.
    resolve_post_evaluations(experiment)
    return experiment


def resolve_post_evaluations(exp: ExperimentConfig) -> list[PostEvaluationConfig]:
    """Resolve ``finetuning.post_evaluations`` paths relative to the train YAML."""
    finetuning = exp.raw.get("finetuning")
    if not isinstance(finetuning, dict):
        return []
    raw_specs = finetuning.get("post_evaluations")
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"finetuning.post_evaluations must be a non-empty list in {exp.path}")
    specs: list[PostEvaluationConfig] = []
    seen: set[Path] = set()
    for index, raw in enumerate(raw_specs):
        if isinstance(raw, str):
            experiment_value = raw
            checkpoint = "best"
            label = None
        elif isinstance(raw, dict):
            unknown = sorted(set(raw) - {"experiment", "config", "checkpoint", "label"})
            if unknown:
                raise ValueError(
                    f"Unknown finetuning.post_evaluations[{index}] key(s) in {exp.path}: "
                    + ", ".join(unknown)
                )
            experiment_value = raw.get("experiment") or raw.get("config")
            if not experiment_value:
                raise ValueError(
                    f"finetuning.post_evaluations[{index}] requires experiment or config in {exp.path}"
                )
            checkpoint = str(raw.get("checkpoint") or "best").strip().lower()
            label_value = raw.get("label")
            label = str(label_value).strip() if label_value is not None else None
            if label_value is not None and not label:
                raise ValueError(
                    f"finetuning.post_evaluations[{index}].label must not be empty in {exp.path}"
                )
        else:
            raise ValueError(
                f"finetuning.post_evaluations[{index}] must be a path string or mapping in {exp.path}"
            )
        if checkpoint not in {"best", "last"}:
            raise ValueError(
                f"finetuning.post_evaluations[{index}].checkpoint must be 'best' or 'last' in {exp.path}"
            )
        experiment = resolve_path(str(experiment_value), exp.path.parent)
        if experiment in seen:
            raise ValueError(f"Duplicate post-evaluation experiment {experiment} in {exp.path}")
        seen.add(experiment)
        specs.append(PostEvaluationConfig(experiment=experiment, checkpoint=checkpoint, label=label))
    return specs


def _runtime_dataset_config(raw: dict[str, Any], split_file: Path, exp_path: Path, repo: Path) -> DatasetConfig:
    dataset_raw = _require(raw, "dataset", exp_path)
    if not isinstance(dataset_raw, dict):
        raise ValueError(f"dataset must be a mapping in {exp_path}")
    dataset_path_raw = _require(dataset_raw, "path", exp_path)
    dataset_path = _resolve_experiment_or_repo_path(dataset_path_raw, exp_path.parent, repo)
    if split_file.is_file():
        split = json.loads(split_file.read_text(encoding="utf-8"))
        dataset = split.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError(f"Split {split_file} must contain dataset metadata")
        dataset_id = str(_require(dataset, "dataset_id", split_file))
    else:
        dataset = {}
        dataset_id = str(dataset_raw.get("dataset_id") or dataset_path.name)
    image_root_raw = dataset_raw.get("image_root")
    if image_root_raw:
        image_root = _resolve_experiment_or_repo_path(image_root_raw, exp_path.parent, repo)
    else:
        image_root = dataset_path / "images"
    identity = dict(dataset.get("identity") or {})
    dataset_db = dataset_path / "dataset.db"
    return DatasetConfig(dataset_path, dataset_id, dataset_db, image_root, identity, dict(dataset_raw))


def _resolve_experiment_or_repo_path(value: str | Path, base: Path, repo: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    experiment_relative = (base / path).resolve()
    if experiment_relative.exists():
        return experiment_relative
    return (repo / path).resolve()


_AUGMENTATION_KEYS = {
    "backend",
    "random_crop",
    "random_resized_crop",
    "horizontal_flip",
    "affine",
    "color_jitter",
    "grayscale",
    "image_compression",
    "gaussian_blur",
    "coarse_dropout",
    "perspective",
    "desaturation",
    "square_mask",
    "center_crop_in",
    "gaussian_noise",
}


def training_repeat_seeds(finetuning: dict[str, Any]) -> tuple[int, ...]:
    """Return the predeclared training-repeat seeds, if configured."""
    raw = finetuning.get("training_repeats")
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ValueError("finetuning.training_repeats must be a mapping")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise ValueError(
            "finetuning.training_repeats.seeds must contain at least two seeds"
        )
    normalized: list[int] = []
    for value in seeds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "finetuning.training_repeats.seeds must contain integers"
            )
        normalized.append(int(value))
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "finetuning.training_repeats.seeds must not contain duplicates"
        )
    return tuple(normalized)


def _validate_training_repeats(
    finetuning: dict[str, Any], source: Path
) -> None:
    seeds = training_repeat_seeds(finetuning)
    if not seeds:
        return
    repeats = finetuning["training_repeats"]
    if "require_complete" in repeats and not isinstance(
        repeats["require_complete"], bool
    ):
        raise ValueError(
            f"finetuning.training_repeats.require_complete must be boolean in {source}"
        )
    configuration_id = finetuning.get("configuration_id")
    if not isinstance(configuration_id, str) or not configuration_id.strip():
        raise ValueError(
            "finetuning.configuration_id is required when training_repeats are "
            f"configured in {source}"
        )
    train = finetuning.get("train")
    configured_seed = train.get("seed") if isinstance(train, dict) else None
    if configured_seed is not None and (
        isinstance(configured_seed, bool)
        or not isinstance(configured_seed, int)
        or int(configured_seed) not in seeds
    ):
        raise ValueError(
            "finetuning.train.seed must be one of "
            f"finetuning.training_repeats.seeds in {source}"
        )


def _validate_gradient_cache(finetuning: dict[str, Any], source: Path) -> None:
    train = finetuning.get("train")
    if not isinstance(train, dict):
        return
    if "prefetch_factor" in train:
        prefetch_factor = train["prefetch_factor"]
        if (
            isinstance(prefetch_factor, bool)
            or not isinstance(prefetch_factor, int)
            or prefetch_factor <= 0
        ):
            raise ValueError(f"finetuning.train.prefetch_factor must be a positive integer in {source}")
    raw_cache = train.get("gradient_cache")
    if raw_cache is None:
        return
    if not isinstance(raw_cache, dict):
        raise ValueError(f"finetuning.train.gradient_cache must be a mapping in {source}")
    if "enabled" in raw_cache and not isinstance(raw_cache["enabled"], bool):
        raise ValueError(f"finetuning.train.gradient_cache.enabled must be boolean in {source}")
    if "verify_replay" in raw_cache and not isinstance(raw_cache["verify_replay"], bool):
        raise ValueError(f"finetuning.train.gradient_cache.verify_replay must be boolean in {source}")
    enabled = bool(raw_cache.get("enabled", False))
    if not enabled:
        return
    chunk_size = raw_cache.get("chunk_size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            f"finetuning.train.gradient_cache.chunk_size must be a positive integer in {source}"
        )
    sampler = finetuning.get("sampler")
    if not isinstance(sampler, dict):
        raise ValueError(f"Gradient caching requires finetuning.sampler in {source}")
    classes_per_batch = sampler.get("classes_per_batch", sampler.get("p"))
    images_per_class = sampler.get("images_per_class", sampler.get("k"))
    per_origin = sampler.get("per_origin")
    if isinstance(per_origin, dict) and per_origin:
        origin_total = sum(int(count) for count in per_origin.values())
        if images_per_class is None:
            images_per_class = origin_total
        elif int(images_per_class) != origin_total:
            raise ValueError(
                "sampler.images_per_class must equal the sum of sampler.per_origin "
                f"in {source}"
            )
    quotas = sampler.get("quotas")
    if isinstance(quotas, dict) and quotas:
        quota_total = sum(int(count) for count in quotas.values())
        if images_per_class is None:
            images_per_class = quota_total
        elif int(images_per_class) != quota_total:
            raise ValueError(
                "sampler.images_per_class must equal the sum of sampler.quotas "
                f"in {source}"
            )
    if (
        isinstance(classes_per_batch, bool)
        or not isinstance(classes_per_batch, int)
        or classes_per_batch <= 0
        or isinstance(images_per_class, bool)
        or not isinstance(images_per_class, int)
        or images_per_class <= 0
    ):
        raise ValueError(
            "Gradient caching requires positive integer sampler classes_per_batch and "
            f"images_per_class in {source}"
        )
    effective_batch_size = classes_per_batch * images_per_class
    if chunk_size > effective_batch_size:
        raise ValueError(
            "finetuning.train.gradient_cache.chunk_size cannot exceed the effective sampler "
            f"batch size ({effective_batch_size}) in {source}"
        )


def _validate_protocol_blocks(raw: dict[str, Any], source: Path) -> None:
    if "evaluation" in raw:
        evaluation = raw.get("evaluation") or {}
        _validate_retrieval_tasks(evaluation.get("retrieval_tasks"), source)
        if "threshold_calibration" in evaluation:
            _validate_threshold_calibration(evaluation, source)
        if "top_k_calibration" in evaluation:
            _validate_top_k_calibration(evaluation, source)
        if "threshold_top_k_floor" in evaluation:
            _validate_threshold_top_k_floor(evaluation, source)
        if "profile_diagnostics" in evaluation:
            diagnostics = evaluation.get("profile_diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError(f"evaluation.profile_diagnostics must be a mapping in {source}")
            for key in ("enabled", "required"):
                if key in diagnostics and not isinstance(diagnostics[key], bool):
                    raise ValueError(
                        f"evaluation.profile_diagnostics.{key} must be boolean in {source}"
                    )
            field = str(diagnostics.get("candidate_metadata_field") or "")
            if diagnostics.get("enabled", False) and not field:
                raise ValueError(
                    f"evaluation.profile_diagnostics.candidate_metadata_field is required in {source}"
                )
    if "finetuning" in raw:
        finetuning = raw.get("finetuning")
        if not isinstance(finetuning, dict):
            raise ValueError(f"finetuning must be a mapping in {source}")
        _validate_augmentations(finetuning.get("augmentations"), source)
        _validate_training_repeats(finetuning, source)
        _validate_gradient_cache(finetuning, source)
        data = finetuning.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"finetuning.data is required in {source}")
        train_selector = data.get("train")
        if not isinstance(train_selector, dict):
            raise ValueError(f"finetuning.data.train is required in {source}")
        _validate_selector(train_selector, source, "finetuning.data.train")
        validation_sets = data.get("validation")
        if not isinstance(validation_sets, list) or not validation_sets:
            raise ValueError(f"finetuning.data.validation must be a non-empty list in {source}")
        primary_count = 0
        seen_names: set[str] = set()
        for index, validation_set in enumerate(validation_sets):
            if not isinstance(validation_set, dict):
                raise ValueError(f"Each finetuning.data.validation entry must be a mapping in {source}")
            _validate_selector(validation_set, source, f"finetuning.data.validation[{index}]")
            name = str(validation_set.get("name") or "")
            if not name:
                raise ValueError(f"finetuning.data.validation[{index}].name is required in {source}")
            if name in seen_names:
                raise ValueError(f"Duplicate finetuning.data.validation name {name!r} in {source}")
            seen_names.add(name)
            primary_count += int(bool(validation_set.get("primary", False)))
            if "split" in validation_set:
                split = validation_set["split"]
                if not isinstance(split, dict) or "file" not in split:
                    raise ValueError(f"finetuning.data.validation[{index}].split must contain file in {source}")
            if "dataset" in validation_set:
                dataset = validation_set["dataset"]
                if not isinstance(dataset, dict) or "path" not in dataset:
                    raise ValueError(f"finetuning.data.validation[{index}].dataset must contain path in {source}")
            if "retrieval_tasks" in validation_set:
                _validate_retrieval_tasks(validation_set.get("retrieval_tasks"), source)
        if primary_count != 1:
            raise ValueError(f"Exactly one finetuning.data.validation entry must set primary: true in {source}")
        finetune_eval = finetuning.get("evaluation")
        if isinstance(finetune_eval, dict) and "retrieval_tasks" in finetune_eval:
            _validate_retrieval_tasks(finetune_eval.get("retrieval_tasks"), source)
        _validate_finetuning_checkpoint_selection(
            finetune_eval,
            validation_sets,
            source,
        )
        include = ((raw.get("models") or {}).get("include")) or []
        include_model_ids = [str(entry.get("model_id")) for entry in include if isinstance(entry, dict)]
        _validate_references(finetune_eval, source, known_sets=seen_names, include_model_ids=include_model_ids)


def _validate_finetuning_checkpoint_selection(
    finetune_eval: Any,
    validation_sets: list[dict[str, Any]],
    source: Path,
) -> None:
    if not isinstance(finetune_eval, dict):
        return
    raw_selection = finetune_eval.get("checkpoint_selection")
    if raw_selection is None:
        return
    if not isinstance(raw_selection, dict):
        raise ValueError(f"finetuning.evaluation.checkpoint_selection must be a mapping in {source}")
    metric = str(raw_selection.get("metric") or raw_selection.get("type") or "pc_at_k").strip().lower()
    if metric == "weighted_mean_pc_at_k_and_transfer":
        if "target_pc" not in finetune_eval:
            raise ValueError(
                f"weighted_mean_pc_at_k_and_transfer requires finetuning.evaluation.target_pc in {source}"
            )
        try:
            target_pc = float(finetune_eval["target_pc"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"finetuning.evaluation.target_pc must be numeric in {source}") from exc
        if target_pc <= 0.0 or target_pc > 1.0:
            raise ValueError(f"finetuning.evaluation.target_pc must be in (0, 1] in {source}")
        names = {str(item.get("name") or "") for item in validation_sets}
        calibration_name = str(raw_selection.get("calibration_validation_set") or "")
        selection_name = str(raw_selection.get("selection_validation_set") or "")
        if not calibration_name or calibration_name not in names:
            raise ValueError(f"calibration_validation_set must name a configured validation set in {source}")
        if not selection_name or selection_name not in names:
            raise ValueError(f"selection_validation_set must name a configured validation set in {source}")
        if calibration_name == selection_name:
            raise ValueError(f"Calibration and selection validation sets must differ in {source}")
        fixed_k = raw_selection.get("fixed_k")
        if not isinstance(fixed_k, list) or not fixed_k:
            raise ValueError(f"checkpoint_selection.fixed_k must be a non-empty list in {source}")
        try:
            fixed_k_int = [int(k) for k in fixed_k]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint_selection.fixed_k must contain integers in {source}") from exc
        if any(k <= 0 for k in fixed_k_int) or len(set(fixed_k_int)) != len(fixed_k_int):
            raise ValueError(
                f"checkpoint_selection.fixed_k must contain unique positive integers in {source}"
            )
        top_k = finetune_eval.get("top_k")
        if not isinstance(top_k, list):
            raise ValueError(f"finetuning.evaluation.top_k must be a list in {source}")
        missing_k = sorted(set(fixed_k_int) - {int(k) for k in top_k})
        if missing_k:
            raise ValueError(
                f"checkpoint_selection.fixed_k values must occur in finetuning.evaluation.top_k; "
                f"missing {missing_k} in {source}"
            )
        weights = raw_selection.get("weights")
        weight_keys = {"mean_pc_at_k", "transferred_pc", "transferred_rr"}
        if not isinstance(weights, dict) or set(weights) != weight_keys:
            raise ValueError(
                "checkpoint_selection.weights must define exactly mean_pc_at_k, transferred_pc, "
                f"and transferred_rr in {source}"
            )
        try:
            weight_values = [float(weights[key]) for key in sorted(weight_keys)]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint_selection.weights must be numeric in {source}") from exc
        if any(not math.isfinite(value) or value < 0.0 for value in weight_values):
            raise ValueError(f"checkpoint_selection.weights must be finite and non-negative in {source}")
        if not math.isclose(sum(weight_values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"checkpoint_selection.weights must sum to 1 in {source}")
        if str(raw_selection.get("epoch_tiebreak", "earliest")).strip().lower() != "earliest":
            raise ValueError(
                f"weighted_mean_pc_at_k_and_transfer epoch_tiebreak must be 'earliest' in {source}"
            )
        by_name = {str(item.get("name")): item for item in validation_sets}
        for set_name in (calibration_name, selection_name):
            tasks = by_name[set_name].get("retrieval_tasks")
            if isinstance(tasks, list) and len(tasks) > 1:
                raise ValueError(
                    "weighted_mean_pc_at_k_and_transfer supports at most one retrieval task "
                    f"in {set_name!r} in {source}"
                )
        return
    if metric in {"mean_precision_at_k", "mean_precision@k"}:
        raw_k = raw_selection.get("k", raw_selection.get("fixed_k", [5, 10, 20, 40, 80]))
        if isinstance(raw_k, int):
            raw_k = [raw_k]
        if not isinstance(raw_k, list) or not raw_k:
            raise ValueError(f"checkpoint_selection.k must be a non-empty list in {source}")
        try:
            k_values = [int(k) for k in raw_k]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint_selection.k must contain integers in {source}") from exc
        if any(k <= 0 for k in k_values) or len(set(k_values)) != len(k_values):
            raise ValueError(
                f"checkpoint_selection.k must contain unique positive integers in {source}"
            )
        top_k = finetune_eval.get("top_k")
        if not isinstance(top_k, list):
            raise ValueError(f"finetuning.evaluation.top_k must be a list in {source}")
        missing_k = sorted(set(k_values) - {int(k) for k in top_k})
        if missing_k:
            raise ValueError(
                "checkpoint_selection.k values must occur in finetuning.evaluation.top_k; "
                f"missing {missing_k} in {source}"
            )
        try:
            precision_tie = float(raw_selection.get("precision_tie", 0.01))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint_selection.precision_tie must be numeric in {source}") from exc
        if not math.isfinite(precision_tie) or precision_tie < 0.0 or precision_tie > 1.0:
            raise ValueError(f"checkpoint_selection.precision_tie must be in [0, 1] in {source}")
        if str(raw_selection.get("epoch_tiebreak", "earliest")).strip().lower() != "earliest":
            raise ValueError(f"mean_precision_at_k epoch_tiebreak must be 'earliest' in {source}")
        return
    if metric == "mean_pair_completeness_at_k":
        raw_k = raw_selection.get("k", raw_selection.get("fixed_k", [5, 10, 20, 40, 60, 80]))
        if isinstance(raw_k, int):
            raw_k = [raw_k]
        if not isinstance(raw_k, list) or not raw_k:
            raise ValueError(f"checkpoint_selection.k must be a non-empty list in {source}")
        try:
            k_values = [int(k) for k in raw_k]
            pc_tie = float(raw_selection.get("pc_tie", 0.005))
            rr_tie = float(raw_selection.get("rr_tie", 0.01))
            target_pc = float(finetune_eval["target_pc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "mean_pair_completeness_at_k requires integer k values, numeric pc_tie and rr_tie, "
                f"and numeric finetuning.evaluation.target_pc in {source}"
            ) from exc
        if any(k <= 0 for k in k_values) or len(set(k_values)) != len(k_values):
            raise ValueError(
                f"checkpoint_selection.k must contain unique positive integers in {source}"
            )
        top_k = finetune_eval.get("top_k")
        if not isinstance(top_k, list):
            raise ValueError(f"finetuning.evaluation.top_k must be a list in {source}")
        missing_k = sorted(set(k_values) - {int(k) for k in top_k})
        if missing_k:
            raise ValueError(
                "checkpoint_selection.k values must occur in finetuning.evaluation.top_k; "
                f"missing {missing_k} in {source}"
            )
        if target_pc <= 0.0 or target_pc > 1.0:
            raise ValueError(f"finetuning.evaluation.target_pc must be in (0, 1] in {source}")
        if not math.isfinite(pc_tie) or pc_tie < 0.0 or pc_tie > 1.0:
            raise ValueError(f"checkpoint_selection.pc_tie must be in [0, 1] in {source}")
        if not math.isfinite(rr_tie) or rr_tie < 0.0 or rr_tie > 1.0:
            raise ValueError(f"checkpoint_selection.rr_tie must be in [0, 1] in {source}")
        if str(raw_selection.get("epoch_tiebreak", "earliest")).strip().lower() != "earliest":
            raise ValueError(
                f"mean_pair_completeness_at_k epoch_tiebreak must be 'earliest' in {source}"
            )
        return
    if metric == "calibration_transfer":
        if "target_pc" not in finetune_eval:
            raise ValueError(
                f"calibration_transfer requires finetuning.evaluation.target_pc in {source}"
            )
        try:
            target_pc = float(finetune_eval["target_pc"])
            minimum_pc = float(raw_selection.get("minimum_selection_pc", 0.98))
            rr_margin = float(raw_selection.get("rr_tie_margin", 0.005))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calibration_transfer PC and RR values must be numeric in {source}") from exc
        if target_pc <= 0.0 or target_pc > 1.0:
            raise ValueError(f"finetuning.evaluation.target_pc must be in (0, 1] in {source}")
        if minimum_pc < 0.0 or minimum_pc > 1.0:
            raise ValueError(f"minimum_selection_pc must be in [0, 1] in {source}")
        if rr_margin < 0.0 or rr_margin > 1.0:
            raise ValueError(f"rr_tie_margin must be in [0, 1] in {source}")
        names = {str(item.get("name") or "") for item in validation_sets}
        calibration_name = str(raw_selection.get("calibration_validation_set") or "")
        selection_name = str(raw_selection.get("selection_validation_set") or "")
        if not calibration_name or calibration_name not in names:
            raise ValueError(f"calibration_validation_set must name a configured validation set in {source}")
        if not selection_name or selection_name not in names:
            raise ValueError(f"selection_validation_set must name a configured validation set in {source}")
        if calibration_name == selection_name:
            raise ValueError(f"Calibration and selection validation sets must differ in {source}")
        if str(raw_selection.get("epoch_tiebreak", "earliest")).strip().lower() != "earliest":
            raise ValueError(f"calibration_transfer epoch_tiebreak must be 'earliest' in {source}")
        return
    if metric != "reduction_ratio_at_target_pc":
        return
    if len(validation_sets) != 1:
        raise ValueError(
            "finetuning.evaluation.checkpoint_selection.metric='reduction_ratio_at_target_pc' "
            f"requires exactly one finetuning.data.validation entry in {source}"
        )
    validation_set = validation_sets[0]
    if validation_set.get("primary") is not True:
        raise ValueError(
            "The sole finetuning.data.validation entry must explicitly set primary: true "
            f"for reduction_ratio_at_target_pc in {source}"
        )
    if "target_pc" not in finetune_eval:
        raise ValueError(
            "reduction_ratio_at_target_pc requires a predeclared numeric "
            f"finetuning.evaluation.target_pc in {source}"
        )
    try:
        target_pc = float(finetune_eval["target_pc"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"finetuning.evaluation.target_pc must be numeric in {source}") from exc
    if target_pc <= 0.0 or target_pc > 1.0:
        raise ValueError(
            f"finetuning.evaluation.target_pc must be in (0, 1] in {source}"
        )
    if "k" in raw_selection or "top_k" in raw_selection:
        raise ValueError(
            f"checkpoint_selection.k is not valid for reduction_ratio_at_target_pc in {source}"
        )
    if "target_pc" in raw_selection:
        raise ValueError(
            "checkpoint_selection.target_pc is not allowed; finetuning.evaluation.target_pc "
            f"is the single source of truth in {source}"
        )
    validation_name = str(validation_set.get("name") or "")
    configured_name = raw_selection.get("validation_set", raw_selection.get("set"))
    if configured_name is not None and str(configured_name) != validation_name:
        raise ValueError(
            f"checkpoint_selection.validation_set={configured_name!r} must match the sole "
            f"validation set {validation_name!r} in {source}"
        )
    local_tasks = validation_set.get("retrieval_tasks")
    if isinstance(local_tasks, list) and len(local_tasks) > 1:
        raise ValueError(
            "reduction_ratio_at_target_pc supports at most one retrieval task in the sole "
            f"validation set in {source}"
        )


def _validate_references(
    finetune_eval: Any,
    source: Path,
    *,
    known_sets: set[str],
    include_model_ids: list[str],
) -> None:
    if not isinstance(finetune_eval, dict):
        return
    references = finetune_eval.get("references")
    if references is None:
        return
    if not isinstance(references, list) or not references:
        raise ValueError(f"finetuning.evaluation.references must be a non-empty list in {source}")
    seen_ids: set[str] = set()
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            raise ValueError(f"finetuning.evaluation.references[{index}] must be a mapping in {source}")
        ref_id = str(reference.get("id") or "")
        if not ref_id:
            raise ValueError(f"finetuning.evaluation.references[{index}].id is required in {source}")
        if ref_id in seen_ids:
            raise ValueError(f"Duplicate finetuning.evaluation.references id {ref_id!r} in {source}")
        seen_ids.add(ref_id)
        if "epoch" in reference:
            raise ValueError(f"finetuning.evaluation.references[{index}] must not define an epoch in {source}")
        kind = str(reference.get("kind") or "")
        if kind != "frozen_backbone":
            raise ValueError(
                f"Only finetuning.evaluation.references kind='frozen_backbone' is supported in {source}"
            )
        model_id = str(reference.get("model_id") or "")
        if not model_id:
            raise ValueError(f"finetuning.evaluation.references[{index}].model_id is required in {source}")
        if include_model_ids.count(model_id) != 1:
            raise ValueError(
                f"finetuning.evaluation.references[{index}].model_id={model_id!r} must resolve to "
                f"exactly one models.include entry in {source}"
            )
        if "validation_sets" in reference:
            filter_sets = reference["validation_sets"]
            if not isinstance(filter_sets, list) or not filter_sets:
                raise ValueError(
                    f"finetuning.evaluation.references[{index}].validation_sets must be a "
                    f"non-empty list in {source}"
                )
            unknown = sorted({str(name) for name in filter_sets} - known_sets)
            if unknown:
                raise ValueError(
                    f"finetuning.evaluation.references[{index}].validation_sets references unknown "
                    f"set(s): {', '.join(unknown)} in {source}"
                )


def _validate_selector(selector: dict[str, Any], source: Path, name: str) -> None:
    if "subset" not in selector or "roles" not in selector:
        raise ValueError(f"{name} must contain subset and roles in {source}")
    roles = selector.get("roles")
    if isinstance(roles, str):
        return
    if not isinstance(roles, list) or not roles:
        raise ValueError(f"{name}.roles must be a non-empty list or string in {source}")


def _validate_augmentations(augmentations: Any, source: Path) -> None:
    if augmentations is None:
        return
    if not isinstance(augmentations, dict):
        raise ValueError(f"finetuning.augmentations must be a mapping in {source}")
    if not augmentations:
        return
    unknown = sorted(set(augmentations) - _AUGMENTATION_KEYS)
    if unknown:
        raise ValueError(f"Unsupported finetuning.augmentations key(s) in {source}: {', '.join(unknown)}")
    if str(augmentations.get("backend", "")).strip().lower() != "albumentations":
        raise ValueError(f"Only finetuning.augmentations.backend='albumentations' is supported in {source}")
    for key in sorted(set(augmentations) - {"backend"}):
        if not isinstance(augmentations[key], dict):
            raise ValueError(f"finetuning.augmentations.{key} must be a mapping in {source}")


_WILDCARD_SUBSET_TOKENS = {"all_subsets", "all-subsets", "allsubsets", "*"}


def _validate_threshold_calibration(evaluation: dict[str, Any], source: Path) -> None:
    calibration = evaluation.get("threshold_calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"evaluation.threshold_calibration must be a mapping in {source}")
    method = str(calibration.get("method") or "")
    if method != "empirical_target_pc":
        raise ValueError(
            f"Only evaluation.threshold_calibration.method='empirical_target_pc' is supported in {source}"
        )
    calibration_id = str(calibration.get("calibration_id") or "")
    if not calibration_id:
        raise ValueError(f"evaluation.threshold_calibration.calibration_id is required in {source}")
    targets = calibration.get("target_pc")
    if not isinstance(targets, list) or not targets:
        raise ValueError(
            f"evaluation.threshold_calibration.target_pc must be a non-empty list in {source}"
        )
    for value in targets:
        target = float(value)
        if target <= 0.0 or target > 1.0:
            raise ValueError(
                f"evaluation.threshold_calibration.target_pc values must be in (0, 1] in {source}"
            )
    task = calibration.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"evaluation.threshold_calibration.task must be a mapping in {source}")
    # Reuse the retrieval-task shape validation for the calibration task.
    _validate_retrieval_tasks([task], source)
    for selector_key in ("query", "candidates"):
        subset = str((task.get(selector_key) or {}).get("subset") or "")
        if subset.lower() in _WILDCARD_SUBSET_TOKENS:
            raise ValueError(
                f"evaluation.threshold_calibration.task.{selector_key} must not use an "
                f"ALL_SUBSETS wildcard in {source}"
            )
        if subset.lower() == "test":
            raise ValueError(
                f"evaluation.threshold_calibration.task.{selector_key} must not use the test "
                f"subset (calibration may not touch the held-out test set) in {source}"
            )
    calibration_task_id = str(task.get("task_id") or "")
    test_task_ids = {
        str(entry.get("task_id"))
        for entry in (evaluation.get("retrieval_tasks") or [])
        if isinstance(entry, dict)
    }
    if calibration_task_id in test_task_ids:
        raise ValueError(
            f"evaluation.threshold_calibration.task.task_id={calibration_task_id!r} must differ "
            f"from every evaluation.retrieval_tasks task_id in {source}"
        )


def _validate_threshold_top_k_floor(evaluation: dict[str, Any], source: Path) -> None:
    floor = evaluation.get("threshold_top_k_floor")
    if not isinstance(floor, dict):
        raise ValueError(f"evaluation.threshold_top_k_floor must be a mapping in {source}")
    if "threshold_calibration" not in evaluation:
        raise ValueError(
            f"evaluation.threshold_top_k_floor requires evaluation.threshold_calibration in {source}"
        )
    method = str(floor.get("method") or "")
    if method != "frozen_calibrated_threshold_or_top_k_floor":
        raise ValueError(
            "Only evaluation.threshold_top_k_floor.method="
            f"'frozen_calibrated_threshold_or_top_k_floor' is supported in {source}"
        )
    values = floor.get("minimum_k")
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"evaluation.threshold_top_k_floor.minimum_k must be a non-empty list in {source}"
        )
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "evaluation.threshold_top_k_floor.minimum_k values must be positive integers "
                f"in {source}"
            )
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"evaluation.threshold_top_k_floor.minimum_k values must be unique in {source}"
        )
    unknown = sorted(set(floor) - {"method", "minimum_k"})
    if unknown:
        raise ValueError(
            "Unsupported evaluation.threshold_top_k_floor key(s): "
            f"{', '.join(unknown)} in {source}"
        )


def _validate_top_k_calibration(evaluation: dict[str, Any], source: Path) -> None:
    calibration = evaluation.get("top_k_calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"evaluation.top_k_calibration must be a mapping in {source}")
    # The leakage and target checks are identical to threshold calibration.
    proxy = dict(evaluation)
    proxy["threshold_calibration"] = calibration
    try:
        _validate_threshold_calibration(proxy, source)
    except ValueError as exc:
        raise ValueError(str(exc).replace("threshold_calibration", "top_k_calibration")) from exc
    policy = str(calibration.get("on_unattained") or "report_only")
    if policy not in {"report_only", "least_restrictive"}:
        raise ValueError(
            f"evaluation.top_k_calibration.on_unattained must be 'report_only' or "
            f"'least_restrictive' in {source}"
        )


def _validate_retrieval_tasks(tasks: Any, source: Path) -> None:
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"evaluation.retrieval_tasks must be a non-empty list in {source}")
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"Each evaluation.retrieval_tasks entry must be a mapping in {source}")
        for key in ("task_id", "query", "candidates", "positive_policy", "exclude_self"):
            if key not in task:
                raise ValueError(f"Retrieval task missing {key!r} in {source}")
        for selector_key in ("query", "candidates"):
            selector = task[selector_key]
            if not isinstance(selector, dict):
                raise ValueError(f"Retrieval task {selector_key} must be a mapping in {source}")
            if "subset" not in selector:
                raise ValueError(f"Retrieval task {selector_key} missing 'subset' in {source}")
            has_role = "role" in selector
            has_roles = "roles" in selector
            if has_role == has_roles:
                raise ValueError(
                    f"Retrieval task {selector_key} must contain exactly one of "
                    f"'role' or 'roles' in {source}"
                )
            if has_roles:
                roles = selector["roles"]
                if not isinstance(roles, list) or not roles or any(
                    not isinstance(role, str) or not role for role in roles
                ):
                    raise ValueError(
                        f"Retrieval task {selector_key}.roles must be a non-empty list "
                        f"of strings in {source}"
                    )
        if str(task["positive_policy"]) != "same_painting":
            raise ValueError("Only positive_policy='same_painting' is supported")


def to_resolved_dict(
    repo: Path, raw: dict[str, Any], dataset: DatasetConfig, split_file: Path, models: list[ModelConfig]
) -> dict[str, Any]:
    resolved = dict(raw)
    resolved["dataset"] = {
        **dict(raw.get("dataset") or {}),
        "dataset_id": dataset.dataset_id,
        "path": str(dataset.path),
        "dataset_db": str(dataset.dataset_db),
        "image_root": str(dataset.image_root) if dataset.image_root else None,
    }
    resolved["split"] = {**dict(raw.get("split") or {}), "file": str(split_file)}
    resolved["models"] = {
        **dict(raw.get("models") or {}),
        "include": [_resolved_model_entry(model) for model in models],
    }
    resolved["environment"] = environment_provenance(models)
    resolved["repo_root"] = str(repo)
    return resolved


def _resolved_model_entry(model: ModelConfig) -> dict[str, Any]:
    entry = {"config_path": str(model.path), **model.raw}
    entry["model_id"] = model.model_id
    entry["model_storage_key"] = model.storage_key
    entry["adapter_name"] = model.adapter.adapter_name
    entry["adapter_kwargs"] = dict(model.adapter.kwargs)
    entry["revision"] = model.revision
    return entry
