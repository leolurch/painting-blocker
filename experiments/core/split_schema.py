"""Static JSON split schema, validation, and materialization helpers.

Schema v2 makes each split self-contained: image IDs are filenames relative to a
flat dataset ``image_root``, subset membership is expressed as named roles, and
per-image labels live in ``split["images"]``. Runtime code never reads the
SQLite dataset DB or a sidecar image manifest.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import EXPERIMENTS_VERSION
from .artifacts import atomic_write_json, sha256_file, utc_now_iso
from .config_schema import DatasetConfig, validate_dataset_config
from .db import (
    all_same_painting_ids,
    class_to_image_ids,
    connect,
    fetch_same_painting_records,
    ids_for_role_or_tag,
    ids_for_split,
    split_exists,
)

ALL_ROLES = "ALL_ROLES"
ALL_SUBSETS = "ALL_SUBSETS"
_ALL_ROLE_TOKENS = {"all_roles", "all-roles", "allroles", "*"}
_ALL_SUBSET_TOKENS = {"all_subsets", "all-subsets", "allsubsets", "*"}


@dataclass(frozen=True)
class SplitImage:
    """DB-free per-image record hydrated from a split JSON file."""

    file_id: str
    class_id: int

    def absolute_path(self, image_root: Path | None) -> Path:
        if image_root is None:
            raise ValueError(f"image_root is required to resolve image {self.file_id!r}")
        return Path(image_root) / self.file_id


def is_all_roles(value: Any) -> bool:
    """Return true for role wildcard tokens (``ALL_ROLES``/``all_roles``/``*``)."""
    return str(value).strip().lower() in _ALL_ROLE_TOKENS


def is_all_subsets(value: Any) -> bool:
    """Return true for subset wildcard tokens (``ALL_SUBSETS``/``all_subsets``/``*``)."""
    return str(value).strip().lower() in _ALL_SUBSET_TOKENS


def manifest_path_for(split_path: Path | str) -> Path:
    """Legacy sidecar path helper retained for callers that report old paths."""
    path = Path(split_path)
    return path.with_name(f"{path.stem}.images.json")


def load_split(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_shape_v2(data)
    validate_split_manifest(data)
    return data


def write_split(split_path: Path | str, split: dict[str, Any]) -> Path:
    """Write a materialized schema-v2 split JSON without any sidecar manifest."""
    validate_split_shape_v2(split)
    validate_split_manifest(split)
    atomic_write_json(Path(split_path), split)
    return Path(split_path)


def split_image_root(split: dict[str, Any]) -> Path | None:
    dataset = split.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Split is missing dataset metadata")
    image_root = dataset.get("image_root")
    return Path(str(image_root)).expanduser() if image_root else None


def records_from_split(split: dict[str, Any]) -> dict[str, SplitImage]:
    """Build DB-free image records from ``split['images']``."""
    images = split.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("Split is missing images labels")
    records: dict[str, SplitImage] = {}
    for file_id, raw in images.items():
        if isinstance(raw, dict):
            class_id = raw.get("class_id")
        else:  # tolerate pre-v2 in-memory fixtures after explicit migration tests
            class_id = raw
        if class_id is None:
            raise ValueError(f"Split image {file_id!r} is missing class_id")
        records[str(file_id)] = SplitImage(str(file_id), int(class_id))
    return records


def ids_for_selector(split: dict[str, Any], subset: str, role: str) -> list[str]:
    """Return image IDs for a single subset/role selector.

    ``subset`` may be ``ALL_SUBSETS``/``all_subsets``/``*`` to select the requested
    role across every materialized subset. ``role`` may be ``ALL_ROLES``/``all_roles``/``*``
    to select every materialized role under the selected subset(s), without requiring
    duplicate ``all`` entries in the split.
    """
    if is_all_subsets(subset):
        subsets = split.get("subsets")
        if not isinstance(subsets, dict) or not subsets:
            raise ValueError("Split contains no subsets")
        ids: list[str] = []
        for subset_name in sorted(str(key) for key in subsets):
            ids.extend(ids_for_selector(split, subset_name, role))
        return _unique_sorted(ids)

    subset_roles = _roles_for_subset(split, subset)
    if is_all_roles(role):
        return _unique_sorted(value for ids in subset_roles.values() for value in ids)
    if role not in subset_roles:
        available = ", ".join(sorted(subset_roles))
        raise ValueError(f"Split subset {subset!r} does not contain role {role!r}; available: {available}")
    return _unique_sorted(subset_roles[role])


def ids_for_multirole_selector(split: dict[str, Any], subset: str, roles: Iterable[str] | str) -> list[str]:
    if isinstance(roles, str):
        role_values = [roles]
    else:
        role_values = [str(role) for role in roles]
    if not role_values:
        raise ValueError(f"Selector for subset {subset!r} must contain at least one role")
    ids: list[str] = []
    for role in role_values:
        ids.extend(ids_for_selector(split, subset, role))
    return _unique_sorted(ids)


def resolve_retrieval_tasks(split: dict[str, Any], task_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve experiment retrieval task selectors to explicit-ID tasks."""
    if not isinstance(task_specs, list) or not task_specs:
        raise ValueError("evaluation.retrieval_tasks must be a non-empty list")
    resolved = []
    for task in task_specs:
        if not isinstance(task, dict):
            raise ValueError("Each retrieval task must be a mapping")
        task_id = str(_require(task, "task_id", "retrieval task"))
        positive_policy = str(_require(task, "positive_policy", task_id))
        if positive_policy != "same_painting":
            raise ValueError(f"Task {task_id!r} uses unsupported positive_policy {positive_policy!r}")
        if "exclude_self" not in task:
            raise ValueError(f"Task {task_id!r} missing required key 'exclude_self'")
        query = _selector(task, "query", task_id)
        candidates = _selector(task, "candidates", task_id)
        query_ids = ids_for_multirole_selector(split, query[0], query[1])
        candidate_ids = ids_for_multirole_selector(split, candidates[0], candidates[1])
        if not query_ids:
            raise ValueError(f"Task {task_id!r} resolved to no query_ids")
        if not candidate_ids:
            raise ValueError(f"Task {task_id!r} resolved to no candidate_ids")
        resolved.append(
            {
                "task_id": task_id,
                "query_ids": query_ids,
                "candidate_ids": candidate_ids,
                "exclude_self": bool(task["exclude_self"]),
                "positive_policy": positive_policy,
            }
        )
    return resolved


def validate_split_shape(data: dict[str, Any]) -> None:
    """Compatibility alias for the only supported split schema."""
    validate_split_shape_v2(data)


def validate_split_shape_v2(data: dict[str, Any]) -> None:
    if data.get("schema_version") != 2:
        raise ValueError("Unsupported split schema_version; expected 2")
    if not data.get("split_id"):
        raise ValueError("Split JSON must contain split_id")
    dataset = data.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("dataset_id"):
        raise ValueError("Split JSON must contain dataset.dataset_id")
    if not dataset.get("image_root"):
        raise ValueError("Split JSON must contain dataset.image_root")
    if "retrieval_tasks" in data:
        raise ValueError("Split schema v2 must not contain retrieval_tasks")
    subsets = data.get("subsets")
    if not isinstance(subsets, dict) or not subsets:
        raise ValueError("Split JSON must contain non-empty subsets")
    for subset_name, subset in subsets.items():
        if not isinstance(subset, dict):
            raise ValueError(f"Split subset {subset_name!r} must be a mapping")
        if "class_ids" in subset or "image_ids" in subset:
            raise ValueError(f"Split subset {subset_name!r} must use roles, not class_ids/image_ids")
        roles = subset.get("roles")
        if not isinstance(roles, dict) or not roles:
            raise ValueError(f"Split subset {subset_name!r} must contain non-empty roles")
        for role, image_ids in roles.items():
            if is_all_roles(role):
                raise ValueError(f"Split subset {subset_name!r} must not materialize wildcard role {role!r}")
            _validate_id_list(image_ids, f"subsets.{subset_name}.roles.{role}")
    images = data.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("Split JSON must contain non-empty images")
    for file_id, record in images.items():
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("Split image IDs must be non-empty strings")
        if not isinstance(record, dict) or "class_id" not in record:
            raise ValueError(f"Split image {file_id!r} must be a mapping with class_id")
        int(record["class_id"])


def validate_split_manifest(split: dict[str, Any]) -> None:
    """Every ID referenced by subset roles must have an ``images`` label."""
    validate_split_shape_v2(split)
    known = set(split["images"])
    missing: list[str] = []
    for subset_name, subset in split["subsets"].items():
        for role, image_ids in subset["roles"].items():
            missing.extend(set(str(value) for value in image_ids) - known)
            if len(image_ids) != len(set(image_ids)):
                raise ValueError(f"Duplicate IDs in subsets.{subset_name}.roles.{role}")
    if missing:
        raise ValueError(f"Split references image IDs absent from images: {sorted(set(missing))[:10]}")


def _roles_for_subset(split: dict[str, Any], subset: str) -> dict[str, list[str]]:
    subsets = split.get("subsets")
    if not isinstance(subsets, dict) or subset not in subsets:
        available = ", ".join(sorted(str(key) for key in (subsets or {}))) if isinstance(subsets, dict) else ""
        raise ValueError(f"Split does not contain subset {subset!r}; available: {available}")
    roles = subsets[subset].get("roles") if isinstance(subsets[subset], dict) else None
    if not isinstance(roles, dict) or not roles:
        raise ValueError(f"Split subset {subset!r} has no roles")
    return {str(role): [str(value) for value in ids] for role, ids in roles.items()}


def _selector(task: dict[str, Any], key: str, task_id: str) -> tuple[str, list[str]]:
    raw = _require(task, key, task_id)
    if not isinstance(raw, dict):
        raise ValueError(f"Task {task_id!r} {key} must be a mapping")
    subset = str(_require(raw, "subset", f"{task_id}.{key}"))
    has_role, has_roles = "role" in raw, "roles" in raw
    if has_role == has_roles:
        raise ValueError(
            f"Task {task_id!r} {key} must contain exactly one of 'role' or 'roles'"
        )
    roles_raw = [raw["role"]] if has_role else raw["roles"]
    if not isinstance(roles_raw, list) or not roles_raw:
        raise ValueError(f"Task {task_id!r} {key}.roles must be a non-empty list")
    roles = [str(role) for role in roles_raw]
    if any(not role for role in roles):
        raise ValueError(f"Task {task_id!r} {key}.roles contains an empty role")
    return subset, roles


def _require(data: dict[str, Any], key: str, source: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required key {key!r} in {source}")
    return data[key]


def _validate_id_list(value: Any, source: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{source} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{source} must contain non-empty string image IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{source} contains duplicate image IDs")


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values})


def _snapshot(dataset: DatasetConfig) -> dict[str, Any]:
    counts = validate_dataset_config(dataset)
    return {**counts, "source_db_sha256": sha256_file(dataset.dataset_db)}


def _dataset_block(dataset: DatasetConfig) -> dict[str, Any]:
    snapshot = _snapshot(dataset)
    return {
        "dataset_id": dataset.dataset_id,
        "image_root": str(dataset.image_root) if dataset.image_root else None,
        "identity": {
            "num_classes": int(snapshot["num_classes"]),
            "num_images": int(snapshot["num_images"]),
        },
        "source_db_sha256": snapshot["source_db_sha256"],
    }


def _finalize_split(split: dict[str, Any], dataset: DatasetConfig) -> dict[str, Any]:
    """Rewrite DB ``file_id`` references to flat image filenames and attach labels."""
    file_ids: set[str] = set()
    for subset in split["subsets"].values():
        for role_ids in subset["roles"].values():
            file_ids.update(str(file_id) for file_id in role_ids)
    records = fetch_same_painting_records(dataset.dataset_db, sorted(file_ids), statuses=None)
    missing = sorted(file_ids - set(records))
    if missing:
        raise ValueError(f"Split references unknown same_painting IDs: {missing[:10]}")
    role_by_id: dict[str, str] = {}
    for subset in split["subsets"].values():
        for role, role_ids in subset["roles"].items():
            for file_id in role_ids:
                role_by_id.setdefault(str(file_id), str(role))
    provenance: dict[str, dict[str, Any]] = {}
    with connect(dataset.dataset_db) as conn:
        has_derivations = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='image_file_derivations'"
        ).fetchone()
        if has_derivations and file_ids:
            placeholders = ",".join("?" for _ in file_ids)
            for child_id, parent_id, params_json in conn.execute(
                f"SELECT child_file_id, parent_file_id, transform_params_json "
                f"FROM image_file_derivations WHERE child_file_id IN ({placeholders})",
                sorted(file_ids),
            ):
                try:
                    params = json.loads(params_json)
                except (TypeError, json.JSONDecodeError):
                    params = {}
                provenance[str(child_id)] = {**params, "parent_file_id": str(parent_id)}

    filename_of: dict[str, str] = {}
    for file_id in sorted(file_ids):
        rel = records[file_id].local_rel_path
        if not rel:
            raise ValueError(f"Image {file_id!r} has no local path to use as its filename ID")
        filename_of[file_id] = Path(rel).name
    if len(set(filename_of.values())) != len(filename_of):
        raise ValueError("Image filenames are not unique across the split; a flat image_root is required")
    for subset in split["subsets"].values():
        for role, role_ids in list(subset["roles"].items()):
            subset["roles"][role] = sorted(filename_of[file_id] for file_id in role_ids)
    split["images"] = {}
    for file_id in sorted(file_ids):
        params = provenance.get(file_id, {})
        dimensions = dict(params.get("dimensions") or {})
        final_size = dimensions.get("final_size") or []
        if len(final_size) == 2:
            dimensions.setdefault("width", int(final_size[0]))
            dimensions.setdefault("height", int(final_size[1]))
        record: dict[str, Any] = {"class_id": int(records[file_id].class_id)}
        if params:
            record.update({
                "role": params.get("role") or role_by_id.get(file_id),
                "origin": params.get("origin") or "synthetic",
            })
        for key in ("profile", "profile_version", "parent_file_id", "assets", "dimensions", "severity", "transforms"):
            if params.get(key) not in (None, {}, []):
                record[key] = params[key]
        severity = params.get("severity") or {}
        if severity:
            record["severity_bucket"] = severity.get("bucket")
            record["severity_score"] = severity.get("score")
        split["images"][filename_of[file_id]] = record
    validate_split_shape_v2(split)
    validate_split_manifest(split)
    return split


def _ids_from_split_or_all(dataset: DatasetConfig, split_name: str) -> list[str]:
    if split_exists(dataset.dataset_db, split_name):
        return ids_for_split(dataset.dataset_db, split_name)
    return all_same_painting_ids(dataset.dataset_db)


def _historic_ids(dataset: DatasetConfig) -> list[str]:
    if split_exists(dataset.dataset_db, "historic"):
        return ids_for_split(dataset.dataset_db, "historic")
    if split_exists(dataset.dataset_db, "historical"):
        return ids_for_split(dataset.dataset_db, "historical")
    return ids_for_role_or_tag(dataset.dataset_db, "historic", "historic", ("historical",))


def _modern_ids(dataset: DatasetConfig) -> list[str]:
    if split_exists(dataset.dataset_db, "modern"):
        return ids_for_split(dataset.dataset_db, "modern")
    return ids_for_role_or_tag(dataset.dataset_db, "modern", "modern")


def _filter_classes_with_both_roles(
    dataset: DatasetConfig, modern_ids: list[str], historic_ids: list[str]
) -> tuple[list[str], list[str], dict[str, Any]]:
    records = fetch_same_painting_records(dataset.dataset_db, sorted(set(modern_ids) | set(historic_ids)))
    modern_classes = {records[file_id].class_id for file_id in modern_ids if file_id in records}
    historic_classes = {records[file_id].class_id for file_id in historic_ids if file_id in records}
    included = modern_classes & historic_classes
    excluded = (modern_classes | historic_classes) - included
    filtered_modern = [file_id for file_id in modern_ids if file_id in records and records[file_id].class_id in included]
    filtered_historic = [file_id for file_id in historic_ids if file_id in records and records[file_id].class_id in included]
    return filtered_modern, filtered_historic, {
        "type": "require_modern_and_historic_per_class",
        "required_roles": ["modern", "historic"],
        "included_num_classes": len(included),
        "excluded_num_classes": len(excluded),
        "excluded_class_ids": sorted(excluded),
        "input_num_modern_images": len(modern_ids),
        "input_num_historic_images": len(historic_ids),
        "included_num_modern_images": len(filtered_modern),
        "included_num_historic_images": len(filtered_historic),
    }


def materialize_baseline_split(dataset: DatasetConfig, preset: str) -> dict[str, Any]:
    all_ids = _ids_from_split_or_all(dataset, "same_painting")
    historic, modern = _historic_ids(dataset), _modern_ids(dataset)
    if not all_ids:
        raise ValueError("No same_painting downloaded/generated image IDs found")
    if not historic or not modern:
        raise ValueError("Baseline split requires both historic and modern image IDs")
    modern, historic, coverage = _filter_classes_with_both_roles(dataset, modern, historic)
    if not modern or not historic:
        raise ValueError("No classes contain both modern and historic image IDs")
    return _finalize_split({
        "schema_version": 2,
        "split_id": f"{dataset.dataset_id}_{preset}_v2",
        "dataset": _dataset_block(dataset),
        "created_at": utc_now_iso(),
        "created_by": {"tool": "experiments.cli make-split", "version": EXPERIMENTS_VERSION, "seed": None},
        "coverage_filter": coverage,
        "subsets": {"eval": {"roles": {"modern": sorted(modern), "historic": sorted(historic)}}},
    }, dataset)


def _partition_classes(class_ids: list[Any], ratios: dict[str, float], seed: int) -> dict[str, list[Any]]:
    shuffled = list(class_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_ratio = float(ratios["train"])
    val_ratio = float(ratios["val"])
    test_ratio = float(ratios["test"])
    if any(ratio < 0.0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Split ratios must be non-negative")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")
    required_subsets = sum(ratio > 0.0 for ratio in (train_ratio, val_ratio, test_ratio))
    if n < required_subsets:
        raise ValueError("Not enough classes to create all requested non-empty subsets")

    train_count = 0 if train_ratio == 0.0 else max(1, int(round(n * train_ratio)))
    val_count = 0 if val_ratio == 0.0 else max(1, int(round(n * val_ratio)))
    if test_ratio > 0.0 and train_count + val_count >= n:
        # Rounding on tiny datasets must not consume the requested test subset.
        if train_ratio > 0.0 and val_ratio > 0.0:
            val_count = 1
            train_count = n - 2
        elif train_ratio > 0.0:
            train_count = n - 1
        elif val_ratio > 0.0:
            val_count = n - 1
    train_end = train_count
    val_end = min(n, train_end + val_count)
    subsets = {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }
    requested_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    missing = [name for name, ratio in requested_ratios.items() if ratio > 0.0 and not subsets[name]]
    if missing:
        raise ValueError(
            "Could not create requested non-empty class-disjoint subsets: " + ", ".join(missing)
        )
    return subsets


def materialize_static_finetune_split(
    dataset: DatasetConfig,
    preset: str,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    grouped = class_to_image_ids(dataset.dataset_db)
    subsets = _partition_classes(sorted(grouped), ratios, seed)
    subset_json = {
        name: {"roles": {"all": sorted(img for cid in ids for img in grouped[cid])}}
        for name, ids in subsets.items()
    }
    return _finalize_split({
        "schema_version": 2,
        "split_id": f"{dataset.dataset_id}_{preset}_class_disjoint_seed_{seed}_v2",
        "dataset": _dataset_block(dataset),
        "created_at": utc_now_iso(),
        "created_by": {"tool": "experiments.cli make-split", "version": EXPERIMENTS_VERSION, "seed": seed},
        "split_strategy": {"type": "class_disjoint", "seed": seed, "ratios": ratios},
        "subsets": subset_json,
    }, dataset)


def _share_id(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


def materialize_class_disjoint_role_split(
    dataset: DatasetConfig,
    preset: str,
    seed: int = 42,
    train_share: float = 0.7,
    validation_share: float | None = None,
) -> dict[str, Any]:
    train_share = float(train_share)
    if not 0.0 <= train_share < 1.0:
        raise ValueError("train_share must be in [0, 1)")
    remainder_share = 1.0 - train_share
    if validation_share is None:
        validation_share = remainder_share / 2.0
    else:
        validation_share = float(validation_share)
        if not 0.0 <= validation_share < 1.0:
            raise ValueError("validation_share must be in [0, 1)")
        if validation_share >= remainder_share:
            raise ValueError("train_share + validation_share must be less than 1")
    ratios = {
        "train": train_share,
        "val": validation_share,
        "test": 1.0 - train_share - validation_share,
    }
    modern, historic = _modern_ids(dataset), _historic_ids(dataset)
    if not modern or not historic:
        raise ValueError("Role split requires both modern and historic image IDs")
    modern, historic, coverage = _filter_classes_with_both_roles(dataset, modern, historic)
    if not modern or not historic:
        raise ValueError("No classes contain both modern and historic image IDs")

    records = fetch_same_painting_records(dataset.dataset_db, sorted(set(modern) | set(historic)), statuses=None)
    modern_by_class: dict[int, list[str]] = {}
    historic_by_class: dict[int, list[str]] = {}
    for file_id in modern:
        modern_by_class.setdefault(records[file_id].class_id, []).append(file_id)
    for file_id in historic:
        historic_by_class.setdefault(records[file_id].class_id, []).append(file_id)

    class_ids = sorted(modern_by_class)
    requested_subset_count = sum(share > 0.0 for share in ratios.values())
    if len(class_ids) < requested_subset_count:
        raise ValueError("Not enough classes to create all requested non-empty subsets")

    partition = _partition_classes(class_ids, ratios, seed)
    subset_json = {
        name: {
            "roles": {
                "modern": sorted(file_id for cid in subset_class_ids for file_id in modern_by_class[cid]),
                "historic": sorted(file_id for cid in subset_class_ids for file_id in historic_by_class[cid]),
            }
        }
        for name, subset_class_ids in partition.items()
        if subset_class_ids
    }
    return _finalize_split({
        "schema_version": 2,
        "split_id": (
            f"{dataset.dataset_id}_{preset}_class_disjoint_"
            f"train_{_share_id(train_share)}_val_{_share_id(validation_share)}_"
            f"test_{_share_id(ratios['test'])}_seed_{seed}_v2"
        ),
        "dataset": _dataset_block(dataset),
        "created_at": utc_now_iso(),
        "created_by": {"tool": "experiments.cli make-split", "version": EXPERIMENTS_VERSION, "seed": seed},
        "split_strategy": {"type": "class_disjoint_role", "seed": seed, "ratios": ratios},
        "coverage_filter": coverage,
        "subsets": subset_json,
    }, dataset)


HARD_SYNTH_ROLES = (
    "modern_original",
    "modern_generated",
    "historic_archival",
    "historic_print",
    "historic_cropped_record",
    "historic_framed_photo",
)


def materialize_hard_synth_split(
    dataset: DatasetConfig,
    preset: str = "hard_synth_class_disjoint_v1",
    seed: int = 42,
    *,
    respect_stored_subsets: bool = True,
) -> dict[str, Any]:
    """Materialize explicit origin/profile roles created by a generation plan."""
    role_ids = {
        role: ids_for_split(dataset.dataset_db, role)
        for role in HARD_SYNTH_ROLES
        if split_exists(dataset.dataset_db, role)
    }
    required = set(HARD_SYNTH_ROLES) - {"modern_generated"}
    missing = sorted(required - set(role_ids))
    if missing:
        raise ValueError(f"Hard-synth dataset is missing explicit roles: {missing}")
    all_ids = sorted({file_id for ids in role_ids.values() for file_id in ids})
    records = fetch_same_painting_records(dataset.dataset_db, all_ids, statuses=None)
    classes_by_role = {
        role: {records[file_id].class_id for file_id in ids if file_id in records}
        for role, ids in role_ids.items()
    }
    complete_classes = set.intersection(*(classes_by_role[role] for role in required))
    if not complete_classes:
        raise ValueError("No classes contain all required hard-synth roles")

    planned: dict[str, set[int]] = {}
    if respect_stored_subsets:
        for subset in ("train", "val", "test"):
            split_name = f"subset_{subset}"
            if split_exists(dataset.dataset_db, split_name):
                planned_ids = ids_for_split(dataset.dataset_db, split_name)
                planned[subset] = {
                    records[file_id].class_id for file_id in planned_ids if file_id in records
                } & complete_classes
    if set(planned) != {"train", "val", "test"}:
        planned = {
            name: set(ids)
            for name, ids in _partition_classes(
                sorted(complete_classes), {"train": 0.7, "val": 0.15, "test": 0.15}, seed
            ).items()
        }
    if any(planned[a] & planned[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Generation-plan class assignments are not disjoint")

    subset_json: dict[str, Any] = {}
    for subset, class_ids in planned.items():
        roles: dict[str, list[str]] = {}
        for role, ids in role_ids.items():
            if subset in {"val", "test"} and role == "modern_generated":
                continue
            selected = sorted(
                file_id for file_id in ids
                if file_id in records and records[file_id].class_id in class_ids
            )
            if selected:
                roles[role] = selected
        subset_json[subset] = {"roles": roles}

    summary_path = dataset.dataset_db.parent / "summary.json"
    build = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            build = dict(summary.get("reproducibility") or {})
        except (OSError, json.JSONDecodeError):
            build = {}
    return _finalize_split({
        "schema_version": 2,
        "split_id": f"{dataset.dataset_id}_{preset}_seed_{seed}_v2",
        "dataset": _dataset_block(dataset),
        "created_at": utc_now_iso(),
        "created_by": {"tool": "experiments.cli make-split", "version": EXPERIMENTS_VERSION, "seed": seed},
        "split_strategy": {"type": "hard_synth_class_disjoint", "seed": seed},
        "build_reproducibility": build,
        "subsets": subset_json,
    }, dataset)


def materialize_split(
    dataset: DatasetConfig,
    preset: str,
    seed: int = 42,
    *,
    train_share: float | None = None,
    validation_share: float | None = None,
) -> dict[str, Any]:
    if preset in {"class_disjoint_role_v1"}:
        return materialize_class_disjoint_role_split(
            dataset,
            preset,
            seed,
            train_share=0.7 if train_share is None else train_share,
            validation_share=validation_share,
        )
    if train_share is not None or validation_share is not None:
        raise ValueError(
            "--train-share and --validation-share are only supported by the "
            "class_disjoint_role_v1 preset"
        )
    if preset in {"baseline_full", "baseline_mini10", "mini10_baseline"}:
        return materialize_baseline_split(dataset, preset)
    if preset in {"static_finetune_v1", "finetune_static_v1"}:
        return materialize_static_finetune_split(dataset, preset, seed)
    if preset == "hard_synth_class_disjoint_v1":
        return materialize_hard_synth_split(dataset, preset, seed)
    raise ValueError(f"Unknown split preset: {preset}")
