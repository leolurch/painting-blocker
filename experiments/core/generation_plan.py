"""Immutable hard-synthetic generation plans and holdout partitioning."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, canonical_json_bytes, sha256_file, utc_now_iso
from .config_schema import DatasetConfig
from .db import class_to_image_ids
from .split_schema import _partition_classes

_ASSET_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_FILE_ID_CLEANUP = re.compile(r"[^A-Za-z0-9._-]+")


def _hash(path: Path) -> str:
    return "sha256:" + sha256_file(path)


def build_asset_registry(directories: dict[str, Path]) -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {}
    for asset_type, directory in sorted(directories.items()):
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in _ASSET_EXTENSIONS:
                registry[_hash(path)] = {"type": asset_type, "name": path.name}
    return registry


def partition_assets(
    registry: dict[str, dict[str, str]], seed: int
) -> dict[str, str]:
    """Partition each asset type independently so every type can be held out."""
    by_type: dict[str, list[str]] = {}
    for digest, metadata in registry.items():
        by_type.setdefault(metadata["type"], []).append(digest)
    assignments: dict[str, str] = {}
    for asset_type, digests in sorted(by_type.items()):
        values = sorted(digests)
        random.Random(f"{seed}:{asset_type}").shuffle(values)
        for index, digest in enumerate(values):
            # Round-robin guarantees disjoint sets and gives tiny registries a
            # deterministic result; missing asset types are validated later.
            assignments[digest] = ("train", "val", "test")[index % 3]
    return assignments


def image_directory_sources(image_dir: Path) -> list[tuple[str, Path]]:
    """Return the exact stable IDs used by the image-directory DB bootstrap."""
    paths = sorted(
        (
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _ASSET_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not paths:
        raise ValueError(f"No supported images found in {image_dir}")
    used: set[str] = set()
    result: list[tuple[str, Path]] = []
    for path in paths:
        base = _FILE_ID_CLEANUP.sub("_", path.stem.strip()).strip("._-") or "image"
        file_id, index = base, 2
        while file_id in used:
            file_id = f"{base}_{index:03d}"
            index += 1
        used.add(file_id)
        result.append((file_id, path))
    return result


def image_directory_hash(image_dir: Path) -> str:
    manifest = [
        {"file_id": file_id, "filename": path.name, "sha256": sha256_file(path)}
        for file_id, path in image_directory_sources(image_dir)
    ]
    return "sha256:" + hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _policies(held_out_profile: str | None) -> tuple[list[str], dict[str, Any]]:
    profiles = ["archival", "print", "cropped_record", "framed_photo"]
    if held_out_profile is not None and held_out_profile not in profiles:
        raise ValueError(f"Unknown held-out profile: {held_out_profile}")
    return profiles, {
        "protocol": "leave_one_profile_out" if held_out_profile else "compound_ood",
        "profile_version": 1,
        "held_out_profile": held_out_profile,
        "train": [profile for profile in profiles if profile != held_out_profile],
        "val": [profile for profile in profiles if profile != held_out_profile],
        "test": profiles,
    }


def _asset_blocks(asset_directories: dict[str, Path] | None, seed: int) -> tuple[dict, dict, str]:
    registry = build_asset_registry(asset_directories or {})
    procedural_fonts = {
        "sha256:" + hashlib.sha256(f"Pillow.load_default:v1:{subset}".encode()).hexdigest(): {
            "type": "font", "name": f"procedural_handwriting_{subset}"
        }
        for subset in ("train", "val", "test")
    }
    registry.update(procedural_fonts)
    assignments = {
        **partition_assets({
            digest: metadata for digest, metadata in registry.items()
            if metadata["type"] != "font"
        }, seed),
        **{
            digest: metadata["name"].rsplit("_", 1)[-1]
            for digest, metadata in procedural_fonts.items()
        },
    }
    registry_hash = "sha256:" + hashlib.sha256(
        json.dumps(registry, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return registry, assignments, registry_hash


def make_generation_plan(
    dataset: DatasetConfig,
    recipe_path: Path,
    *,
    seed: int = 42,
    asset_directories: dict[str, Path] | None = None,
    held_out_profile: str | None = None,
) -> dict[str, Any]:
    _, profile_policy = _policies(held_out_profile)
    class_ids = sorted(class_to_image_ids(dataset.dataset_db))
    if len(class_ids) < 3:
        raise ValueError("A generation plan requires at least three classes")
    partition = _partition_classes(
        class_ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed
    )
    class_assignments = {
        str(class_id): subset
        for subset, ids in partition.items()
        for class_id in ids
    }
    registry, asset_assignments, registry_hash = _asset_blocks(asset_directories, seed)
    recipe_hash = _hash(recipe_path)
    return {
        "schema_version": 1,
        "plan_id": "hard_synth_class_disjoint_v1",
        "created_at": utc_now_iso(),
        "seed": seed,
        "source_dataset_hash": "sha256:" + sha256_file(dataset.dataset_db),
        "recipe_hash": recipe_hash,
        "asset_registry_hash": registry_hash,
        "class_assignments": class_assignments,
        "asset_registry": registry,
        "asset_assignments": asset_assignments,
        "severity_policy": {
            "train": ["mild", "medium"],
            "val": ["medium"],
            "test": ["hard", "extreme"],
        },
        "profile_policy": profile_policy,
        "assignment_key": "class_id",
    }


def make_image_directory_generation_plan(
    image_dir: Path,
    recipe_path: Path,
    *,
    seed: int = 42,
    asset_directories: dict[str, Path] | None = None,
    held_out_profile: str | None = None,
) -> dict[str, Any]:
    """Plan a one-file-per-class raw directory using stable filename IDs."""
    sources = image_directory_sources(image_dir)
    if len(sources) < 3:
        raise ValueError("A generation plan requires at least three source images")
    source_ids = [file_id for file_id, _ in sources]
    partition = _partition_classes(
        source_ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed
    )
    assignments = {
        file_id: subset for subset, ids in partition.items() for file_id in ids
    }
    _, profile_policy = _policies(held_out_profile)
    registry, asset_assignments, registry_hash = _asset_blocks(asset_directories, seed)
    return {
        "schema_version": 1,
        "plan_id": "hard_synth_class_disjoint_v1",
        "created_at": utc_now_iso(),
        "seed": seed,
        "source_kind": "image_dir_one_file_per_class",
        "source_dataset_hash": image_directory_hash(image_dir),
        "recipe_hash": _hash(recipe_path),
        "asset_registry_hash": registry_hash,
        "class_assignments": assignments,
        "assignment_key": "parent_file_id",
        "source_files": {file_id: path.name for file_id, path in sources},
        "asset_registry": registry,
        "asset_assignments": asset_assignments,
        "severity_policy": {
            "train": ["mild", "medium"],
            "val": ["medium"],
            "test": ["hard", "extreme"],
        },
        "profile_policy": profile_policy,
    }


def write_generation_plan(path: Path, plan: dict[str, Any]) -> None:
    atomic_write_json(path, plan)
