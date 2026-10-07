"""Machine validator for the hard-synth class/asset/severity protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .split_schema import load_split, resolve_retrieval_tasks, split_image_root

PROFILES = ("archival", "print", "cropped_record", "framed_photo")
HISTORIC_ROLES = tuple(f"historic_{profile}" for profile in PROFILES)
EXPECTED_COUNTS = {"modern_original": 1, **{role: 2 for role in HISTORIC_ROLES}}


def _severity_from_transforms(transforms: dict[str, Any]) -> tuple[str, float]:
    crop = transforms.get("crop") or {}
    resolution = transforms.get("resolution_loss") or {}
    score = max(0.0, 1.0 - float(crop.get("retained_area_fraction", 1.0))) * 0.40
    score += max(0.0, min(1.0, (192 - int(resolution.get("longest_side_pixels", 1024))) / 160.0)) * 0.45
    score += 0.10 if (transforms.get("perspective") or {}).get("applied") else 0.0
    score += min(0.10, float((transforms.get("occlusion") or {}).get("coverage", 0.0)))
    score += 0.05 if (transforms.get("annotations") or {}).get("applied") else 0.0
    score += 0.05 if (transforms.get("paper_document") or {}).get("applied") else 0.0
    score += 0.08 if int((transforms.get("jpeg") or {}).get("quality", 100)) <= 72 else 0.0
    score = round(min(1.0, score), 6)
    bucket = "extreme" if score >= 0.72 else "hard" if score >= 0.50 else "medium" if score >= 0.25 else "mild"
    return bucket, score


def retrieval_task_specs(subset: str = "test") -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"{subset}_original_to_{profile}",
            "query": {"subset": subset, "role": "modern_original"},
            "candidates": {"subset": subset, "role": f"historic_{profile}"},
            "positive_policy": "same_painting",
            "exclude_self": False,
        }
        for profile in PROFILES
    ]


def _classes(split: dict, subset: str) -> set[int]:
    ids = {
        image_id
        for role_ids in split["subsets"][subset]["roles"].values()
        for image_id in role_ids
    }
    return {int(split["images"][image_id]["class_id"]) for image_id in ids}


def validate_hard_synth_benchmark(
    split: dict[str, Any],
    generation_plan: dict[str, Any],
    *,
    inspect_headers: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    images = split["images"]
    root = split_image_root(split)

    # Class and parent leakage.
    subset_classes = {name: _classes(split, name) for name in ("train", "val", "test")}
    class_assignments = generation_plan.get("class_assignments") or {}
    assignment_key = generation_plan.get("assignment_key", "class_id")
    for subset, class_ids in subset_classes.items():
        if assignment_key == "class_id":
            for class_id in class_ids:
                planned = class_assignments.get(str(class_id))
                if planned is not None and planned != subset:
                    errors.append(f"class {class_id} materialized in {subset}, planned {planned}")
        else:
            subset_ids = {
                value for values in split["subsets"][subset]["roles"].values() for value in values
            }
            for parent_id in {
                str(images[value].get("parent_file_id")) for value in subset_ids
                if images[value].get("parent_file_id")
            }:
                planned = class_assignments.get(parent_id)
                if planned != subset:
                    errors.append(f"source {parent_id} materialized in {subset}, planned {planned}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = subset_classes[left] & subset_classes[right]
        if overlap:
            errors.append(f"class leakage {left}/{right}: {sorted(overlap)[:10]}")
    parents: dict[str, set[str]] = {}
    for subset in ("train", "val", "test"):
        ids = {
            value for values in split["subsets"][subset]["roles"].values() for value in values
        }
        parents[subset] = {
            str(images[value].get("parent_file_id"))
            for value in ids if images[value].get("parent_file_id")
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = parents[left] & parents[right]
        if overlap:
            errors.append(f"parent leakage {left}/{right}: {sorted(overlap)[:5]}")

    # Exact benchmark structure and provenance role agreement.
    test_roles = split["subsets"]["test"]["roles"]
    by_class_role: dict[int, Counter] = defaultdict(Counter)
    for role, ids in test_roles.items():
        for image_id in ids:
            record = images[image_id]
            class_id = int(record["class_id"])
            by_class_role[class_id][role] += 1
            if role in EXPECTED_COUNTS and record.get("role") != role:
                errors.append(f"role/provenance mismatch: {image_id}: {role} != {record.get('role')}")
            if role == "modern_original":
                if record.get("origin") != "source_original" or not record.get("parent_file_id"):
                    errors.append(f"modern_original lacks source derivation: {image_id}")
                if record.get("profile") != "modern_query":
                    errors.append(f"modern_original has wrong profile: {image_id}")
    for class_id in sorted(subset_classes["test"]):
        for role, expected in EXPECTED_COUNTS.items():
            actual = by_class_role[class_id][role]
            if actual != expected:
                errors.append(f"class {class_id} role {role}: expected {expected}, got {actual}")

    # Dimensions: trust headers, not generated metadata.
    checked_headers = 0
    for image_id, record in images.items():
        dimensions = record.get("dimensions") or {}
        if not dimensions:
            errors.append(f"missing dimension provenance: {image_id}")
            continue
        maximum = int(dimensions.get("configured_max_output_dimension", 0))
        cap = int(dimensions.get("realized_cap", 0))
        configured_build_max = (split.get("build_reproducibility") or {}).get("max_output_dimension")
        if configured_build_max is not None and maximum != int(configured_build_max):
            errors.append(f"dimension/build setting mismatch: {image_id}: {maximum} != {configured_build_max}")
        if maximum < 1 or not round(0.85 * maximum) <= cap <= maximum:
            errors.append(f"invalid realized cap: {image_id}: cap={cap}, maximum={maximum}")
        width = int(dimensions.get("width", dimensions.get("final_size", [0, 0])[0]))
        height = int(dimensions.get("height", dimensions.get("final_size", [0, 0])[1]))
        if inspect_headers:
            if root is None:
                errors.append("split has no image_root for header validation")
                break
            path = root / image_id
            if not path.is_file():
                errors.append(f"missing local image: {path}")
                continue
            with Image.open(path) as image:
                width, height = image.size
            checked_headers += 1
        if width > cap or height > cap or max(width, height) > maximum:
            errors.append(f"dimension violation: {image_id}: {width}x{height}, cap={cap}, max={maximum}")

    # Asset hashes must respect the immutable plan and train/test disjointness.
    asset_usage: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for subset in ("train", "val", "test"):
        ids = {value for values in split["subsets"][subset]["roles"].values() for value in values}
        for image_id in ids:
            for asset_type, digest in (images[image_id].get("assets") or {}).items():
                asset_usage[asset_type][str(digest)].append(f"{subset}:{image_id}")
                assigned = generation_plan.get("asset_assignments", {}).get(str(digest))
                if assigned is not None and assigned != subset:
                    errors.append(f"asset assignment violation: {asset_type} {digest} in {subset}, planned {assigned}")
    for asset_type, hashes in asset_usage.items():
        train = {digest for digest, examples in hashes.items() if any(x.startswith("train:") for x in examples)}
        test = {digest for digest, examples in hashes.items() if any(x.startswith("test:") for x in examples)}
        for digest in sorted(train & test):
            errors.append(f"asset leakage: {asset_type} {digest}: {hashes[digest][:4]}")

    # Declared severity ranges apply to historic generated records.
    severity_policy = generation_plan.get("severity_policy") or {}
    for subset in ("train", "val", "test"):
        allowed = set(severity_policy.get(subset) or [])
        if not allowed:
            continue
        for role, ids in split["subsets"][subset]["roles"].items():
            if not role.startswith("historic_"):
                continue
            for image_id in ids:
                preset = (images[image_id].get("severity") or {}).get("preset")
                if preset not in allowed:
                    errors.append(f"severity holdout violation: {subset}:{image_id} preset={preset}")

    # Profile/version holdout declared by the generation plan.
    profile_policy = generation_plan.get("profile_policy") or {}
    held_out = profile_policy.get("held_out_profile")
    if held_out:
        held_role = f"historic_{held_out}"
        if split["subsets"]["train"]["roles"].get(held_role):
            errors.append(f"held-out profile appears in train: {held_out}")
        for image_id in test_roles.get(held_role, []):
            if int(images[image_id].get("profile_version", -1)) != int(profile_policy.get("profile_version", 1)):
                errors.append(f"held-out profile version mismatch: {image_id}")

    # Numeric severity and required effects for test candidates.
    for role in HISTORIC_ROLES:
        for image_id in test_roles.get(role, []):
            record = images[image_id]
            severity = record.get("severity") or {}
            if severity.get("preset") not in {"hard", "extreme"}:
                errors.append(f"test item has non-hard preset: {image_id}")
            score = float(severity.get("score", -1))
            if not 0.0 <= score <= 1.0:
                errors.append(f"invalid severity score: {image_id}")
            if severity.get("bucket") not in {"hard", "extreme"}:
                errors.append(f"test item classified below hard: {image_id}")
            transforms = record.get("transforms") or {}
            computed_bucket, computed_score = _severity_from_transforms(transforms)
            if severity.get("bucket") != computed_bucket or abs(score - computed_score) > 1e-5:
                errors.append(
                    f"severity provenance disagrees with transforms: {image_id}: "
                    f"declared={severity.get('bucket')}/{score}, computed={computed_bucket}/{computed_score}"
                )
            resolution = transforms.get("resolution_loss") or {}
            limit = 96 if severity.get("preset") == "extreme" else 160
            if not resolution.get("applied") or int(resolution.get("longest_side_pixels", 10**9)) > limit:
                errors.append(f"required resolution loss missing/out of range: {image_id}")
            crop = transforms.get("crop") or {}
            retained = float(crop.get("retained_area_fraction", 1.0))
            max_retained = 0.60 if severity.get("preset") == "extreme" else 0.75
            if not crop.get("applied") or retained > max_retained + 0.01:
                errors.append(f"required hard crop missing/out of range: {image_id}")
            perspective = transforms.get("perspective") or {}
            if perspective.get("applied") and float(perspective.get("visible_fraction", 0.0)) <= 0.0:
                errors.append(f"invalid perspective visibility: {image_id}")
            profile = record.get("profile")
            if profile == "archival":
                if not transforms.get("vintage", {}).get("applied") or not perspective.get("applied"):
                    errors.append(f"archival requirements missing: {image_id}")
            elif profile == "print":
                if not transforms.get("paper_document", {}).get("applied") or not transforms.get("annotations", {}).get("applied"):
                    errors.append(f"print requirements missing: {image_id}")
            elif profile == "cropped_record":
                has_context = transforms.get("annotations", {}).get("applied") or transforms.get("paper_document", {}).get("applied")
                if not has_context:
                    errors.append(f"cropped-record context missing: {image_id}")
            elif profile == "framed_photo":
                if not transforms.get("frame") or not perspective.get("applied"):
                    errors.append(f"framed-photo frame/perspective missing: {image_id}")
                if not (transforms.get("glare", {}).get("applied") or transforms.get("occlusion", {}).get("applied")):
                    errors.append(f"framed-photo glare/occlusion missing: {image_id}")
            occlusion = transforms.get("occlusion") or {}
            if occlusion.get("applied") and not 0.0 < float(occlusion.get("coverage", 0.0)) < 0.5:
                errors.append(f"invalid occlusion coverage: {image_id}")

    # Resolve the public protocol and prove exactly two positives/query/task.
    tasks = resolve_retrieval_tasks(split, retrieval_task_specs())
    candidate_counts = set()
    for task, profile in zip(tasks, PROFILES):
        candidate_counts.add(len(task["candidate_ids"]))
        query_ids = task["query_ids"]
        candidates = task["candidate_ids"]
        if any(images[value].get("role") != "modern_original" for value in query_ids):
            errors.append(f"task {task['task_id']} contains a non-original query")
        if any(images[value].get("role") != f"historic_{profile}" for value in candidates):
            errors.append(f"task {task['task_id']} contains wrong-profile candidates")
        candidate_class_counts = Counter(int(images[value]["class_id"]) for value in candidates)
        for query in query_ids:
            positives = candidate_class_counts[int(images[query]["class_id"])]
            if positives != 2:
                errors.append(f"task {task['task_id']} query {query}: expected 2 positives, got {positives}")
    if len(candidate_counts) != 1:
        errors.append(f"candidate counts differ by profile: {sorted(candidate_counts)}")

    required_repro = {
        "generator_commit", "generator_version", "recipe_sha256",
        "generation_plan_sha256", "source_dataset_sha256", "asset_registry_sha256",
        "random_seed", "dimension_jitter_fraction", "max_output_dimension",
    }
    build_repro = split.get("build_reproducibility") or {}
    missing_repro = sorted(required_repro - set(build_repro))
    if missing_repro:
        errors.append(f"missing reproducibility fields: {missing_repro}")
    for plan_key, build_key in (
        ("recipe_hash", "recipe_sha256"),
        ("source_dataset_hash", "source_dataset_sha256"),
        ("asset_registry_hash", "asset_registry_sha256"),
        ("_file_sha256", "generation_plan_sha256"),
    ):
        expected = generation_plan.get(plan_key)
        if expected is not None and build_repro.get(build_key) != expected:
            errors.append(
                f"reproducibility hash mismatch: {build_key}: "
                f"{build_repro.get(build_key)!r} != {expected!r}"
            )
    if float(build_repro.get("dimension_jitter_fraction", -1)) != 0.15:
        errors.append("dimension jitter constant is not 0.15")

    if errors:
        raise ValueError("Hard-synth benchmark validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "ok",
        "num_test_classes": len(subset_classes["test"]),
        "num_images": len(images),
        "headers_inspected": checked_headers,
        "retrieval_tasks": [task["task_id"] for task in tasks],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--generation-plan", required=True, type=Path)
    parser.add_argument("--skip-image-headers", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.generation_plan.read_text(encoding="utf-8"))
    import hashlib
    plan["_file_sha256"] = "sha256:" + hashlib.sha256(args.generation_plan.read_bytes()).hexdigest()
    report = validate_hard_synth_benchmark(
        load_split(args.split),
        plan,
        inspect_headers=not args.skip_image_headers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
