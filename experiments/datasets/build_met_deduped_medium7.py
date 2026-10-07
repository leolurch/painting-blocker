#!/usr/bin/env python3
"""Build a Met-deduped SYN set: 1 original, 3 modern, 4 historic, all medium.

Manual review accepted one residual pair, ARTIC ``artic_111872`` against Met
``metmuseum_437830``. That class is omitted. Every remaining class keeps the
museum source. The generator then writes one identity image, three medium
modern synthetics, and one medium image of each historic profile. Train, val,
and test all contain those eight images.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("data/work")
sys.path.insert(0, str(REPO))

SRC_IMAGES = Path(
    "data/datasets/"
    "hard_synth_v1_wikidata_overlap_removed_class_sources/images"
)
PRIOR_SPLIT = Path(
    "data/experiments/configs/splits/"
    "museum_deduplication_hard_synth_v1_met_identity_removed/"
    "hard_synth_class_disjoint_v1.json"
)
STAGING = Path(
    "data/datasets/"
    "museum_deduplication_hard_synth_v1_met_deduped_medium7_v1_sources"
)
DATASET_ID = "museum_deduplication_hard_synth_v1_met_deduped_medium_o1m3h4_v1"
OUTPUT = Path("data/datasets") / DATASET_ID
PLAN = Path("data/datasets") / f"{DATASET_ID}_generation_plan.json"
SPLIT_OUT = (
    Path("data/experiments/configs/splits")
    / DATASET_ID
    / "hard_synth_class_disjoint_v1.json"
)
DROPPED = {"artic_111872"}
HISTORIC_PROFILES = ["archival", "print", "cropped_record", "framed_photo"]
ROLES = ["modern_original", "modern_generated", *[f"historic_{name}" for name in HISTORIC_PROFILES]]
RECIPE = REPO / "experiments/configs/synthetic/hard_synth_medium_o1m3h4_v1.yml"
FRAMES = REPO / "experiments/datasets/image-distorter/horizontal_frames"
OVERLAYS = REPO / "experiments/datasets/image-distorter/reflection_overlay"
TEXTURES = REPO / "experiments/datasets/image-distorter/wall_textures"
GENERATOR = REPO / "experiments/datasets/image-distorter/generate_synth_dataset.py"


def class_subsets() -> dict[str, str]:
    split = json.loads(PRIOR_SPLIT.read_text())
    assigned: dict[str, str] = {}
    for subset, body in split["subsets"].items():
        for name in body["roles"].get("modern_original", []):
            record = split["images"][name]
            label = str(record.get("parent_file_id") or Path(name).stem)
            if label in DROPPED:
                continue
            assigned[label] = subset
    return assigned


def stage_sources(assigned: dict[str, str]) -> list[str]:
    if STAGING.exists() and any(STAGING.glob("*.jpg")):
        print("reusing staged sources", STAGING, flush=True)
        return []
    if STAGING.exists():
        raise SystemExit(f"refusing to overwrite {STAGING}")
    STAGING.mkdir(parents=True)
    missing = []
    for label in sorted(assigned):
        src = SRC_IMAGES / f"{label}.jpg"
        if not src.is_file():
            missing.append(label)
            continue
        os.link(src, STAGING / f"{label}.jpg")
    print("staged", len(assigned) - len(missing), "missing", len(missing), flush=True)
    return missing


def write_plan(assigned: dict[str, str]) -> None:
    from experiments.core.generation_plan import (
        make_image_directory_generation_plan,
        write_generation_plan,
    )
    plan = make_image_directory_generation_plan(
        STAGING,
        RECIPE,
        seed=42,
        asset_directories={
            "frame": FRAMES,
            "overlay": OVERLAYS,
            "wall_texture": TEXTURES,
        },
    )
    staged = {path.stem for path in STAGING.glob("*.jpg")}
    plan["class_assignments"] = {
        label: subset for label, subset in assigned.items() if label in staged
    }
    plan["plan_id"] = "met_deduped_medium_o1m3h4_v1"
    plan["severity_policy"] = {subset: ["medium"] for subset in ("train", "val", "test")}
    plan["profile_policy"] = {subset: list(HISTORIC_PROFILES) for subset in ("train", "val", "test")}
    plan["composition"] = {
        "originals_per_class": 1,
        "modern_synthetics_per_class": 3,
        "historic_synthetics_per_class": 4,
        "historic_profiles": list(HISTORIC_PROFILES),
        "severity_preset": "medium",
        "dropped_manual_accepts": sorted(DROPPED),
    }
    write_generation_plan(PLAN, plan)
    print("plan classes", len(plan["class_assignments"]), flush=True)


def generate() -> None:
    command = [
        sys.executable,
        str(GENERATOR),
        str(STAGING),
        "--output-root",
        str(OUTPUT),
        "--source-kind",
        "image-dir",
        "--name",
        DATASET_ID,
        "--recipe",
        "hard-synth-v1",
        "--recipe-config",
        str(RECIPE),
        "--generation-plan",
        str(PLAN),
        "--modern-variants",
        "3",
        "--historic-variants",
        "4",
        "--workers",
        "16",
        "--process-start-method",
        "fork",
        "--seed",
        "42",
        "--max-output-dimension",
        "1024",
        "--frames-horizontal",
        str(FRAMES),
        "--overlays",
        str(OVERLAYS),
        "--wall-textures",
        str(TEXTURES),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_split() -> None:
    import sqlite3

    db = OUTPUT / "dataset.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    plan = json.loads(PLAN.read_text())
    assignments = plan["class_assignments"]
    placeholders = ",".join("?" for _ in ROLES)
    rows = con.execute(
        f"""
        SELECT f.file_id, f.local_rel_path, c.class_id, c.label, s.split_name, d.transform_params_json
        FROM image_files f
        JOIN image_file_classes j ON j.file_id = f.file_id
        JOIN classes c ON c.class_id = j.class_id
        JOIN splits s ON s.file_id = f.file_id
        LEFT JOIN image_file_derivations d ON d.child_file_id = f.file_id
        WHERE s.split_name IN ({placeholders})
        """,
        ROLES,
    ).fetchall()
    images: dict[str, dict] = {}
    roles: dict[str, dict[str, list[str]]] = {
        subset: {role: [] for role in ROLES}
        for subset in ("train", "val", "test")
    }
    for row in rows:
        label = row["label"]
        subset = assignments.get(label)
        if subset not in roles:
            continue
        filename = Path(row["local_rel_path"]).name
        params = {}
        if row["transform_params_json"]:
            params = json.loads(row["transform_params_json"])
        severity = params.get("severity") or {}
        record = {
            "class_id": int(row["class_id"]),
            "role": row["split_name"],
            "origin": params.get("origin") or "synthetic",
            "parent_file_id": params.get("parent_file_id") or label,
            "profile": params.get("profile"),
            "profile_version": params.get("profile_version") or 1,
            "severity_bucket": severity.get("bucket"),
            "severity_score": severity.get("score"),
            "severity": severity,
        }
        if params.get("dimensions"):
            record["dimensions"] = params["dimensions"]
        if params.get("transforms"):
            record["transforms"] = params["transforms"]
        images[filename] = record
        roles[subset][row["split_name"]].append(filename)
    for subset in roles.values():
        for role, names in subset.items():
            subset[role] = sorted(set(names))
    counts = {
        subset: {role: len(names) for role, names in body.items()}
        for subset, body in roles.items()
    }
    split = {
        "schema_version": 2,
        "split_id": f"{DATASET_ID}_seed_42",
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_db": str(db),
            "image_root": str(OUTPUT / "images"),
        },
        "split_strategy": {
            "type": "class_disjoint_role",
            "seed": 42,
            "images_per_class": {
                "modern_original": 1,
                "modern_generated": 3,
                "historic_archival": 1,
                "historic_print": 1,
                "historic_cropped_record": 1,
                "historic_framed_photo": 1,
            },
            "severity": "medium",
        },
        "subsets": {name: {"roles": body} for name, body in roles.items()},
        "images": images,
        "role_counts": counts,
    }
    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_OUT.write_text(json.dumps(split))
    print(json.dumps(counts, indent=2), flush=True)


def main() -> None:
    if "--write-split" in sys.argv:
        write_split()
        return
    assigned = class_subsets()
    print("assigned", len(assigned), flush=True)
    stage_sources(assigned)
    write_plan(assigned)
    generate()
    write_split()


if __name__ == "__main__":
    main()
