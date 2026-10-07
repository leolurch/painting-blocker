#!/usr/bin/env python3
"""Write the SynView split the trainer reads.

The class assignment is ``data/syn/class_split.csv``. Image names come from
the generated dataset, after ``generate_synth_dataset.py`` has run. An
existing split file is left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET_ID = "museum_deduplication_hard_synth_v1_met_deduped_medium_o1m3h4_v1"
DEFAULT_OUTPUT = (
    REPO
    / "experiments/configs/splits"
    / DATASET_ID
    / "hard_synth_class_disjoint_v1.json"
)
DEFAULT_DATASET = REPO / "data/datasets" / DATASET_ID
DEFAULT_CLASSES = REPO / "data/syn/class_split.csv"
ROLES = (
    "modern_original",
    "modern_generated",
    "historic_archival",
    "historic_print",
    "historic_cropped_record",
    "historic_framed_photo",
)
IMAGES_PER_CLASS = {
    "modern_original": 1,
    "modern_generated": 3,
    "historic_archival": 1,
    "historic_print": 1,
    "historic_cropped_record": 1,
    "historic_framed_photo": 1,
}


def load_assignments(path: Path) -> dict[str, str]:
    assigned: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subset = row["subset"]
            if subset not in {"train", "val", "test"}:
                raise SystemExit(f"bad subset {subset!r} for {row['parent_file_id']}")
            assigned[row["parent_file_id"]] = subset
    if not assigned:
        raise SystemExit(f"no paintings in {path}")
    return assigned


def write_synview_split(
    *,
    dataset_root: Path,
    class_split: Path,
    output: Path,
) -> dict[str, dict[str, int]]:
    """Build the split JSON. Refuses to replace ``output`` when it already exists."""
    if output.exists():
        raise SystemExit(f"leaving existing split in place: {output}")
    database = dataset_root / "dataset.db"
    if not database.is_file():
        raise SystemExit(f"generated dataset not found: {database}")
    assigned = load_assignments(class_split)
    placeholders = ",".join("?" for _ in ROLES)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        f"""
        SELECT f.local_rel_path, c.class_id, d.parent_file_id, s.split_name, d.transform_params_json
        FROM image_files f
        JOIN image_file_classes j ON j.file_id = f.file_id
        JOIN classes c ON c.class_id = j.class_id
        JOIN splits s ON s.file_id = f.file_id
        JOIN image_file_derivations d ON d.child_file_id = f.file_id
        WHERE j.primary_label = 'same_painting' AND s.split_name IN ({placeholders})
        """,
        ROLES,
    ).fetchall()
    connection.close()
    images: dict[str, dict] = {}
    roles: dict[str, dict[str, list[str]]] = {
        subset: {role: [] for role in ROLES} for subset in ("train", "val", "test")
    }
    per_class: dict[str, Counter[str]] = {label: Counter() for label in assigned}
    for row in rows:
        label = str(row["parent_file_id"])
        subset = assigned.get(label)
        if subset is None:
            raise SystemExit(f"generated painting {label} is not in {class_split}")
        filename = Path(row["local_rel_path"]).name
        if filename in images:
            raise SystemExit(f"duplicate generated file {filename}")
        params = json.loads(row["transform_params_json"]) if row["transform_params_json"] else {}
        severity = params.get("severity") or {}
        record = {
            "class_id": int(row["class_id"]),
            "role": row["split_name"],
            "origin": params.get("origin") or "synthetic",
            "parent_file_id": label,
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
        per_class[label][row["split_name"]] += 1
    missing = sorted(label for label, present in per_class.items() if present != IMAGES_PER_CLASS)
    if missing:
        raise SystemExit(
            f"{len(missing)} paintings do not have the eight generated views, first {missing[:3]}"
        )
    for subset in roles.values():
        for role, names in subset.items():
            subset[role] = sorted(set(names))
    counts = {
        subset: {role: len(names) for role, names in body.items()}
        for subset, body in roles.items()
    }
    image_root = dataset_root / "images"
    split = {
        "schema_version": 2,
        "split_id": f"{DATASET_ID}_seed_42",
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_db": str(database),
            "image_root": str(image_root),
        },
        "split_strategy": {
            "type": "class_disjoint_role",
            "seed": 42,
            "images_per_class": dict(IMAGES_PER_CLASS),
            "severity": "medium",
        },
        "subsets": {name: {"roles": body} for name, body in roles.items()},
        "images": images,
        "role_counts": counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--class-split", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    counts = write_synview_split(
        dataset_root=args.dataset_root,
        class_split=args.class_split,
        output=args.output,
    )
    print(json.dumps(counts, indent=2))
    print(args.output)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise
