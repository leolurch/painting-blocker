#!/usr/bin/env python3
"""Build the Met checkpoint-selection and model-selection halves.

Visitor identities from the official Met val and test query lists are shuffled
once with seed 42 and cut into two identity-disjoint halves. The first half is
checkpoint selection. The second half is model selection. Distractors are not
part of the cut. Gallery candidates are Met catalog photos of the identities in
that half. Queries are the labeled visitor photos.

Pass --painting-ids to drop every visitor identity that is not in that list
before the same shuffle and cut. The SynGallery painting classes are that list.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [dict(row) for row in data]


def _class_id(row: dict[str, Any]) -> int | None:
    raw = row.get("MET_id", row.get("id"))
    if raw is None:
        return None
    return int(raw)


def _file_id(row: dict[str, Any]) -> str:
    path = str(row["path"]).lstrip("/")
    if path.startswith("images/"):
        path = path[len("images/") :]
    return path


def _identity_allowlist(path: Path) -> set[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("identities", data.get("ids"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of identities in {path}")
    allowed: set[int] = set()
    for item in data:
        if isinstance(item, dict):
            class_id = _class_id(item)
            if class_id is None:
                raise ValueError(f"Identity row in {path} has no id")
            allowed.add(class_id)
        else:
            allowed.add(int(item))
    if not allowed:
        raise ValueError(f"No identities in {path}")
    return allowed


def build(root: Path, seed: int, painting_ids: set[int] | None = None) -> dict[str, Any]:
    info = root / "ground_truth"
    queries = _load(info / "valset.json") + _load(info / "testset.json")
    catalog = _load(info / "MET_database.json")
    by_identity: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        class_id = _class_id(row)
        if class_id is None:
            continue
        if painting_ids is not None and class_id not in painting_ids:
            continue
        by_identity[class_id].append(row)
    if painting_ids is not None and not by_identity:
        raise ValueError("Painting filter removed every visitor identity")
    identities = sorted(by_identity)
    rng = random.Random(seed)
    order = list(identities)
    rng.shuffle(order)
    cut = len(order) // 2
    halves = {
        "checkpoint_selection": set(order[:cut]),
        "model_selection": set(order[cut:]),
    }
    catalog_by_class: dict[int, list[str]] = defaultdict(list)
    for row in catalog:
        class_id = _class_id(row)
        if class_id is None:
            continue
        catalog_by_class[class_id].append(_file_id(row))

    images: dict[str, dict[str, Any]] = {}
    subsets: dict[str, dict[str, Any]] = {}
    for name, class_ids in halves.items():
        roles: dict[str, list[str]] = {"query": [], "catalog": []}
        missing = []
        for class_id in sorted(class_ids):
            gallery = catalog_by_class.get(class_id, [])
            if not gallery:
                missing.append(class_id)
                continue
            for file_id in gallery:
                images[file_id] = {"class_id": class_id, "role": "catalog", "origin": "met_catalog"}
                roles["catalog"].append(file_id)
            for row in by_identity[class_id]:
                file_id = _file_id(row)
                images[file_id] = {"class_id": class_id, "role": "query", "origin": "met_visitor"}
                roles["query"].append(file_id)
        if missing:
            raise ValueError(f"{name} identities without catalog photos: {missing[:8]}")
        for role in roles:
            roles[role] = sorted(set(roles[role]))
        subsets[name] = {"roles": roles}
    return {
        "schema_version": 2,
        "split_id": (
            f"met_benchmark_painting_identity_halves_seed_{seed}_v1"
            if painting_ids is not None
            else f"met_benchmark_identity_halves_seed_{seed}_v1"
        ),
        "dataset": {
            "dataset_id": "met_benchmark",
            "image_root": str(root / "images"),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": {
            "tool": "experiments/datasets/met_benchmark/materialize_met_identity_halves.py",
            "seed": seed,
        },
        "split_strategy": {
            "type": "identity_halves",
            "seed": seed,
            "source": "valset.json positive queries plus testset.json positive queries",
            "distractors": "excluded",
            **(
                {
                    "painting_filter": "syngallery_painting_classes",
                    "painting_classes": len(painting_ids),
                    "visitor_identities_after_filter": len(identities),
                }
                if painting_ids is not None
                else {}
            ),
        },
        "images": images,
        "subsets": subsets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--painting-ids",
        type=Path,
        default=None,
        help="JSON list of painting identities. Other visitor identities are dropped before the split.",
    )
    args = parser.parse_args()
    painting_ids = _identity_allowlist(args.painting_ids) if args.painting_ids is not None else None
    split = build(args.root, args.seed, painting_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split) + "\n", encoding="utf-8")
    counts = {}
    for name, subset in split["subsets"].items():
        class_ids = {
            int(split["images"][file_id]["class_id"])
            for file_id in subset["roles"]["query"]
        }
        counts[name] = {
            "identities": len(class_ids),
            **{role: len(ids) for role, ids in subset["roles"].items()},
        }
    print(json.dumps({"output": str(args.output), "images": len(split["images"]), "subsets": counts}, indent=2))


if __name__ == "__main__":
    main()
