"""Assign Lost Art distractors to the painting halves of the test set.

Wikidata class membership is copied from the existing full and 50/50 splits.
Lost Art identities accepted in the review are left out. Every remaining Lost
Art identity is one distractor class, and all of its images stay in one half.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WIKI_SPLIT_DIR = REPO / "experiments/configs/splits/wikidata-1.5"
OUT_DIR = REPO / "experiments/configs/splits/wikidata1.5-lostart-distractors"
NAMES_PATH = Path("data/lostart/filenames.txt")
IMAGE_ROOT = "data/datasets/wikidata1.5-lostart-distractors/images"
DATASET_ID = "wikidata1.5-lostart-distractors"
EXCLUDED_IDENTITIES = frozenset({"91918", "44146"})
CLASS_OFFSET = 10_000_000
SEEDS = (6, 7, 9, 10, 21, 42, 67, 87, 1337, 4711)
IDENTITY_RE = re.compile(r"^LA(\d+)")


def _identity(name: str) -> str:
    match = IDENTITY_RE.match(name)
    if match is None:
        raise ValueError(f"Lost Art filename has no identity id: {name}")
    return match.group(1)


def _load_groups(names_path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in names_path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        groups.setdefault(_identity(name), []).append(name)
    for names in groups.values():
        names.sort()
    return groups


def _groups_from_split(path: Path) -> dict[str, list[str]]:
    """Lost Art filenames already accepted into a built split."""
    split = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = {}
    for name in split["images"]:
        if not IDENTITY_RE.match(name):
            continue
        groups.setdefault(_identity(name), []).append(name)
    for names in groups.values():
        names.sort()
    return groups


def _subset_from_split(path: Path, identities: set[str]) -> dict[str, str]:
    """Identity to val/test, copied from an existing distractor split."""
    split = json.loads(path.read_text(encoding="utf-8"))
    subset_of: dict[str, str] = {}
    for subset_name, subset in split["subsets"].items():
        for name in subset["roles"].get("historic", []):
            if not IDENTITY_RE.match(name):
                continue
            identity = _identity(name)
            if identity not in identities:
                continue
            previous = subset_of.get(identity)
            if previous is not None and previous != subset_name:
                raise ValueError(f"Identity {identity} is in {previous} and {subset_name} of {path.name}")
            subset_of[identity] = subset_name
    missing = sorted(identities - set(subset_of), key=int)
    if missing:
        raise ValueError(f"{path.name} is missing {len(missing)} distractor identities, first {missing[:3]}")
    return subset_of


def _class_id(identity: str, wiki_ids: set[int]) -> int:
    class_id = CLASS_OFFSET + int(identity)
    if class_id in wiki_ids:
        raise ValueError(f"Distractor class id {class_id} collides with Wikidata")
    return class_id


def _assign(identities: list[str], seed: int) -> tuple[set[str], set[str]]:
    order = list(identities)
    random.Random(seed).shuffle(order)
    val_count = len(order) // 2
    return set(order[:val_count]), set(order[val_count:])


def _stamp(
    split: dict,
    *,
    seed: int | None,
    historic_added: int,
    identities_added: int,
    dataset_id: str,
    image_root: str,
) -> None:
    split["dataset"] = {
        "dataset_id": dataset_id,
        "identity": {
            "num_classes": int(split["coverage_filter"]["included_num_classes"]) + identities_added,
            "num_images": len(split["images"]),
        },
        "image_root": image_root,
        "source_db_sha256": split["dataset"]["source_db_sha256"],
    }
    split["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    split["created_by"] = {
        "seed": seed,
        "tool": "experiments.datasets.wikidata_lostart_distractors.build_splits",
        "version": "0.1.0",
    }
    coverage = split["coverage_filter"]
    coverage["included_num_historic_images"] = int(coverage["included_num_historic_images"]) + historic_added
    coverage["input_num_historic_images"] = int(coverage["input_num_historic_images"]) + historic_added
    coverage["distractor_identities_excluded"] = sorted(EXCLUDED_IDENTITIES)
    coverage["distractor_identity_policy"] = "all_images_of_one_lost_art_id_stay_in_one_subset"


def _add_distractors(
    split: dict,
    groups: dict[str, list[str]],
    class_of: dict[str, int],
    subset_of: dict[str, str],
) -> tuple[int, int]:
    images = split["images"]
    subsets = split["subsets"]
    added_images = 0
    added_identities: set[str] = set()
    for identity, subset_name in subset_of.items():
        role = subsets[subset_name]["roles"].setdefault("historic", [])
        for name in groups[identity]:
            if name in images:
                raise ValueError(f"Distractor filename collides with Wikidata: {name}")
            images[name] = {"class_id": class_of[identity], "lost_art_id": identity}
            role.append(name)
            added_images += 1
        added_identities.add(identity)
        subsets[subset_name]["roles"]["historic"] = sorted(set(role))
    return added_images, len(added_identities)


def _check(split: dict, groups: dict[str, list[str]], subset_of: dict[str, str], out_dir: Path) -> None:
    from experiments.core.split_schema import load_split

    path = out_dir / "_check.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    try:
        load_split(path)
    finally:
        path.unlink()
    seen: dict[str, str] = {}
    for subset_name, subset in split["subsets"].items():
        for role, names in subset["roles"].items():
            for name in names:
                if name in seen:
                    raise ValueError(f"{name} is in {seen[name]} and {subset_name}/{role}")
                seen[name] = f"{subset_name}/{role}"
                if IDENTITY_RE.match(name) and _identity(name) in EXCLUDED_IDENTITIES:
                    raise ValueError(f"Excluded identity remained in the split: {name}")
    for identity, names in groups.items():
        if identity in EXCLUDED_IDENTITIES:
            continue
        places = {seen[name] for name in names}
        if places != {f"{subset_of[identity]}/historic"}:
            raise ValueError(f"Identity {identity} leaked across {sorted(places)}")


def build(
    *,
    wiki_split_dir: Path,
    out_dir: Path,
    image_root: str,
    dataset_id: str,
    names_path: Path | None = None,
    names_from_split: Path | None = None,
    reuse_from: Path | None = None,
) -> dict[str, object]:
    """Write test-only and the ten partitions.

    ``reuse_from`` copies each Lost Art identity's half from an existing
    distractor split directory instead of shuffling again.
    """
    if names_from_split is not None:
        groups = _groups_from_split(names_from_split)
    elif names_path is not None:
        groups = _load_groups(names_path)
    else:
        raise ValueError("names_path or names_from_split is required")
    kept = sorted((identity for identity in groups if identity not in EXCLUDED_IDENTITIES), key=int)
    if names_from_split is None:
        missing = EXCLUDED_IDENTITIES - set(groups)
        if missing:
            raise SystemExit(f"Excluded identities not on disk: {sorted(missing)}")
    wiki_probe = json.loads((wiki_split_dir / "test-only.json").read_text(encoding="utf-8"))
    wiki_ids = {int(record["class_id"]) for record in wiki_probe["images"].values()}
    class_of = {identity: _class_id(identity, wiki_ids) for identity in kept}
    source_sha = str(wiki_probe["dataset"]["source_db_sha256"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def prepare(name: str) -> dict:
        split = json.loads((wiki_split_dir / name).read_text(encoding="utf-8"))
        if split["dataset"]["source_db_sha256"] != source_sha:
            raise ValueError(f"{name} does not share the Wikidata source hash")
        return split

    excluded_images: list[str] = []
    if names_from_split is None:
        excluded_images = sorted(name for identity in EXCLUDED_IDENTITIES for name in groups[identity])
    summary: dict[str, object] = {
        "excluded_identities": sorted(EXCLUDED_IDENTITIES),
        "excluded_images": excluded_images,
        "distractor_identities": len(kept),
        "distractor_images": sum(len(groups[identity]) for identity in kept),
        "assignment": "copied" if reuse_from is not None else "shuffled",
        "reuse_from": str(reuse_from) if reuse_from is not None else None,
        "splits": {},
    }

    full = prepare("test-only.json")
    full_before = {role: set(names) for role, names in full["subsets"]["test"]["roles"].items()}
    full_assign = {identity: "test" for identity in kept}
    added_images, added_identities = _add_distractors(full, groups, class_of, full_assign)
    for role, names in full_before.items():
        if not names <= set(full["subsets"]["test"]["roles"][role]):
            raise ValueError("Full split dropped a Wikidata image")
    full["split_id"] = f"{dataset_id}_test_only_v1"
    _stamp(
        full,
        seed=None,
        historic_added=added_images,
        identities_added=added_identities,
        dataset_id=dataset_id,
        image_root=image_root,
    )
    _check(full, groups, full_assign, out_dir)
    (out_dir / "test-only.json").write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
    summary["splits"]["test-only"] = {
        "test_identities": len(kept),
        "val_identities": 0,
        "distractor_images": added_images,
    }

    for seed in SEEDS:
        name = f"class-disjoint-val-test-seed-{seed}.json"
        split = prepare(name)
        before = {
            subset_name: {role: set(names) for role, names in subset["roles"].items()}
            for subset_name, subset in split["subsets"].items()
        }
        if reuse_from is not None:
            subset_of = _subset_from_split(reuse_from / name, set(kept))
            val_ids = {identity for identity, subset_name in subset_of.items() if subset_name == "val"}
            test_ids = {identity for identity, subset_name in subset_of.items() if subset_name == "test"}
        else:
            val_ids, test_ids = _assign(kept, seed)
            subset_of = {identity: "val" if identity in val_ids else "test" for identity in kept}
        if val_ids & test_ids or val_ids | test_ids != set(kept):
            raise ValueError(f"Seed {seed} did not partition every identity once")
        added_images, added_identities = _add_distractors(split, groups, class_of, subset_of)
        for subset_name, roles in before.items():
            for role, names in roles.items():
                if not names <= set(split["subsets"][subset_name]["roles"][role]):
                    raise ValueError(f"Seed {seed} dropped Wikidata images from {subset_name}/{role}")
        split["split_id"] = f"{dataset_id}_class_disjoint_val_test_seed_{seed}_v1"
        _stamp(
            split,
            seed=seed,
            historic_added=added_images,
            identities_added=added_identities,
            dataset_id=dataset_id,
            image_root=image_root,
        )
        _check(split, groups, subset_of, out_dir)
        (out_dir / name).write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
        summary["splits"][name] = {
            "val_identities": len(val_ids),
            "test_identities": len(test_ids),
            "val_images": sum(len(groups[identity]) for identity in val_ids),
            "test_images": sum(len(groups[identity]) for identity in test_ids),
        }
        print(
            f"seed {seed}: val identities {len(val_ids)} images {summary['splits'][name]['val_images']}; "
            f"test identities {len(test_ids)} images {summary['splits'][name]['test_images']}"
        )

    manifest_bytes = json.dumps(summary, sort_keys=True).encode("utf-8")
    summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    (out_dir / "distractor_assignment.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"identities {len(kept)} images {summary['distractor_images']} "
        f"excluded {summary['excluded_images']}"
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Wikidata splits with Lost Art historic distractors")
    parser.add_argument("--wiki-split-dir", type=Path, default=WIKI_SPLIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--image-root", default=IMAGE_ROOT)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--names", type=Path, default=None, help="One Lost Art filename per line")
    parser.add_argument("--names-from-split", type=Path, default=None, help="Read LA filenames from this split")
    parser.add_argument("--reuse-from", type=Path, default=None, help="Copy identity halves from this split directory")
    args = parser.parse_args()
    names = args.names
    if names is None and args.names_from_split is None:
        names = NAMES_PATH
    build(
        wiki_split_dir=args.wiki_split_dir,
        out_dir=args.out_dir,
        image_root=args.image_root,
        dataset_id=args.dataset_id,
        names_path=names,
        names_from_split=args.names_from_split,
        reuse_from=args.reuse_from,
    )


if __name__ == "__main__":
    main()
