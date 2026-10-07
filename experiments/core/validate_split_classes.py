"""Validate class-level separation in schema-v2 split JSON files.

The primary guard is that train/validation classes do not leak into the test
subset for class-disjoint finetuning/evaluation splits. This script reads only
the self-contained split JSON; it does not open the dataset DB.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

try:  # Allow both ``python -m ...`` and direct script execution from repo root.
    from .split_schema import load_split
except ImportError:  # pragma: no cover - exercised only for direct script usage.
    from experiments.core.split_schema import load_split

CRITICAL_CLASS_DISJOINT_PAIRS = (("train", "test"), ("val", "test"))


@dataclass(frozen=True)
class ClassOverlap:
    """Class overlap found between two split subsets."""

    left_subset: str
    right_subset: str
    class_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_subset": self.left_subset,
            "right_subset": self.right_subset,
            "num_overlapping_classes": len(self.class_ids),
            "class_ids": list(self.class_ids),
        }


@dataclass(frozen=True)
class SplitClassValidation:
    """Class-disjoint validation result for one split file."""

    split_id: str
    subset_class_counts: dict[str, int]
    subset_image_counts: dict[str, int]
    critical_overlaps: tuple[ClassOverlap, ...]
    all_pair_overlaps: tuple[ClassOverlap, ...]

    @property
    def ok(self) -> bool:
        return not self.critical_overlaps

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "split_id": self.split_id,
            "subset_class_counts": self.subset_class_counts,
            "subset_image_counts": self.subset_image_counts,
            "critical_checks": [overlap.as_dict() for overlap in self.critical_overlaps],
            "all_pair_overlaps": [overlap.as_dict() for overlap in self.all_pair_overlaps],
        }


def class_sets_by_subset(split: dict[str, Any]) -> tuple[dict[str, set[int]], dict[str, int]]:
    """Return class IDs and unique image counts for every subset in ``split``."""
    images = split["images"]
    classes_by_subset: dict[str, set[int]] = {}
    image_counts: dict[str, int] = {}
    for subset_name, subset in split["subsets"].items():
        subset_image_ids = {
            str(image_id)
            for role_ids in subset["roles"].values()
            for image_id in role_ids
        }
        image_counts[str(subset_name)] = len(subset_image_ids)
        classes_by_subset[str(subset_name)] = {
            int(images[image_id]["class_id"])
            for image_id in subset_image_ids
        }
    return classes_by_subset, image_counts


def validate_split_class_disjointness(
    split: dict[str, Any],
    critical_pairs: Iterable[tuple[str, str]] = CRITICAL_CLASS_DISJOINT_PAIRS,
) -> SplitClassValidation:
    """Validate that critical subset pairs have disjoint class IDs."""
    classes_by_subset, image_counts = class_sets_by_subset(split)
    critical_pair_list = list(critical_pairs)
    required_subsets = {name for pair in critical_pair_list for name in pair}
    missing = sorted(required_subsets - set(classes_by_subset))
    if missing:
        raise ValueError(f"Split is missing required subset(s): {', '.join(missing)}")

    all_overlaps = _overlaps_for_pairs(classes_by_subset, combinations(sorted(classes_by_subset), 2))
    critical_overlaps = _overlaps_for_pairs(classes_by_subset, critical_pair_list)
    return SplitClassValidation(
        split_id=str(split.get("split_id", "")),
        subset_class_counts={name: len(ids) for name, ids in sorted(classes_by_subset.items())},
        subset_image_counts={name: image_counts[name] for name in sorted(image_counts)},
        critical_overlaps=tuple(critical_overlaps),
        all_pair_overlaps=tuple(all_overlaps),
    )


def _overlaps_for_pairs(
    classes_by_subset: dict[str, set[int]], pairs: Iterable[tuple[str, str]]
) -> list[ClassOverlap]:
    overlaps: list[ClassOverlap] = []
    for left, right in pairs:
        shared = sorted(classes_by_subset[left] & classes_by_subset[right])
        if shared:
            overlaps.append(ClassOverlap(left, right, tuple(shared)))
    return overlaps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate that a schema-v2 split keeps train/val classes out of test."
    )
    parser.add_argument("split", type=Path, help="Path to a schema-v2 split JSON file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON report instead of text",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        split = load_split(args.split)
        result = validate_split_class_disjointness(split)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        _print_text_report(args.split, result)
    return 0 if result.ok else 1


def _print_text_report(split_path: Path, result: SplitClassValidation) -> None:
    print(f"Split: {split_path}")
    print(f"split_id: {result.split_id}")
    print("Subset counts:")
    for subset in sorted(result.subset_class_counts):
        print(
            f"  {subset}: {result.subset_class_counts[subset]} classes, "
            f"{result.subset_image_counts[subset]} images"
        )

    print("Critical class-disjoint checks:")
    if not result.critical_overlaps:
        print("  OK: classes(train) ∩ classes(test) = ∅")
        print("  OK: classes(val) ∩ classes(test) = ∅")
    else:
        for overlap in result.critical_overlaps:
            preview = list(overlap.class_ids[:20])
            suffix = "..." if len(overlap.class_ids) > len(preview) else ""
            print(
                f"  FAIL: classes({overlap.left_subset}) ∩ classes({overlap.right_subset}) "
                f"has {len(overlap.class_ids)} class(es): {preview}{suffix}"
            )

    critical_pair_keys = {frozenset(pair) for pair in CRITICAL_CLASS_DISJOINT_PAIRS}
    noncritical = [
        overlap for overlap in result.all_pair_overlaps
        if frozenset((overlap.left_subset, overlap.right_subset)) not in critical_pair_keys
    ]
    if noncritical:
        print("Other subset overlaps:")
        for overlap in noncritical:
            print(
                f"  classes({overlap.left_subset}) ∩ classes({overlap.right_subset}) "
                f"has {len(overlap.class_ids)} class(es)"
            )


if __name__ == "__main__":
    raise SystemExit(main())
