"""Write the three overlap-removed SYN partitions and the full Wikidata eval splits.

Stored subset_* rows in the overlap-removed SYN database are the original seed-42
assignment. This campaign repartitions that catalog with the hard-synth class
shuffle for seeds 42, 43, and 44, so the seed actually changes the classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.core.config_schema import load_dataset_config
from experiments.core.split_schema import (
    materialize_class_disjoint_role_split,
    materialize_hard_synth_split,
    write_split,
)


WIKIDATA_SPLIT_SEEDS = (6, 7, 9, 10, 21, 42, 67, 87, 1337, 4711)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn-dataset", type=Path, required=True)
    parser.add_argument("--syn-output-dir", type=Path, required=True)
    parser.add_argument("--wikidata-dataset", type=Path, required=True)
    parser.add_argument("--wikidata-output-dir", type=Path, required=True)
    args = parser.parse_args()
    syn = load_dataset_config(args.syn_dataset)
    args.syn_output_dir.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        split = materialize_hard_synth_split(
            syn,
            "hard_synth_class_disjoint_v1",
            seed,
            respect_stored_subsets=False,
        )
        path = args.syn_output_dir / f"hard_synth_class_disjoint_v1_seed_{seed}.json"
        write_split(path, split)
        print(f"syn seed {seed}: {split['split_id']} -> {path}")

    wikidata = load_dataset_config(args.wikidata_dataset)
    args.wikidata_output_dir.mkdir(parents=True, exist_ok=True)
    full = materialize_class_disjoint_role_split(
        wikidata,
        "class_disjoint_role_v1",
        seed=42,
        train_share=0.0,
        validation_share=0.0,
    )
    write_split(args.wikidata_output_dir / "test-only.json", full)
    print(f"wikidata full: {full['split_id']}")
    for seed in WIKIDATA_SPLIT_SEEDS:
        split = materialize_class_disjoint_role_split(
            wikidata,
            "class_disjoint_role_v1",
            seed=seed,
            train_share=0.0,
            validation_share=0.5,
        )
        path = args.wikidata_output_dir / f"class-disjoint-val-test-seed-{seed}.json"
        write_split(path, split)
        print(f"wikidata seed {seed}: {split['split_id']}")
    reference = materialize_class_disjoint_role_split(
        wikidata,
        "class_disjoint_role_v1",
        seed=42,
        train_share=0.0,
        validation_share=0.5,
    )
    write_split(args.wikidata_output_dir / "class-disjoint-val-test.json", reference)


if __name__ == "__main__":
    main()
