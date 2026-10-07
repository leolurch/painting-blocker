#!/usr/bin/env python3
"""Summarize rank_positions.json files into the numbers the paper reports.

Reads the files written by export_wikidata_rank_positions.py. For each model
on Wikidata 1.5 (wiki), Wikidata 1.5 with Lost Art distractors (wilo), and
the uncapped Wikidata 1.5 check, it reports:

- full catalog: pairs completeness and per-query recall at fixed k, and the
  smallest k that reaches a completeness target (top-k, chosen on the same
  catalog, so not a calibrated operating point);
- ten partitions: k calibrated on the calibration half for 95% and 99% pairs
  completeness, applied to the test half (pairs completeness, attainment with
  a 0.005 tolerance, candidates per query, reduction ratio);
- ten partitions: the same test half at the k calibrated by another model on
  the same partition (frozen backbone, DINOv3-7B), with per-partition wins;
- the share of test-half misses that fall on the paintings with most pairs.

Adapted models with three training seeds are reported as the seed mean and
sample standard deviation. Prints plain text; --json writes the summary.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/runs"
FIXED_K = (1, 5, 10, 20, 40, 80, 160)
TARGETS = (0.95, 0.99)
TOLERANCE = 0.005
FROZEN = {
    "DINOv3-7B-PP": "DINOv3-7B",
    "DINOv3-H+": "DINOv3-H+",
    "SigLIP2-PP": "SigLIP2",
    "Qwen3-VL Embedding 8B": "Qwen3",
    "Qwen3": "Qwen3",
    "OpenCLIP": "OpenCLIP",
    "CLIP": "CLIP",
    "ResNet-50": "ResNet-50",
}
FOLLOWUP_ROLE = {
    0: "full recipe", 1: "full recipe", 2: "full recipe",
    3: "without modern generated", 4: "without modern generated", 5: "without modern generated",
    6: "without archival", 7: "without archival", 8: "without archival",
    9: "without print", 10: "without print", 11: "without print",
    12: "without cropped record", 13: "without cropped record", 14: "without cropped record",
    15: "without framed photo", 16: "without framed photo", 17: "without framed photo",
}
CONTROLS = {
    "finetune_dinov3_vithplus_met_paint_head_only_v1": "control: head only",
    "finetune_dinov3_vithplus_met_paint_modern_only_v1": "control: modern views only",
    "finetune_dinov3_vithplus_met_paint_originals_aug_v1": "control: originals + generic augmentation",
}
SELECTED_HPLUS = "3212906212fe"
SIGLIP2_RUNS = {
    "2603007-4": "SigLIP2 adapted",
    "2597773-3": "SigLIP2 adapted (wave-1 pick)",
}


def catalog_of(task: dict) -> str:
    candidates = int(task["num_candidates"])
    if candidates > 5000:
        return "wilo"
    if candidates in (383,):
        return "wiki13u"
    return "wiki"


def label_of(doc: dict) -> tuple[str, int | None] | None:
    """Model label and training seed, or None for runs outside the paper."""
    run_dir = str(doc.get("run_dir") or "")
    name = str(doc.get("display_name") or "")
    parent = doc.get("parent_training") or {}
    parent_blob = " ".join(str(parent.get(key) or "") for key in ("run_id", "run_dir", "checkpoint_path"))
    blob = f"{run_dir} {parent_blob} {doc.get('model_dir')} {doc.get('model_id')}"
    if name in FROZEN and "finetune" not in run_dir:
        return FROZEN[name], None
    for root, label in CONTROLS.items():
        if run_dir.startswith(root) or root in parent_blob:
            seed = re.search(r"trainseed-(\d+)", f"{run_dir} {parent_blob}")
            if seed is None and parent.get("training_seed") is not None:
                return label, int(parent["training_seed"])
            return label, int(seed.group(1)) if seed else None
    # Wikidata 1.5 eval directories name the checkpoint in the run id. The
    # evaluation records no parent path unless both a configuration id and a
    # training seed were passed.
    control_run = re.search(r"wik15-(head|modern|origaug)-s(\d+)", run_dir)
    if control_run:
        label = {
            "head": "control: head only",
            "modern": "control: modern views only",
            "origaug": "control: originals + generic augmentation",
        }[control_run.group(1)]
        return label, int(control_run.group(2))
    holdout_run = re.search(r"wik15-hold-(\d+)-s(\d+)", run_dir)
    if holdout_run and int(holdout_run.group(1)) in FOLLOWUP_ROLE:
        index = int(holdout_run.group(1))
        role = FOLLOWUP_ROLE[index]
        label = "H+ adapted" if role == "full recipe" else f"H+ {role}"
        return label, int(holdout_run.group(2))
    follow = run_dir if run_dir.startswith("finetune_dinov3_vithplus_met_paint_winner_followup_v1") else ""
    if "winner_followup_v1" in parent_blob:
        follow = f"{follow} {parent_blob}"
    if follow:
        match = re.search(r"run-\d+-(\d+)-trainseed-(\d+)", follow)
        if match and int(match.group(1)) in FOLLOWUP_ROLE:
            index = int(match.group(1))
            role = FOLLOWUP_ROLE[index]
            label = "H+ adapted" if role == "full recipe" else f"H+ {role}"
            return label, int(match.group(2))
    for token, label in SIGLIP2_RUNS.items():
        if token in blob:
            return label, 42
    if "siglip2-w2-" in run_dir:
        return "SigLIP2 adapted", 42
    if SELECTED_HPLUS in blob or "2587228-12" in blob:
        return "H+ selected run", None
    return None


def load(runs: Path) -> dict:
    """data[catalog][label][(seed, kind, partition)] = task views."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    for path in sorted(runs.rglob("rank_positions.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        tag = label_of(doc)
        if tag is None:
            continue
        label, seed = tag
        test = doc["tasks"].get("test")
        if test is None:
            continue
        catalog = catalog_of(test)
        partition = doc.get("split_seed") if "calibration" in doc["tasks"] else None
        if partition is None and "calibration" in doc["tasks"]:
            match = re.search(r"seed-(\d+)$", str(doc.get("run_dir")))
            partition = int(match.group(1)) if match else None
        kind = "part" if "calibration" in doc["tasks"] else "full"
        key = (seed, kind, partition)
        data[catalog][label].setdefault(key, doc["tasks"])
    return data


def pc(view: dict, k: int) -> float:
    ranks = np.asarray(view["positive_rank"])
    return float((ranks <= k).sum() / view["num_positive_pairs"])


def macro_recall(view: dict, k: int) -> float:
    ranks = np.asarray(view["positive_rank"])
    queries = np.asarray(view["positive_query"])
    hits = np.bincount(queries[ranks <= k], minlength=view["num_queries"])
    return float(np.mean(hits / np.asarray(view["query_positives"])))


def smallest_k(view: dict, target: float) -> int:
    ranks = np.sort(np.asarray(view["positive_rank"]))
    needed = int(np.ceil(target * view["num_positive_pairs"] - 1e-9))
    return int(ranks[max(needed, 1) - 1])


def rr_at_k(view: dict, k: int) -> float:
    return 1.0 - view["num_queries"] * min(k, view["num_candidates"]) / view["possible_pairs"]


def misses_by_painting(view: dict, k: int) -> dict[int, int]:
    ranks = np.asarray(view["positive_rank"])
    queries = np.asarray(view["positive_query"])
    classes = np.asarray(view["query_class"])[queries]
    missed = classes[ranks > k]
    values, counts = np.unique(missed, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def seeds_of(entries: dict, kind: str) -> dict:
    by_seed: dict = defaultdict(dict)
    for (seed, entry_kind, partition), tasks in entries.items():
        if entry_kind == kind:
            by_seed[seed][partition] = tasks
    return by_seed


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def full_summary(entries: dict) -> dict:
    per_seed = []
    for seed, parts in seeds_of(entries, "full").items():
        view = parts[None]["test"]
        per_seed.append(
            {
                "pc": {k: pc(view, k) for k in FIXED_K},
                "recall": {k: macro_recall(view, k) for k in FIXED_K},
                "k_for": {t: smallest_k(view, t) for t in (0.90, 0.95, 0.97, 0.98, 0.99)},
                "candidates": view["num_candidates"],
            }
        )
    return {"seeds": len(per_seed), "rows": per_seed}


def partition_summary(entries: dict, donors: dict[str, dict]) -> dict:
    per_seed = []
    for seed, parts in seeds_of(entries, "part").items():
        rows = {}
        for target in TARGETS:
            pcs, ks, rrs, attained = [], [], [], 0
            for partition, tasks in parts.items():
                k = smallest_k(tasks["calibration"], target)
                value = pc(tasks["test"], k)
                pcs.append(value)
                ks.append(k)
                rrs.append(rr_at_k(tasks["test"], k))
                attained += value >= target - TOLERANCE - 1e-12
            rows[target] = {
                "pc": mean_sd(pcs), "k": mean_sd(ks), "k_range": (min(ks), max(ks)),
                "rr": mean_sd(rrs), "attained": attained, "n": len(pcs),
            }
        borrowed = {}
        for donor_label, donor_entries in donors.items():
            donor_parts = seeds_of(donor_entries, "part").get(None) or {}
            for target in TARGETS:
                own, other, wins, ties = [], [], 0, 0
                for partition, tasks in parts.items():
                    if partition not in donor_parts:
                        continue
                    k = smallest_k(donor_parts[partition]["calibration"], target)
                    mine = pc(tasks["test"], k)
                    theirs = pc(donor_parts[partition]["test"], k)
                    own.append(mine)
                    other.append(theirs)
                    wins += mine > theirs + 1e-12
                    ties += abs(mine - theirs) <= 1e-12
                if own:
                    borrowed[(donor_label, target)] = {
                        "pc": st.mean(own), "donor_pc": st.mean(other), "wins": wins, "ties": ties, "n": len(own),
                        "attained": sum(value >= target - TOLERANCE - 1e-12 for value in own),
                        "donor_attained": sum(value >= target - TOLERANCE - 1e-12 for value in other),
                    }
        per_seed.append({"seed": seed, "own": rows, "borrowed": borrowed})
    return {"seeds": len(per_seed), "rows": per_seed}


def _fmt(values: list[float], digits: int = 3) -> str:
    mean, sd = mean_sd(values)
    if len(values) > 1:
        return f"{mean:.{digits}f}±{sd:.{digits}f}"
    return f"{mean:.{digits}f}"


def report(data: dict) -> dict:
    out: dict = {}
    order = [
        "H+ adapted", "H+ selected run", "SigLIP2 adapted", "SigLIP2 adapted (wave-1 pick)",
        "DINOv3-7B", "DINOv3-H+", "SigLIP2", "Qwen3", "OpenCLIP", "CLIP", "ResNet-50",
        "control: head only", "control: modern views only", "control: originals + generic augmentation",
        "H+ without modern generated", "H+ without archival", "H+ without print",
        "H+ without cropped record", "H+ without framed photo",
    ]
    for catalog in ("wilo", "wiki", "wiki13u"):
        models = data.get(catalog) or {}
        if not models:
            continue
        print(f"\n######## {catalog}")
        print("-- full catalog: PC@k [per-query recall@k] | smallest k for PC 0.95 / 0.99")
        for label in order:
            if label not in models:
                continue
            summary = full_summary(models[label])
            if not summary["rows"]:
                continue
            rows = summary["rows"]
            pcs = " ".join(
                f"@{k}:{_fmt([row['pc'][k] for row in rows])}[{_fmt([row['recall'][k] for row in rows])}]"
                for k in (10, 20, 40, 80, 160)
            )
            kneed = " / ".join(_fmt([row["k_for"][t] for row in rows], 0) for t in (0.95, 0.99))
            print(f"  {label:42s} n={summary['seeds']} {pcs} | k {kneed}")
            out.setdefault(catalog, {}).setdefault(label, {})["full"] = summary
        print("-- partitions: own calibrated k on the calibration half, applied to the test half")
        backbone = {"H+": "DINOv3-H+", "SigLIP2": "SigLIP2"}
        for label in order:
            if label not in models:
                continue
            donors = {}
            for prefix, frozen in backbone.items():
                if label.startswith(prefix) and frozen in models:
                    donors[frozen] = models[frozen]
            if "DINOv3-7B" in models and label != "DINOv3-7B":
                donors["DINOv3-7B"] = models["DINOv3-7B"]
            summary = partition_summary(models[label], donors)
            if not summary["rows"]:
                continue
            rows = summary["rows"]
            text = []
            for target in TARGETS:
                pcs = [row["own"][target]["pc"][0] for row in rows]
                ks = [row["own"][target]["k"][0] for row in rows]
                rrs = [row["own"][target]["rr"][0] for row in rows]
                att = [row["own"][target]["attained"] for row in rows]
                lo = min(row["own"][target]["k_range"][0] for row in rows)
                hi = max(row["own"][target]["k_range"][1] for row in rows)
                text.append(
                    f"{int(target * 100)}%: PC {_fmt(pcs)} att {_fmt(att, 1)}/10 k {_fmt(ks, 0)} [{lo}-{hi}] RR {_fmt(rrs)}"
                )
            print(f"  {label:42s} n={summary['seeds']} | " + " | ".join(text))
            for (donor, target) in sorted({key for row in rows for key in row["borrowed"]}):
                cells = [row["borrowed"][(donor, target)] for row in rows if (donor, target) in row["borrowed"]]
                print(
                    f"      at {donor}'s calibrated k ({int(target * 100)}%): PC {_fmt([c['pc'] for c in cells])}"
                    f" vs {donor} {_fmt([c['donor_pc'] for c in cells])};"
                    f" higher in {_fmt([c['wins'] for c in cells], 1)}, tied {_fmt([c['ties'] for c in cells], 1)} of 10;"
                    f" attained {_fmt([c['attained'] for c in cells], 1)} vs {_fmt([c['donor_attained'] for c in cells], 1)}"
                )
            out.setdefault(catalog, {}).setdefault(label, {})["partitions"] = summary
    return out


def painting_report(data: dict) -> None:
    for catalog in ("wilo", "wiki"):
        models = data.get(catalog) or {}
        print(f"\n######## {catalog}: share of test-half misses at the own 99% k on the paintings with most pairs")
        for label in ("H+ selected run", "H+ adapted", "DINOv3-7B", "DINOv3-H+", "SigLIP2", "SigLIP2 adapted"):
            if label not in models:
                continue
            for seed, parts in sorted(seeds_of(models[label], "part").items(), key=lambda item: str(item[0])):
                total, top = 0, 0
                worst = defaultdict(int)
                for partition, tasks in parts.items():
                    k = smallest_k(tasks["calibration"], 0.99)
                    missed = misses_by_painting(tasks["test"], k)
                    total += sum(missed.values())
                    for painting, count in missed.items():
                        worst[painting] += count
                ranked = sorted(worst.values(), reverse=True)
                share3 = sum(ranked[:3]) / total if total else 0.0
                print(f"  {label:22s} seed {seed}: {total} misses over 10 test halves; top-3 paintings hold {share3:.2f}; worst {ranked[:5]}")
                break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    data = load(args.runs)
    summary = report(data)
    painting_report(data)
    if args.json is not None:
        args.json.write_text(json.dumps(summary, default=str, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
