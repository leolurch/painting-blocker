#!/usr/bin/env python3
"""Write every number and plot series that the conference paper reports.

Reads the rank_positions.json files written by
tooling/export_wikidata_rank_positions.py, and the eval.json and curves.json
next to each partition file (the candidate count of the test half at every
similarity threshold). Operating points are certified with conformal risk
control in risk_control.py; differences at fixed budgets get a paired
bootstrap over the paintings. Writes, into
writeups/conference-paper/figures/generated/paper/:

- numbers.tex: one \\pnset{key}{value} line per reported number, and
  \\provisionaltrue or \\provisionalfalse;
- curve.csv: pairs completeness against the candidate budget k on the full
  catalog, for the budget figure;
- concentration.csv: cumulative share of true pairs over the paintings,
  largest first.

The paper prints every value through \\pn{key}. The written file marks the
numbers as final. Pass --draft to highlight them as provisional instead.

Usage:
  python writeups/tooling/build_paper_numbers.py \\
      --split-prefix wikidata1.5-lostart-distractors \\
      --companion-prefix wikidata-1.5
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tooling"))

from summarize_rank_positions import label_of, macro_recall, pc, smallest_k  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_control import (  # noqa: E402
    conformal_feasible,
    conformal_paintings_needed,
    first_safe,
)

RUNS = ROOT / "experiments/runs"
OUT = ROOT / "writeups/conference-paper/figures/generated/paper"
# writeups/references/spsg_server_throughput_summary.md: deployed matcher and scraper.
LIGHTGLUE_PAIRS_PER_SECOND = 41.05
LISTING_IMAGES_PER_DAY = 4899.1
# Lost Art reports added as distractors to the test set.
LOSTART_IDENTITIES = 19_771
FIXED_K = (1, 5, 10, 20, 40, 80, 160)
TARGETS = {95: 0.95, 99: 0.99}
BOOTSTRAP_RESAMPLES = 2000
# e: conformal risk control, a guarantee in expectation.
RULES = ("e",)
# Paired painting bootstraps on the full test set: (model, reference, measures).
PAIRED = (
    ("hplus", "dhp", ("idpc10", "idpc20", "idpc40", "idpc160", "pc10", "pc20", "pc40", "pc160", "rec20", "idk95", "k95")),
    ("hplus", "dsev", ("idpc10", "idpc20", "idpc40", "idpc160", "pc20", "pc160")),
    ("ctlhead", "dhp", ("idpc20", "pc20")),
    ("ctlmodern", "dhp", ("idpc20", "pc20")),
    ("ctlorig", "dhp", ("idpc20", "pc20")),
    ("hplus", "ctlhead", ("idpc20", "pc20")),
    ("hplus", "ctlmodern", ("idpc20", "pc20", "idk95", "k95")),
    ("hplus", "ctlorig", ("idpc20", "pc20")),
    ("sigad", "sig", ("idpc20", "pc20")),
    ("sigad", "dhp", ("idpc20",)),
    ("clipad", "clip", ("idpc20", "idpc40")),
    ("oclipad", "oclip", ("idpc20", "idpc40")),
    ("rnad", "rn", ("idpc20", "idpc40")),
)
COMPANION_PAIRED = ("idpc20", "pc20", "pc40")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 10: "ten"}

SLUGS = {
    "hplus": ["H+ adapted"],
    "sigad": ["SigLIP2 adapted", "SigLIP2 adapted (wave-1 pick)"],
    "dsev": ["DINOv3-7B"],
    "dhp": ["DINOv3-H+"],
    "sig": ["SigLIP2"],
    "qwen": ["Qwen3"],
    "oclip": ["OpenCLIP"],
    "clip": ["CLIP"],
    "rn": ["ResNet-50"],
    "ctlhead": ["control: head only"],
    "ctlmodern": ["control: modern views only"],
    "ctlorig": ["control: originals + generic augmentation"],
    "clipad": ["CLIP adapted"],
    "oclipad": ["OpenCLIP adapted"],
    "rnad": ["ResNet-50 adapted"],
}
HOLDOUTS = (
    "H+ without modern generated",
    "H+ without archival",
    "H+ without print",
    "H+ without cropped record",
    "H+ without framed photo",
)
# Paired intervals, same-budget tests, and the curve lead use frozen DINOv3-H+.
REFERENCE = "dhp"
CURVE_MODELS = ("hplus", "dsev", "dhp", "sigad", "sig", "oclip")
# Series drawn in the budget figure and its smallest budget; the y limit follows from them.
CURVE_PLOTTED = ("hplusidlo", "dsevid", "dhpid", "sigadid", "sigid")
CURVE_KMIN = 5
CURVE_BELOW = ("dsevid", "sigadid", "sigid")
TRADEOFF_MODELS = ("hplus", "dsev", "dhp")
# Lower y limit of the tradeoff figure; the curves below it are clipped.
TRADEOFF_YMIN = 0.85
# The figure marks where the better rule of the first model first reaches the better rule of the second.
TRADEOFF_CROSSING = ("dsev", "hplus")
TRADEOFF_RULES = {"th": "global threshold", "topk": "top-$k$"}
MAIN_MODELS = ("hplus", "sigad", "dsev", "dhp", "sig", "oclip", "qwen", "clip", "rn")
TRADEOFF_FROZEN = ("dsev", "dhp")
# Rows and columns of the result tables, for best.{table}.{key} (\pnb in main.tex). Rows are
# (model, frozen backbone) and must list the rows of the LaTeX table; columns say whether the
# lowest or highest value wins. Values compare as printed, so equal printed values all win.
BEST_TABLES = {
    "main": (
        (("rn", ""), ("clip", ""), ("qwen", ""), ("sig", ""), ("oclip", ""), ("dhp", ""), ("dsev", ""), ("hplus", "")),
        (("{m}.ek95.k", min), ("{m}.ek95.pc", max), ("{m}.et95.cpq", min), ("{m}.et95.pc", max),
         ("{m}.idpc10", max), ("{m}.idpc20", max), ("{m}.idpc40", max), ("{m}.idpc80", max), ("{m}.idpc160", max),
         ("{m}.pc20", max), ("{m}.pc160", max)),
    ),
    "controls": (
        (("hplus", ""), ("ctlhead", ""), ("ctlorig", ""), ("ctlmodern", "")),
        (("{m}.idpc20", max), ("{m}.ek95.k", min), ("{m}.et95.cpq", min),
         ("vsdhp.{m}.idpc20", max), ("vsdhp.{m}.ek95.k", min), ("vsdhp.{m}.et95.cpq", min)),
    ),
    "transfer": (
        (("hplus", "dhp"), ("sigad", "sig"), ("oclipad", "oclip"), ("clipad", "clip"), ("rnad", "rn")),
        (("{m}.idpc20", max), ("gain.{f}.idpc20.signed", max), ("{m}.ek95.k", min), ("gain.{f}.ek95.k", min),
         ("{m}.et95.cpq", min), ("gain.{f}.et95.cpq", min)),
    ),
}


HPLUS_SEED_RUNS = re.compile(r"winner_followup_v1/.*run-(2587822-0|2587823-1|2587823-2)-trainseed-(\d+)")
SIGLIP2_RECIPE = "r-4__la-4__lb-last20__mh-frozen__llr-3e-4__hd-1024__hlr-1e-4__hdo-0"
# Transfer of the recipe: the setting each backbone selected on the Met model-selection half
# (writeups/data-memory.md, "Recipe transfer to other backbones").
TRANSFER_RECIPES = {
    "CLIP adapted": ("finetune_transfer_clip_met_paint_v1", "__lb-all__llr-3e-4__"),
    "OpenCLIP adapted": ("finetune_transfer_openclip_met_paint_v1", "__lb-last24__llr-1e-4__"),
    "ResNet-50 adapted": ("finetune_transfer_resnet50_met_paint_v1", "__lb-last8__llr-1e-4__"),
}


def split_prefix_of(doc: dict) -> str:
    return str(doc.get("split_id") or "")


def paper_label(doc: dict) -> tuple[str, int | None] | None:
    """label_of, plus the selected recipes identified by their training checkpoint."""
    parent = doc.get("parent_training") or {}
    blob = " ".join(str(parent.get(key) or "") for key in ("checkpoint_path", "run_dir", "run_id"))
    seed = parent.get("training_seed")
    match = HPLUS_SEED_RUNS.search(blob)
    if match:
        return "H+ adapted", int(match.group(2))
    if "siglip2" in blob and SIGLIP2_RECIPE in blob:
        found = re.search(r"trainseed-(\d+)", blob)
        return "SigLIP2 adapted", int(seed if seed is not None else found.group(1) if found else 42)
    for label, (root, setting) in TRANSFER_RECIPES.items():
        if root in blob and setting in blob:
            found = re.search(r"trainseed-(\d+)", blob)
            return label, int(seed if seed is not None else found.group(1))
    tag = label_of(doc)
    if tag is not None:
        return tag
    return None


def _merge_missing(into: dict, extra: dict) -> None:
    """Fill labels and seeds that ``into`` does not already have."""
    for label, seeds in extra.items():
        for seed, entry in seeds.items():
            dest = into[label][seed]
            if dest["full"] is None and entry["full"] is not None:
                dest["full"] = entry["full"]
            for partition, value in entry["parts"].items():
                dest["parts"].setdefault(partition, value)
            for partition, directory in entry["dirs"].items():
                dest["dirs"].setdefault(partition, directory)


def fill_run_metadata(doc: dict, path: Path) -> None:
    """Fill the split and model fields that an export left empty from the run's run.json.

    An export that ran while run.json was still being written has no split_id, and the
    split filter would drop its partition.
    """
    if doc.get("split_id") is not None:
        return
    run_path = path.parents[2] / "run.json"
    if not run_path.is_file():
        return
    run = json.loads(run_path.read_text(encoding="utf-8"))
    model = (run.get("model_runs") or [{}])[0]
    split = run.get("split") or {}
    doc.update(
        display_name=doc.get("display_name") or model.get("display_name"),
        model_id=doc.get("model_id") or model.get("model_id"),
        parent_training=doc.get("parent_training") or run.get("parent_training"),
        split_id=split.get("split_id"),
        split_seed=split.get("split_seed"),
    )


def load(prefix: str, *, test_task: str = "test", calibration_task: str = "calibration") -> dict:
    """models[label][seed] = {"full": tasks, "parts": {partition: (tasks, thresholds)}}."""
    models: dict = defaultdict(lambda: defaultdict(lambda: {"full": None, "parts": {}, "dirs": {}}))
    for path in sorted(RUNS.rglob("rank_positions.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        fill_run_metadata(doc, path)
        split_id = split_prefix_of(doc)
        if not split_id.startswith(prefix + "_") and split_id != prefix:
            continue
        tag = paper_label(doc)
        if tag is None:
            continue
        label, seed = tag
        stored = doc["tasks"]
        if test_task not in stored:
            continue
        tasks = {"test": stored[test_task]}
        if calibration_task in stored:
            tasks["calibration"] = stored[calibration_task]
        entry = models[label][seed]
        if "calibration" in tasks:
            partition = doc.get("split_seed")
            if partition in entry["parts"]:
                continue
            entry["parts"][partition] = (tasks, thresholds_of(path.parent / "eval.json"))
            entry["dirs"][partition] = path.parent
        elif entry["full"] is None:
            entry["full"] = tasks
    return models


def thresholds_of(eval_path: Path) -> dict:
    if not eval_path.is_file():
        return {}
    doc = json.loads(eval_path.read_text(encoding="utf-8"))
    out = {}
    for task in doc.get("tasks", []):
        for row in task.get("calibrated_threshold_metrics") or []:
            target = round(float(row["calibration_target_pair_completeness"]), 2)
            out[target] = {
                "pc": float(row["pair_completeness"]),
                "rr": float(row["reduction_ratio"]),
                "cpq": float(row["candidate_pairs"]) / float(task["num_queries"]),
            }
    return out


def resolve(models: dict) -> dict:
    found = {}
    for slug, labels in SLUGS.items():
        for label in labels:
            if label in models:
                found[slug] = models[label]
                break
    return found


def seed_stats(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return float("nan"), float("nan"), 0
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0), len(values)


class Numbers:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        self.values[key] = value

    def num(self, key: str, values: list[float], digits: int = 3) -> None:
        mean, sd, n = seed_stats(values)
        if n == 0:
            return
        self.put(key, f"{mean:.{digits}f}")
        if n > 1:
            self.put(key + ".sd", f"{sd:.{digits}f}")
            self.put(key + ".min", f"{min(values):.{digits}f}")
            self.put(key + ".max", f"{max(values):.{digits}f}")

    def count(self, key: str, values: list[float]) -> None:
        mean, sd, n = seed_stats(values)
        if n == 0:
            return
        self.put(key, grouped(round(mean)))
        if n > 1:
            self.put(key + ".sd", grouped(round(sd)))
            self.put(key + ".min", grouped(round(min(values))))
            self.put(key + ".max", grouped(round(max(values))))

    def tex(self, final: bool) -> str:
        lines = [
            "% Generated by writeups/tooling/build_paper_numbers.py. Do not edit by hand.",
            "\\provisionalfalse" if final else "\\provisionaltrue",
        ]
        for key in sorted(self.values):
            lines.append(f"\\pnset{{{key}}}{{{self.values[key]}}}")
        return "\n".join(lines) + "\n"


def grouped(value: float | int) -> str:
    text = f"{int(value):,}"
    return text.replace(",", "{,}")


def identity_weights(view: dict) -> np.ndarray:
    """Weight of each true pair so that every painting sums to 1 / #paintings."""
    classes = np.asarray(view["query_class"])[np.asarray(view["positive_query"])]
    _, inverse, sizes = np.unique(classes, return_inverse=True, return_counts=True)
    return 1.0 / (len(sizes) * sizes[inverse])


def identity_pc(view: dict, k: int) -> float:
    """Mean over paintings of the share of their true pairs within the top k."""
    ranks = np.asarray(view["positive_rank"])
    return float(identity_weights(view)[ranks <= k].sum())


def smallest_k_identity(view: dict, target: float) -> int:
    ranks = np.asarray(view["positive_rank"])
    order = np.argsort(ranks, kind="stable")
    reached = np.cumsum(identity_weights(view)[order])
    index = min(int(np.searchsorted(reached, target - 1e-9)), len(order) - 1)
    return int(ranks[order][index])


def pair_paintings(view: dict) -> np.ndarray:
    return np.asarray(view["query_class"])[np.asarray(view["positive_query"])]


def painting_hits(view: dict, k: int, paintings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True pairs within the top k, and all true pairs, per painting."""
    index = np.searchsorted(paintings, pair_paintings(view))
    hit = (np.asarray(view["positive_rank"]) <= k).astype(float)
    return np.bincount(index, weights=hit, minlength=len(paintings)), np.bincount(index, minlength=len(paintings)).astype(float)


def fixed_k_numbers(numbers: Numbers, slug: str, seeds: dict) -> None:
    fulls = [entry["full"] for entry in seeds.values() if entry["full"] is not None]
    if not fulls:
        return
    views = [tasks["test"] for tasks in fulls]
    for k in FIXED_K:
        numbers.num(f"{slug}.pc{k}", [pc(view, k) for view in views])
        numbers.num(f"{slug}.rec{k}", [macro_recall(view, k) for view in views])
        numbers.num(f"{slug}.idpc{k}", [identity_pc(view, k) for view in views])
    for name, target in TARGETS.items():
        numbers.count(f"{slug}.k{name}", [smallest_k(view, target) for view in views])
        numbers.count(f"{slug}.idk{name}", [smallest_k_identity(view, target) for view in views])
    numbers.put(f"{slug}.seeds", str(len(views)))


def painting_losses(view: dict, kept: np.ndarray) -> np.ndarray:
    """Loss of every painting (columns, by painting id) at every operating point (rows).

    kept[i, j] says whether true pair j is a candidate at operating point i.
    """
    _, index = np.unique(pair_paintings(view), return_inverse=True)
    onehot = np.zeros((len(index), int(index.max()) + 1))
    onehot[np.arange(len(index)), index] = 1.0
    return 1.0 - (kept.astype(float) @ onehot) / onehot.sum(axis=0)


def kept_shares(view: dict, kept: np.ndarray) -> tuple[float, float]:
    """PC_id and pooled PC of a candidate set, given which true pairs it keeps."""
    return float(identity_weights(view)[kept].sum()), float(kept.mean())


_CURVES: dict[Path, tuple[np.ndarray, np.ndarray] | None] = {}


def test_curve(model_dir: Path, view: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Thresholds and test-half candidate pairs of the threshold curve stored by the evaluation."""
    if model_dir not in _CURVES:
        _CURVES[model_dir] = None
        eval_path, curves_path = model_dir / "eval.json", model_dir / "curves.json"
        if eval_path.is_file() and curves_path.is_file():
            tasks = json.loads(eval_path.read_text(encoding="utf-8")).get("tasks") or []
            task_id = next((row.get("task_id") for row in tasks if row.get("query_ids_hash") == view.get("query_ids_hash")), None)
            curves = json.loads(curves_path.read_text(encoding="utf-8")).get("tasks") or []
            curve = next((row for row in curves if row.get("task_id") == task_id), None)
            points = (curve or {}).get("precision_recall") or []
            if points:
                _CURVES[model_dir] = (
                    np.asarray([row["threshold"] for row in points], dtype=np.float64),
                    np.asarray([row["candidate_pairs"] for row in points], dtype=np.float64),
                )
    return _CURVES[model_dir]


def certify(losses: np.ndarray, alpha: float) -> int | None:
    """First operating point from which conformal risk control holds at every larger candidate set."""
    return first_safe(conformal_feasible(losses, alpha))


_POINTS: dict[Path, dict] = {}


def operating_points(tasks: dict, model_dir: Path) -> dict:
    """Certified budget and threshold of one partition, and what they keep on its test half.

    Keys are ("e", kind, target): conformal risk control, kind "k" for top-k and "t" for the
    threshold. None means that no operating point is certified. Threshold keys are absent without
    true-pair similarities.
    """
    if model_dir in _POINTS:
        return _POINTS[model_dir]
    calibration, test = tasks["calibration"], tasks["test"]
    ranks = np.asarray(calibration["positive_rank"])
    budgets = np.unique(ranks)
    budget_losses = painting_losses(calibration, ranks[None, :] <= budgets[:, None])
    test_ranks = np.asarray(test["positive_rank"])
    curve = None
    if "positive_score" in calibration and "positive_score" in test:
        curve = test_curve(model_dir, test)
    if curve is not None:
        thresholds, candidate_pairs = curve[0][::-1], curve[1][::-1]
        scores = np.asarray(calibration["positive_score"], dtype=np.float32)
        threshold_losses = painting_losses(calibration, scores[None, :] >= thresholds[:, None])
        test_scores = np.asarray(test["positive_score"], dtype=np.float32)
    points: dict = {}
    for name, target in TARGETS.items():
        for rule in RULES:
            index = certify(budget_losses, 1 - target)
            if index is None:
                points[rule, "k", name] = None
            else:
                k = int(budgets[index])
                pcid, pooled = kept_shares(test, test_ranks <= k)
                points[rule, "k", name] = {"k": k, "cand": float(min(k, test["num_candidates"])), "pc": pcid, "pool": pooled}
            if curve is None:
                continue
            index = certify(threshold_losses, 1 - target)
            if index is None:
                points[rule, "t", name] = None
            else:
                pcid, pooled = kept_shares(test, test_scores >= thresholds[index])
                points[rule, "t", name] = {
                    "tau": float(thresholds[index]),
                    "cand": float(candidate_pairs[index] / test["num_queries"]),
                    "pc": pcid,
                    "pool": pooled,
                }
    _POINTS[model_dir] = points
    return points


def guaranteed_numbers(numbers: Numbers, slug: str, seeds: dict, reference: dict | None) -> None:
    """Certified operating points on every partition, as means over partitions and then over seeds.

    {slug}.{rule}{kind}{target}: .k (top-k) or .cpq (threshold) candidates per query on the test
    half, .pc and .worst its mean and lowest PC_id, .met the test halves with PC_id at or above the
    target, .pool its mean pooled PC, .feas the partitions with a certified operating point out of .n.
    {slug}.safe{target}: the smallest k that reaches the target on every test half.
    {slug}.sb95: PC_id on the test half at the conformal 95% budget of the reference model.
    """
    per_seed: dict[str, list[float]] = defaultdict(list)
    # Threshold candidates and completeness on each calibration split, averaged over seeds.
    split_threshold: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"cpq": [], "pc": []})
    for entry in seeds.values():
        parts = entry["parts"]
        if not parts:
            continue
        fitted = {partition: operating_points(tasks, entry["dirs"][partition]) for partition, (tasks, _) in parts.items()}
        for name, target in TARGETS.items():
            per_seed[f"safe{name}"].append(max(smallest_k_identity(tasks["test"], target) for tasks, _ in parts.values()))
            for rule in RULES:
                for kind in ("k", "t"):
                    rows = [points[rule, kind, name] for points in fitted.values() if (rule, kind, name) in points]
                    if not rows:
                        continue
                    base = f"{rule}{kind}{name}"
                    certified = [row for row in rows if row is not None]
                    per_seed[f"{base}.feas"].append(len(certified))
                    per_seed[f"{base}.n"].append(len(rows))
                    if not certified:
                        continue
                    per_seed[f"{base}.{'k' if kind == 'k' else 'cpq'}"].append(st.mean(row["cand"] for row in certified))
                    per_seed[f"{base}.pc"].append(st.mean(row["pc"] for row in certified))
                    if rule == "e" and kind == "t" and name == 95:
                        for partition, points in fitted.items():
                            row = points.get((rule, kind, name))
                            if row is None:
                                continue
                            split_threshold[str(partition)]["cpq"].append(row["cand"])
                            split_threshold[str(partition)]["pc"].append(row["pc"])
                    per_seed[f"{base}.worst"].append(min(row["pc"] for row in certified))
                    per_seed[f"{base}.met"].append(sum(row["pc"] >= target - 1e-12 for row in certified))
                    per_seed[f"{base}.pool"].append(st.mean(row["pool"] for row in certified))
        if reference is None:
            continue
        donor = next(iter(reference.values()))
        mine, theirs, mine_pool, theirs_pool, wins = [], [], [], [], 0
        for partition, (tasks, _) in parts.items():
            if partition not in donor["parts"]:
                continue
            donor_tasks = donor["parts"][partition][0]
            point = operating_points(donor_tasks, donor["dirs"][partition]).get(("e", "k", 95))
            if point is None:
                continue
            a = kept_shares(tasks["test"], np.asarray(tasks["test"]["positive_rank"]) <= point["k"])
            b = kept_shares(donor_tasks["test"], np.asarray(donor_tasks["test"]["positive_rank"]) <= point["k"])
            mine.append(a[0])
            theirs.append(b[0])
            mine_pool.append(a[1])
            theirs_pool.append(b[1])
            wins += a[0] > b[0] + 1e-12
        if mine:
            per_seed["sb95.pc"].append(st.mean(mine))
            per_seed["sb95.donor"].append(st.mean(theirs))
            per_seed["sb95.pool"].append(st.mean(mine_pool))
            per_seed["sb95.donorpool"].append(st.mean(theirs_pool))
            per_seed["sb95.wins"].append(wins)
            per_seed["sb95.n"].append(len(mine))
    for field, digits in (("cpq", 0), ("pc", 3)):
        per_split = [st.mean(fields[field]) for fields in split_threshold.values() if fields[field]]
        if len(per_split) > 1:
            sd = st.stdev(per_split)
            text = grouped(round(sd)) if digits == 0 else f"{sd:.{digits}f}"
            numbers.put(f"{slug}.et95.{field}.splitsd", text)
    for key, values in per_seed.items():
        if key.endswith((".feas", ".n", ".met", ".wins")):
            numbers.num(f"{slug}.{key}", values, 0 if len(set(values)) == 1 else 1)
        elif key.endswith((".k", ".cpq")) or key.startswith("safe"):
            numbers.count(f"{slug}.{key}", values)
        else:
            numbers.num(f"{slug}.{key}", values)


def dataset_numbers(numbers: Numbers, reference: dict, prefix: str = "data") -> np.ndarray | None:
    entry = next(iter(reference.values()))
    if entry["full"] is not None:
        view = entry["full"]["test"]
        numbers.put(f"{prefix}.Q", grouped(view["num_queries"]))
        numbers.put(f"{prefix}.T", grouped(view["num_candidates"]))
        numbers.put(f"{prefix}.M", grouped(view["num_positive_pairs"]))
        numbers.put(f"{prefix}.pairs", grouped(view["possible_pairs"]))
        numbers.put(f"{prefix}.images", grouped(view["num_queries"] + view["num_candidates"]))
        if prefix == "data":
            numbers.put("plot.T", str(int(view["num_candidates"])))
    parts = entry["parts"]
    if parts and prefix == "data":
        for half in ("calibration", "test"):
            numbers.count(f"part.{half}.Q", [t[half]["num_queries"] for t, _ in parts.values()])
            numbers.count(f"part.{half}.T", [t[half]["num_candidates"] for t, _ in parts.values()])
            numbers.num(f"part.{half}.M", [t[half]["num_positive_pairs"] for t, _ in parts.values()], 1)
            numbers.num(f"part.{half}.P", [len(np.unique(pair_paintings(t[half]))) for t, _ in parts.values()], 0)
            numbers.put(f"part.{half}.Q.sdtwo", f"{st.stdev([t[half]['num_queries'] for t, _ in parts.values()]):.2f}")
            numbers.put(f"part.{half}.T.sdtwo", f"{st.stdev([t[half]['num_candidates'] for t, _ in parts.values()]):.2f}")
        numbers.put("part.test.T.int", grouped(round(st.mean(t["test"]["num_candidates"] for t, _ in parts.values()))))
        numbers.put("part.n", str(len(parts)))
    if entry["full"] is None:
        return None
    view = entry["full"]["test"]
    classes = np.asarray(view["query_class"])[np.asarray(view["positive_query"])]
    _, counts = np.unique(classes, return_counts=True)
    counts = np.sort(counts)[::-1]
    total = counts.sum()
    if prefix == "data":
        numbers.put("conc.largest", grouped(counts[0]))
        numbers.put("conc.largest.share", f"{100 * counts[0] / total:.1f}")
        numbers.put("conc.top10.share", f"{100 * counts[:10].sum() / total:.0f}")
        numbers.put("conc.paintings", grouped(len(counts)))
        numbers.put("plot.paintings", str(len(counts)))
        half = int(np.searchsorted(np.cumsum(counts), 0.5 * total)) + 1
        numbers.put("conc.half.paintings", str(half))
        numbers.put("conc.median", f"{np.median(counts):.0f}")
        numbers.put("conc.neff", f"{total ** 2 / float(np.sum(counts.astype(float) ** 2)):.0f}")
        # A pooled loss per painting, misses over the mean pair count, is bounded only by this ratio.
        cap = counts[0] / counts.mean()
        numbers.put("conc.cap", f"{cap:.0f}")
        if parts:
            calibration = st.mean(len(np.unique(pair_paintings(t["calibration"]))) for t, _ in parts.values())
            numbers.put("conc.cap.crc", f"{cap / (calibration + 1):.2f}")
            numbers.put("crc.corr", f"{1 / (round(calibration) + 1):.4f}")
    return counts


def axis_floor(values: np.ndarray, step: float = 0.05) -> str:
    return f"{np.floor(float(np.min(values)) / step + 1e-9) * step:.2f}"


def run_end(grid: np.ndarray, holds: np.ndarray, start: int = 0) -> tuple[int, int | None]:
    """Last grid value of the run where `holds` is true from `start`, and the first one after it."""
    behind = np.flatnonzero(~holds[start:])
    if behind.size == 0:
        return int(grid[-1]), None
    end = start + int(behind[0])
    return int(grid[max(end - 1, start)]), int(grid[end])


def curve_numbers(numbers: Numbers, grid: np.ndarray, columns: dict, catalog_size: int) -> None:
    start = int(np.searchsorted(grid, CURVE_KMIN))
    numbers.put("plot.curve.kmin", str(CURVE_KMIN))
    numbers.put("plot.curve.kmax", str(catalog_size))
    plotted = [columns[name][start:] for name in CURVE_PLOTTED if name in columns]
    if plotted:
        ymin = axis_floor(np.concatenate(plotted))
        numbers.put("plot.curve.ymin", ymin)
        numbers.put("plot.curve.yticks", ",".join(f"{v:.2f}" for v in np.arange(float(ymin), 0.951, 0.05)))
    ref = REFERENCE + "id"
    if "hplusid" in columns and ref in columns:
        diff = columns["hplusid"] - columns[ref]
        shown = diff[start:]
        numbers.put("curve.lead.min", f"{100 * float(np.min(shown)):.2f}")
        numbers.put("curve.lead.max", f"{100 * float(np.max(np.abs(shown))):.1f}")
        behind_from = np.flip(np.logical_and.accumulate(np.flip(diff < 0)))
        if behind_from.any() and int(np.argmax(behind_from)) > start:
            tail = int(np.argmax(behind_from))
            numbers.put("curve.lead.k", grouped(int(grid[tail - 1])))
            numbers.put("curve.lead.behind", grouped(int(grid[tail])))
        below = [columns[name] for name in CURVE_BELOW if name in columns]
        if below:
            pair_low = np.minimum(columns["hplusid"], columns[ref])
            last, _ = run_end(grid, pair_low > np.max(np.vstack(below), axis=0), start)
            numbers.put("curve.above.k", grouped(last))
    reach = {}
    for name in ("hplusid", "dsevid", "dhpid", "sigadid", "sigid"):
        if name in columns and (columns[name] >= 0.99).any():
            reach[name] = int(grid[int(np.argmax(columns[name] >= 0.99))])
    if reach:
        numbers.put("curve.k99.min", grouped(min(reach.values())))
        numbers.put("curve.k99.max", grouped(max(reach.values())))
        numbers.put("curve.k99.first", {"hplusid": "the adapted H+ model", "dsevid": "DINOv3-7B", "dhpid": "DINOv3-H+",
                                        "sigadid": "the adapted SigLIP2 model", "sigid": "SigLIP2"}[min(reach, key=reach.get)])


def write_curve(found: dict, path: Path, catalog_size: int, numbers: Numbers) -> None:
    grid = np.unique(np.round(np.logspace(0, np.log10(catalog_size), 160)).astype(int))
    columns: dict[str, np.ndarray] = {"k": grid}
    for slug in CURVE_MODELS:
        if slug not in found:
            continue
        rows, id_rows = [], []
        for entry in found[slug].values():
            if entry["full"] is None:
                continue
            view = entry["full"]["test"]
            ranks = np.asarray(view["positive_rank"])
            order = np.argsort(ranks, kind="stable")
            rows.append(np.searchsorted(ranks[order], grid, side="right") / view["num_positive_pairs"])
            reached = np.concatenate([[0.0], np.cumsum(identity_weights(view)[order])])
            id_rows.append(reached[np.searchsorted(ranks[order], grid, side="right")])
        if not rows:
            continue
        for suffix, series in (("", rows), ("id", id_rows)):
            stack = np.vstack(series)
            columns[slug + suffix] = stack.mean(axis=0)
            if len(series) > 1:
                columns[slug + suffix + "lo"] = stack.min(axis=0)
                columns[slug + suffix + "hi"] = stack.max(axis=0)
    curve_numbers(numbers, grid, columns, catalog_size)
    header = list(columns)
    lines = [",".join(header)]
    for i in range(len(grid)):
        lines.append(",".join(f"{columns[name][i]:.5f}" if name != "k" else str(int(grid[i])) for name in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tradeoff(found: dict, path: Path, numbers: Numbers) -> None:
    """Test-half PC_id against mean candidates per query: global threshold and top-k."""
    grid = np.unique(np.concatenate([np.round(np.logspace(0, 3.5, 90)).astype(int), [10, 20, 50]]))
    columns: dict[str, np.ndarray] = {"cpq": grid}
    for slug in TRADEOFF_MODELS:
        if slug not in found:
            continue
        threshold_rows, topk_rows = [], []
        for entry in found[slug].values():
            for partition, (tasks, _) in entry["parts"].items():
                test = tasks["test"]
                curve = test_curve(entry["dirs"][partition], test) if "positive_score" in test else None
                if curve is None:
                    continue
                thresholds, candidate_pairs = curve
                weights = identity_weights(test)
                scores = np.asarray(test["positive_score"], dtype=np.float32)
                order = np.argsort(scores)
                cumulative = np.concatenate([[0.0], np.cumsum(weights[order])])
                kept = 1.0 - cumulative[np.searchsorted(scores[order], thresholds, side="left")]
                cpq = candidate_pairs / test["num_queries"]
                by_cost = np.argsort(cpq, kind="stable")
                threshold_rows.append(np.interp(grid, cpq[by_cost], kept[by_cost]))
                ranks = np.asarray(test["positive_rank"])
                by_rank = np.argsort(ranks, kind="stable")
                reached = np.concatenate([[0.0], np.cumsum(weights[by_rank])])
                topk_rows.append(reached[np.searchsorted(ranks[by_rank], grid, side="right")])
        if threshold_rows:
            columns[slug + "th"] = np.vstack(threshold_rows).mean(axis=0)
            columns[slug + "topk"] = np.vstack(topk_rows).mean(axis=0)
            for budget in (10, 20, 50):
                index = int(np.searchsorted(grid, budget))
                if index < len(grid) and grid[index] == budget:
                    numbers.put(f"trade.{slug}.th.c{budget}", f"{columns[slug + 'th'][index]:.3f}")
                    numbers.put(f"trade.{slug}.topk.c{budget}", f"{columns[slug + 'topk'][index]:.3f}")
    if "hplusth" in columns:
        best_topk = np.max(np.vstack([columns[s + "topk"] for s in TRADEOFF_MODELS if s + "topk" in columns]), axis=0)
        ahead = columns["hplusth"] > best_topk
        if ahead[0]:
            first_behind = int(np.argmax(~ahead)) if (~ahead).any() else len(grid)
            last_ahead = max(first_behind - 1, 0)
            numbers.put("trade.cross", str(int(grid[last_ahead])))
            numbers.put("trade.cross.pc", f"{columns['hplusth'][last_ahead]:.2f}")
    best_rule = {}
    for slug in TRADEOFF_CROSSING:
        if slug + "th" in columns:
            # Mean over the log-spaced grid: the average height of the drawn curve.
            best_rule[slug] = max(TRADEOFF_RULES, key=lambda rule: float(columns[slug + rule].mean()))
            numbers.put(f"trade.best.{slug}.rule", TRADEOFF_RULES[best_rule[slug]])
    if len(best_rule) == 2:
        challenger, leader = TRADEOFF_CROSSING
        gap = columns[challenger + best_rule[challenger]] - columns[leader + best_rule[leader]]
        reached = np.flatnonzero(gap >= 0)
        if gap[0] < 0 and len(reached):
            i = int(reached[0])
            share = gap[i - 1] / (gap[i - 1] - gap[i])
            crossing = float(np.exp(np.log(grid[i - 1]) + share * (np.log(grid[i]) - np.log(grid[i - 1]))))
            numbers.put("plot.trade.crossing", f"{crossing:.1f}")
            numbers.put("trade.crossing", grouped(round(crossing)))
            curve = columns[leader + best_rule[leader]]
            numbers.put("trade.crossing.pc", f"{curve[i - 1] + share * (curve[i] - curve[i - 1]):.3f}")
    drawn = [columns[name] for name in columns if name != "cpq"]
    if drawn:
        numbers.put("plot.trade.cpqmin", str(int(grid[0])))
        numbers.put("plot.trade.cpqmax", str(int(grid[-1])))
        numbers.put("plot.trade.ymin", f"{TRADEOFF_YMIN:.2f}")
    frozen = [s for s in TRADEOFF_FROZEN if s + "th" in columns]
    if frozen:
        topk_ahead = np.all(np.vstack([columns[s + "topk"] >= columns[s + "th"] for s in frozen]), axis=0)
        tail = np.flip(np.logical_and.accumulate(np.flip(topk_ahead)))
        if tail.any():
            numbers.put("trade.frozen.from", str(int(grid[int(np.argmax(tail))])))
        for s in frozen:
            numbers.put(f"trade.{s}.th.c1", f"{columns[s + 'th'][0]:.3f}")
            numbers.put(f"trade.{s}.topk.c1", f"{columns[s + 'topk'][0]:.3f}")
    header = list(columns)
    lines = [",".join(header)]
    for i in range(len(grid)):
        lines.append(",".join(str(int(grid[i])) if name == "cpq" else f"{columns[name][i]:.5f}" for name in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_concentration(counts: np.ndarray, path: Path) -> None:
    share = np.cumsum(counts) / counts.sum()
    lines = ["paintings,share"] + [f"{i + 1},{value:.5f}" for i, value in enumerate(share)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def zero_words(excluded: int, n: int) -> str:
    if n == 1:
        return "excludes zero" if excluded else "includes zero"
    if excluded == 0:
        return f"include zero for all {NUMBER_WORDS.get(n, n)} seeds"
    if excluded == n:
        return f"exclude zero for all {NUMBER_WORDS.get(n, n)} seeds"
    return f"exclude zero for {NUMBER_WORDS.get(excluded, excluded)} of the {NUMBER_WORDS.get(n, n)} seeds"


def resample_counts(paintings: int, keep: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """How often each painting appears in each resample; paintings outside keep never appear."""
    pool = np.arange(paintings) if keep is None else np.flatnonzero(keep)
    draws = pool[np.random.default_rng(seed).integers(0, len(pool), size=(BOOTSTRAP_RESAMPLES, len(pool)))]
    return np.stack([np.bincount(row, minlength=paintings) for row in draws]).astype(float)


def resampled(view: dict, measure: str, paintings: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The measure on every resample.

    idpc20, pc20, rec20: PC_id, pooled PC, and per-query recall at k = 20; idk95, k95: the smallest
    k at which PC_id or pooled PC reaches 0.95.
    """
    kind, number = re.fullmatch(r"(idpc|pc|rec|idk|k)(\d+)", measure).groups()
    number = int(number)
    if kind in ("idpc", "pc"):
        hits, totals = painting_hits(view, number, paintings)
        if kind == "idpc":
            return counts @ (hits / totals) / counts.sum(axis=1)
        return (counts @ hits) / (counts @ totals)
    if kind == "rec":
        retrieved = (np.asarray(view["positive_rank"]) <= number).astype(float)
        hit = np.bincount(np.asarray(view["positive_query"]), weights=retrieved, minlength=view["num_queries"])
        recall = hit / np.asarray(view["query_positives"], dtype=float)
        index = np.searchsorted(paintings, np.asarray(view["query_class"]))
        sums = np.bincount(index, weights=recall, minlength=len(paintings))
        queries = np.bincount(index, minlength=len(paintings)).astype(float)
        return (counts @ sums) / (counts @ queries)
    index = np.searchsorted(paintings, pair_paintings(view))
    ranks = np.asarray(view["positive_rank"])
    order = np.argsort(ranks, kind="stable")
    sizes = np.bincount(index, minlength=len(paintings)).astype(float)
    weights = counts[:, index[order]]
    if kind == "idk":
        weights = weights / sizes[index[order]] / counts.sum(axis=1, keepdims=True)
    else:
        weights = weights / (counts @ sizes)[:, None]
    reached = np.cumsum(weights, axis=1)
    first = np.minimum((reached < number / 100 - 1e-9).sum(axis=1), len(order) - 1)
    return ranks[order][first].astype(float)


def paired_interval(
    numbers: Numbers, key: str, mine: dict, theirs: dict, measure: str, paintings: np.ndarray, counts: np.ndarray
) -> None:
    """Percentile interval of mine minus theirs over the resamples, one per training seed of mine.

    A frozen model has one entry and is compared with every seed; two adapted models are paired by
    training seed. Completeness differences are in points, budget differences in candidates.
    """
    rows = []
    for seed, entry in mine.items():
        other = theirs.get(seed) if len(theirs) > 1 else next(iter(theirs.values()))
        if entry["full"] is None or other is None or other["full"] is None:
            continue
        deltas = resampled(entry["full"]["test"], measure, paintings, counts) - resampled(
            other["full"]["test"], measure, paintings, counts
        )
        rows.append((float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)), float(np.mean(deltas > 0))))
    if not rows:
        return
    budget = re.match(r"(idk|k)\d+$", measure) is not None
    show = (lambda value: grouped(round(value))) if budget else (lambda value: f"{100 * value:.1f}")
    excluded = sum(low > 0 or high < 0 for low, high, _ in rows)
    numbers.put(f"{key}.lo", show(min(low for low, _, _ in rows)))
    numbers.put(f"{key}.hi", show(max(high for _, high, _ in rows)))
    numbers.put(f"{key}.excl", str(excluded))
    numbers.put(f"{key}.n", str(len(rows)))
    numbers.put(f"{key}.pos.min", f"{min(share for _, _, share in rows):.2f}")
    numbers.put(f"{key}.pos.max", f"{max(share for _, _, share in rows):.2f}")
    numbers.put(f"{key}.zero", zero_words(excluded, len(rows)))


def bootstrap_numbers(numbers: Numbers, found: dict, prefix: str, pairs) -> None:
    """Paired painting bootstraps on the full test set; keys {prefix}.{model}.{reference}.{measure}."""
    if REFERENCE not in found:
        return
    reference = next(iter(found[REFERENCE].values()))["full"]
    if reference is None:
        return
    paintings, sizes = np.unique(pair_paintings(reference["test"]), return_counts=True)
    counts = resample_counts(len(paintings))
    for mine, theirs, measures in pairs:
        if mine in found and theirs in found:
            for measure in measures:
                paired_interval(numbers, f"{prefix}.{mine}.{theirs}.{measure}", found[mine], found[theirs], measure, paintings, counts)
    if prefix != "pb" or "hplus" not in found:
        return
    without_largest = resample_counts(len(paintings), keep=sizes < sizes.max())
    paired_interval(numbers, f"pb.hplus.{REFERENCE}.pc20nl", found["hplus"], found[REFERENCE], "pc20", paintings, without_largest)
    budget = numbers.values.get(f"{REFERENCE}.ek95.k")
    if budget is not None:
        k = int(budget.replace("{,}", ""))
        numbers.put("pb.sb95.k", grouped(k))
        for kind in ("idpc", "pc"):
            paired_interval(numbers, f"pb.hplus.{REFERENCE}.sb95{kind}", found["hplus"], found[REFERENCE], f"{kind}{k}", paintings, counts)
    numbers.put("boot.n", grouped(BOOTSTRAP_RESAMPLES))


def interval_numbers(numbers: Numbers, found: dict, k: int = 20, seed: int = 0) -> None:
    """95% percentile intervals of PC_id@k and PC@k over paintings, for the seed mean of every model."""
    if REFERENCE not in found:
        return
    reference = next(iter(found[REFERENCE].values()))["full"]
    if reference is None:
        return
    paintings = np.unique(pair_paintings(reference["test"]))
    draws = np.random.default_rng(seed).integers(0, len(paintings), size=(BOOTSTRAP_RESAMPLES, len(paintings)))
    for slug, seeds in found.items():
        views = [entry["full"]["test"] for entry in seeds.values() if entry["full"] is not None]
        if not views:
            continue
        counts = [painting_hits(view, k, paintings) for view in views]
        identity = np.mean([(hits / totals)[draws].mean(axis=1) for hits, totals in counts], axis=0)
        pooled = np.mean([hits[draws].sum(axis=1) / totals[draws].sum(axis=1) for hits, totals in counts], axis=0)
        for key, values in ((f"idpc{k}", identity), (f"pc{k}", pooled)):
            numbers.put(f"{slug}.{key}.lo", f"{np.quantile(values, 0.025):.3f}")
            numbers.put(f"{slug}.{key}.hi", f"{np.quantile(values, 0.975):.3f}")


def largest_painting_numbers(numbers: Numbers, found: dict, k: int = 20) -> None:
    """Pooled PC@k without the painting with most true pairs, and that painting's own completeness."""
    reference = next(iter(found[REFERENCE].values()))["full"]
    if reference is None:
        return
    paintings, sizes = np.unique(pair_paintings(reference["test"]), return_counts=True)
    largest = int(np.argmax(sizes))
    for slug in ("hplus", "dsev", "dhp"):
        if slug not in found:
            continue
        rest, own = [], []
        for entry in found[slug].values():
            if entry["full"] is None:
                continue
            hits, totals = painting_hits(entry["full"]["test"], k, paintings)
            own.append(hits[largest] / totals[largest])
            rest.append((hits.sum() - hits[largest]) / (totals.sum() - totals[largest]))
        numbers.num(f"{slug}.pc{k}.nolargest", rest)
        numbers.num(f"{slug}.largest.pc{k}", own)


def throughput_numbers(numbers: Numbers, path: Path) -> None:
    if not path.is_file():
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    names = {"hplus_lora_merged": "hplus", "dinov3_7b_frozen": "dsev", "dinov3_hplus_frozen": "dhp"}
    for row in doc["results"]:
        slug = names.get(row["name"])
        if slug is None:
            continue
        for batch in ("batch_1", "batch_16"):
            size = batch.split("_")[1]
            numbers.put(f"tp.{slug}.ips{size}", f"{row[batch]['images_per_second']:.1f}")
            numbers.put(f"tp.{slug}.gib{size}", f"{row[batch]['peak_allocated_gib']:.1f}")
    hplus = next(r for r in doc["results"] if r["name"] == "hplus_lora_merged")["batch_1"]["images_per_second"]
    seven = next(r for r in doc["results"] if r["name"] == "dinov3_7b_frozen")["batch_1"]["images_per_second"]
    numbers.put("tp.speedup", f"{hplus / seven:.1f}")


def derived_numbers(numbers: Numbers) -> None:
    def value(key: str) -> float | None:
        text = numbers.values.get(key)
        return None if text is None else float(text.replace("{,}", ""))

    pairs = {
        "gain.dhp.pc20": ("hplus.pc20", "dhp.pc20"),
        "gain.dsev.pc20": ("hplus.pc20", "dsev.pc20"),
        "gain.dsev.pc40": ("hplus.pc40", "dsev.pc40"),
        "gain.dsev.pc160": ("hplus.pc160", "dsev.pc160"),
        "gain.dhp.pc160": ("hplus.pc160", "dhp.pc160"),
        "gain.dsev.rec20": ("hplus.rec20", "dsev.rec20"),
        "gain.dhp.rec20": ("hplus.rec20", "dhp.rec20"),
        "gain.dsev.pc20.nolargest": ("hplus.pc20.nolargest", "dsev.pc20.nolargest"),
        "gain.dhp.pc20.nolargest": ("hplus.pc20.nolargest", "dhp.pc20.nolargest"),
        "gain.dhp.pc40": ("hplus.pc40", "dhp.pc40"),
        "gain.dhp.idpc10": ("hplus.idpc10", "dhp.idpc10"),
        "gain.dhp.idpc40": ("hplus.idpc40", "dhp.idpc40"),
        "gain.dsev.idpc10": ("hplus.idpc10", "dsev.idpc10"),
        "gain.dsev.idpc20": ("hplus.idpc20", "dsev.idpc20"),
        "gain.dsev.idpc40": ("hplus.idpc40", "dsev.idpc40"),
        "gain.dsev.idpc160": ("hplus.idpc160", "dsev.idpc160"),
        "gain.dhp.idpc20": ("hplus.idpc20", "dhp.idpc20"),
        "gain.uncap.idpc20": ("uncap.hplus.idpc20", "uncap.dhp.idpc20"),
        "gain.dhp.idpc160": ("hplus.idpc160", "dhp.idpc160"),
        "gain.sig.pc20": ("sigad.pc20", "sig.pc20"),
        "gain.sig.idpc20": ("sigad.idpc20", "sig.idpc20"),
        "gain.clip.idpc20": ("clipad.idpc20", "clip.idpc20"),
        "gain.oclip.idpc20": ("oclipad.idpc20", "oclip.idpc20"),
        "gain.rn.idpc20": ("rnad.idpc20", "rn.idpc20"),
        "gain.ctlhead.pc20": ("ctlhead.pc20", "dhp.pc20"),
        "gain.ctlmodern.pc20": ("ctlmodern.pc20", "dhp.pc20"),
        "gain.ctlorig.pc20": ("ctlorig.pc20", "dhp.pc20"),
        "gain.ctlhead.idpc20": ("ctlhead.idpc20", "dhp.idpc20"),
        "gain.ctlmodern.idpc20": ("ctlmodern.idpc20", "dhp.idpc20"),
        "gain.ctlorig.idpc20": ("ctlorig.idpc20", "dhp.idpc20"),
    }
    for key, (a, b) in pairs.items():
        if value(a) is not None and value(b) is not None:
            numbers.put(key, f"{100 * (value(a) - value(b)):.1f}")
    for frozen, adapted in (("dhp", "hplus"), ("sig", "sigad"), ("oclip", "oclipad"), ("clip", "clipad"), ("rn", "rnad")):
        gain = value(f"gain.{frozen}.idpc20")
        if gain is not None:
            numbers.put(f"gain.{frozen}.idpc20.signed", f"{gain:+.1f}")
        for field in ("ek95.k", "et95.cpq"):
            mine, theirs = value(f"{adapted}.{field}"), value(f"{frozen}.{field}")
            if mine is not None and theirs is not None:
                delta = round(mine - theirs)
                numbers.put(f"gain.{frozen}.{field}", ("+" if delta > 0 else "") + grouped(delta))
    for trained in ("hplus", "ctlhead", "ctlorig", "ctlmodern"):
        mine, theirs = value(f"{trained}.idpc20"), value(f"{REFERENCE}.idpc20")
        if mine is not None and theirs is not None:
            numbers.put(f"vsdhp.{trained}.idpc20", f"{100 * (mine - theirs):+.1f}")
        for field in ("ek95.k", "et95.cpq"):
            mine, theirs = value(f"{trained}.{field}"), value(f"{REFERENCE}.{field}")
            if mine is not None and theirs is not None:
                delta = round(mine - theirs)
                numbers.put(f"vsdhp.{trained}.{field}", ("+" if delta > 0 else "") + grouped(delta))
    for rule in RULES:
        for donor in ("dsev", "dhp", "sig"):
            mine, theirs = value(f"hplus.{rule}t95.cpq"), value(f"{donor}.{rule}t95.cpq")
            if mine and theirs:
                numbers.put(f"ratio.{rule}t95.{donor}", f"{theirs / mine:.1f}")
    safes = [value(f"{slug}.safe99") for slug in SLUGS if value(f"{slug}.safe99") is not None]
    if safes:
        numbers.put("safe99.min", grouped(min(safes)))
    for rule in RULES:
        ratios = {}
        for slug in SLUGS:
            threshold, budget = value(f"{slug}.{rule}t95.cpq"), value(f"{slug}.{rule}k95.k")
            if threshold and budget:
                ratios[slug] = threshold / budget
                numbers.put(f"ratio.{rule}tk95.{slug}", f"{threshold / budget:.1f}")
        strong = [ratios[s] for s in ("dsev", "dhp", "sig", "oclip") if s in ratios]
        if strong:
            numbers.put(f"ratio.{rule}tk95.strong.min", f"{min(strong):.1f}")
            numbers.put(f"ratio.{rule}tk95.strong.max", f"{max(strong):.1f}")
        adapted = [ratios[s] for s in ("hplus", "sigad", "clipad", "oclipad", "rnad") if s in ratios]
        if adapted:
            numbers.put(f"ratio.{rule}tk95.adapted.min", f"{min(adapted):.1f}")
            numbers.put(f"ratio.{rule}tk95.adapted.max", f"{max(adapted):.1f}")
    for rule in RULES:
        for kind in ("k", "t"):
            for name in TARGETS:
                base = f"{rule}{kind}{name}"
                for field, digits in (("met", None), ("pc", 3), ("worst", 3), (kind if kind == "k" else "cpq", 0)):
                    values = [value(f"{s}.{base}.{field}") for s in MAIN_MODELS if value(f"{s}.{base}.{field}") is not None]
                    if not values:
                        continue
                    for end, picked in (("min", min(values)), ("max", max(values))):
                        if digits == 0:
                            text = grouped(round(picked))
                        elif digits is None:
                            text = f"{picked:.0f}" if picked == int(picked) else f"{picked:.1f}"
                        else:
                            text = f"{picked:.{digits}f}"
                        numbers.put(f"range.{base}.{field}.{end}", text)
    for slug in ("hplus", "dsev", "dhp"):
        targets_and_keys = tuple(
            (f"{rule}{kind}{name}", f"{rule}{kind}{name}.{'k' if kind == 'k' else 'cpq'}")
            for rule in RULES for kind in ("k", "t") for name in TARGETS
        )
        for target, key in targets_and_keys:
            cpq = value(f"{slug}.{key}")
            if cpq:
                seconds = cpq / LIGHTGLUE_PAIRS_PER_SECOND
                numbers.put(f"lg.{slug}.{target}.sec", f"{seconds:.1f}" if seconds < 10 else f"{seconds:.0f}")
                hours = seconds * LISTING_IMAGES_PER_DAY / 3600
                numbers.put(f"lg.{slug}.{target}.hday", f"{hours:.1f}" if hours < 10 else f"{hours:.0f}")
    targets = value("data.T")
    if targets:
        days = LISTING_IMAGES_PER_DAY * targets / LIGHTGLUE_PAIRS_PER_SECOND / 86400
        numbers.put("deploy.exhaustive.days", f"{days:.0f}")
    for k in (20, 160):
        hours = k / LIGHTGLUE_PAIRS_PER_SECOND * LISTING_IMAGES_PER_DAY / 3600
        numbers.put(f"lg.k{k}.hday", f"{hours:.1f}")
    numbers.put("deploy.matcher.pps", f"{LIGHTGLUE_PAIRS_PER_SECOND:.0f}")
    numbers.put("deploy.images.day", grouped(round(LISTING_IMAGES_PER_DAY)))
    gains = [value(f"gain.{slug}.idpc20") for slug in ("sig", "clip", "oclip", "rn")]
    gains = [gain for gain in gains if gain is not None]
    if gains:
        numbers.put("gain.transfer.min", f"{min(gains):.1f}")
        numbers.put("gain.transfer.max", f"{max(gains):.1f}")
        numbers.put("gain.transfer.n", NUMBER_WORDS.get(len(gains), str(len(gains))))


def best_numbers(numbers: Numbers, warn: bool = True) -> None:
    """best.{table}.{key} for every cell that holds the best printed value of its column."""
    for table, (rows, columns) in BEST_TABLES.items():
        for template, pick in columns:
            keys = [template.format(m=model, f=frozen) for model, frozen in rows]
            printed = {key: float(numbers.values[key].replace("{,}", "")) for key in keys if key in numbers.values}
            if warn and len(printed) < len(keys):
                print(f"best.{table}: missing {sorted(set(keys) - set(printed))}", file=sys.stderr)
            if not printed:
                continue
            winner = pick(printed.values())
            for key, value in printed.items():
                if value == winner:
                    numbers.put(f"best.{table}.{key}", "1")


def build(split_prefix: str, companion_prefix: str, uncapped_prefix: str, throughput: Path, out: Path | None) -> Numbers:
    """Every number for one catalog; plot series are written to out, if given."""
    numbers = Numbers()
    found = resolve(load(split_prefix))
    missing = [slug for slug in SLUGS if slug not in found]
    if missing:
        print("missing models:", ", ".join(missing), file=sys.stderr)
    for slug, seeds in found.items():
        fixed_k_numbers(numbers, slug, seeds)
        reference = found.get(REFERENCE) if slug == "hplus" else None
        guaranteed_numbers(numbers, slug, seeds, reference)
    for name, target in TARGETS.items():
        numbers.put(f"need.e{name}", grouped(conformal_paintings_needed(1 - target)))
    catalog = found[REFERENCE]
    counts = dataset_numbers(numbers, catalog)
    catalog_size = int(next(iter(catalog.values()))["full"]["test"]["num_candidates"])
    if out is not None:
        if counts is not None:
            write_concentration(counts, out / "concentration.csv")
        write_curve(found, out / "curve.csv", catalog_size, numbers)
        write_tradeoff(found, out / "tradeoff.csv", numbers)

    companion_raw = load(companion_prefix)
    if split_prefix.startswith("wikidata1.5"):
        # Wiki-only runs fill labels the combined catalog does not have, such as
        # the single-profile holdouts. Models scored only on the combined catalog
        # keep the view with Lost Art candidates removed.
        _merge_missing(
            companion_raw,
            load(split_prefix, test_task="test_wikidata", calibration_task="calibration_wikidata"),
        )
    companion = resolve(companion_raw)
    for slug in ("hplus", "dsev", "dhp", "sigad", "sig"):
        if slug in companion:
            fixed_k_numbers(numbers, "wiki." + slug, companion[slug])
    if REFERENCE in companion:
        dataset_numbers(numbers, companion[REFERENCE], prefix="wikidata")
    count = lambda key: int(numbers.values[key].replace("{,}", ""))
    numbers.put("lostart.identities", grouped(LOSTART_IDENTITIES))
    if "data.T" in numbers.values and "wikidata.T" in numbers.values:
        numbers.put("lostart.T", grouped(count("data.T") - count("wikidata.T")))
    if "conc.paintings" in numbers.values:
        numbers.put("wilo.identities", grouped(count("conc.paintings") + LOSTART_IDENTITIES))
    companion_models = companion_raw
    holdout_pc20 = []
    for label in HOLDOUTS:
        for entry in companion_models.get(label, {}).values():
            if entry["full"] is not None:
                holdout_pc20.append(pc(entry["full"]["test"], 20))
    if holdout_pc20:
        for key, completeness in (("pc20", pc), ("idpc20", identity_pc)):
            seed_means = [
                st.mean(completeness(entry["full"]["test"], 20) for entry in companion_models[label].values() if entry["full"] is not None)
                for label in HOLDOUTS if label in companion_models
            ]
            numbers.put(f"wiki.hold.{key}.min", f"{min(seed_means):.3f}")
            numbers.put(f"wiki.hold.{key}.max", f"{max(seed_means):.3f}")
        numbers.put("wiki.hold.n", str(len(seed_means)))

    targets_full = catalog_size
    for k in (10, 20, 40, 80, 160):
        numbers.put(f"rr.k{k}", f"{1 - k / targets_full:.4f}")

    uncapped = resolve(load(uncapped_prefix))
    for slug in ("hplus", "dsev", "dhp", "sig"):
        if slug in uncapped:
            fixed_k_numbers(numbers, "uncap." + slug, uncapped[slug])

    bootstrap_numbers(numbers, found, "pb", PAIRED)
    bootstrap_numbers(numbers, companion, "pb.wiki", (("hplus", REFERENCE, COMPANION_PAIRED),))
    bootstrap_numbers(numbers, uncapped, "pb.uncap", (("hplus", REFERENCE, COMPANION_PAIRED),))
    largest_painting_numbers(numbers, found)
    interval_numbers(numbers, found)
    throughput_numbers(numbers, throughput)
    derived_numbers(numbers)
    best_numbers(numbers, warn=out is not None)
    numbers.put("source.catalog", split_prefix)
    return numbers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-prefix", default="wikidata1.5-lostart-distractors")
    parser.add_argument("--companion-prefix", default="wikidata-1.5")
    parser.add_argument("--uncapped-prefix", default="wikidata1.5-uncapped-lostart-distractors")
    parser.add_argument("--throughput", type=Path, default=RUNS / "merged_lora_throughput_v1/summary.json")
    parser.add_argument("--final", action="store_true", help="Accepted for compatibility. Numbers are final unless --draft is set.")
    parser.add_argument("--draft", action="store_true", help="Mark the numbers as provisional in the PDF.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    numbers = build(args.split_prefix, args.companion_prefix, args.uncapped_prefix, args.throughput, OUT)
    (OUT / "numbers.tex").write_text(numbers.tex(not args.draft), encoding="utf-8")
    print(f"wrote {len(numbers.values)} numbers to {OUT / 'numbers.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
