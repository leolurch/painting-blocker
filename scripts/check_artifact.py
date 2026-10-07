#!/usr/bin/env python3
"""Check the archive contents. Does not download images or train a model."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    labels = list(csv.DictReader((ROOT / "data/wikidata/labels.csv").open(encoding="utf-8")))
    if len(labels) != MANIFEST["wikidata_images"]:
        raise SystemExit(f"expected {MANIFEST['wikidata_images']} labels, found {len(labels)}")
    qids = set()
    for row in labels:
        if row["role"] not in {"modern", "historic"}:
            raise SystemExit(f"bad role for {row['filename']}")
        if not row["commons_page"].startswith("https://commons.wikimedia.org/"):
            raise SystemExit(f"missing Commons page for {row['filename']}")
        if not row["download_url"].startswith("https://"):
            raise SystemExit(f"missing download URL for {row['filename']}")
        if not str(row.get("class_id", "")).isdigit():
            raise SystemExit(f"missing class id for {row['filename']}")
        if row["half_seed_42"] not in {"val", "test"}:
            raise SystemExit(f"missing partition for {row['filename']}")
        qids.add(row["qid"])
    if len(qids) != MANIFEST["wikidata_paintings"]:
        raise SystemExit(f"expected {MANIFEST['wikidata_paintings']} paintings, found {len(qids)}")

    lost = list(csv.DictReader((ROOT / "data/lostart/distractor_files.csv").open(encoding="utf-8")))
    ids = {row["lost_art_id"] for row in lost}
    if len(ids) != MANIFEST["lostart_reports"]:
        raise SystemExit(f"expected {MANIFEST['lostart_reports']} Lost Art reports, found {len(ids)}")

    syn = list(csv.DictReader((ROOT / "data/syn/class_split.csv").open(encoding="utf-8")))
    subsets = {row["subset"] for row in syn}
    if subsets != {"train", "val", "test"}:
        raise SystemExit(f"SynView split subsets are {subsets}")
    if len(syn) != MANIFEST["syn_paintings"]:
        raise SystemExit(f"expected {MANIFEST['syn_paintings']} SynView paintings, found {len(syn)}")

    for name in (
        "data/dedup/museum/reviewed_pairs.csv",
        "data/dedup/wikidata_self/review_decisions.csv",
        "data/dedup/wikidata_self/review_pairs.csv",
        "data/dedup/lostart/review_decisions.csv",
        "data/dedup/lostart/review_pairs.csv",
    ):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            raise SystemExit(f"{name} is empty")
        if name.endswith("review_decisions.csv") and "accepted" not in (ROOT / name).read_text(encoding="utf-8"):
            raise SystemExit(f"{name} has no accepted row")

    winner = ROOT / "experiments/configs/experiments/finetune_dinov3_vithplus_met_paint_winner_followup_v1/none.yml"
    winner_text = winner.read_text(encoding="utf-8")
    if "seeds: [42, 43, 44]" not in winner_text:
        raise SystemExit("winner config is missing the three seeds")
    sweep = list((ROOT / "experiments/configs/experiments/finetune_dinov3_vithplus_met_paint_wave1_v1").glob("*.yml"))
    if len(sweep) != 64:
        raise SystemExit(f"expected 64 sweep configs, found {len(sweep)}")
    forbidden = (
        "experiments/core/met_protocol",
        "experiments/core/finetuning/models/qwen3_vl_lora.py",
        "experiments/core/finetuning/models/dinov2_salad.py",
        "experiments/datasets/met_benchmark/materialize_syngallery.py",
        "experiments/configs/experiments/eval_wikidata1_5_lostart_distractors_resnet_winner_full_v1",
        "experiments/configs/experiments/finetune_dinov3_vithplus_met_paint_e5_v1",
        "experiments/configs/experiments/finetune_siglip2_giant_met_paint_wave2_v1/14_graphhalf.yml",
    )
    for name in forbidden:
        if (ROOT / name).exists():
            raise SystemExit(f"unreported method still packaged: {name}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("download_wikidata_labels", ROOT / "scripts/download_wikidata_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import tempfile
    sample = labels[0]
    target = module.destination(Path("images"), sample)
    if target != Path("images") / sample["filename"]:
        raise SystemExit(f"downloader path mismatch: {target}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "images"
        db_path = module.write_dataset_db(labels, out)
        import sqlite3
        conn = sqlite3.connect(db_path)
        images = conn.execute("SELECT COUNT(*) FROM image_files").fetchone()[0]
        classes = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
        conn.close()
    if images != len(labels) or classes != len(qids):
        raise SystemExit(f"dataset database has {images} images and {classes} classes")
    print("artifact contents ok")
    print(json.dumps({key: MANIFEST[key] for key in ("wikidata_images", "wikidata_paintings", "lostart_reports", "syn_paintings")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise
