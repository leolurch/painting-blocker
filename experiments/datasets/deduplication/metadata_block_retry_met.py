#!/usr/bin/env python3
"""Refetch Met records that came back HTTP 403 and rebuild the candidate matrix.

The first pass cached those 403 bodies and does not call them again. This
script writes a second cache and leaves the first cache in place. SYN rows
whose id is ``metmuseum_<object id>`` then copy the official Met title.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import metadata_block as block

OUT = block.OUT
RETRY = OUT / "cache" / "met_retry"
PAUSE = 0.45


def refetch() -> None:
    RETRY.mkdir(parents=True, exist_ok=True)
    pending = []
    for path in sorted((OUT / "cache" / "met").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == 200 and (payload.get("body") or {}).get("title"):
            continue
        if (RETRY / path.name).exists():
            continue
        pending.append(path.stem)
    print("met refetch pending", len(pending), flush=True)
    for index, object_id in enumerate(pending, start=1):
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        request = urllib.request.Request(url, headers={"User-Agent": block.UA, "Accept": "application/json"})
        status = 0
        body = None
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                status = response.status
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = exc.code
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            status = 0
            body = {"error": str(exc)}
        (RETRY / f"{object_id}.json").write_text(
            json.dumps({"status": status, "body": body if status == 200 else None})
        )
        if index % 100 == 0 or status != 200:
            print(f"retry {index}/{len(pending)} id={object_id} status={status}", flush=True)
        time.sleep(PAUSE if status == 200 else 2.0)


def load_met() -> list[dict]:
    cards = []
    for path in sorted(block.MET_IMAGES.glob("metmuseum_*.jpg")):
        object_id = path.stem.split("_", 1)[1]
        payload = None
        for cache in (RETRY / f"{object_id}.json", OUT / "cache" / "met" / f"{object_id}.json"):
            if cache.exists():
                payload = json.loads(cache.read_text())
                if payload.get("status") == 200 and (payload.get("body") or {}).get("title"):
                    break
        body = (payload or {}).get("body") or {}
        cards.append(
            {
                "set": "met",
                "id": path.stem,
                "object_id": object_id,
                "title": body.get("title") or "",
                "artist": body.get("artistDisplayName") or "",
                "year": body.get("objectBeginDate"),
                "year_end": body.get("objectEndDate"),
                "accession": body.get("accessionNumber") or "",
                "source": "met",
            }
        )
    return cards


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    refetch()
    mets = load_met()
    block.write_jsonl(OUT / "cards_met.jsonl", mets)
    titled = sum(1 for card in mets if card["title"])
    print("met with title", titled, "of", len(mets), flush=True)
    by_object = {card["object_id"]: card for card in mets}
    syn = load_jsonl(OUT / "cards_syn.jsonl")
    for card in syn:
        if card.get("prefix") == "metmuseum" and card.get("raw_id") in by_object:
            met = by_object[card["raw_id"]]
            card["title"] = met["title"]
            card["artist"] = met["artist"]
            card["year"] = met["year"]
    block.write_jsonl(OUT / "cards_syn.jsonl", syn)
    wik = load_jsonl(OUT / "cards_wikidata.jsonl")
    print("image neighbors", flush=True)
    wik_images = block.image_neighbors(block.WIK_CANDIDATES, "wikidata")
    syn_images = block.image_neighbors(block.SYN_CANDIDATES, "syn")
    from collections import defaultdict

    file_to_qid = {}
    for card in wik:
        for name in card.get("image_ids") or []:
            file_to_qid[Path(name).name] = card["id"]
        card["image_ids"] = [card["id"]]
    qid_images = defaultdict(list)
    for key, pairs in wik_images.items():
        name = Path(key).name
        qid = file_to_qid.get(name) or file_to_qid.get(name[:-4] if name.endswith(".jpg") else name)
        if qid:
            qid_images[qid].extend(pairs)
    wik_rows = block.block(wik, mets, qid_images, "wikidata")
    syn_rows = block.block(syn, mets, syn_images, "syn")
    block.write_jsonl(OUT / "blocked_wikidata_vs_met.jsonl", wik_rows)
    block.write_jsonl(OUT / "blocked_syn_vs_met.jsonl", syn_rows)
    summary = {
        "met_cards": len(mets),
        "met_with_title": titled,
        "wikidata_pairs": len(wik_rows),
        "syn_pairs": len(syn_rows),
        "wikidata_reason_counts": block.count_reasons(wik_rows),
        "syn_reason_counts": block.count_reasons(syn_rows),
        "note": "Rebuilt after Met 403 retries. First-pass 403 cache files were kept.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
