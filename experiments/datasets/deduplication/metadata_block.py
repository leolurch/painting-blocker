#!/usr/bin/env python3
"""Build a metadata-and-image candidate matrix for painting deduplication.

Blocking rules, in priority order:

1. Exact museum object id (SYN ``metmuseum_<id>`` equals a Met original).
2. Exact normalized title from official catalog fields.
3. Title token Jaccard >= 0.80.
4. Title token Jaccard >= 0.50 and a shared artist token of length >= 4.
5. Existing DINOv3 cosine: keep a pair when cosine >= 0.55, and also the
   best pair per query when cosine >= 0.40. At most 20 image neighbors
   per query enter the union.

Each query keeps at most 1000 reference candidates. Pairs are not classified
here. A later local pass reads ``blocked_pairs.jsonl``.

Catalog text comes from the Met, Wikidata, ARTIC, and Cleveland public APIs.
Responses are cached under the output directory. A failed fetch is cached
too, so a rerun does not repeat it.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

OUT = Path("data/datasets/metadata_dedupe_v1")
WIK_DB = Path("data/datasets/wikidata-1.5/dataset.db")
SYN_DB = Path(
    "data/datasets/"
    "museum_deduplication_hard_synth_v1_wikidata_1_5_overlap_removed/dataset.db"
)
MET_IMAGES = Path("data/datasets/met_originals_museum_dedup_v1/images")

IMAGE_KEEP = 0.55
IMAGE_TOP = 0.40
IMAGE_CAP = 20
PAIR_CAP = 1000
UA = "metadata-block/1.0"

_STOP = {"a", "an", "the", "of", "and", "le", "la", "les", "el", "der", "die", "das"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str | None) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKD", text)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = _NON_ALNUM.sub(" ", value.lower()).strip()
    tokens = [tok for tok in value.split() if tok not in _STOP]
    return " ".join(tokens)


def tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def artist_overlap(left: str | None, right: str | None) -> bool:
    a = {tok for tok in tokens(left or "") if len(tok) >= 4}
    b = {tok for tok in tokens(right or "") if len(tok) >= 4}
    return bool(a & b)


def fetch_json(url: str, cache: Path, pause: float = 0.15) -> dict | None:
    if cache.exists():
        payload = json.loads(cache.read_text())
        return payload.get("body")
    cache.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    body = None
    status = 0
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        status = 0
        body = {"error": str(exc)}
    cache.write_text(json.dumps({"status": status, "body": body if status == 200 else None}))
    time.sleep(pause)
    return body if status == 200 else None


def wikidata_cards() -> list[dict]:
    con = sqlite3.connect(f"file:{WIK_DB}?mode=ro", uri=True)
    classes = list(
        con.execute(
            "SELECT class_id, qid, painting_inception_year FROM classes ORDER BY class_id"
        )
    )
    files: dict[int, list[str]] = defaultdict(list)
    for class_id, rel in con.execute(
        """
        SELECT j.class_id, f.local_rel_path
        FROM image_file_classes j
        JOIN image_files f ON f.file_id = j.file_id
        """
    ):
        files[class_id].append(Path(rel or "").name)
    con.close()
    cards = []
    artist_qids: list[str] = []
    cache_root = OUT / "cache" / "wikidata"
    for start in range(0, len(classes), 50):
        chunk = classes[start : start + 50]
        ids = "|".join(qid for _, qid, _ in chunk)
        url = (
            "https://www.wikidata.org/w/api.php?action=wbgetentities"
            f"&ids={urllib.parse.quote(ids)}&props=labels|claims&languages=en&format=json"
        )
        body = fetch_json(url, cache_root / f"batch_{start:04d}.json", pause=0.3)
        entities = (body or {}).get("entities", {})
        for class_id, qid, year in chunk:
            entity = entities.get(qid, {})
            label = ((entity.get("labels") or {}).get("en") or {}).get("value")
            claims = entity.get("claims") or {}
            artist_qid = claim_entity_id(claims.get("P170"))
            if artist_qid:
                artist_qids.append(artist_qid)
            inception = year if year is not None else claim_year(claims.get("P571"))
            cards.append(
                {
                    "set": "wikidata",
                    "id": qid,
                    "class_id": class_id,
                    "title": label or "",
                    "artist": "",
                    "artist_qid": artist_qid,
                    "year": inception,
                    "image_ids": files[class_id],
                    "source": "wikidata",
                }
            )
    labels = entity_labels(artist_qids, cache_root / "artists")
    for card in cards:
        card["artist"] = labels.get(card.get("artist_qid") or "", "")
        card.pop("artist_qid", None)
    return cards


def claim_text(statements: list | None) -> str:
    if not statements:
        return ""
    mainsnak = statements[0].get("mainsnak", {})
    datavalue = (mainsnak.get("datavalue") or {}).get("value")
    if isinstance(datavalue, str):
        return datavalue
    if isinstance(datavalue, dict):
        return str(datavalue.get("id") or datavalue.get("text") or "")
    return ""


def claim_year(statements: list | None) -> int | None:
    if not statements:
        return None
    value = ((statements[0].get("mainsnak") or {}).get("datavalue") or {}).get("value")
    if not isinstance(value, dict):
        return None
    time_value = str(value.get("time") or "")
    match = re.search(r"(\d{3,4})", time_value)
    return int(match.group(1)) if match else None


def claim_entity_id(statements: list | None) -> str:
    if not statements:
        return ""
    value = ((statements[0].get("mainsnak") or {}).get("datavalue") or {}).get("value")
    if isinstance(value, dict):
        return str(value.get("id") or "")
    return ""


def entity_labels(qids: list[str], cache_root: Path) -> dict[str, str]:
    unique = list(dict.fromkeys(qid for qid in qids if qid))
    labels: dict[str, str] = {}
    for start in range(0, len(unique), 50):
        chunk = unique[start : start + 50]
        ids = "|".join(chunk)
        url = (
            "https://www.wikidata.org/w/api.php?action=wbgetentities"
            f"&ids={urllib.parse.quote(ids)}&props=labels&languages=en&format=json"
        )
        body = fetch_json(url, cache_root / f"batch_{start:04d}.json", pause=0.3)
        entities = (body or {}).get("entities", {})
        for qid in chunk:
            labels[qid] = ((entities.get(qid, {}).get("labels") or {}).get("en") or {}).get("value") or ""
    return labels


def claim_item_label(statements: list | None, entities: dict) -> str:
    if not statements:
        return ""
    value = ((statements[0].get("mainsnak") or {}).get("datavalue") or {}).get("value")
    if not isinstance(value, dict):
        return ""
    qid = value.get("id")
    if not qid:
        return ""
    entity = entities.get(qid) or {}
    return ((entity.get("labels") or {}).get("en") or {}).get("value") or qid


def met_cards() -> list[dict]:
    cards = []
    cache_root = OUT / "cache" / "met"
    for path in sorted(MET_IMAGES.glob("metmuseum_*.jpg")):
        object_id = path.stem.split("_", 1)[1]
        body = fetch_json(
            f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}",
            cache_root / f"{object_id}.json",
        )
        cards.append(
            {
                "set": "met",
                "id": path.stem,
                "object_id": object_id,
                "title": (body or {}).get("title") or "",
                "artist": (body or {}).get("artistDisplayName") or "",
                "year": (body or {}).get("objectBeginDate"),
                "year_end": (body or {}).get("objectEndDate"),
                "accession": (body or {}).get("accessionNumber") or "",
                "source": "met",
            }
        )
    return cards


def syn_cards(met_by_object: dict[str, dict]) -> list[dict]:
    con = sqlite3.connect(f"file:{SYN_DB}?mode=ro", uri=True)
    labels = [row[0] for row in con.execute("SELECT label FROM classes ORDER BY label")]
    con.close()
    cards = []
    for label in labels:
        prefix, _, raw_id = label.partition("_")
        title, artist, year = "", "", None
        if prefix == "metmuseum" and raw_id in met_by_object:
            met = met_by_object[raw_id]
            title, artist, year = met["title"], met["artist"], met["year"]
        elif prefix == "artic":
            body = fetch_json(
                "https://api.artic.edu/api/v1/artworks/"
                f"{raw_id}?fields=id,title,artist_title,date_start,date_end",
                OUT / "cache" / "artic" / f"{raw_id}.json",
            )
            data = (body or {}).get("data") or {}
            title = data.get("title") or ""
            artist = data.get("artist_title") or ""
            year = data.get("date_start")
        elif prefix == "cleveland":
            body = fetch_json(
                f"https://openaccess-api.clevelandart.org/api/artworks/{raw_id}",
                OUT / "cache" / "cleveland" / f"{raw_id}.json",
            )
            data = (body or {}).get("data") or body or {}
            title = data.get("title") or ""
            creators = data.get("creators") or []
            if creators and isinstance(creators[0], dict):
                artist = creators[0].get("description") or ""
            year = data.get("creation_date_earliest")
        cards.append(
            {
                "set": "syn",
                "id": label,
                "prefix": prefix,
                "raw_id": raw_id,
                "title": title,
                "artist": artist,
                "year": year,
                "source": prefix,
            }
        )
    return cards


def string_reasons(query: dict, ref: dict) -> list[str]:
    reasons = []
    if query.get("set") == "syn" and query.get("prefix") == "metmuseum":
        if query.get("raw_id") == ref.get("object_id"):
            reasons.append("exact_object_id")
    q_title = normalize(query.get("title"))
    r_title = normalize(ref.get("title"))
    if q_title and q_title == r_title and len(q_title.split()) >= 2:
        reasons.append("exact_title")
    score = jaccard(query.get("title") or "", ref.get("title") or "")
    if score >= 0.80 and "exact_title" not in reasons:
        reasons.append("title_jaccard_0.8")
    elif score >= 0.50 and artist_overlap(query.get("artist"), ref.get("artist")):
        reasons.append("title_jaccard_0.5_artist")
    return reasons


def image_neighbors(path: Path, query_prefix: str) -> dict[str, list[tuple[str, float]]]:
    """Map a query key to (met stem, cosine), capped and thresholded."""
    best: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not path.exists():
        return best
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cosine = float(row["cosine_similarity"])
            if cosine < IMAGE_TOP:
                continue
            image_a = row["image_a_id"]
            image_b = row["image_b_id"]
            met = image_b if "metmuseum_" in image_b else image_a
            other = image_a if met == image_b else image_b
            met_stem = Path(met).name
            met_stem = met_stem[: -len(".jpg")] if met_stem.endswith(".jpg") else met_stem
            if query_prefix == "wikidata":
                key = other
            else:
                key = Path(other).stem
            best[key].append((met_stem, cosine))
    kept: dict[str, list[tuple[str, float]]] = {}
    for key, pairs in best.items():
        pairs.sort(key=lambda item: item[1], reverse=True)
        strong = [item for item in pairs if item[1] >= IMAGE_KEEP][:IMAGE_CAP]
        if not strong and pairs and pairs[0][1] >= IMAGE_TOP:
            strong = pairs[:1]
        kept[key] = strong[:IMAGE_CAP]
    return kept


def add_pair(bucket: dict, query_id: str, ref_id: str, reason: str, cosine: float | None) -> None:
    slot = bucket[(query_id, ref_id)]
    slot["reasons"].add(reason)
    if cosine is not None:
        slot["cosine"] = max(slot["cosine"] or 0.0, cosine)


def block(queries: list[dict], refs: list[dict], images: dict[str, list[tuple[str, float]]], query_set: str) -> list[dict]:
    ref_by_id = {ref["id"]: ref for ref in refs}
    ref_by_title: dict[str, list[dict]] = defaultdict(list)
    for ref in refs:
        key = normalize(ref.get("title"))
        if key:
            ref_by_title[key].append(ref)
    bucket: dict[tuple[str, str], dict] = defaultdict(lambda: {"reasons": set(), "cosine": None})
    for query in queries:
        title_key = normalize(query.get("title"))
        if title_key:
            for ref in ref_by_title.get(title_key, []):
                for reason in string_reasons(query, ref):
                    add_pair(bucket, query["id"], ref["id"], reason, None)
        if query.get("set") == "syn" and query.get("prefix") == "metmuseum":
            ref_id = f"metmuseum_{query['raw_id']}"
            if ref_id in ref_by_id:
                add_pair(bucket, query["id"], ref_id, "exact_object_id", None)
        image_keys = query["image_ids"] if query_set == "wikidata" else [query["id"]]
        for image_key in image_keys:
            for ref_id, cosine in images.get(image_key, []):
                if ref_id in ref_by_id:
                    add_pair(bucket, query["id"], ref_id, "image_cosine", cosine)
        # Fuzzy pass only against titles that share a token of length >= 5.
        if title_key:
            query_tokens = {tok for tok in title_key.split() if len(tok) >= 5}
            seen = {ref["id"] for ref in ref_by_title.get(title_key, [])}
            for ref in refs:
                if ref["id"] in seen:
                    continue
                ref_tokens = {tok for tok in normalize(ref.get("title")).split() if len(tok) >= 5}
                if not (query_tokens & ref_tokens):
                    continue
                for reason in string_reasons(query, ref):
                    if reason != "exact_title":
                        add_pair(bucket, query["id"], ref["id"], reason, None)
    per_query: dict[str, list] = defaultdict(list)
    for (query_id, ref_id), slot in bucket.items():
        per_query[query_id].append((query_id, ref_id, slot))
    rows = []
    for query_id, items in per_query.items():
        def rank(item: tuple) -> tuple:
            reasons = item[2]["reasons"]
            priority = 0
            if "exact_object_id" in reasons:
                priority = 4
            elif "exact_title" in reasons:
                priority = 3
            elif "title_jaccard_0.8" in reasons or "title_jaccard_0.5_artist" in reasons:
                priority = 2
            elif "image_cosine" in reasons:
                priority = 1
            return (priority, item[2]["cosine"] or 0.0)

        items.sort(key=rank, reverse=True)
        for query_id, ref_id, slot in items[:PAIR_CAP]:
            query = next(card for card in queries if card["id"] == query_id)
            ref = ref_by_id[ref_id]
            rows.append(
                {
                    "query_set": query_set,
                    "query_id": query_id,
                    "ref_id": ref_id,
                    "reasons": sorted(slot["reasons"]),
                    "cosine": slot["cosine"],
                    "title_jaccard": round(jaccard(query.get("title") or "", ref.get("title") or ""), 4),
                    "artist_overlap": artist_overlap(query.get("artist"), ref.get("artist")),
                    "query_title": query.get("title") or "",
                    "query_artist": query.get("artist") or "",
                    "query_year": query.get("year"),
                    "ref_title": ref.get("title") or "",
                    "ref_artist": ref.get("artist") or "",
                    "ref_year": ref.get("year"),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("fetching met", flush=True)
    mets = met_cards()
    write_jsonl(OUT / "cards_met.jsonl", mets)
    print("met cards", len(mets), flush=True)
    print("fetching wikidata", flush=True)
    wik = wikidata_cards()
    write_jsonl(OUT / "cards_wikidata.jsonl", wik)
    print("wikidata cards", len(wik), "with title", sum(1 for c in wik if c["title"]), flush=True)
    met_by_object = {card["object_id"]: card for card in mets}
    print("fetching syn museum catalogs", flush=True)
    syn = syn_cards(met_by_object)
    write_jsonl(OUT / "cards_syn.jsonl", syn)
    titled = sum(1 for card in syn if card["title"])
    print("syn cards", len(syn), "with title", titled, flush=True)
    wik_images = {}
    syn_images = {}
    # Wikidata images are file ids. Attach image cosine via any file of the class.
    wik_by_file: dict[str, list[tuple[str, float]]] = {}
    for card in wik:
        merged: list[tuple[str, float]] = []
        for file_id in card["image_ids"]:
            merged.extend(wik_images.get(f"set_a/{file_id}.jpg", []))
            merged.extend(wik_images.get(file_id, []))
            merged.extend(wik_images.get(f"set_a/{file_id}", []))
        card_images = {card["id"]: merged}
        wik_by_file.update(card_images)
        card["image_ids"] = [card["id"]]
    # Rebuild wik image map under qid after inspecting one candidate key.
    sample_key = next(iter(wik_images), "")
    print("wik image key sample", sample_key, flush=True)
    qid_images: dict[str, list[tuple[str, float]]] = defaultdict(list)
    file_to_qid = {}
    for card in wik:
        # image_ids were replaced; recover from the jsonl we just wrote
        pass
    stored = [json.loads(line) for line in (OUT / "cards_wikidata.jsonl").read_text().splitlines()]
    for card in stored:
        for file_id in card["image_ids"]:
            file_to_qid[file_id] = card["id"]
            file_to_qid[Path(file_id).name] = card["id"]
    for key, pairs in wik_images.items():
        name = Path(key).name
        stem = name[:-4] if name.endswith(".jpg") else name
        qid = file_to_qid.get(stem) or file_to_qid.get(name) or file_to_qid.get(key)
        if qid:
            qid_images[qid].extend(pairs)
    for card in stored:
        card["image_ids"] = [card["id"]]
    wik_rows = block(stored, mets, qid_images, "wikidata")
    syn_rows = block(syn, mets, syn_images, "syn")
    write_jsonl(OUT / "blocked_wikidata_vs_met.jsonl", wik_rows)
    write_jsonl(OUT / "blocked_syn_vs_met.jsonl", syn_rows)
    summary = {
        "met_cards": len(mets),
        "met_with_title": sum(1 for card in mets if card["title"]),
        "wikidata_cards": len(stored),
        "wikidata_with_title": sum(1 for card in stored if card["title"]),
        "syn_cards": len(syn),
        "syn_with_title": titled,
        "wikidata_pairs": len(wik_rows),
        "syn_pairs": len(syn_rows),
        "wikidata_reason_counts": count_reasons(wik_rows),
        "syn_reason_counts": count_reasons(syn_rows),
        "rules": {
            "exact_object_id": "SYN metmuseum id equals Met file stem",
            "exact_title": "normalized official titles equal and have at least two tokens",
            "title_jaccard_0.8": "title token Jaccard >= 0.80",
            "title_jaccard_0.5_artist": "title token Jaccard >= 0.50 and a shared artist token",
            "image_cosine": f"DINOv3 cosine >= {IMAGE_KEEP}, or the top neighbor if >= {IMAGE_TOP}, cap {IMAGE_CAP}",
            "pair_cap": PAIR_CAP,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def count_reasons(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for reason in row["reasons"]:
            counts[reason] += 1
    return dict(counts)


if __name__ == "__main__":
    main()
