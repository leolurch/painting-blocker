#!/usr/bin/env python3
"""Block Lost Art catalog rows against Wikidata titles and artists.

Image bytes are not touched. A later image run can add pairs. This pass only
writes three pair files: obvious false positives, likely matches, and the
uncertain pairs that still need a person.

Blocking, in priority order:

1. Normalized titles equal and contain at least two tokens.
2. Title-token Jaccard at least 0.80.
3. Jaccard at least 0.50 and a shared artist token of length at least 4.

An obvious false positive is a specific title overlap where both sides name an
artist and those artist tokens do not overlap. Generic titles and missing
artists stay uncertain. A likely match needs a near-exact specific title and a
shared non-generic artist token.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

_STOP = {"a", "an", "the", "of", "and", "le", "la", "les", "el", "der", "die", "das", "des", "den", "dem", "ein", "eine", "und", "von", "van", "de"}
_GENERIC_ARTIST = {
    "master", "painter", "andrea", "giovanni", "unknown", "unbekannt", "unbekannte",
    "school", "schule", "werkstatt", "nachfolger", "artist", "maler", "anonymous",
    "anonym", "copy", "kopie", "workshop", "follower", "attributed",
}
_GENERIC_TITLE = {
    "portrait of a man", "portrait of a woman", "portrait of a lady", "portrait of a girl",
    "madonna and child", "madonna with child", "crucifixion", "the crucifixion",
    "landscape", "a landscape", "still life", "self portrait", "head of a man",
    "head of a woman", "flowers", "study", "bildnis", "landschaft", "stillleben",
    "kreuzigung", "madonnenbild",
}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_OBJ_G = re.compile(r'"Objektgruppe": "((?:\\.|[^"\\])*)"')


def normalize(text: str | None) -> str:
    if not text or text == r"\N":
        return ""
    value = unicodedata.normalize("NFKD", text)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = _NON_ALNUM.sub(" ", value.lower()).strip()
    return " ".join(tok for tok in value.split() if tok not in _STOP)


def tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def specific_artist_tokens(name: str) -> set[str]:
    return {tok for tok in tokens(name) if len(tok) >= 4 and tok not in _GENERIC_ARTIST}


def generic_title(text: str) -> bool:
    norm = normalize(text)
    if not norm:
        return True
    if norm in _GENERIC_TITLE:
        return True
    return len(norm.split()) < 3


def iter_copy(path: Path, marker: str):
    in_copy = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not in_copy:
                if line.startswith(marker):
                    in_copy = True
                continue
            if line.startswith("\\."):
                break
            yield line.rstrip("\n")


def load_wikidata_cards(cache_dir: Path) -> list[dict]:
    cards = []
    artist_qids: list[str] = []
    for path in sorted(cache_dir.glob("*.json")):
        body = (json.loads(path.read_text()).get("body") or {})
        for qid, entity in (body.get("entities") or {}).items():
            label = ((entity.get("labels") or {}).get("en") or {}).get("value") or ""
            claims = entity.get("claims") or {}
            p170 = []
            for claim in claims.get("P170") or []:
                try:
                    p170.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                except (KeyError, TypeError):
                    continue
            year = None
            for claim in claims.get("P571") or []:
                try:
                    year = int(claim["mainsnak"]["datavalue"]["value"]["time"][1:5])
                    break
                except (KeyError, TypeError, ValueError):
                    continue
            artist_qids.extend(p170)
            cards.append({"qid": qid, "title": label, "year": year, "artist_qids": p170, "artist": ""})
    names = fetch_artist_names(sorted(set(artist_qids)))
    for card in cards:
        card["artist"] = "; ".join(names.get(qid, "") for qid in card["artist_qids"] if names.get(qid))
    return cards


def fetch_artist_names(qids: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for start in range(0, len(qids), 50):
        chunk = qids[start : start + 50]
        url = (
            "https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels"
            f"&languages=en&format=json&ids={urllib.parse.quote('|'.join(chunk))}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "lostart-metadata/1.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            body = json.loads(response.read().decode("utf-8"))
        for qid, entity in (body.get("entities") or {}).items():
            label = ((entity.get("labels") or {}).get("en") or {}).get("value")
            if label:
                names[qid] = label
    return names


def load_lostart(sql_path: Path) -> list[dict]:
    artists: dict[str, str] = {}
    for line in iter_copy(sql_path, "COPY sm_import_artist "):
        parts = line.split("\t")
        if len(parts) >= 2:
            artists[parts[0]] = parts[1]
    images: dict[str, str] = {}
    for line in iter_copy(sql_path, "COPY sm_import_image_file "):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("lostart/"):
            images[parts[0]] = Path(parts[1]).name
    art_files: dict[str, list[str]] = defaultdict(list)
    for line in iter_copy(sql_path, "COPY sm_import_lost_artwork_image_file "):
        art_id, image_id = line.split("\t")[:2]
        name = images.get(image_id)
        if name:
            art_files[art_id].append(name)
    rows = []
    for line in iter_copy(sql_path, "COPY sm_import_lost_artwork "):
        parts = line.split("\t")
        if len(parts) < 35:
            continue
        files = art_files.get(parts[0])
        if not files:
            continue
        raw = parts[34]
        group = _OBJ_G.search(raw)
        artist_ids = [item for item in parts[3].strip("{}").split(",") if item and item != "NULL"]
        artist = "; ".join(artists.get(item, "") for item in artist_ids if artists.get(item))
        title_en = "" if parts[2] == r"\N" else parts[2]
        year = None
        if parts[12] not in (r"\N", ""):
            try:
                year = int(parts[12])
            except ValueError:
                year = None
        rows.append(
            {
                "lost_artwork_id": parts[0],
                "lost_art_id": "" if parts[27] == r"\N" else parts[27],
                "title": "" if parts[1] == r"\N" else parts[1],
                "title_en": title_en,
                "artist": artist,
                "year": year,
                "object_group": group.group(1) if group else "",
                "files": sorted(set(files)),
            }
        )
    return rows


def titles_of(row: dict) -> list[str]:
    found = []
    for key in ("title", "title_en"):
        text = row.get(key) or ""
        if text and text not in found:
            found.append(text)
    return found


def classify(lost_title: str, wiki_title: str, lost_artist: str, wiki_artist: str, score: float, rule: str) -> str:
    lost_tokens = specific_artist_tokens(lost_artist)
    wiki_tokens = specific_artist_tokens(wiki_artist)
    shared = lost_tokens & wiki_tokens
    specific = not generic_title(lost_title) and not generic_title(wiki_title)
    # The same title on two named artists is a different work, including short
    # subject titles such as Danae or Flora.
    if lost_tokens and wiki_tokens and not shared and score >= 0.8:
        return "obvious_false_positive"
    if specific and shared and (score >= 0.9 or rule == "exact_title"):
        return "likely"
    return "uncertain"


def block(lost_rows: list[dict], wiki_cards: list[dict]) -> list[dict]:
    pairs = []
    for lost in lost_rows:
        lost_titles = titles_of(lost)
        if not lost_titles:
            continue
        for card in wiki_cards:
            best_score = 0.0
            best_title = ""
            best_wiki = card["title"]
            rule = ""
            for lost_title in lost_titles:
                score = jaccard(lost_title, card["title"])
                exact = normalize(lost_title) == normalize(card["title"]) and len(tokens(lost_title)) >= 2
                if exact:
                    best_score = 1.0
                    best_title = lost_title
                    rule = "exact_title"
                    break
                if score > best_score:
                    best_score = score
                    best_title = lost_title
            if rule == "exact_title":
                pass
            elif best_score >= 0.8:
                rule = "jaccard_0.80"
            elif best_score >= 0.5 and specific_artist_tokens(lost["artist"]) & specific_artist_tokens(card["artist"]):
                rule = "jaccard_0.50_shared_artist"
            else:
                continue
            verdict = classify(best_title, best_wiki, lost["artist"], card["artist"], best_score, rule)
            pairs.append(
                {
                    "verdict": verdict,
                    "rule": rule,
                    "jaccard": round(best_score, 4),
                    "lost_artwork_id": lost["lost_artwork_id"],
                    "lost_art_id": lost["lost_art_id"],
                    "lost_title": best_title,
                    "lost_artist": lost["artist"],
                    "lost_year": lost["year"],
                    "lost_object_group": lost["object_group"],
                    "lost_files": lost["files"],
                    "wikidata_qid": card["qid"],
                    "wikidata_title": best_wiki,
                    "wikidata_artist": card["artist"],
                    "wikidata_year": card["year"],
                }
            )
    return pairs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", type=Path, required=True)
    parser.add_argument("--wikidata-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cards = load_wikidata_cards(args.wikidata_cache)
    lost_rows = load_lostart(args.sql)
    pairs = block(lost_rows, cards)
    buckets = {
        "obvious_false_positive": [row for row in pairs if row["verdict"] == "obvious_false_positive"],
        "likely": [row for row in pairs if row["verdict"] == "likely"],
        "uncertain": [row for row in pairs if row["verdict"] == "uncertain"],
    }
    for name, rows in buckets.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
    summary = {
        "lostart_records_with_images": len(lost_rows),
        "wikidata_classes": len(cards),
        "blocked_pairs": len(pairs),
        "obvious_false_positives": len(buckets["obvious_false_positive"]),
        "likely": len(buckets["likely"]),
        "uncertain": len(buckets["uncertain"]),
        "removal_side": "lostart_only",
        "wikidata_unchanged": True,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
