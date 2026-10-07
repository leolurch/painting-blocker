#!/usr/bin/env python3
"""Count reusable National Gallery of Art (NGA) paintings from the open-data repo.

Unlike the Rijksmuseum pipeline (a live Linked-Art Search API), the NGA publishes
its collection as a versioned CSV repository on GitHub -- no API key, no API at
all -- so "listing/discovery" here means reading two flat files and joining them:

    objects.csv            one row per artwork  (objectid, classification, ...)
    published_images.csv   one row per image    (uuid, iiifurl, viewtype,
                                                  openaccess, width, height,
                                                  depictstmsobjectid -> objectid)

The candidate filter that mirrors the download pipeline:

  1. object classification == "Painting"            (objects.classification)
  2. a *primary* published image exists             (published_images.viewtype)
  3. the image is open access                        (published_images.openaccess == 1)

The NGA releases every open-access image under CC0 / for unrestricted reuse, so
unlike the Rijksmuseum there is no per-record license URL to resolve: the single
``openaccess`` flag *is* the rights filter (group 2 of the porting blueprint
collapses to one boolean). One object may have several primary rows (multiple
scans); we keep the lowest ``sequence`` so we count objects, not images.

Every downloadable painting is served by the NGA's IIIF Image API at
``{iiifurl}/full/{size}/0/default.jpg``. The CSV already carries each image's
native ``width``/``height``, which the downloader uses to pick the size
parameter without probing the server.

This module also holds the constants and the ``iter_painting_images`` candidate
iterator that ``download_nga_paintings.py`` imports, mirroring the way the Rijks
downloader reuses ``count_rijks_paintings``.

Usage
-----
    # Headline counts (downloads/caches the two CSVs on first run):
    python count_nga_paintings.py

    # Point at an existing cache dir and write a machine-readable summary:
    python count_nga_paintings.py --cache-dir nga_data --json-out nga_counts.json
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import requests

# Raw CSV endpoints of the official open-data repository (keyless, public).
# https://github.com/NationalGalleryOfArt/opendata
DATA_BASE_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data"
OBJECTS_CSV_URL = f"{DATA_BASE_URL}/objects.csv"
IMAGES_CSV_URL = f"{DATA_BASE_URL}/published_images.csv"

OBJECTS_CSV_NAME = "objects.csv"
IMAGES_CSV_NAME = "published_images.csv"

# objects.classification value for two-dimensional paintings.
PAINTING_CLASSIFICATION = "Painting"

# Only the object's principal reproduction; "alternate" rows are versos, details,
# frame shots, etc. and would inflate an object count.
PRIMARY_VIEWTYPE = "primary"

# Every open-access NGA image is released under CC0; this is the rights label we
# record for downloaded files (there is no per-image license URL in the data).
OPEN_ACCESS_LICENSE = "CC0-1.0 (NGA open access)"

# The published CSVs allow very long free-text fields (provenance runs to
# thousands of characters); the stdlib default field cap would raise on them.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def ensure_csv(url, path, session=None, refresh=False, chunk_size=1 << 20):
    """Download ``url`` to ``path`` once (atomic), or reuse the cached copy.

    Idempotent like the image downloads: an existing non-empty file is reused
    unless ``refresh`` is set, and the download streams to ``*.part`` then
    ``os.replace`` so an interrupted fetch never leaves a truncated CSV behind.
    """
    if not refresh and os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    session = session or requests.Session()
    tmp_path = path + ".part"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading {os.path.basename(path)} ...", flush=True)
    with session.get(url, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    if os.path.getsize(tmp_path) == 0:
        os.remove(tmp_path)
        raise IOError(f"empty download for {url}")
    os.replace(tmp_path, path)
    return path


def load_painting_object_ids(objects_csv_path):
    """Return the set of ``objectid`` values classified as paintings.

    Streams the (large) objects CSV keeping only the id of matching rows, so
    memory stays small regardless of the ~145k total rows.
    """
    ids = set()
    with open(objects_csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("classification") == PAINTING_CLASSIFICATION:
                oid = (row.get("objectid") or "").strip()
                if oid:
                    ids.add(oid)
    return ids


def iter_painting_images(cache_dir, session=None, refresh=False):
    """Yield one candidate image dict per painting with a primary OA image.

    Ensures both CSVs are cached, then joins them: for every painting object,
    yields the lowest-``sequence`` primary open-access image. Each candidate is::

        {objectid, uuid, iiif_url, width, height, sequence, license, image_row}

    Deduplication keeps objects (not images) as the unit, matching the counter.
    """
    objects_path = os.path.join(cache_dir, OBJECTS_CSV_NAME)
    images_path = os.path.join(cache_dir, IMAGES_CSV_NAME)
    ensure_csv(OBJECTS_CSV_URL, objects_path, session=session, refresh=refresh)
    ensure_csv(IMAGES_CSV_URL, images_path, session=session, refresh=refresh)

    painting_ids = load_painting_object_ids(objects_path)

    # Keep the best (lowest-sequence) primary open-access image per object.
    best = {}
    with open(images_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            oid = (row.get("depictstmsobjectid") or "").strip()
            if oid not in painting_ids:
                continue
            if row.get("viewtype") != PRIMARY_VIEWTYPE:
                continue
            if (row.get("openaccess") or "").strip() != "1":
                continue
            iiif_url = (row.get("iiifurl") or "").strip()
            if not iiif_url:
                continue
            try:
                seq = int(row.get("sequence") or 0)
            except ValueError:
                seq = 0
            prev = best.get(oid)
            if prev is None or seq < prev[0]:
                best[oid] = (seq, row)

    for oid, (seq, row) in best.items():
        yield {
            "objectid": oid,
            "uuid": (row.get("uuid") or "").strip(),
            "iiif_url": (row.get("iiifurl") or "").strip(),
            "width": _as_int(row.get("width")),
            "height": _as_int(row.get("height")),
            "sequence": seq,
            "license": OPEN_ACCESS_LICENSE,
        }


def _as_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Count reusable NGA paintings from the open-data CSV repository."
    )
    parser.add_argument(
        "--cache-dir",
        default="nga_data",
        help="Directory to cache objects.csv / published_images.csv (default: nga_data).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the CSVs even if cached copies exist.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write the summary as JSON.",
    )
    args = parser.parse_args()

    session = requests.Session()
    objects_path = os.path.join(args.cache_dir, OBJECTS_CSV_NAME)
    images_path = os.path.join(args.cache_dir, IMAGES_CSV_NAME)
    ensure_csv(OBJECTS_CSV_URL, objects_path, session=session, refresh=args.refresh)
    ensure_csv(IMAGES_CSV_URL, images_path, session=session, refresh=args.refresh)

    # --- Headline counts, computed while streaming the CSVs ---
    classifications = Counter()
    with open(objects_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            classifications[row.get("classification") or "<none>"] += 1
    paintings_total = classifications.get(PAINTING_CLASSIFICATION, 0)
    painting_ids = load_painting_object_ids(objects_path)

    primary_oa_paintings = set()
    any_oa_paintings = set()
    with open(images_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            oid = (row.get("depictstmsobjectid") or "").strip()
            if oid not in painting_ids:
                continue
            if (row.get("openaccess") or "").strip() == "1":
                any_oa_paintings.add(oid)
                if row.get("viewtype") == PRIMARY_VIEWTYPE:
                    primary_oa_paintings.add(oid)

    print("=" * 62)
    print("National Gallery of Art painting counts (open-data CSVs, no key)")
    print("=" * 62)
    print(f"  Objects (all classifications):          {sum(classifications.values()):>7,}")
    print(f"  Paintings (classification=Painting):    {paintings_total:>7,}")
    print(f"  ...with any open-access image:          {len(any_oa_paintings):>7,}")
    print(f"  ...with a PRIMARY open-access image:    {len(primary_oa_paintings):>7,}")
    print("  (the last line is what the downloader will fetch)")

    summary = {
        "objects_total": sum(classifications.values()),
        "paintings_total": paintings_total,
        "paintings_with_any_open_access_image": len(any_oa_paintings),
        "paintings_with_primary_open_access_image": len(primary_oa_paintings),
        "classification_distribution": dict(classifications),
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to {args.json_out}")


if __name__ == "__main__":
    main()
