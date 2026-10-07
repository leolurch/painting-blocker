#!/usr/bin/env python3
"""Count reusable Cleveland Museum of Art (CMA) paintings via the Open Access API.

Cleveland exposes a keyless REST API whose records carry the image assets
*inline* -- so, unlike the Art Institute of Chicago (a IIIF Image server) or the
Rijksmuseum (a Linked-Art crawl), there is **no IIIF resize server to resolve**.
Each record ships direct CDN URLs at three fixed renditions:

    https://openaccess-api.clevelandart.org/api/artworks/   (keyless REST)
    -> images.web   : small JPEG (<= ~1300px on the long side)
    -> images.print : large JPEG capped at 3400px on the long side
    -> images.full  : uncompressed TIFF (tens of MB) -- overkill, not JPEG

The candidate filter that mirrors the download pipeline is three query params:

  1. type == "Painting"       (maps to classification_type == Painting)
  2. cc0 == 1                 (the rights filter -- see below)
  3. has_image == 1           (a source image asset exists)

Rights: every CC0 record is released for unrestricted commercial and
noncommercial reuse, so -- exactly like the NGA's ``openaccess`` flag or ArtIC's
``is_public_domain`` boolean -- the single ``cc0=1`` filter *is* the whole rights
filter (there is no per-record license URL to resolve). We also assert
``share_license_status == "CC0"`` per record as a belt-and-braces check.

Listing (group 1 is source-specific): the API pages cleanly with ``skip`` +
``limit`` and, unlike ArtIC, imposes **no result-window cap** -- a deep
``skip`` near the total still returns rows -- so we simply walk ``skip`` in
pages of ``PAGE_LIMIT`` until the reported ``total`` is exhausted. ``fields``
trims the payload to what the downloader needs.

Image sizing (group 3 is source-specific too, and this is the real deviation
from the IIIF loaders): because CMA serves *pre-rendered* JPEGs rather than an
on-the-fly resize server, the downloader cannot ask for ``/full/2048,/...``.
Instead each candidate carries the direct ``print`` URL (preferred) and ``web``
URL (fallback); the downloader fetches ``print`` and lets Pillow downscale it to
<= 2048px (never upscaling). The ``print`` rendition is always >= ``web`` and is
JPEG, so it is always the best source. We still record the ``print`` pixel
dimensions for observability.

This module also holds the constants and the ``iter_painting_images`` candidate
iterator that ``download_cleveland_paintings.py`` imports, mirroring the way the
ArtIC, NGA and Rijks downloaders reuse their counters.

Usage
-----
    # Headline counts (a handful of API requests, seconds):
    python count_cleveland_paintings.py

    # Write a machine-readable summary:
    python count_cleveland_paintings.py --json-out cleveland_counts.json
"""
import argparse
import json
import time

import requests

# Public Open Access REST API of the Cleveland Museum of Art (keyless).
# https://openaccess-api.clevelandart.org/
API_BASE = "https://openaccess-api.clevelandart.org/api"
ARTWORKS_URL = f"{API_BASE}/artworks/"

# CMA has no stated User-Agent requirement, but we identify ourselves anyway.
USER_AGENT = "painting-dataset-builder/1.0"

# ``type`` value for paintings (the API maps it to classification_type=Painting).
PAINTING_TYPE = "Painting"

# Every CC0 CMA image is released for unrestricted reuse; this is the rights
# label we record for downloaded files (there is no per-image license URL).
OPEN_ACCESS_LICENSE = "CC0 1.0 (Cleveland Museum of Art, public domain)"

# The per-record field CMA sets to "CC0" for public-domain works; we assert it
# per candidate as a second check on top of the ``cc0=1`` query filter.
CC0_STATUS = "CC0"

# Fields we request per artwork (keep the payload small).
API_FIELDS = ["id", "accession_number", "title", "share_license_status", "images"]

# CMA paging: rows per page. The API accepts up to 1000 and imposes no
# result-window cap, so we can walk the whole set by ``skip`` alone.
PAGE_LIMIT = 1000


def make_session(user_agent=USER_AGENT):
    """A requests session pre-loaded with the headers we send to CMA."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


def _base_params():
    """Query params that isolate CC0 paintings with an image asset."""
    return {
        "type": PAINTING_TYPE,
        "cc0": 1,
        "has_image": 1,
    }


def fetch_page(session, skip, limit, retries=4):
    """GET one page of CC0 paintings; return ``(rows, total)`` (with backoff).

    Retries transient failures and honours 429/Retry-After, mirroring the JSON
    fetch helpers in the ArtIC and Rijks counters.
    """
    params = _base_params()
    params.update({"skip": skip, "limit": limit, "fields": ",".join(API_FIELDS)})
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = session.get(ARTWORKS_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", delay)))
                delay *= 1.5
                continue
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data") or []
            total = ((data.get("info") or {}).get("total")) or 0
            return rows, total
        except (requests.RequestException, ValueError):
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= 1.5
    return [], 0


def _rendition_url(images, name):
    """Direct URL of the ``name`` rendition (web/print/full), or None."""
    rendition = (images or {}).get(name) or {}
    url = (rendition.get("url") or "").strip()
    return url or None


def build_candidate(artwork):
    """Turn one artwork record into a candidate dict, or None.

    Returns None when the artwork is not CC0 or carries neither a ``print`` nor
    a ``web`` image asset, so a record without a usable reproduction is skipped.
    The ``print`` rendition (large JPEG, <= 3400px) is the preferred source and
    ``web`` is the fallback; the downloader downscales whatever it gets to the
    <= 2048px cap with Pillow.
    """
    if (artwork.get("share_license_status") or "").strip() != CC0_STATUS:
        return None
    images = artwork.get("images") or {}
    print_url = _rendition_url(images, "print")
    web_url = _rendition_url(images, "web")
    image_url = print_url or web_url
    if not image_url:
        return None
    fallback_url = web_url if image_url != web_url else None
    print_dims = (images.get("print") or {}) if print_url else (images.get("web") or {})
    return {
        "objectid": str(artwork.get("id")),
        "accession_number": (artwork.get("accession_number") or "").strip(),
        "image_url": image_url,
        "fallback_url": fallback_url,
        "width": _as_int(print_dims.get("width")),
        "height": _as_int(print_dims.get("height")),
        "title": (artwork.get("title") or "").strip(),
        "license": OPEN_ACCESS_LICENSE,
    }


def iter_painting_images(session=None, user_agent=USER_AGENT, fetch=None):
    """Yield one candidate image dict per CC0 painting with an image.

    Pages the API by ``skip`` until the reported ``total`` is exhausted. Each
    candidate is::

        {objectid, accession_number, image_url, fallback_url,
         width, height, title, license}

    Deduplicated by ``objectid`` so a work is yielded once. ``fetch`` is
    injectable for offline testing; by default it hits the live API.
    """
    session = session or make_session(user_agent)
    fetch = fetch or (lambda skip, limit: fetch_page(session, skip, limit))

    seen = set()
    skip = 0
    while True:
        rows, total = fetch(skip, PAGE_LIMIT)
        if not rows:
            break
        for artwork in rows:
            cand = build_candidate(artwork)
            if cand and cand["objectid"] not in seen:
                seen.add(cand["objectid"])
                yield cand
        skip += len(rows)
        if len(rows) < PAGE_LIMIT or (total and skip >= total):
            break


def _as_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Count reusable Cleveland Museum of Art paintings via the Open Access API."
    )
    parser.add_argument(
        "--user-agent", default=USER_AGENT,
        help="Value for the User-Agent header we send to CMA.",
    )
    parser.add_argument("--json-out", help="Optional path to write the summary as JSON.")
    args = parser.parse_args()

    session = make_session(args.user_agent)

    def total_for(params):
        merged = dict(params)
        merged.update({"skip": 0, "limit": 1, "fields": "id"})
        resp = session.get(ARTWORKS_URL, params=merged, timeout=30)
        resp.raise_for_status()
        return ((resp.json().get("info") or {}).get("total")) or 0

    # Headline counts: one count request each.
    paintings_all = total_for({"type": PAINTING_TYPE})
    paintings_cc0 = total_for({"type": PAINTING_TYPE, "cc0": 1, "has_image": 1})

    # Downloadable set: CC0 paintings that actually expose an image rendition.
    with_image = sum(1 for _ in iter_painting_images(session=session))

    print("=" * 64)
    print("Cleveland Museum of Art painting counts (Open Access API, no key)")
    print("=" * 64)
    print(f"  Paintings (type=Painting):                {paintings_all:>7,}")
    print(f"  ...that are CC0 with an image asset:      {paintings_cc0:>7,}")
    print(f"  ...with a usable print/web rendition:     {with_image:>7,}")
    print("  (the last line is what the downloader will fetch)")

    summary = {
        "paintings_total": paintings_all,
        "paintings_cc0_with_image": paintings_cc0,
        "paintings_cc0_with_rendition": with_image,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to {args.json_out}")


if __name__ == "__main__":
    main()
