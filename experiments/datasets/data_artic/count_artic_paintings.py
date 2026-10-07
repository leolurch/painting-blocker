#!/usr/bin/env python3
"""Count reusable Art Institute of Chicago (ArtIC) paintings via the public API.

Unlike the Rijksmuseum (a live Linked-Art Search API) or the NGA (flat open-data
CSVs), ArtIC exposes a developer-friendly REST API with an Elasticsearch-backed
search endpoint and its own IIIF Image server -- no API key required:

    https://api.artic.edu/api/v1/artworks/search   (Elasticsearch query DSL)
    https://www.artic.edu/iiif/2/{image_id}/...     (IIIF Image API 2.0)

The candidate filter that mirrors the download pipeline is a single ES query:

  1. artwork_type_title == "Painting"     (term on artwork_type_title.keyword)
  2. is_public_domain == true             (the rights filter -- see below)
  3. a source image exists                (image_id is non-null)

Rights: ArtIC releases every public-domain image under CC0 for unrestricted
reuse, so -- exactly like the NGA's ``openaccess`` flag -- the single
``is_public_domain`` boolean *is* the whole rights filter (there is no per-record
license URL to resolve).

Listing gotcha (group 1 is source-specific): the search endpoint caps pagination
at ``from + size <= 1000`` (page 11 at limit 100 returns HTTP 403 "too many
results"), yet there are ~1.9k public-domain paintings. So we cannot page the
whole set directly. Instead we **partition by ``id`` range**: recursively bisect
the id interval until every bucket holds <= 1000 matches, then page each bucket
(<= 10 pages) under the cap. This stays entirely API-based -- no master data dump
-- and costs only a handful of count probes plus ~20 page requests.

Each painting is served by ArtIC's IIIF server at
``{config.iiif_url}/{image_id}/full/{size}/0/default.jpg``. The search response
carries the native pixel dimensions in ``thumbnail.width``/``thumbnail.height``,
which the downloader uses to pick the IIIF size without probing (ArtIC *refuses*
to upscale -- a ``sizeByW`` above native returns 403 -- so the width must be
known client-side).

This module also holds the constants and the ``iter_painting_images`` candidate
iterator that ``download_artic_paintings.py`` imports, mirroring the way the NGA
and Rijks downloaders reuse their counters.

Usage
-----
    # Headline counts (a handful of API requests, seconds):
    python count_artic_paintings.py

    # Write a machine-readable summary:
    python count_artic_paintings.py --json-out artic_counts.json
"""
import argparse
import json
import time

import requests

# Public REST API of the Art Institute of Chicago (keyless).
# https://api.artic.edu/docs/
API_BASE = "https://api.artic.edu/api/v1"
SEARCH_URL = f"{API_BASE}/artworks/search"

# ArtIC asks clients to identify themselves via this header (docs, "Terms").
USER_AGENT = "painting-dataset-builder/1.0"

# artwork_type_title value for paintings (matched on the .keyword sub-field).
PAINTING_TYPE = "Painting"

# Every public-domain ArtIC image is released under CC0; this is the rights
# label we record for downloaded files (there is no per-image license URL).
OPEN_ACCESS_LICENSE = "CC0-1.0 (Art Institute of Chicago, public domain)"

# Fields we request per artwork (keep the payload small).
SEARCH_FIELDS = [
    "id", "title", "image_id", "is_public_domain", "artwork_type_title", "thumbnail",
]

# ArtIC search pagination limits: at most this many results per page, and the
# Elasticsearch result window caps ``from + size`` at this many total.
PAGE_LIMIT = 100
RESULT_WINDOW = 1000


def make_session(user_agent=USER_AGENT):
    """A requests session pre-loaded with the headers ArtIC expects."""
    session = requests.Session()
    session.headers.update({"AIC-User-Agent": user_agent, "Accept": "application/json"})
    return session


def post_search(session, body, retries=4):
    """POST an Elasticsearch query body to the search endpoint (with backoff).

    Retries transient failures and honours 429/Retry-After, mirroring the JSON
    fetch helper in the Rijks counter.
    """
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = session.post(SEARCH_URL, json=body, timeout=30)
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", delay)))
                delay *= 1.5
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= 1.5
    return None


def painting_query(id_lo=None, id_hi=None):
    """Build the ES ``bool`` query for public-domain paintings.

    When an id range is given, restrict to ``id_lo <= id < id_hi`` so a large
    result set can be partitioned under the pagination window.
    """
    must = [
        {"term": {"is_public_domain": True}},
        {"term": {"artwork_type_title.keyword": PAINTING_TYPE}},
    ]
    if id_lo is not None:
        must.append({"range": {"id": {"gte": id_lo, "lt": id_hi}}})
    return {"bool": {"must": must}}


def iiif_image_identifier(iiif_url, image_id):
    """Full IIIF Image API identifier base: ``{iiif_url}/{image_id}``.

    The downloader appends ``/full/{size}/0/default.jpg`` to this.
    """
    return f"{iiif_url.rstrip('/')}/{image_id}"


def build_candidate(artwork, iiif_url):
    """Turn one search-result artwork into a candidate dict, or None.

    Returns None when the artwork has no source image (``image_id`` null), so a
    public-domain record without a reproduction is simply skipped.
    """
    image_id = (artwork.get("image_id") or "").strip()
    if not image_id:
        return None
    thumb = artwork.get("thumbnail") or {}
    return {
        "objectid": str(artwork.get("id")),
        "image_id": image_id,
        "iiif_url": iiif_image_identifier(iiif_url, image_id),
        "width": _as_int(thumb.get("width")),
        "height": _as_int(thumb.get("height")),
        "title": (artwork.get("title") or "").strip(),
        "license": OPEN_ACCESS_LICENSE,
    }


def plan_id_buckets(count_fn, lo, hi, window=RESULT_WINDOW):
    """Partition ``[lo, hi)`` into id ranges each holding <= ``window`` matches.

    ``count_fn(a, b)`` must return the number of matches with ``a <= id < b``.
    Empty ranges are dropped; a range that cannot be split further (width 1) is
    kept as-is even if it somehow exceeds the window.
    """
    out = []

    def rec(a, b):
        c = count_fn(a, b)
        if c == 0:
            return
        if c <= window or b - a <= 1:
            out.append((a, b))
            return
        mid = (a + b) // 2
        rec(a, mid)
        rec(mid, b)

    rec(lo, hi)
    return out


def _extreme_id(session, order):
    """Smallest (order='asc') or largest (order='desc') painting id."""
    data = post_search(session, {
        "query": painting_query(), "fields": ["id"], "limit": 1,
        "sort": [{"id": order}],
    })
    rows = data.get("data") or []
    return int(rows[0]["id"]) if rows else None


def iter_painting_images(session=None, user_agent=USER_AGENT):
    """Yield one candidate image dict per public-domain painting with an image.

    Discovers the IIIF base and the full id range, partitions the range under
    the pagination window, then pages each bucket. Each candidate is::

        {objectid, image_id, iiif_url, width, height, title, license}

    Deduplicated by ``objectid`` so a work is yielded once.
    """
    session = session or make_session(user_agent)

    head = post_search(session, {"query": painting_query(), "limit": 0})
    iiif_url = head["config"]["iiif_url"]
    if head["pagination"]["total"] == 0:
        return

    lo = _extreme_id(session, "asc")
    hi = _extreme_id(session, "desc") + 1

    def count_fn(a, b):
        d = post_search(session, {"query": painting_query(a, b), "limit": 0})
        return d["pagination"]["total"]

    seen = set()
    for blo, bhi in plan_id_buckets(count_fn, lo, hi):
        page = 1
        while True:
            data = post_search(session, {
                "query": painting_query(blo, bhi), "fields": SEARCH_FIELDS,
                "limit": PAGE_LIMIT, "page": page, "sort": [{"id": "asc"}],
            })
            rows = data.get("data") or []
            for artwork in rows:
                cand = build_candidate(artwork, iiif_url)
                if cand and cand["objectid"] not in seen:
                    seen.add(cand["objectid"])
                    yield cand
            if not rows or page >= data["pagination"]["total_pages"]:
                break
            page += 1


def _as_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Count reusable Art Institute of Chicago paintings via the public API."
    )
    parser.add_argument(
        "--user-agent", default=USER_AGENT,
        help="Value for the AIC-User-Agent header ArtIC asks clients to send.",
    )
    parser.add_argument("--json-out", help="Optional path to write the summary as JSON.")
    args = parser.parse_args()

    session = make_session(args.user_agent)

    # Headline counts: one count request each.
    paintings_all = post_search(
        session,
        {"query": {"term": {"artwork_type_title.keyword": PAINTING_TYPE}}, "limit": 0},
    )["pagination"]["total"]
    paintings_pd = post_search(session, {"query": painting_query(), "limit": 0})[
        "pagination"]["total"]

    # Downloadable set: public-domain paintings that actually have an image_id.
    with_image = sum(1 for _ in iter_painting_images(session=session))

    print("=" * 64)
    print("Art Institute of Chicago painting counts (public API, no key)")
    print("=" * 64)
    print(f"  Paintings (artwork_type=Painting):        {paintings_all:>7,}")
    print(f"  ...that are public domain (CC0):          {paintings_pd:>7,}")
    print(f"  ...with a source IIIF image:              {with_image:>7,}")
    print("  (the last line is what the downloader will fetch)")

    summary = {
        "paintings_total": paintings_all,
        "paintings_public_domain": paintings_pd,
        "paintings_public_domain_with_image": with_image,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to {args.json_out}")


if __name__ == "__main__":
    main()
