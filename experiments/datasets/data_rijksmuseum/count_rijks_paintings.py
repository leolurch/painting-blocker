#!/usr/bin/env python3
"""Estimate how many reusable paintings the Rijksmuseum could supply.

This talks to the modern Rijksmuseum "Data Services" Search API, which is public
and needs no API key:

    https://data.rijksmuseum.nl/search/collection

The endpoint returns a Linked-Art / ActivityStreams ``OrderedCollectionPage``
whose ``partOf.totalItems`` field is the total match count for a query. That
lets us read headline counts with a single request each, instead of harvesting
the whole ~800k-object collection.

What we measure, mirroring the target filter for the pipeline:

  1. object type == painting                          (``type=painting``)
  2. a principal digital image exists                 (``imageAvailable=true``)
  3. the image right permits reuse                    (sampled from records)
  4. a single two-dimensional work per object         (implicit: we count
     objects, not images, so verso/detail/frame extra images do not inflate
     the number)

Rights are NOT a server-side search parameter, so step 3 is estimated by
sampling object records, following ``shows`` -> VisualItem, and reading the
``subject_to`` -> Right -> ``classified_as`` license URL. The reusable fraction
is then applied to the image-available count.

Usage
-----
    # Headline counts only (a handful of requests, seconds):
    python count_rijks_paintings.py

    # Also sample 300 records to estimate the reusable fraction:
    python count_rijks_paintings.py --sample 300

    # Write a machine-readable summary:
    python count_rijks_paintings.py --sample 300 --json-out rijks_painting_counts.json
"""
import argparse
import json
import time
from collections import Counter

import requests

SEARCH_URL = "https://data.rijksmuseum.nl/search/collection"

# License URLs (on a VisualItem's Right.classified_as) that permit reuse.
# The Rijksmuseum publishes freely reusable reproductions under Public Domain
# Mark / CC0; CC-BY is included as reuse-permitting-with-attribution.
REUSE_LICENSES = {
    "https://creativecommons.org/publicdomain/mark/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by-sa/4.0/",
}


def get_json(session, url, params=None, retries=4, backoff=1.5):
    """GET a URL and parse JSON, retrying transient failures with backoff."""
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = session.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", delay))
                time.sleep(retry_after)
                delay *= backoff
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= backoff
    return None


def count_total(session, params):
    """Return ``partOf.totalItems`` for a search query (one request)."""
    data = get_json(session, SEARCH_URL, params=params)
    return int(data.get("partOf", {}).get("totalItems", 0))


def iter_object_ids(session, params, limit):
    """Yield up to ``limit`` object IDs, paging via ``next.id`` (pageToken)."""
    url = SEARCH_URL
    query = dict(params)
    seen = 0
    while url and seen < limit:
        data = get_json(session, url, params=query)
        query = None  # pageToken is baked into subsequent next.id URLs
        for item in data.get("orderedItems", []):
            yield item["id"]
            seen += 1
            if seen >= limit:
                return
        url = (data.get("next") or {}).get("id")


def extract_reuse_license(session, object_id):
    """Follow object -> shows -> VisualItem and return its image license URL.

    Returns the license URL string, or None if the object has no VisualItem or
    no classified license on it.
    """
    obj = get_json(session, object_id)
    shows = obj.get("shows") or []
    if not shows:
        return None
    visual = get_json(session, shows[0]["id"])
    for right in visual.get("subject_to") or []:
        for cls in right.get("classified_as") or []:
            if cls.get("id"):
                return cls["id"]
    return None


def sample_reuse_fraction(session, n, rate_limit):
    """Sample n image-bearing paintings and tally their image licenses.

    Returns (license_counter, sampled, reusable) where ``reusable`` counts
    samples whose license is in ``REUSE_LICENSES``.
    """
    params = {"type": "painting", "imageAvailable": "true"}
    licenses = Counter()
    sampled = 0
    reusable = 0
    interval = 1.0 / rate_limit if rate_limit > 0 else 0.0
    for object_id in iter_object_ids(session, params, n):
        if interval:
            time.sleep(interval)
        lic = extract_reuse_license(session, object_id)
        sampled += 1
        licenses[lic or "<none>"] += 1
        if lic in REUSE_LICENSES:
            reusable += 1
        if sampled % 25 == 0:
            print(f"  sampled {sampled}/{n} (reusable so far: {reusable})")
    return licenses, sampled, reusable


def main():
    parser = argparse.ArgumentParser(
        description="Count reusable Rijksmuseum paintings via the public Search API."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample this many image-bearing paintings to estimate the reusable "
        "(public-domain) fraction. 0 = headline counts only.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=8.0,
        help="Max record requests per second while sampling (default: 8).",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write the summary as JSON.",
    )
    args = parser.parse_args()

    session = requests.Session()

    # --- Headline counts: one request each ---
    counts = {
        "paintings_total": count_total(session, {"type": "painting"}),
        "paintings_with_image": count_total(
            session, {"type": "painting", "imageAvailable": "true"}
        ),
    }
    # A couple of illustrative sub-filters for material breakdown.
    breakdown = {
        "oil_paint": count_total(
            session, {"type": "painting", "material": "oil paint"}
        ),
        "canvas": count_total(session, {"type": "painting", "material": "canvas"}),
        "panel": count_total(session, {"type": "painting", "material": "panel"}),
    }

    print("=" * 60)
    print("Rijksmuseum painting counts (Search API, no key required)")
    print("=" * 60)
    print(f"  Paintings (type=painting):            {counts['paintings_total']:>7,}")
    print(f"  ...with a digital image:              {counts['paintings_with_image']:>7,}")
    print("  material breakdown (of all paintings):")
    for name, val in breakdown.items():
        print(f"    - {name:<10}                        {val:>7,}")

    summary = {"counts": counts, "material_breakdown": breakdown}

    # --- Reusable-fraction estimate via record sampling ---
    if args.sample > 0:
        print(f"\nSampling {args.sample} records to estimate reusable fraction...")
        licenses, sampled, reusable = sample_reuse_fraction(
            session, args.sample, args.rate_limit
        )
        fraction = reusable / sampled if sampled else 0.0
        estimated = round(counts["paintings_with_image"] * fraction)
        print("\n  License distribution in sample:")
        for lic, cnt in licenses.most_common():
            print(f"    {cnt:>4}  {lic}")
        print(f"\n  Reusable in sample:   {reusable}/{sampled} ({fraction:.1%})")
        print(
            f"  => Estimated reusable paintings with image: ~{estimated:,} "
            f"(of {counts['paintings_with_image']:,})"
        )
        summary["reuse_estimate"] = {
            "sampled": sampled,
            "reusable": reusable,
            "reusable_fraction": fraction,
            "estimated_reusable_with_image": estimated,
            "license_distribution": dict(licenses),
        }
    else:
        print(
            "\n(Run with --sample N to estimate how many of the image-bearing "
            "paintings are reusable/public-domain.)"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to {args.json_out}")


if __name__ == "__main__":
    main()
