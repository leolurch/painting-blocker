#!/usr/bin/env python3
"""Offline unit tests for the ArtIC paintings downloader (no network required).

Covers the source-specific edge branches:
  * choose_urls: wide source -> server downscale, narrow/unknown -> original
  * ensure_max_width: downscales when wider than the cap, never upscales
  * build_candidate: IIIF base + native width from the search payload, skip null
  * plan_id_buckets: id-range partition keeps every bucket under the cap window

Run with ``pytest test_artic_downloader.py`` or ``python test_artic_downloader.py``.
"""
import os
import tempfile

from PIL import Image

import count_artic_paintings as counter
import download_artic_paintings as dl


def test_choose_urls_wide_prefers_server_downscale():
    preferred, fallback = dl.choose_urls(
        {"iiif_url": "https://www.artic.edu/iiif/2/abc", "width": 12614}
    )
    assert preferred.endswith("/full/2048,/0/default.jpg")
    assert fallback.endswith("/full/full/0/default.jpg")


def test_choose_urls_narrow_prefers_original():
    # Below the cap: fetch the original so we never ask ArtIC to upscale (403).
    preferred, fallback = dl.choose_urls(
        {"iiif_url": "https://www.artic.edu/iiif/2/xyz", "width": 1152}
    )
    assert preferred.endswith("/full/full/0/default.jpg")
    assert fallback.endswith("/full/2048,/0/default.jpg")


def test_choose_urls_unknown_width_treated_as_wide():
    preferred, _ = dl.choose_urls(
        {"iiif_url": "https://www.artic.edu/iiif/2/nnn", "width": None}
    )
    assert preferred.endswith("/full/2048,/0/default.jpg")


def test_ensure_max_width_downscales():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "big.jpg")
        Image.new("RGB", (4000, 3000)).save(path)
        dl.ensure_max_width(path)
        assert Image.open(path).size == (2048, 1536)


def test_ensure_max_width_never_upscales():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "small.jpg")
        Image.new("RGB", (800, 600)).save(path)
        dl.ensure_max_width(path)
        assert Image.open(path).size == (800, 600)


def test_build_candidate_maps_fields():
    iiif = "https://www.artic.edu/iiif/2"
    artwork = {
        "id": 28560,
        "title": "The Bedroom",
        "image_id": "6644829f-image",
        "is_public_domain": True,
        "artwork_type_title": "Painting",
        "thumbnail": {"width": 12614, "height": 9875},
    }
    cand = counter.build_candidate(artwork, iiif)
    assert cand["objectid"] == "28560"
    assert cand["image_id"] == "6644829f-image"
    assert cand["iiif_url"] == "https://www.artic.edu/iiif/2/6644829f-image"
    assert cand["width"] == 12614
    assert cand["height"] == 9875
    assert cand["title"] == "The Bedroom"
    assert cand["license"] == counter.OPEN_ACCESS_LICENSE


def test_build_candidate_skips_null_image():
    cand = counter.build_candidate(
        {"id": 1, "image_id": None, "thumbnail": {}}, "https://www.artic.edu/iiif/2"
    )
    assert cand is None


def test_plan_id_buckets_partitions_under_window():
    # 1500 painting ids evenly spread over [0, 3000); each is present once.
    present = set(range(0, 3000, 2))  # 1500 ids

    def count_fn(lo, hi):
        return sum(1 for i in present if lo <= i < hi)

    buckets = counter.plan_id_buckets(count_fn, 0, 3000, window=1000)
    # Every bucket is under the window, and the union covers all ids exactly once.
    assert all(count_fn(lo, hi) <= 1000 for lo, hi in buckets)
    total = sum(count_fn(lo, hi) for lo, hi in buckets)
    assert total == len(present)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
