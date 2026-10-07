#!/usr/bin/env python3
"""Offline unit tests for the NGA paintings downloader (no network required).

Covers the source-specific edge branches:
  * choose_urls: wide source -> server downscale, narrow/unknown -> original
  * ensure_max_width: downscales when wider than the cap, never upscales
  * iter_painting_images: CSV join, primary+open-access filter, lowest-sequence dedup

Run with ``pytest test_nga_downloader.py`` or ``python test_nga_downloader.py``.
"""
import os
import tempfile

from PIL import Image

import count_nga_paintings as counter
import download_nga_paintings as dl


def test_choose_urls_wide_prefers_server_downscale():
    preferred, fallback = dl.choose_urls(
        {"iiif_url": "https://api.nga.gov/iiif/ABC", "width": 8763}
    )
    assert preferred.endswith("/full/2048,/0/default.jpg")
    assert fallback.endswith("/full/full/0/default.jpg")


def test_choose_urls_narrow_prefers_original():
    # Below the cap: fetch the original so the server never upscales it.
    preferred, fallback = dl.choose_urls(
        {"iiif_url": "https://api.nga.gov/iiif/XYZ", "width": 665}
    )
    assert preferred.endswith("/full/full/0/default.jpg")
    assert fallback.endswith("/full/2048,/0/default.jpg")


def test_choose_urls_unknown_width_treated_as_wide():
    preferred, _ = dl.choose_urls(
        {"iiif_url": "https://api.nga.gov/iiif/NNN", "width": None}
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


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def test_iter_painting_images_join_filter_and_dedup():
    objects = (
        "objectid,classification\n"
        "1,Painting\n"      # painting, has primary OA image -> kept
        "2,Painting\n"      # painting, but only an alternate/non-OA image -> dropped
        "3,Print\n"         # not a painting -> dropped
        "4,Painting\n"      # painting with two primary OA rows -> dedup to seq 0
    )
    images = (
        "uuid,iiifurl,viewtype,sequence,width,height,openaccess,depictstmsobjectid\n"
        "u1,https://api.nga.gov/iiif/u1,primary,0,3000,4000,1,1\n"
        "u2a,https://api.nga.gov/iiif/u2a,alternate,0,3000,4000,1,2\n"  # not primary
        "u2b,https://api.nga.gov/iiif/u2b,primary,0,3000,4000,0,2\n"    # not open access
        "u3,https://api.nga.gov/iiif/u3,primary,0,3000,4000,1,3\n"      # print
        "u4b,https://api.nga.gov/iiif/u4b,primary,1,3000,4000,1,4\n"    # higher seq
        "u4a,https://api.nga.gov/iiif/u4a,primary,0,3000,4000,1,4\n"    # lower seq wins
    )
    with tempfile.TemporaryDirectory() as tmp:
        _write(os.path.join(tmp, counter.OBJECTS_CSV_NAME), objects)
        _write(os.path.join(tmp, counter.IMAGES_CSV_NAME), images)
        # refresh=False + cached files present -> no network.
        cands = {c["objectid"]: c for c in counter.iter_painting_images(tmp)}

    assert set(cands) == {"1", "4"}
    assert cands["1"]["uuid"] == "u1"
    assert cands["1"]["width"] == 3000
    assert cands["1"]["license"] == counter.OPEN_ACCESS_LICENSE
    # Object 4 keeps the lowest-sequence primary image.
    assert cands["4"]["uuid"] == "u4a"
    assert cands["4"]["sequence"] == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
