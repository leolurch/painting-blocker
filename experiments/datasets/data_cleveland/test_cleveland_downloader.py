#!/usr/bin/env python3
"""Offline unit tests for the Cleveland (CMA) paintings downloader (no network).

Covers the source-specific edge branches:
  * choose_urls: print preferred, web as fallback (no IIIF size to build)
  * ensure_max_width: downscales when wider than the cap, never upscales
  * build_candidate: print/web URLs + native width, skip non-CC0 / imageless
  * iter_painting_images: pages by skip, dedups, stops at total (fake fetch)

Run with ``pytest test_cleveland_downloader.py`` or
``python test_cleveland_downloader.py``.
"""
import os
import tempfile

from PIL import Image

import count_cleveland_paintings as counter
import download_cleveland_paintings as dl


def _record(id_, print_w=2849, print_url="p", web_url="w", cc0=True):
    images = {}
    if print_url:
        images["print"] = {"url": f"https://cdn/{print_url}.jpg",
                            "width": str(print_w), "height": "3400"}
    if web_url:
        images["web"] = {"url": f"https://cdn/{web_url}.jpg",
                         "width": "748", "height": "893"}
    return {
        "id": id_,
        "accession_number": f"{id_}.1",
        "title": f"Work {id_}",
        "share_license_status": "CC0" if cc0 else "Copyrighted",
        "images": images,
    }


def test_choose_urls_prefers_print_fallback_web():
    cand = counter.build_candidate(_record(1, print_url="big", web_url="small"))
    preferred, fallback = dl.choose_urls(cand)
    assert preferred.endswith("/big.jpg")
    assert fallback.endswith("/small.jpg")


def test_choose_urls_web_only_has_no_distinct_fallback():
    cand = counter.build_candidate(_record(2, print_url=None, web_url="only"))
    preferred, fallback = dl.choose_urls(cand)
    assert preferred.endswith("/only.jpg")
    assert fallback == preferred  # single source: fallback collapses to preferred


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
    cand = counter.build_candidate(_record(94979, print_w=2849))
    assert cand["objectid"] == "94979"
    assert cand["accession_number"] == "94979.1"
    assert cand["image_url"].endswith("/p.jpg")
    assert cand["fallback_url"].endswith("/w.jpg")
    assert cand["width"] == 2849
    assert cand["height"] == 3400
    assert cand["title"] == "Work 94979"
    assert cand["license"] == counter.OPEN_ACCESS_LICENSE


def test_build_candidate_skips_non_cc0():
    assert counter.build_candidate(_record(1, cc0=False)) is None


def test_build_candidate_skips_imageless():
    assert counter.build_candidate(_record(1, print_url=None, web_url=None)) is None


def test_iter_painting_images_pages_and_dedups():
    # 2500 records spread across pages; one duplicate id to exercise dedup.
    records = [_record(i) for i in range(2500)]
    records.append(_record(0))  # duplicate of the first object
    total = 2500

    def fake_fetch(skip, limit):
        return records[skip:skip + limit], total

    got = list(counter.iter_painting_images(session=object(), fetch=fake_fetch))
    ids = [c["objectid"] for c in got]
    assert len(ids) == 2500          # duplicate collapsed
    assert len(set(ids)) == 2500     # all unique
    assert ids[0] == "0" and ids[-1] == "2499"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
