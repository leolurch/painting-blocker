"""Procedural, provenance-aware archival occlusions."""

from __future__ import annotations

import random
from typing import Tuple

from PIL import Image, ImageDraw


def apply_archival_occlusion(
    image: Image.Image,
    rng: random.Random,
    fraction_range: Tuple[float, float] = (0.04, 0.15),
) -> tuple[Image.Image, dict]:
    """Cover a bounded portion with a plausible label, tape, or page edge."""
    low, high = fraction_range
    if not (0.0 <= low <= high < 1.0):
        raise ValueError("occlusion fraction range must satisfy 0 <= low <= high < 1")
    result = image.convert("RGBA").copy()
    w, h = result.size
    target = rng.uniform(low, high)
    kind = rng.choice(("catalogue_label", "paper_edge", "tape", "page_clip"))
    horizontal = rng.random() < 0.5
    if horizontal:
        box_h = max(1, min(h - 1, int(round(h * target))))
        box_w = max(1, int(round(w * rng.uniform(0.65, 1.0))))
    else:
        box_w = max(1, min(w - 1, int(round(w * target))))
        box_h = max(1, int(round(h * rng.uniform(0.65, 1.0))))
    x0 = rng.choice((0, max(0, w - box_w))) if kind == "paper_edge" else rng.randint(0, max(0, w - box_w))
    y0 = rng.choice((0, max(0, h - box_h))) if kind == "paper_edge" else rng.randint(0, max(0, h - box_h))
    box = (x0, y0, x0 + box_w, y0 + box_h)
    colours = {
        "catalogue_label": (235, 226, 196, 245),
        "paper_edge": (220, 216, 203, 255),
        "tape": (202, 181, 119, 205),
        "page_clip": (91, 94, 96, 245),
    }
    ImageDraw.Draw(result, "RGBA").rectangle(box, fill=colours[kind])
    coverage = (box_w * box_h) / float(max(1, w * h))
    return result, {
        "applied": True,
        "type": kind,
        "asset_sha256": None,
        "bounding_region": list(box),
        "target_fraction": round(target, 6),
        "coverage": round(coverage, 6),
    }
