"""Class-independent archival annotations with explicit provenance."""

from __future__ import annotations

import hashlib
import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

_BUILTIN_FONT_HASH = "sha256:" + hashlib.sha256(b"Pillow.load_default:v1").hexdigest()
_TOKENS = ("INV", "CAT", "PL", "No.", "REF", "ARCH", "BOX", "FIG")


def apply_archival_annotations(
    image: Image.Image,
    rng: random.Random,
    count_range: Tuple[int, int] = (1, 3),
    *,
    surface: str = "document",
) -> tuple[Image.Image, dict]:
    """Draw random inventory-like marks that never encode class identity."""
    low, high = count_range
    if low < 0 or high < low:
        raise ValueError("annotation count range must satisfy 0 <= low <= high")
    result = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(result, "RGBA")
    font = ImageFont.load_default()
    w, h = result.size
    count = rng.randint(low, high)
    records: list[dict] = []
    covered = 0
    for _ in range(count):
        kind = rng.choice(("inventory_number", "underline", "arrow", "stamp", "marginal_note"))
        text = f"{rng.choice(_TOKENS)} {rng.randint(1, 9999):04d}"
        x = rng.randint(0, max(0, w - max(20, len(text) * 6)))
        y = rng.randint(0, max(0, h - 12))
        colour = rng.choice(((48, 39, 32, 210), (103, 32, 27, 205), (28, 45, 82, 205)))
        bbox = draw.textbbox((x, y), text, font=font)
        draw.text((x, y), text, fill=colour, font=font)
        if kind == "underline":
            draw.line((bbox[0], bbox[3] + 1, bbox[2], bbox[3] + 1), fill=colour, width=1)
        elif kind == "stamp":
            draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), outline=colour, width=1)
        covered += max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
        records.append({"type": kind, "text": text, "box": list(bbox), "surface": surface})
    return result, {
        "applied": count > 0,
        "count": count,
        "coverage": round(covered / float(max(1, w * h)), 6),
        "font": _BUILTIN_FONT_HASH,
        "items": records,
    }
