#!/usr/bin/env python3
"""Select and stage the open-access photographs SynView is trained on.

``data/syn/class_split.csv`` names every painting. Each museum downloader
saves ``{object id}.jpg``. This module maps those files to the class id the
generator expects, ``{prefix}{object id}.jpg``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PREFIXES = {
    "artic_": "artic",
    "cleveland_": "cleveland",
    "nationalgallery_": "nga",
    "rijksmuseum_": "rijks",
}


def object_ids(class_split: Path, prefix: str) -> set[str]:
    """Object ids in ``class_split`` whose painting id starts with ``prefix``."""
    class_split = Path(class_split)
    if prefix not in PREFIXES:
        raise SystemExit(f"unknown museum prefix {prefix!r}")
    wanted: set[str] = set()
    with class_split.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["parent_file_id"]
            if name.startswith(prefix):
                wanted.add(name[len(prefix) :])
    if not wanted:
        raise SystemExit(f"no {prefix} paintings in {class_split}")
    return wanted


def stage_sources(class_split: Path, image_root: Path, out_dir: Path) -> tuple[int, list[str]]:
    """Link each downloaded museum file to ``out_dir/{parent_file_id}.jpg``.

    Existing destination files are left in place. Returns the number linked
    and the painting ids that had no downloaded file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    linked = 0
    with class_split.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        name = row["parent_file_id"]
        prefix = next((item for item in PREFIXES if name.startswith(item)), None)
        if prefix is None:
            missing.append(name)
            continue
        source = image_root / PREFIXES[prefix] / f"{name[len(prefix):]}.jpg"
        dest = out_dir / f"{name}.jpg"
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        if not source.is_file() or source.stat().st_size == 0:
            missing.append(name)
            continue
        try:
            os.link(source, dest)
        except OSError:
            dest.write_bytes(source.read_bytes())
        linked += 1
    return linked, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-split", type=Path, default=Path("data/syn/class_split.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/images/sources"))
    args = parser.parse_args()
    linked, missing = stage_sources(args.class_split, args.image_root, args.out_dir)
    print(f"staged {linked}")
    print(args.out_dir)
    if missing:
        print(f"missing {len(missing)}", file=sys.stderr)
        for name in missing[:20]:
            print(name, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
