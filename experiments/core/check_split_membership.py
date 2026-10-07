#!/usr/bin/env python3
"""Report which split subset/role each image in a CSV belongs to.

Pass a CSV containing image IDs in one column and a schema-v2 split JSON. For every
ID the script prints the ``subset/role`` memberships it finds (an image is normally
in one class-disjoint subset), or ``NOT_FOUND``. Split image IDs are flat filenames
(``Path(local_rel_path).name``), so matching is on basename by default; use
``--exact`` to require a full-string match. A cell holding several ``;``/``,``-joined
IDs (e.g. ``matched_synth_files``) is expanded and each ID checked separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

_ID_CANDIDATES = ("image_id", "file_id", "filename", "image", "id", "rep_file", "name")


def build_index(split: dict, *, exact: bool) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subset_name, subset in (split.get("subsets") or {}).items():
        for role, ids in ((subset or {}).get("roles") or {}).items():
            for image_id in ids:
                key = image_id if exact else os.path.basename(str(image_id))
                index[key].append((subset_name, role))
    return index


def pick_column(explicit: str | None, fieldnames: list[str] | None) -> str:
    if explicit:
        if not fieldnames or explicit not in fieldnames:
            raise SystemExit(f"Column {explicit!r} not in CSV columns {fieldnames}")
        return explicit
    if fieldnames and len(fieldnames) == 1:
        return fieldnames[0]
    for candidate in _ID_CANDIDATES:
        if fieldnames and candidate in fieldnames:
            return candidate
    raise SystemExit(f"Could not autodetect an ID column from {fieldnames}; pass --column")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("split_path", type=Path)
    ap.add_argument("--column", help="CSV column holding image IDs (autodetected if omitted)")
    ap.add_argument("--exact", action="store_true", help="Match full IDs instead of basenames")
    args = ap.parse_args()

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    index = build_index(split, exact=args.exact)

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        column = pick_column(args.column, reader.fieldnames)
        subset_counts: Counter[str] = Counter()
        found = missing = 0
        for row in reader:
            for raw in (v.strip() for v in re.split(r"[;,]", row.get(column) or "") if v.strip()):
                key = raw if args.exact else os.path.basename(raw)
                memberships = index.get(key, [])
                if memberships:
                    found += 1
                    for subset_name, _ in memberships:
                        subset_counts[subset_name] += 1
                    print(f"{raw}\t" + ";".join(f"{s}/{r}" for s, r in memberships))
                else:
                    missing += 1
                    print(f"{raw}\tNOT_FOUND")

    print(f"\n# split_id={split.get('split_id')} column={column} exact={args.exact}")
    print(f"# found={found} not_found={missing} by_subset={dict(subset_counts)}")


if __name__ == "__main__":
    main()
