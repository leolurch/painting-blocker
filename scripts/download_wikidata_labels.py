#!/usr/bin/env python3
"""Download the Wikidata 1.5 images and write the dataset database.

``data/wikidata/labels.csv`` is the full labeled list for the test set.
Each file is saved as ``{out}/{filename}``. ``dataset.db`` is written next
to that image directory, which is the path the evaluation configs use.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/wikidata/labels.csv"
DEFAULT_OUT = ROOT / "data/datasets/wikidata-1.5/images"

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE classes(
    class_id INTEGER PRIMARY KEY,
    qid TEXT NOT NULL UNIQUE,
    label TEXT
);
CREATE TABLE image_files(
    file_id TEXT PRIMARY KEY,
    local_rel_path TEXT,
    download_status TEXT NOT NULL,
    source_role TEXT
);
CREATE TABLE image_file_classes(
    image_file_class_id INTEGER PRIMARY KEY,
    file_id TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    primary_label TEXT
);
CREATE TABLE image_file_class_tags(
    image_file_class_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY(image_file_class_id, tag)
);
CREATE TABLE splits(
    split_name TEXT NOT NULL,
    file_id TEXT NOT NULL,
    PRIMARY KEY(split_name, file_id)
);
"""


def destination(out_dir: Path, row: dict[str, str]) -> Path:
    role = row["role"]
    if role not in {"modern", "historic"}:
        raise ValueError(f"bad role {role!r} for {row['filename']}")
    if not str(row.get("class_id", "")).isdigit():
        raise ValueError(f"missing class id for {row['filename']}")
    return out_dir / row["filename"]


def write_dataset_db(rows: list[dict[str, str]], out_dir: Path) -> Path:
    """Build dataset.db beside ``out_dir`` from the labeled image list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir.parent / "dataset.db"
    if db_path.exists():
        db_path.unlink()
    classes: dict[int, str] = {}
    for row in rows:
        class_id = int(row["class_id"])
        previous = classes.get(class_id)
        if previous is not None and previous != row["qid"]:
            raise SystemExit(f"class {class_id} has qids {previous} and {row['qid']}")
        classes[class_id] = row["qid"]
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO classes(class_id, qid) VALUES(?, ?)",
        sorted(classes.items()),
    )
    for index, row in enumerate(rows, start=1):
        filename = row["filename"]
        target = out_dir / filename
        status = "downloaded" if target.is_file() and target.stat().st_size > 0 else "pending"
        conn.execute(
            "INSERT INTO image_files(file_id, local_rel_path, download_status, source_role) VALUES(?, ?, ?, ?)",
            (filename, f"images/{filename}", status, row["role"]),
        )
        conn.execute(
            "INSERT INTO image_file_classes(image_file_class_id, file_id, class_id, primary_label) VALUES(?, ?, ?, 'same_painting')",
            (index, filename, int(row["class_id"])),
        )
        for tag in str(row.get("tags") or "").split(","):
            tag = tag.strip()
            if tag:
                conn.execute(
                    "INSERT INTO image_file_class_tags(image_file_class_id, tag) VALUES(?, ?)",
                    (index, tag),
                )
        conn.execute(
            "INSERT INTO splits(split_name, file_id) VALUES(?, ?)",
            (row["role"], filename),
        )
    conn.commit()
    conn.close()
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="Download at most this many missing files")
    parser.add_argument("--skip-download", action="store_true", help="Write dataset.db from the label list only")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    fetched = 0
    if not args.skip_download:
        for row in rows:
            target = destination(args.out_dir, row)
            if target.exists() and target.stat().st_size > 0:
                continue
            if args.limit and fetched >= args.limit:
                break
            target.parent.mkdir(parents=True, exist_ok=True)
            request = Request(row["download_url"], headers={"User-Agent": "edbt2027-artifact/1.0"})
            with urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
            fetched += 1
            print(target)
    db_path = write_dataset_db(rows, args.out_dir)
    print(f"downloaded {fetched}")
    print(db_path)


if __name__ == "__main__":
    main()
