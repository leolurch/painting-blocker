from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[2] / "wikimedia" / "export_wikidata_dedup_queries.py"
SPEC = importlib.util.spec_from_file_location("export_wikidata_dedup_queries", MODULE_PATH)
assert SPEC and SPEC.loader
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE classes(class_id INTEGER PRIMARY KEY, qid TEXT, label TEXT);
        CREATE TABLE image_files(
          file_id TEXT PRIMARY KEY, file_ext TEXT, local_rel_path TEXT,
          download_status TEXT, width INTEGER, height INTEGER, bytes_on_disk INTEGER
        );
        CREATE TABLE image_file_classes(
          image_file_class_id INTEGER PRIMARY KEY, file_id TEXT, class_id INTEGER,
          primary_label TEXT
        );
        CREATE TABLE image_file_class_tags(
          image_file_class_id INTEGER, tag TEXT, PRIMARY KEY(image_file_class_id, tag)
        );
        """
    )
    return connection


def test_wikidata_export_selection_order_and_resume(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (10, 10), "red").save(root / "few-tags.png")
    Image.new("RGB", (30, 30), "blue").save(root / "more-tags.png")
    Image.new("RGB", (10, 10), "black").save(root / "small.png")
    Image.new("RGB", (20, 20), "black").save(root / "large.png")
    Image.new("RGB", (20, 20), "black").save(root / "compact.png", optimize=True)
    # Same dimensions, deliberately less compressible/larger bytes.
    noisy = Image.effect_noise((20, 20), 100).convert("RGB")
    noisy.save(root / "noisy.png")

    db = tmp_path / "dataset.db"
    connection = _database(db)
    connection.executemany(
        "INSERT INTO classes VALUES(?,?,?)",
        [(1, "Q1", "one"), (2, "Q2", "two"), (3, "Q3", "three"), (4, "Q4", "four")],
    )
    files = [
        ("few", "png", "few-tags.png", "downloaded"),
        ("more", "png", "more-tags.png", "downloaded"),
        ("small", "png", "small.png", "downloaded"),
        ("large", "png", "large.png", "downloaded"),
        ("compact", "png", "compact.png", "downloaded"),
        ("noisy", "png", "noisy.png", "downloaded"),
        ("missing", "jpg", "missing.jpg", "missing"),
    ]
    connection.executemany(
        "INSERT INTO image_files VALUES(?,?,?,?,0,0,0)", files
    )
    relations = [
        (1, "few", 1), (2, "more", 1),
        (3, "small", 2), (4, "large", 2),
        (5, "compact", 3), (6, "noisy", 3),
        (7, "missing", 4),
    ]
    connection.executemany(
        "INSERT INTO image_file_classes VALUES(?,?,?,'same_painting')", relations
    )
    tags = [(relation_id, "modern") for relation_id, _, _ in relations]
    tags.append((2, "poor_quality"))
    connection.executemany("INSERT INTO image_file_class_tags VALUES(?,?)", tags)
    connection.commit()
    connection.close()

    output = tmp_path / "export"
    first = EXPORTER.export_queries(db, root, output)
    rows = {int(row["class_id"]): row for row in EXPORTER.csv.DictReader((output / "manifest.csv").open())}
    assert first["total_class_count"] == 4
    assert first["selected_class_count"] == 3
    assert rows[1]["source_file_id"] == "few"       # tag count dominates resolution
    assert rows[2]["source_file_id"] == "large"    # resolution descending
    assert rows[3]["source_file_id"] == "compact"  # bytes ascending
    excluded = list(EXPORTER.csv.DictReader((output / "excluded_classes.csv").open()))
    assert excluded == [{
        "class_id": "4", "qid": "Q4", "class_label": "four",
        "reason": "no_available_valid_candidate", "details": "missing:not_downloaded",
    }]

    second = EXPORTER.export_queries(db, root, output)
    assert second["copied_count"] == 0
    assert second["reused_count"] == 3
