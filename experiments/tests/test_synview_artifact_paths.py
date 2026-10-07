"""The packaged SynView and Lost Art rebuilds do not replace existing data."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from experiments.datasets.synview_sources import object_ids, stage_sources
from experiments.datasets.wikidata_lostart_distractors.build_splits import (
    EXCLUDED_IDENTITIES,
    _load_groups,
    build,
)
from experiments.datasets.write_synview_split import (
    DEFAULT_OUTPUT,
    IMAGES_PER_CLASS,
    ROLES,
    write_synview_split,
)

WINNER = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/finetune_dinov3_vithplus_met_paint_winner_followup_v1/none.yml"
)


def test_winner_split_path_is_the_writer_default():
    text = WINNER.read_text(encoding="utf-8")
    assert "hard_synth_class_disjoint_v1.json" in text
    assert DEFAULT_OUTPUT.name == "hard_synth_class_disjoint_v1.json"
    assert DEFAULT_OUTPUT.parent.name in text


def test_existing_synview_split_is_not_replaced(tmp_path: Path):
    output = tmp_path / "split.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="leaving existing split"):
        write_synview_split(
            dataset_root=tmp_path,
            class_split=tmp_path / "class_split.csv",
            output=output,
        )
    assert output.read_text(encoding="utf-8") == "keep"


def test_checked_in_synview_split_is_not_opened_for_write():
    if not DEFAULT_OUTPUT.is_file():
        pytest.skip("training split is not in this checkout")
    before = DEFAULT_OUTPUT.stat().st_mtime_ns
    with pytest.raises(SystemExit, match="leaving existing split"):
        write_synview_split(
            dataset_root=DEFAULT_OUTPUT.parent,
            class_split=DEFAULT_OUTPUT,
            output=DEFAULT_OUTPUT,
        )
    assert DEFAULT_OUTPUT.stat().st_mtime_ns == before


def _generated_database(root: Path, label: str, class_id: int) -> None:
    database = root / "dataset.db"
    (root / "images").mkdir()
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE classes(class_id INTEGER PRIMARY KEY, qid TEXT, label TEXT);
        CREATE TABLE image_files(file_id TEXT PRIMARY KEY, local_rel_path TEXT);
        CREATE TABLE image_file_classes(
            file_id TEXT, class_id INTEGER, primary_label TEXT
        );
        CREATE TABLE splits(split_name TEXT, file_id TEXT);
        CREATE TABLE image_file_derivations(
            child_file_id TEXT, parent_file_id TEXT, transform_params_json TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO classes(class_id, qid, label) VALUES(?, ?, ?)",
        (class_id, label, label),
    )
    index = 0
    for role, count in IMAGES_PER_CLASS.items():
        for _ in range(count):
            index += 1
            file_id = f"{label}_{role}_v{index:03d}"
            filename = f"{file_id}.jpg"
            (root / "images" / filename).write_bytes(b"jpg")
            connection.execute(
                "INSERT INTO image_files(file_id, local_rel_path) VALUES(?, ?)",
                (file_id, f"images/{filename}"),
            )
            connection.execute(
                "INSERT INTO image_file_classes(file_id, class_id, primary_label) VALUES(?, ?, 'same_painting')",
                (file_id, class_id),
            )
            connection.execute(
                "INSERT INTO splits(split_name, file_id) VALUES(?, ?)",
                (role, file_id),
            )
            params = json.dumps({"origin": "synthetic", "profile": role, "severity": {"bucket": "medium"}})
            connection.execute(
                "INSERT INTO image_file_derivations(child_file_id, parent_file_id, transform_params_json) VALUES(?, ?, ?)",
                (file_id, label, params),
            )
    connection.commit()
    connection.close()


def test_synview_split_uses_the_class_assignment(tmp_path: Path):
    label = "artic_100026"
    dataset = tmp_path / "generated"
    dataset.mkdir()
    _generated_database(dataset, label, 7)
    class_split = tmp_path / "class_split.csv"
    with class_split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parent_file_id", "subset"])
        writer.writeheader()
        writer.writerow({"parent_file_id": label, "subset": "train"})
    output = tmp_path / "hard_synth_class_disjoint_v1.json"
    counts = write_synview_split(dataset_root=dataset, class_split=class_split, output=output)
    assert counts["train"]["modern_generated"] == 3
    assert counts["val"]["modern_original"] == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert set(document["subsets"]["train"]["roles"]) == set(ROLES)
    assert document["images"]
    assert all(record["class_id"] == 7 for record in document["images"].values())


def test_stage_sources_links_museum_files_and_skips_existing(tmp_path: Path):
    class_split = tmp_path / "class_split.csv"
    with class_split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parent_file_id", "subset"])
        writer.writeheader()
        writer.writerow({"parent_file_id": "artic_11", "subset": "train"})
        writer.writerow({"parent_file_id": "nationalgallery_1", "subset": "val"})
    images = tmp_path / "images"
    (images / "artic").mkdir(parents=True)
    (images / "artic" / "11.jpg").write_bytes(b"artic")
    out = tmp_path / "sources"
    linked, missing = stage_sources(class_split, images, out)
    assert linked == 1
    assert missing == ["nationalgallery_1"]
    assert (out / "artic_11.jpg").read_bytes() == b"artic"
    linked_again, missing_again = stage_sources(class_split, images, out)
    assert linked_again == 0
    assert missing_again == ["nationalgallery_1"]
    assert object_ids(class_split, "artic_") == {"11"}


def _wiki_split() -> dict:
    return {
        "schema_version": 2,
        "split_id": "wiki",
        "dataset": {
            "dataset_id": "wikidata-1.5",
            "image_root": "images",
            "source_db_sha256": "abc",
            "identity": {"num_classes": 1, "num_images": 2},
        },
        "coverage_filter": {
            "included_num_classes": 1,
            "included_num_historic_images": 1,
            "input_num_historic_images": 1,
        },
        "subsets": {
            "val": {"roles": {"modern": ["modern.jpg"], "historic": ["historic-val.jpg"]}},
            "test": {"roles": {"modern": ["modern-test.jpg"], "historic": ["historic.jpg"]}},
        },
        "images": {
            "modern.jpg": {"class_id": 1},
            "historic-val.jpg": {"class_id": 1},
            "modern-test.jpg": {"class_id": 2},
            "historic.jpg": {"class_id": 2},
        },
    }


def _write_wiki(directory: Path) -> None:
    directory.mkdir()
    full = _wiki_split()
    full["subsets"] = {"test": full["subsets"]["test"]}
    full["images"] = {
        name: record
        for name, record in full["images"].items()
        if name in {"modern-test.jpg", "historic.jpg"}
    }
    (directory / "test-only.json").write_text(json.dumps(full), encoding="utf-8")
    parted = _wiki_split()
    for seed in (6, 7, 9, 10, 21, 42, 67, 87, 1337, 4711):
        (directory / f"class-disjoint-val-test-seed-{seed}.json").write_text(
            json.dumps(parted), encoding="utf-8"
        )


def _halves(path: Path) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    halves = {}
    for subset, body in document["subsets"].items():
        for name in body["roles"].get("historic", []):
            if name.startswith("LA"):
                halves[name] = subset
    return halves


def test_distractor_csv_without_removed_identities_keeps_the_same_halves(tmp_path: Path):
    wiki = tmp_path / "wiki"
    _write_wiki(wiki)
    names = tmp_path / "names.txt"
    names.write_text("LA10_0.jpg\nLA11_0.jpg\nLA44146_0.jpg\nLA91918_0.jpg\n", encoding="utf-8")
    csv_path = tmp_path / "distractor_files.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lost_art_id", "filename", "report_url"])
        writer.writeheader()
        writer.writerow({"lost_art_id": "10", "filename": "LA10_0.jpg", "report_url": "https://example.test/10"})
        writer.writerow({"lost_art_id": "11", "filename": "LA11_0.jpg", "report_url": "https://example.test/11"})
    assert set(_load_groups(csv_path)) == {"10", "11"}
    assert EXCLUDED_IDENTITIES.isdisjoint(_load_groups(csv_path))
    with_removed = build(
        wiki_split_dir=wiki,
        out_dir=tmp_path / "with",
        image_root="images",
        dataset_id="wikidata1.5-lostart-distractors",
        names_path=names,
    )
    without = build(
        wiki_split_dir=wiki,
        out_dir=tmp_path / "without",
        image_root="images",
        dataset_id="wikidata1.5-lostart-distractors",
        names_path=csv_path,
    )
    assert with_removed["distractor_identities"] == without["distractor_identities"] == 2
    seed = "class-disjoint-val-test-seed-42.json"
    assert _halves(tmp_path / "with" / seed) == _halves(tmp_path / "without" / seed)
