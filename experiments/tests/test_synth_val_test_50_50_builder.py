from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from experiments.datasets.synthetic_val_test_50_50.build_original_query_class_preserving_val_test_50_50_medium_v1 import (
    DATASET_ID,
    EXPECTED_ROLES,
    SelectedSource,
    build_generation_plan,
    create_source_stage,
    derive_asset_plan_from_reference,
    select_reference_sources,
    validate_materialized_split,
)


def _reference_split() -> dict:
    images = {}
    subsets = {}
    for subset, classes in (("val", (11, 12)), ("test", (21, 22))):
        roles = {role: [] for role in EXPECTED_ROLES}
        for class_id in classes:
            query = f"p{class_id}_modern-original-modern-query_no_frame_no_wall-v001.jpg"
            roles["modern_original"].append(query)
            images[query] = {
                "class_id": class_id,
                "role": "modern_original",
                "origin": "source_original",
                "profile": "modern_query",
                "parent_file_id": f"p{class_id}",
                "severity": {"preset": "mild", "bucket": "mild", "score": 0.1},
                "dimensions": {"final_size": [8, 8]},
                "transforms": {"crop": {"applied": True}},
            }
            for role in EXPECTED_ROLES[1:]:
                for index in range(2):
                    image_id = f"p{class_id}_{role}_{index}.jpg"
                    roles[role].append(image_id)
                    images[image_id] = {
                        "class_id": class_id,
                        "role": role,
                        "parent_file_id": f"p{class_id}",
                    }
        subsets[subset] = {"roles": roles}
    return {
        "schema_version": 2,
        "split_id": "reference",
        "dataset": {"dataset_id": "museum_deduplication_hard_synth_v1"},
        "subsets": subsets,
        "images": images,
    }


def _generated_split(reference: dict) -> dict:
    images = {}
    subsets = {}
    for subset in ("val", "test"):
        roles = {role: [] for role in EXPECTED_ROLES}
        query_ids = reference["subsets"][subset]["roles"]["modern_original"]
        for query_id in query_ids:
            class_id = int(reference["images"][query_id]["class_id"])
            parent = str(reference["images"][query_id]["parent_file_id"])
            roles["modern_original"].append(query_id)
            images[query_id] = {
                **reference["images"][query_id],
                "preserved_reference_query": {"image_id": query_id},
            }
            for role in EXPECTED_ROLES[1:]:
                for index in range(2):
                    image_id = f"new_{subset}_{class_id}_{role}_{index}.jpg"
                    roles[role].append(image_id)
                    images[image_id] = {
                        "class_id": class_id,
                        "role": role,
                        "parent_file_id": parent,
                        "severity": {"preset": "medium", "bucket": "medium", "score": 0.3},
                    }
        subsets[subset] = {"roles": roles}
    return {
        "schema_version": 2,
        "split_id": "generated",
        "dataset": {"dataset_id": DATASET_ID},
        "subsets": subsets,
        "images": images,
    }


def test_reference_selection_preserves_subsets_and_pristine_parent_ids():
    selected = select_reference_sources(_reference_split())
    assert {(item.class_id, item.subset) for item in selected} == {
        (11, "val"), (12, "val"), (21, "test"), (22, "test")
    }
    assert {item.parent_file_id for item in selected} == {"p11", "p12", "p21", "p22"}


def test_reference_selection_rejects_class_leakage():
    reference = _reference_split()
    test_query = reference["subsets"]["test"]["roles"]["modern_original"][0]
    reference["images"][test_query]["class_id"] = 11
    for role in EXPECTED_ROLES[1:]:
        for image_id in reference["subsets"]["test"]["roles"][role][:2]:
            reference["images"][image_id]["class_id"] = 11
    with pytest.raises(ValueError, match="classes overlap"):
        select_reference_sources(reference)


def test_generation_plan_is_class_preserving_and_medium_for_both_subsets(tmp_path: Path):
    selected = select_reference_sources(_reference_split())
    stage_db = tmp_path / "dataset.db"
    stage_db.write_bytes(b"stable staged DB")
    recipe = tmp_path / "recipe.yml"
    recipe.write_text("recipe_id: hard_synth_v1\n", encoding="utf-8")
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_split()), encoding="utf-8")
    source_plan_path = tmp_path / "source-plan.json"
    source_plan_path.write_text(
        json.dumps({
            "seed": 42,
            "asset_registry_hash": "sha256:registry",
            "asset_registry": {"sha256:val": {"type": "frame", "name": "v"}},
            "asset_assignments": {"sha256:val": "val", "sha256:test": "test"},
        }),
        encoding="utf-8",
    )

    plan = build_generation_plan(
        selected,
        staging_db=stage_db,
        reference_split_path=reference_path,
        source_plan_path=source_plan_path,
        recipe_path=recipe,
        seed=42,
    )
    assert plan["class_assignments"] == {
        "11": "val", "12": "val", "21": "test", "22": "test"
    }
    assert plan["selected_sources"]["p11"] == {
        "class_id": 11,
        "subset": "val",
        "reference_query_image_id": (
            "p11_modern-original-modern-query_no_frame_no_wall-v001.jpg"
        ),
    }
    assert plan["severity_policy"] == {"val": ["medium"], "test": ["medium"]}
    assert plan["profile_policy"]["train"] == []
    assert plan["asset_assignments"] == {"sha256:val": "val", "sha256:test": "test"}


def test_asset_plan_can_be_recovered_without_original_generation_plan(tmp_path: Path):
    directories = {}
    reference = _reference_split()
    for asset_type in ("frame", "overlay", "wall_texture"):
        directory = tmp_path / asset_type
        directory.mkdir()
        directories[asset_type] = directory
        for subset in ("val", "test"):
            path = directory / f"{subset}.png"
            color = {
                ("frame", "val"): (255, 0, 0),
                ("frame", "test"): (0, 0, 255),
                ("overlay", "val"): (200, 0, 0),
                ("overlay", "test"): (0, 0, 200),
                ("wall_texture", "val"): (150, 0, 0),
                ("wall_texture", "test"): (0, 0, 150),
            }[(asset_type, subset)]
            Image.new("RGB", (2, 2), color).save(path)
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            query = reference["subsets"][subset]["roles"]["modern_original"][0]
            reference["images"][query].setdefault("assets", {})[asset_type] = digest

    plan = derive_asset_plan_from_reference(
        reference,
        frames=directories["frame"],
        overlays=directories["overlay"],
        textures=directories["wall_texture"],
    )
    assert plan["seed"] == 42
    counts = Counter(plan["asset_assignments"].values())
    assert counts["val"] == 4  # three raster types and one procedural font
    assert counts["test"] == 4
    assert plan["recovered_from_reference_split"] is True


def _create_source_db(root: Path) -> None:
    root.mkdir()
    (root / "images").mkdir()
    connection = sqlite3.connect(root / "dataset.db")
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE classes(class_id INTEGER PRIMARY KEY, qid TEXT, label TEXT,
          painting_inception_year INTEGER, sitelinks INTEGER);
        CREATE TABLE image_files(file_id TEXT PRIMARY KEY, file_ext TEXT NOT NULL,
          canonical_title TEXT, file_page_url TEXT, source_url TEXT, download_url TEXT,
          local_rel_path TEXT, download_status TEXT NOT NULL, mime_type TEXT,
          width INTEGER, height INTEGER, bytes_on_disk INTEGER NOT NULL DEFAULT 0,
          year INTEGER, epoch_bucket TEXT, source_role TEXT, english_description TEXT,
          description_source TEXT, has_p6243 INTEGER, is_photo INTEGER, p180_count INTEGER,
          pair_count INTEGER, historical_count INTEGER, modern_count INTEGER,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE image_file_classes(image_file_class_id INTEGER PRIMARY KEY AUTOINCREMENT,
          file_id TEXT NOT NULL REFERENCES image_files(file_id) ON DELETE CASCADE,
          class_id INTEGER NOT NULL REFERENCES classes(class_id), primary_label TEXT,
          reviewed_at TEXT, updated_at TEXT NOT NULL, UNIQUE(file_id,class_id));
        CREATE TABLE image_file_class_tags(image_file_class_id INTEGER NOT NULL
          REFERENCES image_file_classes(image_file_class_id) ON DELETE CASCADE,
          tag TEXT NOT NULL, PRIMARY KEY(image_file_class_id,tag));
        CREATE TABLE splits(split_name TEXT NOT NULL, file_id TEXT NOT NULL
          REFERENCES image_files(file_id) ON DELETE CASCADE, PRIMARY KEY(split_name,file_id));
        CREATE TABLE synthetic_runs(synthetic_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL, tool TEXT NOT NULL, config_json TEXT NOT NULL,
          source_dataset_name TEXT, created_at TEXT NOT NULL);
        CREATE TABLE image_file_derivations(child_file_id TEXT PRIMARY KEY
          REFERENCES image_files(file_id) ON DELETE CASCADE, parent_file_id TEXT NOT NULL
          REFERENCES image_files(file_id), synthetic_run_id INTEGER NOT NULL
          REFERENCES synthetic_runs(synthetic_run_id) ON DELETE CASCADE,
          transform_family TEXT NOT NULL, transform_params_json TEXT NOT NULL);
        CREATE TABLE dataset_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    now = "now"
    for class_id in (11, 12, 21, 22, 99):
        parent = f"p{class_id}"
        Image.new("RGB", (8, 8), "red").save(root / "images" / f"{parent}.jpg")
        connection.execute(
            "INSERT INTO classes VALUES(?,?,?,?,?)", (class_id, str(class_id), parent, None, None)
        )
        connection.execute(
            "INSERT INTO image_files(file_id,file_ext,local_rel_path,download_status,bytes_on_disk,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (parent, ".jpg", f"images/{parent}.jpg", "source_input", 10, now, now),
        )
        connection.execute(
            "INSERT INTO image_file_classes(file_id,class_id,primary_label,updated_at) VALUES(?,?,?,?)",
            (parent, class_id, "same_painting", now),
        )
        connection.execute("INSERT INTO splits VALUES('source_inputs',?)", (parent,))
    connection.execute("INSERT INTO dataset_meta VALUES('dataset_name','source')")
    connection.commit()
    connection.close()


def test_source_stage_keeps_only_selected_parents_and_original_class_ids(tmp_path: Path):
    source_root = tmp_path / "source"
    stage_root = tmp_path / "stage"
    _create_source_db(source_root)
    selected = select_reference_sources(_reference_split())

    create_source_stage(source_root, stage_root, selected, workers=2)

    connection = sqlite3.connect(stage_root / "dataset.db")
    parents = {row[0] for row in connection.execute("SELECT file_id FROM image_files")}
    classes = {row[0] for row in connection.execute("SELECT class_id FROM classes")}
    assignments = set(connection.execute("SELECT split_name,file_id FROM splits"))
    name = connection.execute(
        "SELECT value FROM dataset_meta WHERE key='dataset_name'"
    ).fetchone()[0]
    connection.close()
    assert parents == {"p11", "p12", "p21", "p22"}
    assert classes == {11, 12, 21, 22}
    assert assignments == {
        (split_name, parent)
        for split_name in ("source_inputs", "all_files")
        for parent in parents
    }
    assert name == DATASET_ID
    assert all((stage_root / "images" / f"{parent}.jpg").is_file() for parent in parents)


def test_materialized_split_validator_detects_reassignment():
    reference = _reference_split()
    generated = _generated_split(reference)
    assert validate_materialized_split(reference, generated) == []

    query = generated["subsets"]["test"]["roles"]["modern_original"][0]
    generated["images"][query]["class_id"] = 11
    errors = validate_materialized_split(reference, generated)
    assert any("test class set differs" in error for error in errors)
