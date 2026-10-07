"""Stage 10 — materialize the dataset and create the pipeline's only SQLite database.

No other module in this package imports sqlite3.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import file_io


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _load_schema() -> str:
    file_io.ensure_lightglue_verified_importable()
    from image_matching.keypoint.lightglue_scores_to_dataset import SCHEMA

    return SCHEMA


def _build_db(path: Path, rows: list[dict[str, str]], summary_inputs: dict[str, str]) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_load_schema())
        now = _utc_now()
        meta = {
            "dataset_name": "museum_deduplicated",
            "created_at_utc": now,
            "builder": "experiments/datasets/deduplication/deduplication/export_dataset.py",
            "image_root": "images",
            **summary_inputs,
        }
        conn.executemany("INSERT INTO dataset_meta(key, value) VALUES(?, ?)", meta.items())
        class_rows = [
            (int(row["class_id"]), row["class_key"], row["class_key"], None, None) for row in rows
        ]
        conn.executemany(
            "INSERT INTO classes(class_id,qid,label,painting_inception_year,sitelinks) VALUES(?,?,?,?,?)",
            class_rows,
        )
        image_rows = []
        for row in rows:
            ext = Path(row["output_rel_path"]).suffix.lower() or ".jpg"
            image_rows.append(
                (
                    row["file_id"], ext, None, None, None, None, row["output_rel_path"],
                    "downloaded", row["mime_type"], int(row["width"]), int(row["height"]),
                    int(row["bytes_on_disk"]), None, None, row["source_name"], None, None,
                    None, None, None, None, now, now,
                )
            )
        conn.executemany(
            """INSERT INTO image_files(file_id,file_ext,canonical_title,file_page_url,source_url,
            download_url,local_rel_path,download_status,mime_type,width,height,bytes_on_disk,year,
            epoch_bucket,source_role,has_p6243,is_photo,p180_count,pair_count,historical_count,
            modern_count,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            image_rows,
        )
        conn.executemany(
            "INSERT INTO image_file_classes(file_id,class_id,primary_label,reviewed_at,updated_at) VALUES(?,?,?,?,?)",
            [(row["file_id"], int(row["class_id"]), "same_painting", None, now) for row in rows],
        )
        relation_ids = {
            row["file_id"]: int(
                conn.execute(
                    "SELECT image_file_class_id FROM image_file_classes WHERE file_id=?",
                    (row["file_id"],),
                ).fetchone()[0]
            )
            for row in rows
        }
        conn.executemany(
            "INSERT INTO image_file_class_tags(image_file_class_id,tag) VALUES(?,?)",
            [(relation_ids[row["file_id"]], row["source_name"]) for row in rows],
        )
        split_rows = []
        for row in rows:
            split_rows.extend(
                [("all_files", row["file_id"]), ("same_painting", row["file_id"]), (row["source_name"], row["file_id"])]
            )
        conn.executemany("INSERT OR IGNORE INTO splits(split_name,file_id) VALUES(?,?)", split_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _validate_inputs(manifest_path: Path, excluded_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = file_io.read_csv_rows(manifest_path)
    excluded = file_io.read_csv_rows(excluded_path)
    if not rows:
        raise ValueError("Dataset manifest has no keepers")
    for field in ("class_id", "class_key", "file_id", "shared_rel_path", "output_rel_path", "sha256"):
        values = [row.get(field, "") for row in rows]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError(f"Dataset manifest field {field} must be non-empty and unique")
    expected_class_ids = list(range(1, len(rows) + 1))
    if sorted(int(row["class_id"]) for row in rows) != expected_class_ids:
        raise ValueError("Dataset manifest class_id values must be contiguous from 1")
    included_paths = {row["shared_rel_path"] for row in rows}
    for row in rows:
        source = file_io.resolve_from_project(row["shared_rel_path"])
        if not source.is_file() or file_io.sha256_file(source) != row["sha256"]:
            raise ValueError(f"Keeper image is missing or changed: {row['shared_rel_path']}")
        if source.stat().st_size != int(row["bytes_on_disk"]):
            raise ValueError(f"Keeper byte count changed: {row['shared_rel_path']}")
        output = Path(row["output_rel_path"])
        if output.is_absolute() or ".." in output.parts or output.parts[:1] != ("images",):
            raise ValueError(f"Unsafe output_rel_path: {output}")
    excluded_ids: set[str] = set()
    for row in excluded:
        image_id = row.get("excluded_image_id", "")
        if not image_id or image_id in excluded_ids:
            raise ValueError(f"Duplicate/missing excluded image id: {image_id!r}")
        excluded_ids.add(image_id)
        source = file_io.resolve_from_project(row["excluded_shared_rel_path"])
        if not source.is_file() or file_io.sha256_file(source) != row["excluded_sha256"]:
            raise ValueError(f"Excluded image is missing or changed: {image_id}")
        if row["excluded_shared_rel_path"] in included_paths:
            raise ValueError(f"Excluded image {image_id} is also present as a keeper")
        if row["keeper_shared_rel_path"] not in included_paths:
            raise ValueError(f"Excluded image {image_id} maps to an unknown keeper")
        if not row.get("supporting_accepted_pair_ids"):
            raise ValueError(f"Excluded image {image_id} has no accepted-pair support")
    return rows, excluded


def export_dataset(manifest_path: Path, excluded_path: Path, output: Path) -> dict[str, Any]:
    """Build and atomically publish a validated experiment dataset directory."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset output: {output}")
    rows, excluded = _validate_inputs(manifest_path, excluded_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        for row in rows:
            source = file_io.resolve_from_project(row["shared_rel_path"])
            destination = temporary / row["output_rel_path"]
            _link_or_copy(source, destination)
            if file_io.sha256_file(destination) != row["sha256"]:
                raise RuntimeError(f"Materialized image hash mismatch: {row['file_id']}")
        shutil.copy2(manifest_path, temporary / "dataset_manifest.csv")
        shutil.copy2(excluded_path, temporary / "excluded_images.csv")
        manifest_hash = file_io.sha256_file(manifest_path)
        excluded_hash = file_io.sha256_file(excluded_path)
        provenance_inputs = {
            "dataset_manifest_sha256": manifest_hash,
            "excluded_images_sha256": excluded_hash,
        }
        manifest_sidecar_path = manifest_path.with_suffix(".json")
        manifest_sidecar: dict[str, Any] | None = None
        if manifest_sidecar_path.is_file():
            candidate = file_io.read_json(manifest_sidecar_path)
            if candidate.get("selection_strategy") == "minimum_vertex_cover":
                if (
                    candidate.get("dataset_manifest_sha256") != manifest_hash
                    or candidate.get("excluded_images_sha256") != excluded_hash
                ):
                    raise ValueError("Vertex-cover export manifests do not match their sidecar")
                manifest_sidecar = candidate
                shutil.copy2(manifest_sidecar_path, temporary / "dataset_manifest.json")
                for key in (
                    "selection_strategy",
                    "solver_status",
                    "review_snapshot_sha256",
                    "accepted_edge_count",
                    "minimum_removal_count",
                    "selection_sha256",
                ):
                    provenance_inputs[key] = str(candidate[key])
        db_path = temporary / "dataset.db"
        _build_db(db_path, rows, provenance_inputs)

        file_io.ensure_experiments_importable()
        from experiments.core import db as experiment_db
        from experiments.core.config_schema import DatasetConfig, validate_dataset_config

        with experiment_db.connect(db_path) as conn:
            experiment_db.ensure_schema(conn)
        counts = experiment_db.dataset_counts(db_path)
        dataset_config = DatasetConfig(
            path=temporary,
            dataset_id="museum_deduplicated",
            dataset_db=db_path,
            image_root=temporary / "images",
            identity={
                "expected_num_classes": len(rows),
                "expected_num_images": len(rows),
            },
            raw={},
        )
        validated_counts = validate_dataset_config(dataset_config)
        if counts != validated_counts:
            raise RuntimeError("Dataset validation accessors returned inconsistent counts")
        summary = {
            "created_at": _utc_now(),
            "dataset_id": "museum_deduplicated",
            "num_classes": counts["num_classes"],
            "num_images": counts["num_images"],
            "num_excluded_images": len(excluded),
            "dataset_manifest_sha256": manifest_hash,
            "excluded_images_sha256": excluded_hash,
            "dataset_db_sha256": file_io.sha256_file(db_path),
            "selection": manifest_sidecar,
            "validated_with": [
                "experiments.core.db.ensure_schema",
                "experiments.core.db.dataset_counts",
                "experiments.core.config_schema.validate_dataset_config",
            ],
        }
        file_io.atomic_write_json(temporary / "build_summary.json", summary)
        os.replace(temporary, output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
