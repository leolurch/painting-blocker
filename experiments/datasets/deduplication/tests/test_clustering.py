from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from deduplication import file_io, manifest
from deduplication.clustering import build_clusters, prepare_dataset_export
from deduplication.export_dataset import export_dataset


def _setup_images(project: Path) -> dict[str, dict[str, str]]:
    image_dir = project / "work/images"
    image_dir.mkdir(parents=True)
    specs = [
        ("alpha/a.jpg", "alpha", "a.jpg", (20, 20), "red"),
        ("beta/b.jpg", "beta", "b.jpg", (30, 30), "green"),
        ("beta/c.jpg", "beta", "c.jpg", (10, 10), "blue"),
        ("beta/single.jpg", "beta", "single.jpg", (8, 8), "white"),
    ]
    result = {}
    for image_id, source, filename, size, color in specs:
        path = image_dir / filename
        Image.new("RGB", size, color).save(path)
        result[image_id] = {
            "image_id": image_id,
            "source_name": source,
            "raw_relative_path": filename,
            "shared_filename": filename,
            "shared_rel_path": f"work/images/{filename}",
            "mime_type": "image/jpeg",
            "width": str(size[0]),
            "height": str(size[1]),
            "bytes_on_disk": str(path.stat().st_size),
            "sha256": file_io.sha256_file(path),
        }
    file_io.atomic_write_csv(project / "work/images.csv", manifest.IMAGES_FIELDS, result.values())
    return result


def _review_pair(pair_id: str, a: str, b: str, decision: str) -> dict[str, str]:
    row = {field: "" for field in manifest.REVIEWED_PAIRS_FIELDS}
    row.update({"pair_id": pair_id, "image_a_id": a, "image_b_id": b, "decision": decision})
    return row


def test_conflict_keeper_manifests_and_final_export(project, config):
    _setup_images(project)
    review_dir = project / "work/review"
    review_dir.mkdir()
    reviewed = review_dir / "reviewed_pairs.csv"
    rows = [
        _review_pair("ab", "alpha/a.jpg", "beta/b.jpg", "accepted"),
        _review_pair("bc", "beta/b.jpg", "beta/c.jpg", "accepted"),
        _review_pair("ac", "alpha/a.jpg", "beta/c.jpg", "rejected"),
    ]
    file_io.atomic_write_csv(reviewed, manifest.REVIEWED_PAIRS_FIELDS, rows)
    file_io.atomic_write_json(reviewed.with_suffix(".json"), {
        "reviewed_pairs_sha256": file_io.sha256_file(reviewed), "complete": True
    })
    clusters = project / "work/clusters.csv"
    summary = build_clusters(config, project / "work/images.csv", reviewed, clusters)
    assert summary["cluster_count"] == 2
    assert summary["conflict_count"] == 1
    with pytest.raises(ValueError, match="conflicts"):
        prepare_dataset_export(clusters, project / "work/dataset_manifest.csv", project / "work/excluded_images.csv")

    # Resolve the internal rejected edge, then rebuild against the changed review snapshot.
    rows[-1]["decision"] = "accepted"
    file_io.atomic_write_csv(reviewed, manifest.REVIEWED_PAIRS_FIELDS, rows)
    file_io.atomic_write_json(reviewed.with_suffix(".json"), {
        "reviewed_pairs_sha256": file_io.sha256_file(reviewed), "complete": True
    })
    summary = build_clusters(config, project / "work/images.csv", reviewed, clusters)
    assert summary["conflict_count"] == 0
    cluster_rows = file_io.read_csv_rows(clusters)
    duplicate_component = [row for row in cluster_rows if row["component_size"] == "3"]
    assert sum(row["is_keeper"] == "1" for row in duplicate_component) == 1
    assert next(row for row in duplicate_component if row["is_keeper"] == "1")["image_id"] == "alpha/a.jpg"

    dataset_manifest = project / "work/dataset_manifest.csv"
    excluded = project / "work/excluded_images.csv"
    prepared = prepare_dataset_export(clusters, dataset_manifest, excluded)
    assert prepared["keeper_count"] == 2
    assert prepared["excluded_count"] == 2
    assert not list(project.rglob("*.db"))

    output = project / "work/dataset"
    built = export_dataset(dataset_manifest, excluded, output)
    assert built["num_images"] == built["num_classes"] == 2
    assert (output / "dataset.db").is_file()
    assert len(list((output / "images").iterdir())) == 2
    with pytest.raises(FileExistsError):
        export_dataset(dataset_manifest, excluded, output)

    file_io.ensure_experiments_importable()
    from experiments.core.db import dataset_counts, fetch_same_painting_records

    assert dataset_counts(output / "dataset.db") == {"num_classes": 2, "num_images": 2}
    assert len(fetch_same_painting_records(output / "dataset.db")) == 2
