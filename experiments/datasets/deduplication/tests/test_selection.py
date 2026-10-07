from __future__ import annotations

from pathlib import Path

from PIL import Image

from deduplication import file_io, manifest
from deduplication.export_dataset import export_dataset
from deduplication.selection import prepare_vertex_cover_export, select_minimum_vertex_cover


def test_exact_vertex_cover_and_export_manifests(project: Path):
    image_dir = project / "work/images"
    image_dir.mkdir(parents=True)
    image_rows = []
    for index, image_id in enumerate(("source/a.jpg", "source/b.jpg", "source/c.jpg", "source/d.jpg")):
        filename = image_id.split("/")[-1]
        path = image_dir / filename
        Image.new("RGB", (10 + index, 10 + index), (index * 40, 0, 0)).save(path)
        image_rows.append(
            {
                "image_id": image_id,
                "source_name": "source",
                "raw_relative_path": filename,
                "shared_filename": filename,
                "shared_rel_path": f"work/images/{filename}",
                "mime_type": "image/jpeg",
                "width": str(10 + index),
                "height": str(10 + index),
                "bytes_on_disk": str(path.stat().st_size),
                "sha256": file_io.sha256_file(path),
            }
        )
    file_io.atomic_write_csv(project / "work/images.csv", manifest.IMAGES_FIELDS, image_rows)

    review_dir = project / "work/review"
    review_dir.mkdir()
    reviewed = review_dir / "reviewed_pairs.csv"
    rows = []
    for pair_id, a, b, decision in (
        ("ab", "source/a.jpg", "source/b.jpg", "accepted"),
        ("bc", "source/b.jpg", "source/c.jpg", "accepted"),
        ("ac", "source/a.jpg", "source/c.jpg", "rejected"),
    ):
        row = {field: "" for field in manifest.REVIEWED_PAIRS_FIELDS}
        row.update({"pair_id": pair_id, "image_a_id": a, "image_b_id": b, "decision": decision})
        rows.append(row)
    file_io.atomic_write_csv(reviewed, manifest.REVIEWED_PAIRS_FIELDS, rows)
    file_io.atomic_write_json(
        reviewed.with_suffix(".json"),
        {"reviewed_pairs_sha256": file_io.sha256_file(reviewed), "complete": True},
    )

    cover = project / "work/optimization/minimum_vertex_cover.csv"
    summary = select_minimum_vertex_cover(
        project / "work/images.csv", reviewed, cover, workers=1, time_limit=30
    )
    assert summary["status"] == "OPTIMAL"
    assert summary["selected_image_count"] == 1
    selected = file_io.read_csv_rows(cover)
    assert selected[0]["image_id"] == "source/b.jpg"
    assert selected[0]["witness_keeper_image_id"] in {"source/a.jpg", "source/c.jpg"}

    dataset_manifest = project / "work/dataset_manifest.csv"
    excluded = project / "work/excluded_images.csv"
    prepared = prepare_vertex_cover_export(cover, dataset_manifest, excluded)
    assert prepared["minimum_removal_count"] == 1
    assert prepared["retained_image_count"] == 3
    assert len(file_io.read_csv_rows(dataset_manifest)) == 3
    assert file_io.read_csv_rows(excluded)[0]["excluded_image_id"] == "source/b.jpg"

    output = project / "work/dataset"
    built = export_dataset(dataset_manifest, excluded, output)
    assert built["num_images"] == 3
    assert built["selection"]["selection_strategy"] == "minimum_vertex_cover"
    assert (output / "dataset_manifest.json").is_file()
