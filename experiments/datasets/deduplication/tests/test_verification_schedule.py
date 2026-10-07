from __future__ import annotations

from pathlib import Path

from PIL import Image

from deduplication import file_io, manifest
from deduplication.verification_schedule import REFERENCE_FIELD, iter_reference_chunks, prepare_bipartite_schedules


def test_reference_shards_and_worker_schedules_keep_groups_together(project):
    image_dir = project / "work/images"
    image_dir.mkdir(parents=True)
    image_rows = []
    for source, names in (("set_a", ("a0", "a1", "a2")), ("set_b", ("b0", "b1"))):
        for name in names:
            path = image_dir / f"{name}.jpg"
            Image.new("RGB", (4, 4), "red").save(path)
            image_rows.append(
                {
                    "image_id": f"{source}/{name}.jpg",
                    "source_name": source,
                    "raw_relative_path": f"{name}.jpg",
                    "shared_filename": path.name,
                    "shared_rel_path": f"work/images/{path.name}",
                    "mime_type": "image/jpeg",
                    "width": 4,
                    "height": 4,
                    "bytes_on_disk": path.stat().st_size,
                    "sha256": file_io.sha256_file(path),
                }
            )
    file_io.atomic_write_csv(project / "work/images.csv", manifest.IMAGES_FIELDS, image_rows)
    images = manifest.load_images(project / "work/images.csv")
    candidates = []
    for query in ("a0", "a1", "a2"):
        for reference in ("b0", "b1"):
            a_id = f"set_a/{query}.jpg"
            b_id = f"set_b/{reference}.jpg"
            row = {field: "" for field in manifest.CANDIDATES_FIELDS}
            row.update(
                {
                    "pair_id": f"{query}-{reference}",
                    "image_a_id": a_id,
                    "image_b_id": b_id,
                    "cosine_similarity": "0.9",
                }
            )
            candidates.append(row)
    path = project / "work/candidates/candidates.csv"
    file_io.atomic_write_csv(path, manifest.CANDIDATES_FIELDS, candidates)
    paths, summary = prepare_bipartite_schedules(
        path,
        file_io.sha256_file(path),
        images,
        num_shards=2,
        workers=2,
        chunk_rows=2,
    )
    assert sum(summary["counts"].values()) == 6
    owner: dict[str, Path] = {}
    for schedule in paths:
        for row in file_io.iter_csv_rows(schedule):
            reference = row[REFERENCE_FIELD]
            assert owner.setdefault(reference, schedule) == schedule
    assert set(owner) == {"set_b/b0.jpg", "set_b/b1.jpg"}
    for schedule in paths:
        for _, chunk in iter_reference_chunks(schedule, part_size=2):
            # A group may exceed part_size, but one reference is never split/mixed accidentally.
            assert len({row[REFERENCE_FIELD] for row in chunk}) == 1
