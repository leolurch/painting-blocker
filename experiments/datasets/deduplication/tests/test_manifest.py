from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from deduplication import file_io, manifest
from deduplication.ingest import SourceSpec, ingest, parse_source_arg


def test_identities_are_stable_and_content_specific():
    assert manifest.make_image_id("met", "nested\\painting.jpg") == "met/nested/painting.jpg"
    first = manifest.make_pair_id("a", "1", "b", "2")
    assert first == manifest.make_pair_id("a", "1", "b", "2")
    assert first != manifest.make_pair_id("a", "changed", "b", "2")
    with pytest.raises(ValueError, match="Self-pair"):
        manifest.order_endpoints("a", "1", "a", "2")


def test_ingest_multiple_sources_invalid_and_orphan(project, config, tmp_path):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    Image.new("RGB", (20, 10), "red").save(alpha / "one.jpg")
    Image.new("RGB", (12, 14), "blue").save(beta / "two.png")
    (beta / "bad.jpg").write_bytes(b"not an image")
    config.images_dir.mkdir(parents=True)
    (config.images_dir / "orphan.bin").write_bytes(b"orphan")

    summary = ingest(config, [SourceSpec("alpha", alpha), SourceSpec("beta", beta)])
    assert summary["num_valid_images"] == 2
    assert summary["num_invalid_images"] == 1
    assert (config.images_dir / "alpha_one.jpg").is_file()
    assert (config.images_dir / "beta_two.png").is_file()
    assert not any(str(alpha.resolve()) in path.read_text() for path in project.glob("work/*.csv"))
    orphans = file_io.read_csv_rows(project / "work/orphaned_shared_images.csv")
    assert [row["shared_filename"] for row in orphans] == ["orphan.bin"]

    # Restart is deterministic and does not rewrite with a renamed target.
    before = file_io.sha256_file(project / "work/images.csv")
    ingest(config, [SourceSpec("alpha", alpha), SourceSpec("beta", beta)])
    assert file_io.sha256_file(project / "work/images.csv") == before


def test_collision_preflight_copies_nothing(project, config, tmp_path):
    source = tmp_path / "raw"
    (source / "x").mkdir(parents=True)
    (source / "y").mkdir()
    Image.new("RGB", (2, 2)).save(source / "x/same.jpg")
    Image.new("RGB", (2, 2)).save(source / "y/same.jpg")
    with pytest.raises(ValueError, match="multiple raw files"):
        ingest(config, [SourceSpec("raw", source)])
    assert list(config.images_dir.iterdir()) == []


def test_external_work_directory_keeps_manifest_paths_portable(project, tmp_path):
    selected = file_io.configure_work_dir(tmp_path / "large-storage" / "museum_smoke")
    assert selected == (tmp_path / "large-storage" / "museum_smoke").resolve()
    image = selected / "images/a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    assert file_io.resolve_from_project("work/images/a.jpg") == image.resolve()
    assert file_io.rel_to_project(image) == "work/images/a.jpg"
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("fixture")
    with pytest.raises(ValueError, match="not a directory"):
        file_io.configure_work_dir(not_a_directory)


def test_external_work_symlink_keeps_manifest_paths_portable(project, tmp_path):
    external = tmp_path / "large-storage-work"
    external.mkdir()
    (project / "work").symlink_to(external, target_is_directory=True)
    image = external / "images/a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fixture")
    assert file_io.resolve_from_project("work/images/a.jpg") == image.resolve()
    assert file_io.rel_to_project(image) == "work/images/a.jpg"


def test_parse_named_source(tmp_path):
    source = tmp_path / "raw images"
    source.mkdir()
    assert parse_source_arg(f"museum={source}") == SourceSpec("museum", source.resolve())
    assert parse_source_arg(str(source)).name == "raw_images"
