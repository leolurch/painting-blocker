from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from deduplication import file_io, verification
from deduplication.embedding import embed
from deduplication.ingest import SourceSpec, ingest


class ReuseEmbedder:
    output_dim = 3
    calls = 0

    def __init__(self, _model):
        type(self).calls += 1

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0] for _ in paths], dtype=np.float32)

    def close(self):
        pass


class MustNotRunEmbedder:
    output_dim = 3

    def __init__(self, _model):
        raise AssertionError("content-addressed DINO reuse should avoid model construction")


def test_dino_reuse_maps_changed_image_ids_by_sha(project, config, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (8, 8), "red").save(source / "image.jpg")
    ingest(config, [SourceSpec("old_name", source)])
    ReuseEmbedder.calls = 0
    embed(config, embedder_factory=ReuseEmbedder)
    assert ReuseEmbedder.calls == 1
    old_work = project / "work"

    new_work = project / "new-work"
    file_io.configure_work_dir(new_work)
    new_config = config.__class__(
        **{**config.__dict__, "images_dir": new_work / "images"}
    )
    ingest(new_config, [SourceSpec("new_name", source)])
    metadata = embed(
        new_config,
        embedder_factory=MustNotRunEmbedder,
        reuse_work_dirs=[old_work],
    )
    assert metadata["reused_embedding_count"] == 1
    assert metadata["computed_embedding_count"] == 0


def test_superpoint_migration_accepts_legacy_image_id_key(project, config, tmp_path):
    old_work = tmp_path / "old"
    old_features = old_work / "verification/features"
    old_features.mkdir(parents=True)
    image = project / "work/images/image.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    sha = file_io.sha256_file(image)
    row = {
        "image_id": "old/image.jpg",
        "source_name": "old",
        "raw_relative_path": "image.jpg",
        "shared_filename": "image.jpg",
        "shared_rel_path": str(image),
        "mime_type": "image/jpeg",
        "width": 8,
        "height": 8,
        "bytes_on_disk": image.stat().st_size,
        "sha256": sha,
    }
    from deduplication import manifest

    file_io.atomic_write_csv(old_work / "images.csv", manifest.IMAGES_FIELDS, [row])
    fingerprint = verification.superpoint_fingerprint(config)
    legacy_key = file_io.sha256_json([row["image_id"], sha, fingerprint])
    (old_features / f"{legacy_key}.pt").write_bytes(b"fixture-feature")
    target = tmp_path / "content-features"
    migrated = verification._migrate_feature_reuse(config, target, [old_work])
    content_key = verification.feature_cache_key(sha, fingerprint)
    assert migrated == 1
    assert (target / f"{content_key}.pt").read_bytes() == b"fixture-feature"
