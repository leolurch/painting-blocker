from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from deduplication import file_io
from deduplication.embedding import embed, letterbox_image
from deduplication.ingest import SourceSpec, ingest
from deduplication.retrieval import _part_rows, retrieve


class FixtureEmbedder:
    output_dim = 3
    calls = 0

    def __init__(self, model):
        type(self).calls += 1

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0, 0.0] if "duplicate" in path.name else [0.0, 1.0, 0.0] for path in paths],
            dtype=np.float32,
        )

    def close(self):
        pass


def test_letterbox_contract():
    result = letterbox_image(Image.new("L", (20, 10), 255), 32)
    assert result.mode == "RGB"
    assert result.size == (32, 32)
    assert result.getpixel((0, 0)) != result.getpixel((0, 16))


def test_embed_resume_and_symmetric_retrieval(project, config, tmp_path):
    source = tmp_path / "alpha"
    source.mkdir()
    Image.new("RGB", (10, 10), "red").save(source / "duplicate_a.jpg")
    Image.new("RGB", (10, 10), "red").save(source / "duplicate_b.jpg")
    Image.new("RGB", (10, 10), "blue").save(source / "single.jpg")
    ingest(config, [SourceSpec("alpha", source)])

    FixtureEmbedder.calls = 0
    metadata = embed(config, embedder_factory=FixtureEmbedder)
    assert metadata["shape"] == [3, 3]
    assert FixtureEmbedder.calls == 1
    embed(config, embedder_factory=FixtureEmbedder)
    assert FixtureEmbedder.calls == 1  # all finalized parts were reused

    summary = retrieve(config)
    # Every threshold match OR each query's two-neighbour minimum window is retained.
    assert summary["unique_unordered_pair_count"] == 3
    rows = file_io.read_csv_rows(project / "work/candidates/candidates.csv")
    duplicate = next(
        row for row in rows
        if "duplicate_a" in row["image_a_id"] and "duplicate_b" in row["image_b_id"]
    )
    assert duplicate["image_a_id"] < duplicate["image_b_id"]
    assert duplicate["retrieved_from_a"] == duplicate["retrieved_from_b"] == "1"
    assert duplicate["rank_a_to_b"] == duplicate["rank_b_to_a"] == "1"
    assert duplicate["dino_threshold_match"] == "1"


def test_embed_shards_merge_to_the_same_matrix(project, config, tmp_path):
    source = tmp_path / "alpha"
    source.mkdir()
    for name, color in (("a.jpg", "red"), ("b.jpg", "green"), ("c.jpg", "blue")):
        Image.new("RGB", (8, 8), color).save(source / name)
    ingest(config, [SourceSpec("alpha", source)])

    FixtureEmbedder.calls = 0
    first = embed(config, embedder_factory=FixtureEmbedder, num_shards=2, shard_index=0)
    second = embed(config, embedder_factory=FixtureEmbedder, num_shards=2, shard_index=1)
    assert first["complete"] and second["complete"]
    assert first["owned_part_count"] == 1 and second["owned_part_count"] == 1
    assert not (project / "work/embeddings/embeddings.npy").exists()
    merged = embed(config, embedder_factory=FixtureEmbedder, merge_only=True)
    assert merged["shape"] == [3, 3]
    assert FixtureEmbedder.calls == 2


def test_threshold_boundary_and_self_exclusion(config):
    embeddings = np.asarray([[1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=np.float32)
    ids = ["a", "b"]
    rows, directed, _ = _part_rows(embeddings, ids, {"a": "aa", "b": "bb"}, 0, 2, config, "cfg")
    assert len(rows) == 1
    assert directed == 2
    assert rows[0]["image_a_id"] != rows[0]["image_b_id"]
