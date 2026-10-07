from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from deduplication import file_io
from deduplication.embedding import embed
from deduplication.ingest import SourceSpec, ingest
from deduplication.pipeline import detect_set_mode
from deduplication.retrieval import _part_rows, retrieve, write_exact_hash_artifacts


class NameEmbedder:
    output_dim = 3

    def __init__(self, _model):
        pass

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        values = []
        for path in paths:
            if "red" in path.name:
                values.append([1.0, 0.0, 0.0])
            elif "green" in path.name:
                values.append([0.0, 1.0, 0.0])
            else:
                values.append([0.0, 0.0, 1.0])
        return np.asarray(values, dtype=np.float32)

    def close(self):
        pass


def test_threshold_or_window_and_deterministic_ties(config):
    cfg = replace(config, retrieval=replace(config.retrieval, min_retrieval_window=1))
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.7, np.sqrt(0.51)], [0.0, 1.0]], dtype=np.float32
    )
    ids = ["q", "b", "a", "z"]
    hashes = {image_id: image_id * 32 for image_id in ids}
    rows, threshold_count, window_only = _part_rows(
        embeddings,
        ids,
        hashes,
        0,
        1,
        cfg,
        "retrieval",
        query_indices=[0],
        candidate_indices=[1, 2, 3],
        mask_self=False,
    )
    # Both scores over 0.5 survive even though the minimum window is only one.
    assert threshold_count == 2
    assert window_only == 0
    assert {row["image_b_id"] if row["image_a_id"] == "q" else row["image_a_id"] for row in rows} == {"a", "b"}

    tied = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    tied_ids = ["q", "b", "a"]
    tied_hashes = {image_id: image_id * 32 for image_id in tied_ids}
    rows, _, _ = _part_rows(
        tied,
        tied_ids,
        tied_hashes,
        0,
        1,
        replace(cfg, retrieval=replace(cfg.retrieval, cosine_threshold=0.5)),
        "retrieval",
        query_indices=[0],
        candidate_indices=[1, 2],
        mask_self=False,
    )
    assert len(rows) == 1
    assert rows[0]["image_a_id"] == "a"  # image-id tie breaker beats input index


def test_bipartite_retrieval_emits_cross_set_pairs_only(project, config, tmp_path):
    set_a = tmp_path / "a"
    set_b = tmp_path / "b"
    set_a.mkdir()
    set_b.mkdir()
    Image.new("RGB", (8, 8), "red").save(set_a / "red_a.jpg")
    Image.new("RGB", (8, 8), "blue").save(set_a / "blue_a.jpg")
    Image.new("RGB", (8, 8), "red").save(set_b / "red_b.jpg")
    Image.new("RGB", (8, 8), "green").save(set_b / "green_b.jpg")
    Image.new("RGB", (8, 8), "blue").save(set_b / "blue_b.jpg")

    ingest(config, [SourceSpec("set_a", set_a), SourceSpec("set_b", set_b)])
    embed(config, embedder_factory=NameEmbedder)
    exact = write_exact_hash_artifacts(mode="bipartite")
    assert exact["exact_hash_pair_count"] == 2
    summary = retrieve(config, mode="bipartite")
    assert summary["maximum_possible_pair_count"] == 6
    rows = file_io.read_csv_rows(project / "work/candidates/candidates.csv")
    assert rows
    images = {row["image_id"]: row["source_name"] for row in file_io.read_csv_rows(project / "work/images.csv")}
    assert all({images[row["image_a_id"]], images[row["image_b_id"]]} == {"set_a", "set_b"} for row in rows)


def test_set_mode_detects_samefile_and_rejects_nested(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    nested = a / "nested"
    a.mkdir()
    b.mkdir()
    nested.mkdir()
    assert detect_set_mode(a, a)[0] == "symmetric"
    assert detect_set_mode(a, b)[0] == "bipartite"
    try:
        detect_set_mode(a, nested)
    except ValueError as exc:
        assert "contain" in str(exc)
    else:
        raise AssertionError("nested sets were not rejected")
