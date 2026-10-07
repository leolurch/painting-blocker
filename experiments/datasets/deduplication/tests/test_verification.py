from __future__ import annotations

import shutil

import numpy as np
from PIL import Image

from deduplication import file_io, manifest
from deduplication import verification
from deduplication.review import prepare_review
from deduplication.retrieval import retrieval_config_sha256
from deduplication.verification import merge_verification, superpoint_fingerprint, verify


def test_score_feature_contract_matches_reference():
    file_io.ensure_lightglue_verified_importable()
    from image_matching.keypoint.lightglue_score_pseudo_pairs import extract_score_features

    scores = np.asarray([0.1, 0.4, 0.9], dtype=np.float64)
    result = extract_score_features(scores)
    assert result.shape == (17,)
    assert result[0] == 3
    assert result[1] == np.mean(scores)
    assert result[-1] == np.sum(scores)
    empty = extract_score_features(np.asarray([]))
    assert empty[0] == 1 and np.all(empty[1:] == 0)


def test_classifier_and_superpoint_fingerprints(config):
    assert file_io.sha256_file(config.classifier_path) == config.classifier_sha256
    assert superpoint_fingerprint(config) == superpoint_fingerprint(config)


def test_cuda_process_pool_uses_spawn_context(monkeypatch):
    captured = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(verification, "ProcessPoolExecutor", FakeExecutor)
    executor = verification._cuda_process_pool(
        max_workers=3,
        initializer=lambda: None,
        initargs=("fixture",),
    )

    assert isinstance(executor, FakeExecutor)
    assert captured["max_workers"] == 3
    assert captured["initializer"] is not None
    assert captured["initargs"] == ("fixture",)
    assert captured["mp_context"].get_start_method() == "spawn"


def test_verification_retains_outcomes_and_retries_failures(project, config, monkeypatch):
    image_dir = project / "work/images"
    image_dir.mkdir(parents=True)
    image_rows = []
    for name, color in (("a.jpg", "red"), ("b.jpg", "blue"), ("fail.jpg", "white")):
        path = image_dir / name
        Image.new("RGB", (4, 4), color).save(path)
        image_rows.append(
            {
                "image_id": f"source/{name}", "source_name": "source", "raw_relative_path": name,
                "shared_filename": name, "shared_rel_path": f"work/images/{name}",
                "mime_type": "image/jpeg", "width": 4, "height": 4,
                "bytes_on_disk": path.stat().st_size, "sha256": file_io.sha256_file(path),
            }
        )
    file_io.atomic_write_csv(project / "work/images.csv", manifest.IMAGES_FIELDS, image_rows)
    candidate_dir = project / "work/candidates"
    candidates = []
    for pair_id, a, b in (("positive", "source/a.jpg", "source/b.jpg"), ("failure", "source/a.jpg", "source/fail.jpg")):
        row = {field: "" for field in manifest.CANDIDATES_FIELDS}
        row.update({"pair_id": pair_id, "image_a_id": a, "image_b_id": b, "cosine_similarity": "0.9"})
        candidates.append(row)
    candidate_path = candidate_dir / "candidates.csv"
    file_io.atomic_write_csv(candidate_path, manifest.CANDIDATES_FIELDS, candidates)
    embedding_metadata = {
        "embeddings_sha256": "fixture-embeddings",
        "image_ids_sha256": "fixture-image-ids",
    }
    file_io.atomic_write_json(project / "work/embeddings/metadata.json", embedding_metadata)
    file_io.atomic_write_json(candidate_dir / "summary.json", {
        "candidates_sha256": file_io.sha256_file(candidate_path),
        "unique_unordered_pair_count": len(candidates),
        "retrieval_config_sha256": retrieval_config_sha256(config, embedding_metadata),
    })

    class Cuda:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def empty_cache(): pass
    class Torch:
        cuda = Cuda()
    class Store:
        def get(self, key, path): return path.name

    fail = {"enabled": True}
    def match(_matcher, a, b):
        if fail["enabled"] and "fail" in b:
            raise RuntimeError("decode failed")
        return np.asarray([0.9, 0.8]) if b == "b.jpg" else np.asarray([0.1])
    def classify(_classifier, scores): return bool(np.mean(scores) > 0.5), float(np.mean(scores))
    monkeypatch.setattr(
        verification,
        "_load_runtime",
        lambda _config, _cache_size=32, _feature_dir=None: (Torch, None, None, None, Store(), match, classify),
    )

    first = verify(config)
    assert first["positive_count"] == 1
    assert first["failed_count"] == 1
    fail["enabled"] = False
    second = verify(config)
    assert second["failed_count"] == 0
    assert second["positive_count"] == 1
    assert second["negative_count"] == 1
    review_path = project / "work/review/review_pairs.csv"
    prepared = prepare_review(review_path)
    assert prepared["review_pair_count"] == 1
    assert file_io.read_csv_rows(review_path)[0]["pair_id"] == "positive"

    # The same candidate set can be split across independently resumable node shards and merged.
    shutil.rmtree(project / "work/verification")
    verify(config, num_shards=2, shard_index=0, merge=False)
    verify(config, num_shards=2, shard_index=1, merge=False)
    merged = merge_verification(config, num_shards=2)
    assert merged["candidate_count"] == 2
    assert merged["failed_count"] == 0
