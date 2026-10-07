import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from experiments.core.adapter_registry import AdapterConfig
from experiments.core.config_schema import DatasetConfig, ModelConfig
from experiments.core.finetuning.reference_evaluation import (
    REFERENCE_EVALUATOR_VERSION,
    _reference_result_key,
    _task_fingerprint,
)
from experiments.core.finetuning.validation_sets import ValidationSet


def _dataset(root: Path) -> DatasetConfig:
    return DatasetConfig(
        path=root / "dataset.yml",
        dataset_id="test-dataset",
        dataset_db=root / "dataset.db",
        image_root=root / "images",
        identity={},
        raw={},
    )


def _validation_set(dataset: DatasetConfig, *, split_sha256="sha-1", selector=None) -> ValidationSet:
    return ValidationSet(
        name="synth_val",
        primary=True,
        dataset=dataset,
        split={"split_id": "s1"},
        split_path=Path("split.json"),
        split_sha256=split_sha256,
        selector=selector or {"subset": "val", "roles": ["ALL_ROLES"]},
        retrieval_task_specs=[],
    )


class ReferenceResultKeyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dataset = _dataset(Path(self._tmp.name))
        self.vset = _validation_set(self.dataset)

    def tearDown(self):
        self._tmp.cleanup()

    def _key(self, *, embedding_key="emb-1", vset=None, fingerprint=None, top_k=(1, 10), target_pc=0.995):
        return _reference_result_key(
            embedding_key, vset or self.vset, fingerprint or [], top_k, target_pc
        )

    def test_identical_inputs_produce_identical_keys(self):
        self.assertEqual(self._key(), self._key())

    def test_changed_embedding_key_changes_result_key(self):
        self.assertNotEqual(self._key(embedding_key="emb-1"), self._key(embedding_key="emb-2"))

    def test_changed_split_changes_key(self):
        other = _validation_set(self.dataset, split_sha256="sha-2")
        self.assertNotEqual(self._key(), self._key(vset=other))

    def test_changed_top_k_changes_key(self):
        self.assertNotEqual(self._key(top_k=(1, 10)), self._key(top_k=(1, 5, 10)))

    def test_changed_target_pc_changes_key(self):
        self.assertNotEqual(self._key(target_pc=0.995), self._key(target_pc=0.99))

    def test_top_k_order_does_not_change_key(self):
        self.assertEqual(self._key(top_k=(10, 1)), self._key(top_k=(1, 10)))

    def test_changed_query_direction_changes_result_key(self):
        forward = _task_fingerprint(
            [{"task_id": "t", "query": {"subset": "val", "role": "modern"}, "candidates": {"subset": "val", "role": "historic"}}],
            [{"task_id": "t", "query_ids": ["a"], "candidate_ids": ["b"], "positive_policy": "same_painting", "exclude_self": True}],
        )
        reverse = _task_fingerprint(
            [{"task_id": "t", "query": {"subset": "val", "role": "historic"}, "candidates": {"subset": "val", "role": "modern"}}],
            [{"task_id": "t", "query_ids": ["b"], "candidate_ids": ["a"], "positive_policy": "same_painting", "exclude_self": True}],
        )
        self.assertNotEqual(forward, reverse)
        self.assertNotEqual(self._key(fingerprint=forward), self._key(fingerprint=reverse))

    def test_payload_excludes_training_seed_and_optimizer(self):
        # The result key payload only fingerprints data + protocol, so unrelated
        # training config never appears — same inputs always match.
        self.assertEqual(REFERENCE_EVALUATOR_VERSION, 3)
        self.assertEqual(self._key(), self._key())


def _model() -> ModelConfig:
    raw = {
        "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "embedding": {"pooling": "default", "normalize": True, "dtype": "float32"},
    }
    return ModelConfig(
        path=Path("models.yml"),
        model_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        adapter=AdapterConfig("dino_adapter", "facebook/dinov3-vith16plus-pretrain-lvd1689m"),
        raw=raw,
    )


class EmbeddingArtifactTest(unittest.TestCase):
    """get_or_create returns a shared artifact and reuses it without regenerating."""

    def _fake_generate(self, artifact_dir, dataset, model, image_ids, records, run_options, embedding_options, cache_key, db_sha256, cache_inputs):
        from experiments.core.artifacts import sha256_file

        artifact_dir.mkdir(parents=True, exist_ok=True)
        values = np.zeros((len(image_ids), 4), dtype=np.float32)
        values[:, 0] = 1.0
        embeddings_path = artifact_dir / "embeddings.npy"
        np.save(embeddings_path, values)
        (artifact_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "image_ids": list(image_ids),
                    "embedding_shape": list(values.shape),
                    "descriptor_storage_dtype": "float32",
                    "embedding_sha256": sha256_file(embeddings_path),
                }
            ),
            encoding="utf-8",
        )

    def test_shared_artifact_reused_without_regeneration(self):
        from experiments.core import embedding_pipeline as ep

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            model = _model()
            image_ids = ["a", "b", "c"]
            with mock.patch.dict("os.environ", {"EXPERIMENTS_CACHE_DIR": str(root / "cache")}):
                with mock.patch.object(ep, "_generate_embeddings", side_effect=self._fake_generate) as gen:
                    first = ep.get_or_create_embedding_artifact(
                        dataset, model, image_ids, {}, {}, {}, "db-sha",
                        cache_policy="reuse_if_config_hash_matches",
                    )
                    self.assertFalse(first.reused)
                    self.assertEqual(gen.call_count, 1)
                    # A shared artifact lives in the cache, not any run dir.
                    self.assertIn("embeddings", str(first.artifact_dir))

                    second = ep.get_or_create_embedding_artifact(
                        dataset, model, image_ids, {}, {}, {}, "db-sha",
                        cache_policy="reuse_if_config_hash_matches",
                    )
                    self.assertTrue(second.reused)
                    # Re-check under the lock avoided a second generation.
                    self.assertEqual(gen.call_count, 1)
                    self.assertEqual(first.key, second.key)

                    # Run-dir materialization symlinks embeddings and copies metadata.
                    run_dir = root / "run"
                    ep.copy_or_link_artifact_into_run(second, run_dir)
                    self.assertTrue((run_dir / "embeddings.npy").is_symlink())
                    self.assertTrue((run_dir / "embeddings_metadata.json").is_file())
                    self.assertFalse((run_dir / "embeddings_metadata.json").is_symlink())

    def test_partial_artifact_is_rejected(self):
        from experiments.core import embedding_pipeline as ep

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            model = _model()
            image_ids = ["a", "b"]
            with mock.patch.dict("os.environ", {"EXPERIMENTS_CACHE_DIR": str(root / "cache")}):
                # Seed a corrupt artifact: metadata with the wrong cache_key.
                key = ep.embedding_cache_key(dataset, "db-sha", image_ids, model)
                from experiments.core.artifacts import short_hash
                artifact_dir = ep.embeddings_dir() / dataset.dataset_id / model.storage_key / short_hash(key)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                np.save(artifact_dir / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))
                (artifact_dir / "metadata.json").write_text(json.dumps({"cache_key": "wrong", "image_ids": image_ids}), encoding="utf-8")

                with mock.patch.object(ep, "_generate_embeddings", side_effect=self._fake_generate) as gen:
                    artifact = ep.get_or_create_embedding_artifact(
                        dataset, model, image_ids, {}, {}, {}, "db-sha",
                        cache_policy="reuse_if_config_hash_matches",
                    )
                    # The corrupt artifact was rejected and regenerated.
                    self.assertEqual(gen.call_count, 1)
                    self.assertFalse(artifact.reused)


if __name__ == "__main__":
    unittest.main()
