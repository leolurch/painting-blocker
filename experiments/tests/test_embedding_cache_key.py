import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.core.adapter_registry import AdapterConfig
from experiments.core.config_schema import DatasetConfig, ModelConfig
from experiments.core.embedding_pipeline import (
    DESCRIPTOR_POSTPROCESSOR_VERSION,
    EMBEDDING_EVALUATOR_VERSION,
    embedding_cache_inputs,
    embedding_cache_key,
)


class EmbeddingCacheKeyTest(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path) -> DatasetConfig:
        return DatasetConfig(
            path=root / "dataset.yml",
            dataset_id="test-dataset",
            dataset_db=root / "dataset.db",
            image_root=root / "images",
            identity={},
            raw={},
        )

    @staticmethod
    def _model(checkpoint_kwargs: dict[str, str]) -> ModelConfig:
        raw = {
            "model_id": "local/test-checkpoint",
            "adapter_kwargs": dict(checkpoint_kwargs),
            "embedding": {"pooling": "default", "normalize": True, "dtype": "float32"},
        }
        adapter = AdapterConfig(
            adapter_name="dinov3_projection_adapter",
            model_id="local/test-checkpoint",
            kwargs=dict(checkpoint_kwargs),
        )
        return ModelConfig(
            path=Path("models.yml"),
            model_id="local/test-checkpoint",
            adapter=adapter,
            raw=raw,
        )

    def test_direct_checkpoint_content_changes_embedding_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"first-checkpoint")
            model = self._model({"checkpoint_path": str(checkpoint)})
            dataset = self._dataset(root)

            first_inputs = embedding_cache_inputs(dataset, "db-sha", ["a.jpg"], model)
            first_key = embedding_cache_key(dataset, "db-sha", ["a.jpg"], model)
            checkpoint.write_bytes(b"second-checkpoint")
            second_inputs = embedding_cache_inputs(dataset, "db-sha", ["a.jpg"], model)
            second_key = embedding_cache_key(dataset, "db-sha", ["a.jpg"], model)

        self.assertNotEqual(first_inputs["local_checkpoint"]["sha256"], second_inputs["local_checkpoint"]["sha256"])
        self.assertNotEqual(first_key, second_key)

    def test_environment_checkpoint_content_changes_embedding_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"version-one")
            model = self._model({"checkpoint_path_env": "SMARTMATCH_CACHE_KEY_TEST_CHECKPOINT"})
            dataset = self._dataset(root)

            with mock.patch.dict(os.environ, {"SMARTMATCH_CACHE_KEY_TEST_CHECKPOINT": str(checkpoint)}):
                first_key = embedding_cache_key(dataset, "db-sha", ["a.jpg"], model)
                checkpoint.write_bytes(b"version-two")
                second_key = embedding_cache_key(dataset, "db-sha", ["a.jpg"], model)

        self.assertNotEqual(first_key, second_key)

    def test_presentation_overrides_do_not_invalidate_embedding_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            adapter = AdapterConfig("siglip2_adapter", "fake/frozen")
            unstyled = ModelConfig(
                path=Path("models.yml"),
                model_id="fake/frozen",
                adapter=adapter,
                raw={"model_id": "fake/frozen"},
            )
            styled = ModelConfig(
                path=Path("models.yml"),
                model_id="fake/frozen",
                adapter=adapter,
                raw={
                    "model_id": "fake/frozen",
                    "label": "Frozen backbone",
                    "color": "#F58518",
                },
                presentation_overrides=frozenset({"label", "color"}),
            )

            self.assertEqual(
                embedding_cache_key(dataset, "db-sha", ["a.jpg"], styled),
                embedding_cache_key(dataset, "db-sha", ["a.jpg"], unstyled),
            )

    def test_embedding_evaluator_version_is_explicit_and_invalidates_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            model = ModelConfig(
                path=Path("models.yml"),
                model_id="fake/frozen",
                adapter=AdapterConfig("siglip2_adapter", "fake/frozen"),
                raw={"model_id": "fake/frozen"},
            )
            inputs = embedding_cache_inputs(dataset, "db-sha", ["a.jpg"], model)
            first_key = embedding_cache_key(dataset, "db-sha", ["a.jpg"], model)
            with mock.patch("experiments.core.embedding_pipeline.EMBEDDING_EVALUATOR_VERSION", EMBEDDING_EVALUATOR_VERSION + 1):
                second_key = embedding_cache_key(dataset, "db-sha", ["a.jpg"], model)

        self.assertEqual(inputs["embedding_evaluator_version"], EMBEDDING_EVALUATOR_VERSION)
        self.assertEqual(
            inputs["descriptor_postprocessing"],
            {
                "descriptor_postprocess_dtype": "float32",
                "normalized": True,
                "normalization": "l2",
                "normalization_axis": 1,
                "normalization_dtype": "float32",
                "descriptor_storage_dtype": "float32",
                "postprocessor_version": DESCRIPTOR_POSTPROCESSOR_VERSION,
            },
        )
        self.assertIsNone(inputs["local_checkpoint"])
        self.assertNotEqual(first_key, second_key)


if __name__ == "__main__":
    unittest.main()
