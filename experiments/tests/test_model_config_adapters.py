import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from PIL import Image

from experiments.core.adapter_registry import AdapterConfig, instantiate_adapter
from experiments.core.config_schema import DatasetConfig, ModelConfig, load_experiment_config, load_model_reference
from experiments.core.split_schema import SplitImage
from experiments.core.embedding_pipeline import _run_batches


class ModelConfigAdapterTest(unittest.TestCase):
    def test_model_yaml_controls_adapter_constructor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "siglip2_adapter",
                        "models": [
                            {
                                "model_id": "google/siglip2-giant-opt-patch16-384",
                                "adapter_kwargs": {"attn_implementation": "sdpa"},
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            model = load_model_reference(
                {"config": str(model_path), "model_id": "google/siglip2-giant-opt-patch16-384"},
                Path(tmp),
            )

            self.assertEqual(model.model_id, "google/siglip2-giant-opt-patch16-384")
            self.assertEqual(model.adapter.adapter_name, "siglip2_adapter")
            self.assertEqual(model.adapter.model_id, model.model_id)
            self.assertEqual(model.adapter.kwargs, {"attn_implementation": "sdpa"})
            self.assertEqual(model.adapter.identity(), "siglip2_adapter:google/siglip2-giant-opt-patch16-384")
            self.assertNotIn("/", model.storage_key)

    def test_experiment_model_label_and_color_are_resolved_without_changing_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "siglip2_adapter",
                        "models": [
                            {
                                "model_id": "google/siglip2-giant-opt-patch16-384",
                                "revision": "abc123",
                                "embedding": {"normalize": True},
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            base_reference = {
                "config": str(model_path),
                "model_id": "google/siglip2-giant-opt-patch16-384",
            }
            unstyled = load_model_reference(base_reference, Path(tmp))
            styled = load_model_reference(
                {**base_reference, "label": "Frozen backbone", "color": "#F58518"},
                Path(tmp),
            )

            self.assertEqual(styled.raw["label"], "Frozen backbone")
            self.assertEqual(styled.raw["color"], "#F58518")
            self.assertEqual(styled.presentation_overrides, frozenset({"label", "color"}))
            self.assertNotIn("label", styled.identity_raw)
            self.assertNotIn("color", styled.identity_raw)
            self.assertEqual(styled.storage_key, unstyled.storage_key)

    def test_experiment_model_group_is_presentation_only_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "siglip2_adapter",
                        "models": [{"model_id": "google/siglip2-giant-opt-patch16-384"}],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            base_reference = {
                "config": str(model_path),
                "model_id": "google/siglip2-giant-opt-patch16-384",
            }
            plain = load_model_reference(base_reference, Path(tmp))
            grouped = load_model_reference({**base_reference, "group": "frozen"}, Path(tmp))

            self.assertEqual(grouped.raw["group"], "frozen")
            self.assertIn("group", grouped.presentation_overrides)
            self.assertNotIn("group", grouped.identity_raw)
            # group is styling only: the storage key (and embedding caches) are unchanged.
            self.assertEqual(grouped.storage_key, plain.storage_key)

            with self.assertRaises(ValueError):
                load_model_reference({**base_reference, "group": "adapted"}, Path(tmp))

    def test_experiment_variant_id_changes_result_identity_not_adapter_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "siglip2_adapter",
                        "models": [
                            {
                                "model_id": "google/siglip2-giant-opt-patch16-384",
                                "revision": "abc123",
                                "preprocessing": {"resize": {"mode": "letterbox", "size": 384}},
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            variant = load_model_reference(
                {
                    "config": str(model_path),
                    "model_id": "google/siglip2-giant-opt-patch16-384",
                    "variant_id": "model-default/google/siglip2-giant-opt-patch16-384",
                    "preprocessing": {"resize": {"mode": "model_default"}},
                },
                Path(tmp),
            )

            self.assertEqual(variant.model_id, "model-default/google/siglip2-giant-opt-patch16-384")
            self.assertEqual(variant.adapter.model_id, "google/siglip2-giant-opt-patch16-384")
            self.assertEqual(variant.raw["source_model_id"], "google/siglip2-giant-opt-patch16-384")
            self.assertEqual(variant.raw["preprocessing"], {"resize": {"mode": "model_default"}})
            self.assertIn("model-default_google_siglip2", variant.storage_key)

    def test_experiment_model_color_must_be_a_non_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "siglip2_adapter",
                        "models": [{"model_id": "google/siglip2-giant-opt-patch16-384"}],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "color must be a non-empty string"):
                load_model_reference(
                    {
                        "config": str(model_path),
                        "model_id": "google/siglip2-giant-opt-patch16-384",
                        "color": " ",
                    },
                    Path(tmp),
                )

    def test_string_model_references_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "models.include entries must be mappings"):
            load_model_reference("models.yml::toy", Path("."))

    def test_model_file_without_adapter_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {"schema_version": 1, "models": [{"model_id": "toy"}]},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Missing required key 'adapter_name'"):
                load_model_reference({"config": str(model_path), "model_id": "toy"}, Path(tmp))

    def test_adapter_kwargs_reject_duplicate_model_variant_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "siglip2_adapter",
                        "models": [
                            {
                                "model_id": "google/siglip2-giant-opt-patch16-384",
                                "adapter_kwargs": {"model_id": "other", "size_key": "b"},
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "adapter_kwargs must not contain"):
                load_model_reference(
                    {"config": str(model_path), "model_id": "google/siglip2-giant-opt-patch16-384"},
                    Path(tmp),
                )

    def test_adapter_runtime_metadata_excludes_injected_objects_and_tokens(self) -> None:
        class FakeAdapter:
            SUPPORTED_POOLINGS = frozenset({"default"})

            def __init__(self, model_id, geometry=None, hf_token=None, use_compile=True, custom=None, revision=None):
                self.model_id = model_id
                self.geometry = geometry
                self.hf_token = hf_token
                self.use_compile = use_compile
                self.custom = custom
                self.revision = revision

        with mock.patch(
            "experiments.core.adapter_registry._load_adapter_class",
            return_value=FakeAdapter,
        ):
            adapter, _pooling, metadata = instantiate_adapter(
                AdapterConfig("fake_adapter", "global/model", {"custom": "x"}, revision="abc123"),
                geometry=object(),
                pooling="default",
                embedding_options={"hf_token": "secret"},
            )

        self.assertEqual(adapter.model_id, "global/model")
        self.assertEqual(adapter.hf_token, "secret")
        self.assertEqual(adapter.custom, "x")
        self.assertEqual(adapter.revision, "abc123")
        self.assertEqual(metadata["model_id"], "global/model")
        self.assertEqual(metadata["revision"], "abc123")
        self.assertEqual(metadata["kwargs"], {"custom": "x"})

    def test_batch_adapter_errors_are_not_hidden_by_single_image_fallback(self) -> None:
        class FailingBatchAdapter:
            def generate_embeddings_batch_from_pil(self, _images, _pooling_modes):
                raise RuntimeError("batch failed")

            def generate_embedding_from_pil(self, _image, _pooling_modes):
                raise AssertionError("single-image fallback should not be used")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(root / "a.jpg")
            dataset = DatasetConfig(Path("dataset.yml"), "toy", root / "dataset.db", root, {}, {})
            model = ModelConfig(
                Path("model.yml"),
                "toy/model",
                AdapterConfig("siglip2_adapter", "toy/model"),
                {"embedding": {"batch_size": 1}},
            )
            records = {"a.jpg": SplitImage("a.jpg", 1)}

            with self.assertRaisesRegex(RuntimeError, "batch failed"):
                _run_batches(
                    FailingBatchAdapter(),
                    ["a.jpg"],
                    records,
                    dataset,
                    model,
                    ("default",),
                    {"fail_on_missing_images": True},
                    {},
                )

    def test_checked_in_configs_use_file_level_adapter_names_and_global_model_ids(self) -> None:
        root = Path(__file__).resolve().parents[1]
        experiment_path = root / "configs" / "experiments" / "baseline_mini10.yml"
        experiment = load_experiment_config(experiment_path)

        self.assertTrue(experiment.models)
        for model in experiment.models:
            with self.subTest(model_id=model.model_id):
                self.assertIn("/", model.model_id)
                self.assertEqual(model.adapter.model_id, model.model_id)
                self.assertEqual(model.adapter.revision, model.revision)
                self.assertIsNotNone(model.revision)
                self.assertNotIn("adapter", model.raw)
                for duplicate_key in ("adapter_selector", "backend", "family", "model_name", "pretrained", "size_key"):
                    self.assertNotIn(duplicate_key, model.raw)
                self.assertNotIn("model_id", model.adapter.kwargs)
                self.assertNotIn("size_key", model.adapter.kwargs)
        self.assertIn("environment", experiment.resolved)

    def test_resolved_config_records_env_derived_behavior(self) -> None:
        root = Path(__file__).resolve().parents[1]
        experiment_path = root / "configs" / "experiments" / "baseline_mini10.yml"
        with mock.patch.dict("os.environ", {"SIGLIP2_MODEL_ID": "google/siglip2-giant-opt-patch16-384", "HF_TOKEN": "super-token"}):
            experiment = load_experiment_config(experiment_path)
        env = experiment.resolved["environment"]
        self.assertEqual(env["variables"]["SIGLIP2_MODEL_ID"], "google/siglip2-giant-opt-patch16-384")
        self.assertTrue(env["secret_presence"]["HF_TOKEN"])
        self.assertNotIn("super-token", yaml.safe_dump(env))


if __name__ == "__main__":
    unittest.main()
