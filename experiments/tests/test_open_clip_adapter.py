import unittest
from types import SimpleNamespace
from unittest import mock

from experiments.core.adapter_registry import known_adapter_names
from experiments.core.validate.embedding_adapters.dino_adapter import DinoV3Adapter
from experiments.core.validate.embedding_adapters.open_clip_adapter import OpenClipAdapter
from experiments.core.validate.embedding_pooling import normalize_pooling_modes


class OpenClipAdapterConfigTest(unittest.TestCase):
    def test_supported_model_ids_match_requested_laion_checkpoints(self):
        self.assertEqual(
            list(OpenClipAdapter.MODEL_ID_BY_SIZE_KEY.values()),
            [
                "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
                "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
                "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            ],
        )

    def test_full_hf_ids_resolve_as_known_variants(self):
        for size_key, model_id in OpenClipAdapter.MODEL_ID_BY_SIZE_KEY.items():
            with self.subTest(model_id=model_id):
                self.assertEqual(OpenClipAdapter._resolve_size_key(None, model_id), size_key)
                self.assertEqual(
                    OpenClipAdapter._resolve_size_key(None, f"hf-hub:{model_id}"), size_key
                )
                self.assertEqual(OpenClipAdapter._resolve_model_id(model_id, None), model_id)

    def test_registry_uses_adapter_module_filenames(self):
        self.assertIn("open_clip_adapter", known_adapter_names())

    def test_invalid_model_variant_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported OpenCLIP size key"):
            OpenClipAdapter._resolve_size_key(None, "vit-z-99")

    def test_adapter_poolings_are_normalized_by_adapter_class(self):
        modes = normalize_pooling_modes(["default", "mean", "max"], OpenClipAdapter.SUPPORTED_POOLINGS)
        self.assertEqual(modes, ("default", "avg", "max"))
        self.assertEqual(OpenClipAdapter.SUPPORTED_POOLINGS, DinoV3Adapter.SUPPORTED_POOLINGS)

    def test_unsupported_pooling_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported pooling mode"):
            normalize_pooling_modes(["default", "avg"], frozenset({"default"}))

    def test_force_image_size_is_forwarded_to_open_clip_loader(self):
        adapter = object.__new__(OpenClipAdapter)
        adapter.model_id = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        adapter.revision = "a" * 40
        adapter.device = "cuda"
        adapter.force_image_size = 512
        adapter._patch_open_clip_hf_revision = mock.Mock()
        model = SimpleNamespace()
        open_clip = SimpleNamespace(
            create_model_and_transforms=mock.Mock(
                return_value=(model, None, "preprocess")
            )
        )

        loaded_model, preprocess = adapter._load_model(open_clip)

        self.assertIs(loaded_model, model)
        self.assertEqual(preprocess, "preprocess")
        open_clip.create_model_and_transforms.assert_called_once_with(
            f"hf-hub:{adapter.model_id}@{adapter.revision}",
            device="cuda",
            force_image_size=512,
        )
        self.assertEqual(OpenClipAdapter._validate_force_image_size(512), 512)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            OpenClipAdapter._validate_force_image_size(0)


if __name__ == "__main__":
    unittest.main()
