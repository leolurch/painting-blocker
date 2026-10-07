import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from experiments.core.adapter_registry import known_adapter_names
from experiments.core.config_schema import load_model_reference
from experiments.core.validate.embedding_adapters.resnet50_hf_adapter import ResNet50HFAdapter
from experiments.core.validate.embedding_geometry import RESIZE_AND_PAD, ImageGeometry
from experiments.core.validate.embedding_pooling import normalize_pooling_modes


class FakeProcessor:
    crop_size = {"height": 224, "width": 224}

    def __init__(self):
        self.last_images = []
        self.last_kwargs = {}

    def __call__(self, images, return_tensors, **kwargs):
        self.last_images = images
        self.last_kwargs = kwargs
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(hidden_sizes=[2])
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        batch_size = int(inputs["pixel_values"].shape[0])
        values = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)
        return SimpleNamespace(pooler_output=values[:batch_size].reshape(batch_size, 2, 1, 1))


class ResNet50HFAdapterTest(unittest.TestCase):
    def test_checked_in_model_config_uses_resnet_adapter(self):
        root = Path(__file__).resolve().parents[1]
        model = load_model_reference(
            {"config": str(root / "configs" / "models" / "resnet50_hf.yml"), "model_id": "microsoft/resnet-50"},
            root,
        )

        self.assertIn("resnet50_hf_adapter", known_adapter_names())
        self.assertEqual(model.adapter.adapter_name, "resnet50_hf_adapter")
        self.assertEqual(model.raw["embedding"]["output_dim"], 2048)
        self.assertEqual(model.raw["embedding"]["dtype"], "float32")
        self.assertEqual(model.raw["embedding"]["pooling"], "pooler_output")

    def test_pooler_output_is_flattened_without_adapter_normalization(self):
        fake_processor = FakeProcessor()
        fake_model = FakeModel()
        with mock.patch.dict("os.environ", {"HF_TOKEN": ""}), mock.patch(
            "experiments.core.validate.embedding_adapters.resnet50_hf_adapter.torch.cuda.is_available",
            return_value=False,
        ), mock.patch(
            "experiments.core.validate.embedding_adapters.resnet50_hf_adapter.torch.backends.mps.is_available",
            return_value=False,
        ), mock.patch(
            "experiments.core.validate.embedding_adapters.resnet50_hf_adapter.AutoImageProcessor.from_pretrained",
            return_value=fake_processor,
        ) as processor_from_pretrained, mock.patch(
            "experiments.core.validate.embedding_adapters.resnet50_hf_adapter.AutoModel.from_pretrained",
            return_value=fake_model,
        ) as model_from_pretrained:
            adapter = ResNet50HFAdapter(model_id="microsoft/resnet-50")
            embeddings = adapter.generate_embeddings_batch_from_pil(
                [Image.new("L", (2, 2)), Image.new("RGB", (2, 2))],
                ["default", "pooler_output"],
            )

        processor_from_pretrained.assert_called_once_with(
            "microsoft/resnet-50", use_fast=True
        )
        model_from_pretrained.assert_called_once_with("microsoft/resnet-50")
        self.assertEqual(adapter.device, "cpu")
        self.assertEqual(fake_model.device, "cpu")
        self.assertTrue(fake_model.eval_called)
        self.assertEqual(adapter.get_dimension(), 2)
        self.assertTrue(all(image.mode == "RGB" for image in fake_processor.last_images))
        np.testing.assert_allclose(
            embeddings["default"],
            np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
            rtol=1e-6,
        )
        np.testing.assert_allclose(embeddings["pooler_output"], embeddings["default"])
        self.assertEqual(fake_processor.last_kwargs, {})

    def test_letterbox_disables_native_resize_and_center_crop(self):
        adapter = object.__new__(ResNet50HFAdapter)
        adapter.geometry = ImageGeometry(RESIZE_AND_PAD, max_width=224)
        adapter.processor = FakeProcessor()
        adapter.device = "cpu"

        adapter._prepare_inputs([Image.new("RGB", (400, 200), "red")])

        self.assertEqual(adapter.processor.last_images[0].size, (224, 224))
        self.assertEqual(
            adapter.processor.last_kwargs,
            {"do_resize": False, "do_center_crop": False},
        )

    def test_model_id_and_pooling_validation_are_clear(self):
        self.assertEqual(ResNet50HFAdapter._resolve_model_id(None), "microsoft/resnet-50")
        self.assertEqual(ResNet50HFAdapter._resolve_size_key("resnet-50"), "resnet50")
        with self.assertRaisesRegex(ValueError, "Unsupported ResNet-50 HF model_id"):
            ResNet50HFAdapter._resolve_model_id("microsoft/resnet-101")
        with self.assertRaisesRegex(ValueError, "Unsupported pooling mode"):
            normalize_pooling_modes(["cls"], ResNet50HFAdapter.SUPPORTED_POOLINGS)


if __name__ == "__main__":
    unittest.main()
