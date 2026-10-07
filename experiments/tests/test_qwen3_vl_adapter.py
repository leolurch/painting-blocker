import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
from PIL import Image

from experiments.core.validate.embedding_geometry import RESIZE_AND_PAD, ImageGeometry


MODULE = "experiments.core.validate.embedding_adapters.qwen3_vl_adapter"


class FakeProcessor:
    def __init__(self):
        self.conversations = None
        self.processor_kwargs = None
        self.image_processor = types.SimpleNamespace(image_mean=[0.5, 0.5, 0.5])

    def apply_chat_template(
        self, conversations, *, add_generation_prompt, tokenize
    ):
        self.conversations = conversations
        assert add_generation_prompt is True
        assert tokenize is False
        return ["formatted prompt"] * len(conversations)

    def __call__(self, **kwargs):
        self.processor_kwargs = kwargs
        return {"attention_mask": torch.ones((1, 3), dtype=torch.long)}


def _placeholder_process_vision_info(*_args, **_kwargs):
    raise AssertionError("test process_vision_info replacement was not installed")


_package = types.ModuleType("qwen_vl_utils")
_package.__path__ = []
_vision_process = types.ModuleType("qwen_vl_utils.vision_process")
_vision_process.process_vision_info = _placeholder_process_vision_info
with mock.patch.dict(
    sys.modules,
    {
        "qwen_vl_utils": _package,
        "qwen_vl_utils.vision_process": _vision_process,
    },
):
    ADAPTER_MODULE = importlib.import_module(MODULE)


class Qwen3VLEmbeddingAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.process_calls = []
        self.module = ADAPTER_MODULE

        def fake_process_vision_info(conversations, **kwargs):
            self.process_calls.append((conversations, kwargs))
            images = [
                conversation[1]["content"][0]["image"]
                for conversation in conversations
            ]
            return images, None, {"do_sample_frames": False}

        self.module.process_vision_info = fake_process_vision_info

    def _adapter(self, geometry=None):
        adapter = self.module.Qwen3VLEmbeddingAdapter.__new__(
            self.module.Qwen3VLEmbeddingAdapter
        )
        adapter.geometry = geometry
        adapter.processor = FakeProcessor()
        return adapter

    def test_model_dtype_supports_explicit_bf16_without_flash_attention(self) -> None:
        adapter_cls = self.module.Qwen3VLEmbeddingAdapter
        self.assertIs(adapter_cls._resolve_model_dtype("bfloat16"), torch.bfloat16)
        self.assertIs(adapter_cls._resolve_model_dtype("bf16"), torch.bfloat16)
        self.assertIs(adapter_cls._resolve_model_dtype("float16"), torch.float16)
        self.assertIs(adapter_cls._resolve_model_dtype("float32"), torch.float32)
        with self.assertRaises(ValueError):
            adapter_cls._resolve_model_dtype("float64")

    def test_adapter_advertises_the_8b_model(self) -> None:
        adapter_cls = self.module.Qwen3VLEmbeddingAdapter
        self.assertEqual(adapter_cls.SIZE_KEYS, ["8b"])
        self.assertEqual(
            adapter_cls.MODEL_ID_BY_SIZE_KEY,
            {"8b": "Qwen/Qwen3-VL-Embedding-8B"},
        )
        self.assertEqual(adapter_cls.DEFAULT_SIZE_KEY, "8b")
        self.assertEqual(
            adapter_cls.DEFAULT_INSTRUCTION, "Represent the user's input."
        )

    def test_default_preprocessing_uses_qwen_dynamic_resolution_path(self) -> None:
        adapter = self._adapter()
        source = Image.new("RGB", (37, 19), "red")

        adapter._preprocess_batch([source])

        conversation = adapter.processor.conversations[0]
        self.assertEqual(
            conversation[0],
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Represent the user's input."}
                ],
            },
        )
        image_content = conversation[1]["content"][0]
        self.assertEqual(image_content["image"].size, (37, 19))
        self.assertEqual(image_content["min_pixels"], 4096)
        self.assertEqual(image_content["max_pixels"], 1_843_200)

        _, vision_kwargs = self.process_calls[0]
        self.assertEqual(vision_kwargs["image_patch_size"], 16)
        self.assertTrue(vision_kwargs["return_video_metadata"])
        self.assertTrue(vision_kwargs["return_video_kwargs"])
        self.assertFalse(adapter.processor.processor_kwargs["do_resize"])
        self.assertEqual(adapter.processor.processor_kwargs["max_length"], 8192)

    def test_explicit_model_geometry_is_applied_before_qwen_preprocessing(self) -> None:
        adapter = self._adapter(ImageGeometry(RESIZE_AND_PAD, max_width=32))
        source = Image.new("RGB", (40, 20), "blue")

        adapter._preprocess_batch([source])

        processed_image = self.process_calls[0][0][0][1]["content"][0]["image"]
        self.assertEqual(processed_image.size, (32, 32))
        self.assertEqual(processed_image.getpixel((16, 0)), (128, 128, 128))

    def test_pooling_selects_last_non_padding_token_without_adapter_normalization(self) -> None:
        adapter = self._adapter()
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 4.0], [9.0, 9.0]],
                [[0.0, 1.0], [0.0, 2.0], [0.0, 5.0]],
            ]
        )
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        pooled = adapter._pooling_last(hidden, attention_mask)
        result = adapter._embedding_map(pooled, ["default"])["default"]

        torch.testing.assert_close(pooled, torch.tensor([[3.0, 4.0], [0.0, 5.0]]))
        torch.testing.assert_close(result, torch.tensor([[3.0, 4.0], [0.0, 5.0]]))

    def test_checked_in_model_config_uses_project_preprocessing(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs" / "models" / "qwen3_vl.yml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(config["models"]), 1)
        model = config["models"][0]
        self.assertEqual(model["model_id"], "Qwen/Qwen3-VL-Embedding-8B")
        self.assertEqual(model["embedding"]["output_dim"], 4096)
        self.assertEqual(model["preprocessing"]["resize"], {"mode": "max_side", "size": 512})


if __name__ == "__main__":
    unittest.main()
