"""Unit tests for the SigLIP2 q/v LoRA embedding model.

A tiny randomly initialized ``SiglipVisionModel`` replaces the pretrained
weights, so the real SigLIP module names, PEFT wrapping, attention-pooling
head, and position interpolation are exercised without a download or a GPU.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from experiments.core.checkpoint_naming import checkpoint_name_from_finetuning, parse_checkpoint_name
from experiments.core.finetuning.checkpointing import build_trainable_model
from experiments.core.finetuning.models import siglip2 as siglip2_models
from experiments.core.finetuning.models.base import trainable_parameter_groups
from experiments.core.finetuning.models.dinov3 import _layer_index
from experiments.core.finetuning.models.projection import ProjectionHead
from experiments.core.finetuning.train import _unexpected_lora_trainable_names

HIDDEN = 32
LAYERS = 6
IMAGE = 32


def _tiny_backbone(*_args, **_kwargs) -> SiglipVisionModel:
    torch.manual_seed(0)
    config = SiglipVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        image_size=IMAGE,
        patch_size=8,
    )
    return SiglipVisionModel(config).eval()


def _config(*, trainable: bool = False, pooling: str = "map", span: str = "last3") -> dict:
    return {
        "type": "siglip2_lora_qv_projection",
        "backbone": "google/siglip2-giant-opt-patch16-384",
        "revision": "a713301b217d38485fb2204c808367d10bc3cc40",
        "pooling": {"type": pooling, "trainable": trainable},
        "lora": {"target_modules": ["q_proj", "v_proj"], "span": span, "rank": 4, "alpha": 4},
        "head": {"type": "mlp_projection", "dropout": 0.0},
    }


def _build(config: dict):
    with mock.patch.object(siglip2_models.Siglip2LoRAEmbeddingModel, "_load_backbone", side_effect=_tiny_backbone):
        return build_trainable_model(config)


class LayerIndexTest(unittest.TestCase):
    def test_dinov3_and_siglip_block_names(self):
        self.assertEqual(_layer_index("model.layer.31.attention.q_proj"), 31)
        self.assertEqual(_layer_index("vision_model.encoder.layers.7.self_attn.v_proj"), 7)
        self.assertIsNone(_layer_index("vision_model.head.attention.out_proj"))


class Siglip2LoRAModelTest(unittest.TestCase):
    def test_lora_targets_last_blocks_only(self):
        model = _build(_config(span="last3"))
        self.assertEqual(model.lora_config["adapted_layer_indices"], [3, 4, 5])
        self.assertEqual(len(model.lora_config["matched_modules"]), 6)
        for name in model.lora_config["matched_modules"]:
            self.assertTrue(name.startswith("vision_model.encoder.layers."), name)
            self.assertTrue(name.endswith(("self_attn.q_proj", "self_attn.v_proj")), name)

    def test_default_head_doubles_hidden_and_keeps_width(self):
        model = _build(_config())
        self.assertIsInstance(model.projection, ProjectionHead)
        self.assertEqual(model.head_hidden_dim, 2 * HIDDEN)
        self.assertEqual(model.projection_dim, HIDDEN)

    def test_frozen_pooling_head_trains_only_lora_and_head(self):
        model = _build(_config(trainable=False))
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name or name.startswith("projection.") for name in trainable), trainable)
        self.assertEqual(_unexpected_lora_trainable_names(model), [])

    def test_trainable_pooling_head_uses_backbone_learning_rate_group(self):
        model = _build(_config(trainable=True))
        map_params = [p for name, p in model.named_parameters() if siglip2_models.POOLING_HEAD_PREFIX in name]
        self.assertTrue(map_params)
        self.assertTrue(all(p.requires_grad for p in map_params))
        self.assertEqual(_unexpected_lora_trainable_names(model), [])
        backbone_group, head_group = trainable_parameter_groups(model)
        backbone_ids = {id(p) for p in backbone_group}
        head_ids = {id(p) for p in head_group}
        self.assertTrue(all(id(p) in backbone_ids for p in map_params))
        projection_ids = {id(p) for p in model.projection.parameters()}
        self.assertEqual(head_ids, projection_ids)

    def test_map_descriptor_matches_frozen_pooler_output_before_training(self):
        model = _build(_config()).eval()
        reference = _tiny_backbone()
        pixels = torch.randn(2, 3, IMAGE, IMAGE)
        with torch.no_grad():
            adapted = model.backbone(pixel_values=pixels).pooler_output
            frozen = reference(pixel_values=pixels).pooler_output
        self.assertTrue(torch.allclose(adapted, frozen, atol=1e-6))

    def test_forward_native_and_interpolated_sizes(self):
        model = _build(_config()).eval()
        with torch.no_grad():
            native = model(torch.randn(3, 3, IMAGE, IMAGE))
            larger = model(torch.randn(3, 3, IMAGE + 16, IMAGE + 16))
        for out in (native, larger):
            self.assertEqual(out.shape, (3, HIDDEN))
            norms = out.norm(dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))

    def test_patch_mean_pooling(self):
        model = _build(_config(pooling="patch_mean")).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, IMAGE, IMAGE))
        self.assertEqual(out.shape, (2, HIDDEN))

    def test_patch_mean_rejects_trainable_pooling_head(self):
        with self.assertRaises(ValueError):
            _build(_config(pooling="patch_mean", trainable=True))

    def test_state_dict_round_trip(self):
        model = _build(_config(trainable=True))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.add_(0.01)
        restored = _build(_config(trainable=True))
        restored.load_state_dict(model.state_dict())
        pixels = torch.randn(2, 3, IMAGE, IMAGE)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.eval()(pixels), restored.eval()(pixels), atol=1e-6))
        metadata = model.checkpoint_metadata()
        self.assertEqual(metadata["model_type"], "siglip2_lora_qv_projection")
        self.assertTrue(metadata["train_pooling_head"])
        self.assertEqual(metadata["native_image_size"], IMAGE)


class Siglip2CheckpointNameTest(unittest.TestCase):
    def _finetuning(self, trainable: bool) -> dict:
        model = _config(trainable=trainable, span="last20")
        model["head"]["hidden_dim"] = 3072
        return {
            "finetuning": {
                "model": model,
                "sampler": {"type": "pk", "classes_per_batch": 64, "images_per_class": 4},
                "optimizer": {"lr_lora": 3e-5, "lr_head": 1e-4},
                "train": {"epochs": 5},
            }
        }

    def test_name_records_span_and_pooling_head(self):
        frozen = checkpoint_name_from_finetuning(self._finetuning(False))
        trained = checkpoint_name_from_finetuning(self._finetuning(True))
        self.assertEqual(
            frozen,
            "siglip2loramlp__p-64__k-4__r-4__la-4__lb-last20__mh-frozen__llr-3e-5"
            "__hd-3072__hlr-1e-4__hdo-0__sam-pk__ep-5",
        )
        self.assertIn("__mh-train__", trained)
        parsed = parse_checkpoint_name(f"local/finetuned/x/{frozen}__run-1-0/best")
        self.assertEqual(parsed["architecture"], "siglip2loramlp")
        self.assertEqual(parsed["mh"], "frozen")


if __name__ == "__main__":
    unittest.main()
