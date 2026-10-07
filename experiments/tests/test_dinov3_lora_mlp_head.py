"""Unit tests for the DINOv3 q/v LoRA embedding head variants.

These exercise the newly added MLP projection-head option on
``DinoV3LoRAEmbeddingModel`` without pulling in the real DINOv3 weights or a GPU:
the backbone loader and PEFT wrapping are stubbed, so only the head wiring,
forward shape/normalization, and the checkpoint config round-trip are covered.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn

from experiments.core.finetuning.models import dinov3 as dinov3_models
from experiments.core.finetuning.models.dinov3 import DinoV3LoRAEmbeddingModel, build_dinov3_lora_model
from experiments.core.finetuning.models.projection import ProjectionHead


class _FakeConfig:
    def __init__(self, hidden_size: int, num_register_tokens: int = 4) -> None:
        self.hidden_size = hidden_size
        self.num_register_tokens = num_register_tokens


class _FakeBackbone(nn.Module):
    """Minimal stand-in exposing q/v Linear leaves and a token-sequence output."""

    def __init__(self, hidden_size: int = 32, seq_len: int = 9) -> None:
        super().__init__()
        self.config = _FakeConfig(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self._hidden_size = hidden_size
        self._seq_len = seq_len

    def forward(self, pixel_values: torch.Tensor):  # noqa: D401 - matches HF call shape
        batch = pixel_values.shape[0]
        hidden = torch.randn(batch, self._seq_len, self._hidden_size)
        return types.SimpleNamespace(last_hidden_state=hidden)


def _fake_peft_module() -> types.ModuleType:
    module = types.ModuleType("peft")
    module.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.get_peft_model = lambda backbone, config: backbone  # identity wrap
    return module


def _build(config: dict) -> DinoV3LoRAEmbeddingModel:
    with mock.patch.object(
        dinov3_models.DinoV3ProjectionModel, "_load_backbone", return_value=_FakeBackbone()
    ), mock.patch.dict(sys.modules, {"peft": _fake_peft_module()}):
        return build_dinov3_lora_model(config)


_BASE_CONFIG = {
    "backbone": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "pooling": {"type": "cls"},
    "projection_dim": 768,
    "lora": {"target_modules": ["q_proj", "v_proj"], "rank": 4, "alpha": 8},
}


class DinoV3LoRAHeadTest(unittest.TestCase):
    def test_linear_head_is_default(self):
        model = _build(dict(_BASE_CONFIG))
        self.assertIsInstance(model.projection, nn.Linear)
        self.assertEqual(model.head_hidden_dim, 0)

    def test_mlp_head_from_head_type(self):
        cfg = dict(_BASE_CONFIG)
        cfg["head"] = {"type": "mlp_projection", "hidden_dim": 1024, "dropout": 0.1}
        model = _build(cfg)
        self.assertIsInstance(model.projection, ProjectionHead)
        self.assertEqual(model.head_hidden_dim, 1024)

    def test_mlp_head_default_hidden_dim(self):
        cfg = dict(_BASE_CONFIG)
        cfg["head"] = {"type": "mlp_projection"}
        model = _build(cfg)
        self.assertIsInstance(model.projection, ProjectionHead)
        self.assertEqual(model.head_hidden_dim, 1024)

    def test_mlp_forward_shape_and_normalization(self):
        cfg = dict(_BASE_CONFIG)
        cfg["head"] = {"type": "mlp_projection", "hidden_dim": 256, "dropout": 0.0}
        model = _build(cfg).eval()
        out = model(torch.randn(3, 3, 224, 224))
        self.assertEqual(out.shape, (3, 768))
        norms = out.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))

    def test_checkpoint_metadata_records_hidden_dim(self):
        cfg = dict(_BASE_CONFIG)
        cfg["head"] = {"type": "mlp_projection", "hidden_dim": 512}
        model = _build(cfg)
        self.assertEqual(model.checkpoint_metadata()["head_hidden_dim"], 512)


if __name__ == "__main__":
    unittest.main()
