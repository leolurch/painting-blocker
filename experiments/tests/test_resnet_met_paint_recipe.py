"""ResNet Multi-Similarity recipe units."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments.core.finetuning.models.resnet import build_resnet_model


class ResNetRecipeModelTest(unittest.TestCase):
    def test_whitening_head_matches_the_trunk_and_freezes_batchnorm_stats(self):
        model = build_resnet_model(
            {
                "type": "resnet_gem_whitening",
                "backbone": "resnet18",
                "weights": "none",
                "pooling": {"type": "gem", "p_init": 3.0, "learn_p": True},
                "freeze_backbone": False,
                "freeze_bn_running_stats": True,
                "pretrained": False,
            }
        )

        self.assertEqual(model.projection_dim, 512)
        self.assertEqual(tuple(model.projection.weight.shape), (512, 512))
        model.train()
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))
        output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 512))

    def test_mlp_head_expands_to_twice_the_trunk_and_returns_the_trunk_width(self):
        model = build_resnet_model(
            {
                "type": "resnet_gem_mlp",
                "backbone": "resnet50",
                "weights": "none",
                "pooling": {"type": "gem", "p_init": 3.0, "learn_p": True},
                "projection_dim": 2048,
                "projection_hidden_dim": 4096,
                "dropout": 0.0,
                "freeze_backbone": True,
                "pretrained": False,
            }
        )

        self.assertEqual(model.projection_dim, 2048)
        self.assertEqual(model.projection.hidden_dim, 4096)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 2048))

    def test_resnet152_swsl_is_rejected(self):
        with self.assertRaises(ValueError):
            build_resnet_model(
                {
                    "type": "resnet_gem_mlp",
                    "backbone": "resnet152",
                    "weights": "swsl",
                    "pooling": {"type": "gem"},
                    "pretrained": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
