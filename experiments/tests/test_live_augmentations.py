"""Pixel-level checks for the live training augmentations."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _augmentations():
    path = Path(__file__).resolve().parents[1] / "core" / "finetuning" / "augmentations.py"
    spec = importlib.util.spec_from_file_location("finetune_augmentations_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_AUG = _augmentations()
_CallableStep = _AUG._CallableStep
_augmentation_config = _AUG._augmentation_config
apply_square_mask = _AUG.apply_square_mask
center_crop_in_image = _AUG.center_crop_in_image
desaturate_rgb = _AUG.desaturate_rgb
resolve_augmentation_roles = _AUG.resolve_augmentation_roles
square_mask_side = _AUG.square_mask_side


class LiveAugmentationTests(unittest.TestCase):
    def test_new_keys_are_accepted(self) -> None:
        config = _augmentation_config(
            {
                "augmentations": {
                    "backend": "albumentations",
                    "desaturation": {"p": 0.5},
                    "square_mask": {"p": 1.0},
                    "center_crop_in": {"p": 0.5},
                    "gaussian_noise": {"p": 0.5},
                }
            }
        )
        self.assertEqual(
            set(config) - {"backend"},
            {"desaturation", "square_mask", "center_crop_in", "gaussian_noise"},
        )

    def test_historic_role_group(self) -> None:
        roles = resolve_augmentation_roles(["historic"])
        assert roles is not None
        self.assertIn("historic_print", roles)
        self.assertNotIn("modern_original", roles)
        self.assertIsNone(resolve_augmentation_roles(None))

    def test_desaturation_keeps_luma_and_reduces_color(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = 220
        image[..., 1] = 20
        image[..., 2] = 20
        faded = desaturate_rgb(image, 0.25)
        self.assertLess(int(faded[..., 0].mean()), int(image[..., 0].mean()))
        self.assertGreater(int(faded[..., 1].mean()), int(image[..., 1].mean()))

    def test_desaturation_step_skips_modern_roles(self) -> None:
        image = np.zeros((6, 6, 3), dtype=np.uint8)
        image[..., 0] = 200
        step = _CallableStep(lambda array: desaturate_rgb(array, 0.0), resolve_augmentation_roles(["historic"]))
        modern = step(image, "modern_generated")
        historic = step(image, "historic_archival")
        self.assertTrue(np.array_equal(modern, image))
        self.assertFalse(np.array_equal(historic, image))

    def test_center_crop_in_keeps_the_middle(self) -> None:
        image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        cropped = center_crop_in_image(image, 0.30)
        self.assertEqual(cropped.shape[0], 14)
        self.assertEqual(cropped.shape[1], 21)
        self.assertTrue(np.array_equal(cropped, image[3:17, 4:25]))

    def test_square_mask_covers_the_requested_area(self) -> None:
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        side = square_mask_side(100, 100, 0.16)
        self.assertEqual(side, 40)
        covered = apply_square_mask(image, 0.16, top=10, left=15)
        self.assertEqual(int((covered == 0).all(axis=2).sum()), 40 * 40)
        self.assertTrue(np.all(covered[10:50, 15:55] == 0))
        self.assertTrue(np.all(covered[0:10] == 128))


if __name__ == "__main__":
    unittest.main()
