"""The trainer's center crop and PIL letterbox reproduce the frozen rows' pixels."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from experiments.core.finetuning.augmentations import (
    build_image_transform,
    pil_letterbox,
    shortest_side_center_crop,
)
from experiments.core.validate.embedding_geometry import RESIZE_AND_PAD, ImageGeometry

SIZES = ((640, 427), (427, 640), (1001, 333), (300, 300), (200, 150))


def _image(width: int, height: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class TransferPreprocessingTests(unittest.TestCase):
    def test_pil_letterbox_matches_frozen_geometry(self) -> None:
        geometry = ImageGeometry(mode=RESIZE_AND_PAD, max_width=512)
        for seed, (width, height) in enumerate(SIZES):
            array = _image(width, height, seed)
            expected = np.asarray(geometry.apply(Image.fromarray(array), (512, 512), (128, 128, 128)))
            np.testing.assert_array_equal(pil_letterbox(array, 512, (128, 128, 128)), expected)

    def test_center_crop_matches_torchvision(self) -> None:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        for resize_size, crop_size in ((256, 224), (336, 336), (224, 224)):
            reference = transforms.Compose(
                [transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(crop_size)]
            )
            for seed, (width, height) in enumerate(SIZES):
                array = _image(width, height, seed)
                expected = np.asarray(reference(Image.fromarray(array)))
                np.testing.assert_array_equal(shortest_side_center_crop(array, resize_size, crop_size), expected)

    def test_eval_transform_uses_center_crop(self) -> None:
        config = {
            "image_size": 224,
            "preprocessing": {"resize": {"mode": "center_crop", "size": 224, "resize_size": 256}},
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        transform = build_image_transform(config, train=False)
        image = Image.fromarray(_image(640, 427, 0))
        self.assertEqual(tuple(transform(image).shape), (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
