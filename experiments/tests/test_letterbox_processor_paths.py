from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers.models.clip.image_processing_clip_fast import CLIPImageProcessorFast
from transformers.models.siglip.image_processing_siglip_fast import SiglipImageProcessorFast

from experiments.core.validate.embedding_adapters.open_clip_adapter import OpenClipAdapter
from experiments.core.validate.embedding_adapters.openai_clip_adapter import OpenAIClipAdapter
from experiments.core.validate.embedding_adapters.siglip2_adapter import Siglip2Adapter
from experiments.core.validate.embedding_adapters.siglip_adapter import SiglipAdapter
from experiments.core.validate.embedding_geometry import RESIZE_AND_PAD, ImageGeometry


class _ProcessorWrapper:
    """Image-only stand-in for the HF composite processors used by the adapters."""

    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.calls = []

    def __call__(self, *, images, return_tensors, **kwargs):
        self.calls.append(dict(kwargs))
        image_kwargs = dict(kwargs)
        image_kwargs.pop("padding", None)
        return self.image_processor(
            images=images,
            return_tensors=return_tensors,
            **image_kwargs,
        )


def _bordered_image() -> Image.Image:
    pixels = np.full((40, 80, 3), 255, dtype=np.uint8)
    border = 8
    pixels[:, :border] = (255, 0, 0)
    pixels[:, -border:] = (0, 255, 0)
    pixels[:border, border:-border] = (0, 0, 255)
    pixels[-border:, border:-border] = (255, 255, 0)
    return Image.fromarray(pixels, mode="RGB")


def _manual_rescale_and_normalize(
    prepared: Image.Image,
    *,
    rescale_factor: float,
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(prepared).copy()).permute(2, 0, 1).float()
    values = values * float(rescale_factor)
    mean_tensor = torch.tensor(mean, dtype=values.dtype)[:, None, None]
    std_tensor = torch.tensor(std, dtype=values.dtype)[:, None, None]
    return (values - mean_tensor) / std_tensor


def _assert_controlled_letterbox_output(
    *,
    prepared: Image.Image,
    actual: torch.Tensor,
    target: int,
    pad_color: tuple[int, int, int],
    rescale_factor: float,
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
) -> None:
    assert prepared.size == (target, target)
    assert tuple(actual.shape) == (3, target, target)

    prepared_pixels = np.asarray(prepared)
    non_padding = np.any(
        prepared_pixels != np.asarray(pad_color, dtype=np.uint8), axis=2
    )
    ys, xs = np.where(non_padding)
    assert (int(xs.min()), int(xs.max())) == (0, target - 1)
    assert int(ys.max()) - int(ys.min()) + 1 == target // 2

    expected = _manual_rescale_and_normalize(
        prepared,
        rescale_factor=rescale_factor,
        mean=mean,
        std=std,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-7)

    mean_tensor = torch.tensor(mean, dtype=actual.dtype)[:, None, None]
    std_tensor = torch.tensor(std, dtype=actual.dtype)[:, None, None]
    reconstructed = ((actual * std_tensor + mean_tensor) / rescale_factor).round()
    reconstructed = reconstructed.clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
    np.testing.assert_array_equal(reconstructed.cpu().numpy(), prepared_pixels)

    reconstructed_pixels = reconstructed.cpu().numpy()
    for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)):
        assert np.any(
            np.all(
                reconstructed_pixels == np.asarray(color, dtype=np.uint8),
                axis=2,
            )
        )

    # Mean-colored project padding should normalize to approximately zero.
    assert torch.max(torch.abs(actual[:, 0, target // 2])).item() < 0.005


def _siglip_image_processor(target: int = 384) -> SiglipImageProcessorFast:
    return SiglipImageProcessorFast(
        size={"height": target, "width": target},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        rescale_factor=1 / 255,
    )


def test_siglip_letterbox_skips_only_native_resize_and_preserves_pixels() -> None:
    target = 384
    image = _bordered_image()
    image_processor = _siglip_image_processor(target)
    wrapper = _ProcessorWrapper(image_processor)
    adapter = object.__new__(SiglipAdapter)
    adapter.geometry = ImageGeometry(RESIZE_AND_PAD, max_width=target)
    adapter.processor = wrapper

    prepared = adapter._prepare_images([image])[0]
    with mock.patch.object(
        image_processor, "preprocess", wraps=image_processor.preprocess
    ) as preprocess:
        actual = adapter._prepare_inputs([image])["pixel_values"][0]

    assert preprocess.call_args.kwargs["do_resize"] is False
    assert "do_center_crop" not in preprocess.call_args.kwargs
    assert wrapper.calls == []
    _assert_controlled_letterbox_output(
        prepared=prepared,
        actual=actual,
        target=target,
        pad_color=adapter.PAD_COLOR,
        rescale_factor=image_processor.rescale_factor,
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    )

    adapter.geometry = None
    native = adapter._prepare_inputs([image])["pixel_values"]
    expected_native = image_processor(images=[image], return_tensors="pt")[
        "pixel_values"
    ]
    torch.testing.assert_close(native, expected_native)
    assert wrapper.calls == [{"padding": True}]


def test_siglip2_fixres_letterbox_skips_only_native_resize_and_preserves_pixels() -> None:
    target = 384
    image = _bordered_image()
    image_processor = _siglip_image_processor(target)
    wrapper = _ProcessorWrapper(image_processor)
    adapter = object.__new__(Siglip2Adapter)
    adapter.geometry = ImageGeometry(RESIZE_AND_PAD, max_width=target)
    adapter.processor = wrapper
    adapter.model = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(image_size=target))
    )
    adapter.device = "cpu"
    adapter.max_num_patches = None

    prepared = adapter._prepare_images([image])[0]
    with mock.patch.object(
        image_processor, "preprocess", wraps=image_processor.preprocess
    ) as preprocess:
        actual = adapter._prepare_inputs([image])["pixel_values"][0]

    assert preprocess.call_args.kwargs["do_resize"] is False
    assert "do_center_crop" not in preprocess.call_args.kwargs
    assert wrapper.calls == []
    _assert_controlled_letterbox_output(
        prepared=prepared,
        actual=actual,
        target=target,
        pad_color=adapter.PAD_COLOR,
        rescale_factor=image_processor.rescale_factor,
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    )

    adapter.geometry = None
    native = adapter._prepare_inputs([image])["pixel_values"]
    expected_native = image_processor(images=[image], return_tensors="pt")[
        "pixel_values"
    ]
    torch.testing.assert_close(native, expected_native)
    assert wrapper.calls == [{}]


def test_siglip2_fixres_can_force_512_with_position_interpolation() -> None:
    native_target = 384
    forced_target = 512
    image = _bordered_image()
    image_processor = _siglip_image_processor(native_target)
    wrapper = _ProcessorWrapper(image_processor)
    get_image_features = mock.Mock(return_value=torch.ones((1, 8)))
    adapter = object.__new__(Siglip2Adapter)
    adapter.geometry = ImageGeometry(RESIZE_AND_PAD, max_width=forced_target)
    adapter.processor = wrapper
    adapter.model = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(image_size=native_target)),
        get_image_features=get_image_features,
    )
    adapter.device = "cpu"
    adapter.max_num_patches = None
    adapter.force_image_size = forced_target

    inputs = adapter._prepare_inputs([image])
    assert tuple(inputs["pixel_values"].shape[-2:]) == (forced_target, forced_target)
    adapter._image_features(inputs, ["default"])
    assert get_image_features.call_args.kwargs["interpolate_pos_encoding"] is True
    assert Siglip2Adapter._validate_force_image_size(forced_target) == forced_target
    with pytest.raises(ValueError, match="positive integer"):
        Siglip2Adapter._validate_force_image_size(0)


def test_siglip2_naflex_letterbox_requires_separate_handling() -> None:
    class Siglip2ImageProcessorFast:
        def __call__(self, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("NaFlex processor should not receive FixRes inputs")

    adapter = object.__new__(Siglip2Adapter)
    adapter.geometry = ImageGeometry(RESIZE_AND_PAD, max_width=384)
    adapter.processor = Siglip2ImageProcessorFast()
    adapter.model = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(image_size=384))
    )
    adapter.device = "cpu"
    adapter.max_num_patches = 256

    with pytest.raises(ValueError, match="NaFlex requires its native patch-aware path"):
        adapter._prepare_inputs([_bordered_image()])


def test_openai_clip_letterbox_skips_resize_and_crop_and_preserves_pixels() -> None:
    target = 336
    image = _bordered_image()
    image_processor = CLIPImageProcessorFast(
        size={"shortest_edge": target},
        crop_size={"height": target, "width": target},
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        rescale_factor=1 / 255,
    )
    wrapper = _ProcessorWrapper(image_processor)
    adapter = object.__new__(OpenAIClipAdapter)
    adapter.geometry = ImageGeometry(RESIZE_AND_PAD, max_width=target)
    adapter.processor = wrapper
    adapter.input_size = target

    prepared = adapter._prepare_images([image])[0]
    with mock.patch.object(
        image_processor, "preprocess", wraps=image_processor.preprocess
    ) as preprocess:
        actual = adapter._prepare_inputs([image])["pixel_values"][0]

    assert preprocess.call_args.kwargs["do_resize"] is False
    assert preprocess.call_args.kwargs["do_center_crop"] is False
    assert wrapper.calls == []
    _assert_controlled_letterbox_output(
        prepared=prepared,
        actual=actual,
        target=target,
        pad_color=adapter.PAD_COLOR,
        rescale_factor=image_processor.rescale_factor,
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    )

    adapter.geometry = None
    native = adapter._prepare_inputs([image])["pixel_values"]
    expected_native = image_processor(images=[image], return_tensors="pt")[
        "pixel_values"
    ]
    torch.testing.assert_close(native, expected_native)
    assert wrapper.calls == [{"padding": True}]


def test_openclip_letterbox_removes_geometry_transforms_and_preserves_pixels() -> None:
    target = 224
    image = _bordered_image()
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    native_preprocess = Compose(
        [
            Resize(target, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((target, target)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    adapter = object.__new__(OpenClipAdapter)
    adapter.geometry = ImageGeometry(RESIZE_AND_PAD, max_width=target)
    adapter.preprocess = native_preprocess
    adapter._letterbox_preprocess = adapter._without_geometry_transforms(
        native_preprocess
    )
    adapter.input_size = target
    adapter.device = "cpu"

    retained_names = [
        transform.__class__.__name__
        for transform in adapter._letterbox_preprocess.transforms
    ]
    assert retained_names == ["ToTensor", "Normalize"]
    assert [
        transform.__class__.__name__ for transform in adapter.preprocess.transforms
    ] == ["Resize", "CenterCrop", "ToTensor", "Normalize"]

    prepared = adapter._prepare_images([image])[0]
    actual = adapter._prepare_tensor([image])[0]
    _assert_controlled_letterbox_output(
        prepared=prepared,
        actual=actual,
        target=target,
        pad_color=adapter.PAD_COLOR,
        rescale_factor=1 / 255,
        mean=mean,
        std=std,
    )

    adapter.geometry = None
    native = adapter._prepare_tensor([image])[0]
    expected_native = native_preprocess(image)
    torch.testing.assert_close(native, expected_native)
