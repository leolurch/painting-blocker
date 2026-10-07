from types import SimpleNamespace
from unittest import mock

import pytest

from experiments.core.validate.embedding_adapters.dino_adapter import DinoV3Adapter


class DINOv3ViTImageProcessorFast:
    pass


class GenericImageProcessor:
    pass


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(image_size=224, hidden_size=2)
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_dinov3_processor_load_failure_refuses_manual_fallback() -> None:
    with mock.patch.object(
        DinoV3Adapter, "_select_device", return_value="cpu"
    ), mock.patch(
        "experiments.core.validate.embedding_adapters.dino_adapter.AutoImageProcessor.from_pretrained",
        side_effect=OSError("processor unavailable"),
    ), mock.patch(
        "experiments.core.validate.embedding_adapters.dino_adapter.AutoModel.from_pretrained"
    ) as model_from_pretrained:
        with pytest.raises(
            RuntimeError,
            match=(
                "Expected DINOv3 processor could not be loaded; "
                "refusing to use a non-equivalent fallback"
            ),
        ) as error:
            DinoV3Adapter(
                model_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
                use_compile=False,
            )

    assert isinstance(error.value.__cause__, OSError)
    model_from_pretrained.assert_not_called()


def test_dinov3_rejects_an_unexpected_processor_class() -> None:
    with mock.patch.object(
        DinoV3Adapter, "_select_device", return_value="cpu"
    ), mock.patch(
        "experiments.core.validate.embedding_adapters.dino_adapter.AutoImageProcessor.from_pretrained",
        return_value=GenericImageProcessor(),
    ), mock.patch(
        "experiments.core.validate.embedding_adapters.dino_adapter.AutoModel.from_pretrained"
    ) as model_from_pretrained:
        with pytest.raises(RuntimeError) as error:
            DinoV3Adapter(
                model_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
                use_compile=False,
            )

    assert isinstance(error.value.__cause__, TypeError)
    assert "unexpected processor class" in str(error.value.__cause__)
    model_from_pretrained.assert_not_called()


def test_dinov3_accepts_expected_processor_before_loading_model() -> None:
    processor = DINOv3ViTImageProcessorFast()
    model = _FakeModel()
    with mock.patch.object(
        DinoV3Adapter, "_select_device", return_value="cpu"
    ), mock.patch(
        "experiments.core.validate.embedding_adapters.dino_adapter.AutoImageProcessor.from_pretrained",
        return_value=processor,
    ) as processor_from_pretrained, mock.patch(
        "experiments.core.validate.embedding_adapters.dino_adapter.AutoModel.from_pretrained",
        return_value=model,
    ) as model_from_pretrained:
        adapter = DinoV3Adapter(
            model_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
            revision="revision-1",
            hf_token="test-token",
            use_compile=False,
        )

    processor_from_pretrained.assert_called_once_with(
        "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        trust_remote_code=True,
        use_fast=True,
        token="test-token",
        revision="revision-1",
    )
    model_from_pretrained.assert_called_once()
    assert adapter.processor is processor
    assert adapter.model is model
    assert model.device == "cpu"
    assert model.eval_called
