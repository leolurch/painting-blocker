"""DINOv3 embedding adapter."""

from collections.abc import Sequence
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ..embedding_geometry import ImageGeometry, pad_color_from_mean
from ..embedding_pooling import TOKEN_POOLING_MODES, aggregate_tokens, normalize_pooling_modes
from transformers import AutoImageProcessor, AutoModel

from .base import BaseEmbeddingModel

from experiments.core.env import (
    env_hf_token,
    env_non_gpu_inference_allowed,
    env_str,
)


class DinoV3Adapter(BaseEmbeddingModel):
    SIZE_KEYS = ["s", "splus", "l", "hplus", "7b"]
    SIZE_KEY_ALIASES = {
        "s+": "splus",
        "vit-s+": "splus",
        "h+": "hplus",
        "vit-h+": "hplus",
        "vit-7b": "7b",
    }
    MODEL_ID_BY_SIZE_KEY = {
        "s": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "splus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "hplus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "7b": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    }
    SIZE_LABEL_BY_SIZE_KEY = {
        "s": "ViT-S (21M)",
        "splus": "ViT-S+ (29M)",
        "l": "ViT-L (300M)",
        "hplus": "ViT-H+ (840M)",
        "7b": "ViT-7B (6716M)",
    }
    DEFAULT_SIZE_KEY = "7b"
    DEFAULT_POOLING = "default"
    SUPPORTED_POOLINGS = frozenset({"default", "cls"}) | TOKEN_POOLING_MODES
    PAD_COLOR = pad_color_from_mean((0.485, 0.456, 0.406))
    EXPECTED_PROCESSOR_CLASSES = frozenset(
        {"DINOv3ViTImageProcessor", "DINOv3ViTImageProcessorFast"}
    )

    def __init__(
        self,
        use_compile: bool = True,
        hf_token: Optional[str] = None,
        geometry: Optional[ImageGeometry] = None,
        model_id: Optional[str] = None,
        size_key: Optional[str] = None,
        revision: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
    ):
        self.config_model_id = model_id
        self.model_id = self._resolve_model_id(model_name_or_path or model_id, size_key)
        self.revision = revision
        self.size_key = self._resolve_size_key(size_key, self.model_id)
        self.device = self._select_device()
        self.hf_token = env_hf_token(hf_token)
        self.geometry = geometry
        print(f"Loading {self.model_id} on {self.device}...")
        if self.hf_token:
            print("  Using Hugging Face token for authenticated download/access")

        processor_kwargs = {"use_fast": True}
        model_kwargs = {"trust_remote_code": True}
        if self.hf_token:
            processor_kwargs["token"] = self.hf_token
            model_kwargs["token"] = self.hf_token
        if self.revision:
            processor_kwargs["revision"] = self.revision
            model_kwargs["revision"] = self.revision
        if self.device == "cuda":
            model_kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                **processor_kwargs,
            )
            processor_class = self.processor.__class__.__name__
            if processor_class not in self.EXPECTED_PROCESSOR_CLASSES:
                raise TypeError(
                    f"unexpected processor class {processor_class!r}; expected one of "
                    f"{sorted(self.EXPECTED_PROCESSOR_CLASSES)}"
                )
        except Exception as exc:
            raise RuntimeError(
                "Expected DINOv3 processor could not be loaded; "
                "refusing to use a non-equivalent fallback."
            ) from exc
        print("  Loaded DINOv3 AutoImageProcessor")

        self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs).to(
            self.device
        )
        self.model.eval()

        if use_compile and self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"  Applied torch.compile() to {self.model_id}")
            except Exception as exc:
                print(f"  torch.compile() not available: {exc}")

    @classmethod
    def _resolve_size_key(cls, size_key: Optional[str], model_id: Optional[str] = None) -> str:
        key = size_key or cls._size_key_for_model_id(model_id) or env_str("DINOV3_SIZE_KEY", cls.DEFAULT_SIZE_KEY)
        normalized = cls.SIZE_KEY_ALIASES.get(key.strip().lower(), key.strip().lower())
        if normalized not in cls.SIZE_KEYS:
            valid = ", ".join(cls.SIZE_KEYS)
            raise ValueError(
                f"Unsupported DINOv3 size key {key!r}. Expected one of: {valid}"
            )
        return normalized

    @classmethod
    def _size_key_for_model_id(cls, model_id: Optional[str]) -> Optional[str]:
        if not model_id:
            return None
        normalized = model_id.strip().lower()
        for size_key, known_model_id in cls.MODEL_ID_BY_SIZE_KEY.items():
            known = known_model_id.lower()
            repo_name = known.split("/", 1)[1]
            if normalized in {known, repo_name}:
                return size_key
        return None

    @classmethod
    def _resolve_model_id(cls, model_id: Optional[str], size_key: Optional[str]) -> str:
        if model_id:
            return model_id
        if size_key is None:
            env_model_id = env_str("DINOV3_MODEL_ID")
            if env_model_id:
                return env_model_id
        return cls.MODEL_ID_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @classmethod
    def size_label(cls, size_key: str) -> str:
        return cls.SIZE_LABEL_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if not env_non_gpu_inference_allowed():
            raise RuntimeError(
                "CUDA GPU not detected. Set ALLOW_NON_GPU_INFERENCE=1 to allow "
                "MPS/CPU inference fallback."
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(
                "  Warning: CUDA unavailable; ALLOW_NON_GPU_INFERENCE is set, "
                "using MPS."
            )
            return "mps"
        print(
            "  Warning: CUDA unavailable; ALLOW_NON_GPU_INFERENCE is set, "
            "using CPU."
        )
        return "cpu"

    def _image_size(self) -> int:
        image_size = getattr(self.model.config, "image_size", 518)
        if isinstance(image_size, (tuple, list)) and image_size:
            image_size = image_size[0]
        try:
            return int(image_size)
        except Exception:
            return 518

    def _target_image_size(self) -> int:
        if self.geometry is not None and self.geometry.max_width is not None:
            return int(self.geometry.max_width)
        return self._image_size()

    def _prepare_images(self, images: List[Image.Image]) -> List[Image.Image]:
        if self.geometry is None:
            return images
        target_size = self._target_image_size()
        target = (target_size, target_size)
        return [self.geometry.apply(image, target, self.PAD_COLOR) for image in images]

    def _prepare_inputs(self, images: List[Image.Image]) -> dict:
        prepared = self._prepare_images(images)
        processor_kwargs = {}
        if self.geometry is not None:
            processor_kwargs = {"do_resize": False, "do_center_crop": False}
        batch = self.processor(
            images=prepared,
            return_tensors="pt",
            **processor_kwargs,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def _normalize_pooling(self, pooling: Sequence[str]) -> tuple[str, ...]:
        return normalize_pooling_modes(pooling, self.SUPPORTED_POOLINGS)

    def _patch_token_start(self) -> int:
        register_count = getattr(self.model.config, "num_register_tokens", 0) or 0
        return 1 + int(register_count)

    def _pool_hidden_state(
        self, hidden_state: torch.Tensor, pooling: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        modes = self._normalize_pooling(pooling)
        patch_tokens = None
        pooled: dict[str, torch.Tensor] = {}
        for mode in modes:
            if mode in {"default", "cls"}:
                embedding = hidden_state[:, 0, :]
            else:
                if patch_tokens is None:
                    patch_tokens = hidden_state[:, self._patch_token_start() :, :]
                embedding = aggregate_tokens(patch_tokens, mode)
            pooled[mode] = embedding
        return pooled

    def _to_numpy_batch(self, pooled: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        self.descriptor_tensor_dtypes = sorted({str(value.dtype).removeprefix("torch.") for value in pooled.values()})
        return {mode: value.float().cpu().numpy() for mode, value in pooled.items()}

    def generate_embedding(
        self, image_path: str, pooling: Sequence[str]
    ) -> dict[str, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        return self.generate_embedding_from_pil(image, pooling)

    def generate_embedding_from_pil(
        self, image: Image.Image, pooling: Sequence[str]
    ) -> dict[str, np.ndarray]:
        embeddings = self.generate_embeddings_batch_from_pil([image], pooling)
        return {mode: values[0] for mode, values in embeddings.items()}

    def generate_embeddings_batch(
        self, image_paths: List[str], pooling: Sequence[str]
    ) -> dict[str, np.ndarray]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return self.generate_embeddings_batch_from_pil(images, pooling)

    def generate_embeddings_batch_from_pil(
        self, images: List[Image.Image], pooling: Sequence[str]
    ) -> dict[str, np.ndarray]:
        inputs = self._prepare_inputs(images)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self._pool_hidden_state(outputs.last_hidden_state, pooling)
        return self._to_numpy_batch(pooled)

    def get_model_name(self) -> str:
        return self.model_id.replace("/", "_").replace("-", "_")

    def get_dimension(self) -> int:
        return int(self.model.config.hidden_size)
