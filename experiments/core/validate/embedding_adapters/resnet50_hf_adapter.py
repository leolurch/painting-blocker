"""Hugging Face ResNet-50 pooler embedding adapter."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ..embedding_geometry import RESIZE_AND_PAD, ImageGeometry, pad_color_from_mean
from ..embedding_pooling import normalize_pooling_modes
from .base import BaseEmbeddingModel


class ResNet50HFAdapter(BaseEmbeddingModel):
    """Frozen ImageNet-pretrained ResNet-50 global embedding baseline."""

    MODEL_ID = "microsoft/resnet-50"
    SIZE_KEYS = ["resnet50"]
    MODEL_ID_BY_SIZE_KEY = {"resnet50": MODEL_ID}
    SIZE_LABEL_BY_SIZE_KEY = {"resnet50": "ResNet-50 (25.6M)"}
    DEFAULT_SIZE_KEY = "resnet50"
    DEFAULT_POOLING = "default"
    SUPPORTED_POOLINGS = frozenset({"default", "pooler_output"})
    PAD_COLOR = pad_color_from_mean((0.485, 0.456, 0.406))

    def __init__(
        self,
        model_id: str | None = None,
        hf_token: str | None = None,
        geometry: ImageGeometry | None = None,
        revision: str | None = None,
    ):
        self.model_id = self._resolve_model_id(model_id)
        self.revision = revision
        self.size_key = self._resolve_size_key(self.model_id)
        self.device = self._select_device()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.geometry = geometry
        print(f"Loading {self.model_id} on {self.device}...")

        kwargs = {"token": self.hf_token} if self.hf_token else {}
        processor_kwargs = {"use_fast": True}
        if self.hf_token:
            processor_kwargs["token"] = self.hf_token
        if self.revision:
            kwargs["revision"] = self.revision
            processor_kwargs["revision"] = self.revision
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id, **processor_kwargs
        )
        self.model = AutoModel.from_pretrained(self.model_id, **kwargs).to(self.device)
        self.model.eval()
        self._dimension = self._resolve_dimension()

    @classmethod
    def _resolve_size_key(cls, model_id: str | None) -> str:
        normalized = (model_id or cls.MODEL_ID).strip().lower()
        known = cls.MODEL_ID.lower()
        repo_name = known.split("/", 1)[1]
        if normalized in {known, repo_name}:
            return cls.DEFAULT_SIZE_KEY
        raise ValueError(f"Unsupported ResNet-50 HF model_id {model_id!r}. Expected {cls.MODEL_ID!r}")

    @classmethod
    def _resolve_model_id(cls, model_id: str | None) -> str:
        if model_id is None or not model_id.strip():
            return cls.MODEL_ID
        cls._resolve_size_key(model_id)
        return model_id.strip()

    @classmethod
    def size_label(cls, size_key: str) -> str:
        normalized = size_key.strip().lower()
        if normalized not in cls.SIZE_KEYS:
            raise ValueError(f"Unsupported ResNet-50 HF size key {size_key!r}")
        return cls.SIZE_LABEL_BY_SIZE_KEY[normalized]

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dimension(self) -> int:
        hidden_sizes = getattr(self.model.config, "hidden_sizes", None)
        if hidden_sizes:
            return int(hidden_sizes[-1])
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        raise ValueError(f"Could not determine embedding dimension for {self.model_id}")

    def _target_size(self) -> tuple[int, int] | None:
        return self._coerce_target_size(
            getattr(self.processor, "crop_size", None)
            or getattr(self.processor, "size", None)
        )

    @staticmethod
    def _coerce_target_size(size) -> tuple[int, int] | None:
        if size is None:
            return None
        if isinstance(size, dict):
            width = size.get("width") or size.get("shortest_edge")
            height = size.get("height") or size.get("shortest_edge")
            if width is None or height is None:
                return None
            return int(width), int(height)
        if isinstance(size, (tuple, list)) and len(size) >= 2:
            return int(size[0]), int(size[1])
        if isinstance(size, (tuple, list)) and len(size) == 1:
            size = size[0]
        value = int(size)
        return value, value

    def _prepare_images(self, images: List[Image.Image]) -> List[Image.Image]:
        prepared = [
            image if image.mode == "RGB" else image.convert("RGB") for image in images
        ]
        if self.geometry is None:
            return prepared
        return [
            self.geometry.apply(image, self._target_size(), self.PAD_COLOR)
            for image in prepared
        ]

    def _prepare_inputs(self, images: List[Image.Image]) -> dict:
        processor_kwargs = {}
        if self.geometry is not None and self.geometry.mode == RESIZE_AND_PAD:
            processor_kwargs = {"do_resize": False, "do_center_crop": False}
        batch = self.processor(
            images=self._prepare_images(images),
            return_tensors="pt",
            **processor_kwargs,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def _normalize_pooling(self, pooling: Sequence[str]) -> tuple[str, ...]:
        return normalize_pooling_modes(pooling, self.SUPPORTED_POOLINGS)

    @staticmethod
    def _pooler_embedding(outputs: object) -> torch.Tensor:
        embedding = getattr(outputs, "pooler_output", None)
        if embedding is None:
            raise ValueError("ResNetModel output did not contain pooler_output")
        if embedding.ndim < 2:
            raise ValueError(f"Expected pooler_output with at least 2 dimensions, got {tuple(embedding.shape)}")
        if embedding.ndim > 2:
            embedding = torch.flatten(embedding, start_dim=1)
        return embedding

    def _image_features(self, inputs: dict, pooling: Sequence[str]) -> dict[str, torch.Tensor]:
        modes = self._normalize_pooling(pooling)
        outputs = self.model(**inputs)
        embedding = self._pooler_embedding(outputs)
        return {mode: embedding for mode in modes}

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
        with torch.inference_mode():
            pooled = self._image_features(inputs, pooling)
        return self._to_numpy_batch(pooled)

    def get_model_name(self) -> str:
        return self.model_id.replace("/", "_").replace("-", "_")

    def get_dimension(self) -> int:
        return self._dimension
