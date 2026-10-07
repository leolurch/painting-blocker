"""OpenAI CLIP embedding adapter."""

import os
from collections.abc import Sequence
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ..embedding_geometry import RESIZE_AND_PAD, ImageGeometry, pad_color_from_mean
from ..embedding_pooling import TOKEN_POOLING_MODES, aggregate_tokens, normalize_pooling_modes
from transformers import CLIPModel, CLIPProcessor

from .base import BaseEmbeddingModel


class OpenAIClipAdapter(BaseEmbeddingModel):
    SIZE_KEYS = ["l14", "l14-336"]
    SIZE_KEY_ALIASES = {
        "large14": "l14",
        "large-patch14": "l14",
        "clip-vit-large-patch14": "l14",
        "openai/clip-vit-large-patch14": "l14",
        "vit-large-patch14": "l14",
        "l-14": "l14",
        "large14-336": "l14-336",
        "large-patch14-336": "l14-336",
        "clip-vit-large-patch14-336": "l14-336",
        "openai/clip-vit-large-patch14-336": "l14-336",
        "vit-large-patch14-336": "l14-336",
        "l14_336": "l14-336",
        "l-14-336": "l14-336",
    }
    MODEL_ID_BY_SIZE_KEY = {
        "l14": "openai/clip-vit-large-patch14",
        "l14-336": "openai/clip-vit-large-patch14-336",
    }
    SIZE_LABEL_BY_SIZE_KEY = {
        "l14": "ViT-L/14",
        "l14-336": "ViT-L/14@336px",
    }
    INPUT_SIZE_BY_SIZE_KEY = {
        "l14": 224,
        "l14-336": 336,
    }
    DEFAULT_SIZE_KEY = "l14-336"
    DEFAULT_POOLING = "default"
    SUPPORTED_POOLINGS = frozenset({"default", "cls"}) | TOKEN_POOLING_MODES
    PAD_COLOR = pad_color_from_mean((0.48145466, 0.4578275, 0.40821073))

    def __init__(
        self,
        use_compile: bool = True,
        hf_token: Optional[str] = None,
        geometry: Optional[ImageGeometry] = None,
        model_id: Optional[str] = None,
        size_key: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.model_id = self._resolve_model_id(model_id, size_key)
        self.revision = revision
        self.size_key = self._resolve_size_key(size_key, self.model_id)
        self.input_size = self.INPUT_SIZE_BY_SIZE_KEY[self.size_key]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.geometry = geometry
        print(f"Loading {self.model_id} on {self.device}...")

        model_kwargs = {"token": self.hf_token} if self.hf_token else {}
        processor_kwargs = {"use_fast": True}
        if self.hf_token:
            processor_kwargs["token"] = self.hf_token
        if self.revision:
            model_kwargs["revision"] = self.revision
            processor_kwargs["revision"] = self.revision
        self.model = CLIPModel.from_pretrained(self.model_id, **model_kwargs).to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained(
            self.model_id, **processor_kwargs
        )
        self.input_size = self._resolve_input_size()
        self.model.eval()

        if use_compile and self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"  Applied torch.compile() to {self.model_id}")
            except Exception as exc:
                print(f"  torch.compile() not available: {exc}")

    @classmethod
    def _resolve_size_key(cls, size_key: Optional[str], model_id: Optional[str] = None) -> str:
        key = size_key or model_id or os.getenv("OPENAI_CLIP_SIZE_KEY") or cls.DEFAULT_SIZE_KEY
        normalized = cls.SIZE_KEY_ALIASES.get(key.strip().lower(), key.strip().lower())
        if normalized not in cls.SIZE_KEYS:
            valid = ", ".join(cls.SIZE_KEYS)
            raise ValueError(
                f"Unsupported OpenAI CLIP size key {key!r}. Expected one of: {valid}"
            )
        return normalized

    @classmethod
    def _resolve_model_id(cls, model_id: Optional[str], size_key: Optional[str]) -> str:
        if model_id:
            return model_id
        if size_key is None:
            env_model_id = os.getenv("OPENAI_CLIP_MODEL_ID")
            if env_model_id:
                return env_model_id
        return cls.MODEL_ID_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @classmethod
    def size_label(cls, size_key: str) -> str:
        return cls.SIZE_LABEL_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    def _resolve_input_size(self) -> int:
        image_processor = getattr(self.processor, "image_processor", None)
        crop_size = getattr(image_processor, "crop_size", None)
        if isinstance(crop_size, dict):
            size = crop_size.get("height") or crop_size.get("width")
            if size is not None:
                return int(size)
        if isinstance(crop_size, (tuple, list)) and crop_size:
            return int(crop_size[0])
        if isinstance(crop_size, int):
            return crop_size
        return self.INPUT_SIZE_BY_SIZE_KEY[self.size_key]

    def _prepare_images(self, images: List[Image.Image]) -> List[Image.Image]:
        if self.geometry is None:
            return images
        target = (self.input_size, self.input_size)
        return [self.geometry.apply(image, target, self.PAD_COLOR) for image in images]

    def _prepare_inputs(self, images: List[Image.Image]) -> dict:
        prepared = self._prepare_images(images)
        if self.geometry is not None and self.geometry.mode == RESIZE_AND_PAD:
            image_processor = getattr(self.processor, "image_processor", self.processor)
            return image_processor(
                images=prepared,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )
        return self.processor(images=prepared, return_tensors="pt", padding=True)

    def _normalize_pooling(self, pooling: Sequence[str]) -> tuple[str, ...]:
        return normalize_pooling_modes(pooling, self.SUPPORTED_POOLINGS)

    def _model_for_pooling(self):
        return getattr(self.model, "_orig_mod", self.model)

    def _image_features(
        self, inputs: dict, pooling: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        modes = self._normalize_pooling(pooling)
        if modes == ("default",):
            embedding = self.model.get_image_features(**inputs)
            return {"default": embedding}
        model = self._model_for_pooling()
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled: dict[str, torch.Tensor] = {}
        patch_tokens = None
        for mode in modes:
            if mode in {"default", "cls"}:
                embedding = vision_outputs.pooler_output
            else:
                if patch_tokens is None:
                    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
                embedding = aggregate_tokens(patch_tokens, mode)
                embedding = model.vision_model.post_layernorm(embedding)
            embedding = model.visual_projection(embedding)
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
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            pooled = self._image_features(inputs, pooling)
        return self._to_numpy_batch(pooled)

    def get_model_name(self) -> str:
        return self.model_id.replace("/", "_")

    def get_dimension(self) -> int:
        return self.model.config.projection_dim
