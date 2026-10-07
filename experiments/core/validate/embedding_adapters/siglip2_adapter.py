"""SigLIP2 image-only embedding adapter."""

import os
from collections.abc import Sequence
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ..embedding_geometry import RESIZE_AND_PAD, ImageGeometry, pad_color_from_mean
from ..embedding_pooling import TOKEN_POOLING_MODES, aggregate_tokens, normalize_pooling_modes
from transformers import AutoModel, AutoProcessor

from .base import BaseEmbeddingModel


class Siglip2Adapter(BaseEmbeddingModel):
    """Generate raw pooled image descriptors with SigLIP2."""

    SIZE_KEYS = ["l", "so400m", "g"]
    MODEL_ID_BY_SIZE_KEY = {
        "l": "google/siglip2-large-patch16-384",
        "so400m": "google/siglip2-so400m-patch14-384",
        "g": "google/siglip2-giant-opt-patch16-384",
    }
    SIZE_LABEL_BY_SIZE_KEY = {
        "l": "ViT-L (303M)",
        "so400m": "So400m (400M)",
        "g": "ViT-g (1B)",
    }
    DEFAULT_SIZE_KEY = "so400m"
    DEFAULT_POOLING = "default"
    SUPPORTED_POOLINGS = frozenset({"default"}) | TOKEN_POOLING_MODES
    PAD_COLOR = pad_color_from_mean((0.5, 0.5, 0.5))

    def __init__(
        self,
        use_compile: bool = True,
        hf_token: Optional[str] = None,
        geometry: Optional[ImageGeometry] = None,
        model_id: Optional[str] = None,
        size_key: Optional[str] = None,
        max_num_patches: Optional[int] = None,
        force_image_size: Optional[int] = None,
        attn_implementation: str = "sdpa",
        revision: Optional[str] = None,
    ):
        self.model_id = self._resolve_model_id(model_id, size_key)
        self.revision = revision
        self.size_key = self._resolve_size_key(size_key, self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.geometry = geometry
        self.max_num_patches = self._validate_max_num_patches(max_num_patches)
        self.force_image_size = self._validate_force_image_size(force_image_size)
        print(f"Loading {self.model_id} on {self.device}...")

        model_kwargs = {}
        processor_kwargs = {"use_fast": True}
        if self.hf_token:
            model_kwargs["token"] = self.hf_token
            processor_kwargs["token"] = self.hf_token
        if self.revision:
            model_kwargs["revision"] = self.revision
            processor_kwargs["revision"] = self.revision
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs).to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, **processor_kwargs
        )
        self.model.eval()
        self._dimension = self._resolve_dimension()

        if use_compile and self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"  Applied torch.compile() to {self.model_id}")
            except Exception as exc:
                print(f"  torch.compile() not available: {exc}")

    @classmethod
    def _resolve_size_key(cls, size_key: Optional[str], model_id: Optional[str] = None) -> str:
        key = size_key or cls._size_key_for_model_id(model_id) or os.getenv("SIGLIP2_SIZE_KEY") or cls.DEFAULT_SIZE_KEY
        normalized = key.strip().lower()
        if normalized not in cls.SIZE_KEYS:
            valid = ", ".join(cls.SIZE_KEYS)
            raise ValueError(
                f"Unsupported SigLIP2 size key {key!r}. Expected one of: {valid}"
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
            env_model_id = os.getenv("SIGLIP2_MODEL_ID")
            if env_model_id:
                return env_model_id
        return cls.MODEL_ID_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @classmethod
    def size_label(cls, size_key: str) -> str:
        return cls.SIZE_LABEL_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @staticmethod
    def _validate_max_num_patches(max_num_patches: Optional[int]) -> Optional[int]:
        if max_num_patches is None:
            return None
        if max_num_patches <= 0:
            raise ValueError("max_num_patches must be a positive integer or None")
        return int(max_num_patches)

    @staticmethod
    def _validate_force_image_size(force_image_size: Optional[int]) -> Optional[int]:
        if force_image_size is None:
            return None
        size = int(force_image_size)
        if size <= 0:
            raise ValueError("force_image_size must be a positive integer or None")
        return size

    def _vision_config(self):
        return getattr(self.model.config, "vision_config", self.model.config)

    def _native_target_size(self) -> Optional[tuple[int, int]]:
        image_processor = getattr(self.processor, "image_processor", None)
        return self._coerce_target_size(
            getattr(self._vision_config(), "image_size", None)
            or getattr(self.processor, "size", None)
            or getattr(image_processor, "size", None)
        )

    def _target_size(self) -> Optional[tuple[int, int]]:
        forced = getattr(self, "force_image_size", None)
        return (forced, forced) if forced is not None else self._native_target_size()

    def _requires_position_interpolation(self) -> bool:
        forced = getattr(self, "force_image_size", None)
        native = self._native_target_size()
        return forced is not None and native is not None and native != (forced, forced)

    @staticmethod
    def _coerce_target_size(size) -> Optional[tuple[int, int]]:
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
        size = int(size)
        return size, size

    def _prepare_images(self, images: List[Image.Image]) -> List[Image.Image]:
        prepared = [
            image if image.mode == "RGB" else image.convert("RGB") for image in images
        ]
        if self.geometry is None:
            return prepared
        target = self._target_size()
        return [
            self.geometry.apply(image, target, self.PAD_COLOR) for image in prepared
        ]

    @staticmethod
    def _is_fixed_resolution_processor(image_processor: object) -> bool:
        return image_processor.__class__.__name__ in {
            "SiglipImageProcessor",
            "SiglipImageProcessorFast",
        }

    def _prepare_inputs(self, images: List[Image.Image]) -> dict:
        letterbox = self.geometry is not None and self.geometry.mode == RESIZE_AND_PAD
        image_processor = getattr(self.processor, "image_processor", self.processor)
        if letterbox and not self._is_fixed_resolution_processor(image_processor):
            raise ValueError(
                "Letterbox preprocessing is only supported for fixed-resolution "
                "SigLIP2 processors; NaFlex requires its native patch-aware path."
            )

        prepared = self._prepare_images(images)
        if letterbox:
            batch = image_processor(
                images=prepared,
                return_tensors="pt",
                do_resize=False,
            )
        else:
            kwargs = {"return_tensors": "pt"}
            if self.max_num_patches is not None:
                kwargs["max_num_patches"] = self.max_num_patches
            batch = self.processor(images=prepared, **kwargs)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def _resolve_dimension(self) -> int:
        projection = getattr(self.model, "visual_projection", None)
        out_features = getattr(projection, "out_features", None)
        if out_features is not None:
            return int(out_features)
        configs = (self.model.config, self._vision_config())
        for config in configs:
            for name in ("projection_dim", "projection_size", "hidden_size"):
                value = getattr(config, name, None)
                if value is not None:
                    return int(value)
        raise ValueError(
            f"Could not determine embedding dimension for {self.model_id}"
        )

    def _normalize_pooling(self, pooling: Sequence[str]) -> tuple[str, ...]:
        return normalize_pooling_modes(pooling, self.SUPPORTED_POOLINGS)

    def _model_for_pooling(self):
        return getattr(self.model, "_orig_mod", self.model)

    @staticmethod
    def _token_mask(inputs: dict, token_count: int) -> torch.Tensor | None:
        mask = inputs.get("pixel_attention_mask")
        if mask is None or tuple(mask.shape)[:2] != (mask.shape[0], token_count):
            return None
        return mask

    def _image_features(
        self, inputs: dict, pooling: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        modes = self._normalize_pooling(pooling)
        interpolate = self._requires_position_interpolation()
        if modes == ("default",):
            feature_inputs = dict(inputs)
            if interpolate:
                feature_inputs["interpolate_pos_encoding"] = True
            embedding = self.model.get_image_features(**feature_inputs)
            return {"default": embedding}
        model = self._model_for_pooling()
        vision_kwargs = {
            "pixel_values": inputs["pixel_values"],
            "pixel_attention_mask": inputs.get("pixel_attention_mask"),
            "spatial_shapes": inputs.get("spatial_shapes"),
        }
        if interpolate:
            vision_kwargs["interpolate_pos_encoding"] = True
        vision_outputs = model.vision_model(**vision_kwargs)
        token_mask = self._token_mask(inputs, vision_outputs.last_hidden_state.shape[1])
        pooled: dict[str, torch.Tensor] = {}
        for mode in modes:
            if mode == "default":
                embedding = vision_outputs.pooler_output
            else:
                embedding = aggregate_tokens(
                    vision_outputs.last_hidden_state, mode, token_mask=token_mask
                )
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
            pooled = self._image_features(inputs, pooling)
        return self._to_numpy_batch(pooled)

    def get_model_name(self) -> str:
        return self.model_id.replace("/", "_").replace("-", "_")

    def get_dimension(self) -> int:
        return self._dimension
