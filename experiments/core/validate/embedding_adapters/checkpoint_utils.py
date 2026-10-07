"""Utilities for checkpoint-backed validation adapters."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.core.finetuning.augmentations import build_image_transform
from experiments.core.finetuning.checkpointing import load_model_from_checkpoint

from .base import BaseEmbeddingModel


class CheckpointProjectionAdapter(BaseEmbeddingModel):
    """Base adapter that exposes a finetuned torch checkpoint as NumPy embeddings."""

    SUPPORTED_POOLINGS = frozenset({"default", "cls", "pooler_output", "gem", "patch_mean", "patch_gem", "salad"})
    ALLOWED_MODEL_TYPES: frozenset[str] = frozenset()

    def __init__(
        self,
        model_id: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_path_env: str | None = None,
        geometry: Any | None = None,
        use_compile: bool = False,
        hf_token: str | None = None,
    ) -> None:
        del hf_token
        self.checkpoint_path = self._resolve_checkpoint_path(model_id, checkpoint_path, checkpoint_path_env)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.checkpoint = load_model_from_checkpoint(self.checkpoint_path, map_location="cpu")
        self._validate_model_type()
        self.model.to(self.device).eval()
        if use_compile and self.device == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.geometry = geometry
        transform_config = {
            **dict(self.checkpoint.get("preprocessing_config") or {}),
            "normalization": dict(self.checkpoint.get("normalization_config") or {}),
        }
        self.transform = build_image_transform(transform_config, train=False)
        self.dimension = int(self.checkpoint.get("projection_dim") or 0)

    @staticmethod
    def _resolve_checkpoint_path(
        model_id: str | None,
        checkpoint_path: str | None,
        checkpoint_path_env: str | None = None,
    ) -> Path:
        if checkpoint_path and checkpoint_path_env:
            raise ValueError("Configure only one of checkpoint_path or checkpoint_path_env")
        if checkpoint_path_env:
            env_name = str(checkpoint_path_env).strip()
            if not env_name or any(char.isspace() for char in env_name):
                raise ValueError("checkpoint_path_env must be one environment variable name")
            value = os.getenv(env_name)
            if not value:
                raise ValueError(f"Environment variable {env_name} must point to a checkpoint file")
        else:
            value = checkpoint_path or model_id
        if not value:
            raise ValueError("checkpoint-backed adapter requires checkpoint_path, checkpoint_path_env, or model_id")
        expanded = os.path.expandvars(str(value))
        if "$" in expanded:
            raise ValueError(f"Checkpoint path contains an unset environment variable: {value}")
        path = Path(expanded).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    def _validate_model_type(self) -> None:
        if not self.ALLOWED_MODEL_TYPES:
            return
        model_type = str(self.checkpoint.get("model_type") or "")
        if model_type not in self.ALLOWED_MODEL_TYPES:
            allowed = ", ".join(sorted(self.ALLOWED_MODEL_TYPES))
            raise ValueError(f"Checkpoint model_type={model_type!r} is not supported by this adapter. Expected: {allowed}")

    def _prepare_images(self, images: list[Image.Image]) -> torch.Tensor:
        prepared = [image.convert("RGB") for image in images]
        if self.geometry is not None:
            size = int((self.checkpoint.get("preprocessing_config") or {}).get("image_size") or 224)
            prepared = [self.geometry.apply(image, (size, size)) for image in prepared]
        return torch.stack([self.transform(image) for image in prepared]).to(self.device)

    @staticmethod
    def _pooling_keys(pooling: Sequence[str] | str) -> tuple[str, ...]:
        values = [pooling] if isinstance(pooling, str) else list(pooling)
        return tuple(dict.fromkeys(str(value) for value in values)) or ("default",)

    def generate_embedding(self, image_path: str, pooling: Sequence[str]) -> dict[str, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        return self.generate_embedding_from_pil(image, pooling)

    def generate_embedding_from_pil(self, image: Image.Image, pooling: Sequence[str]) -> dict[str, np.ndarray]:
        embeddings = self.generate_embeddings_batch_from_pil([image], pooling)
        return {mode: values[0] for mode, values in embeddings.items()}

    def generate_embeddings_batch(self, image_paths: list[str], pooling: Sequence[str]) -> dict[str, np.ndarray]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return self.generate_embeddings_batch_from_pil(images, pooling)

    def generate_embeddings_batch_from_pil(self, images: list[Image.Image], pooling: Sequence[str]) -> dict[str, np.ndarray]:
        pixel_values = self._prepare_images(images)
        with torch.inference_mode():
            descriptor_tensor = self.model(pixel_values)
        self.descriptor_tensor_dtypes = [
            str(descriptor_tensor.dtype).removeprefix("torch.")
        ]
        # Finetuned retrieval models normalize inside their trained forward pass.
        # The central FP32 postprocessor re-normalizes, but cannot undo this
        # model-intrinsic operation, so expose it in artifact metadata.
        self.intrinsic_normalization = True
        self.intrinsic_normalization_dtype = self.descriptor_tensor_dtypes[0]
        embeddings = descriptor_tensor.float().cpu().numpy()
        keys = self._pooling_keys(pooling)
        return {mode: embeddings for mode in keys}

    def get_model_name(self) -> str:
        return self.checkpoint_path.stem

    def get_dimension(self) -> int:
        return self.dimension
