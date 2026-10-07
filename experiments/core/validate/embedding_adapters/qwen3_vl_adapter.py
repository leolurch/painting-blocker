"""Official-style adapter for the Qwen3-VL multimodal embedding model.

The image preprocessing, chat prompt, and last-token pooling mirror Qwen's
``Qwen3VLEmbedder`` reference implementation. Shared FP32 normalization is
applied centrally by the embedding pipeline. Model-config image
geometry remains available as an explicit preprocessing override and is applied
before Qwen's native dynamic-resolution preprocessing.

Requirements:
    transformers>=4.57.0
    qwen-vl-utils>=0.0.14
    torch>=2.0.0
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ..embedding_geometry import RESIZE_AND_PAD, RESIZE_LONGEST_SIDE, ImageGeometry, pad_color_from_mean
from ..embedding_pooling import normalize_pooling_modes
from .base import BaseEmbeddingModel


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    """Outputs needed by Qwen's canonical embedding pooling step."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Qwen3-VL base model without a language-model generation head."""

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class Qwen3VLEmbeddingAdapter(BaseEmbeddingModel):
    """Image adapter for the retrieval-trained Qwen3-VL-Embedding models.

    The frozen model is the 8B checkpoint.
    """

    SIZE_KEYS = ["8b"]
    SIZE_KEY_ALIASES: dict[str, str] = {}
    MODEL_ID_BY_SIZE_KEY = {
        "8b": "Qwen/Qwen3-VL-Embedding-8B",
    }
    SIZE_LABEL_BY_SIZE_KEY = {
        "8b": "8B",
    }
    DEFAULT_SIZE_KEY = "8b"

    # Defaults from Qwen's scripts/qwen3_vl_embedding.py implementation.
    IMAGE_BASE_FACTOR = 16
    IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
    MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
    MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
    MAX_LENGTH = 8192
    DEFAULT_INSTRUCTION = "Represent the user's input."
    DEFAULT_POOLING = "default"
    SUPPORTED_POOLINGS = frozenset({"default"})

    def __init__(
        self,
        use_flash_attention: bool = False,
        model_dtype: str | None = None,
        use_compile: bool = False,
        hf_token: Optional[str] = None,
        geometry: Optional[ImageGeometry] = None,
        model_id: Optional[str] = None,
        size_key: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.model_id = self._resolve_model_id(model_id, size_key)
        self.revision = revision
        self.size_key = self._resolve_size_key(size_key, self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.geometry = geometry
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        print(f"Loading {self.model_id} on {self.device}...")
        if self.hf_token:
            print("  Using Hugging Face token for authenticated download/access")

        model_kwargs = {"trust_remote_code": True}
        if self.hf_token:
            model_kwargs["token"] = self.hf_token
        if self.revision:
            model_kwargs["revision"] = self.revision
        if model_dtype is not None:
            model_kwargs["dtype"] = self._resolve_model_dtype(model_dtype)

        if use_flash_attention and self.device == "cuda":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print(
                    "  Warning: flash_attention_2 requested but not installed. "
                    "Using Qwen's default attention implementation."
                )
            else:
                # Preserve the adapter's historical recommended FP16 path when
                # explicitly requested. The B200 latency add-on does not use it.
                model_kwargs["dtype"] = torch.float16
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("  Using Qwen's recommended float16 + flash_attention_2")

        try:
            self.model = Qwen3VLForEmbedding.from_pretrained(
                self.model_id, **model_kwargs
            ).to(self.device)

            processor_kwargs = {"padding_side": "right"}
            if self.hf_token:
                processor_kwargs["token"] = self.hf_token
            if self.revision:
                processor_kwargs["revision"] = self.revision
            self.processor = Qwen3VLProcessor.from_pretrained(
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

            print(f"  Successfully loaded {self.model_id}")
        except Exception as exc:
            message = f"Failed to load Qwen3-VL embedding model: {exc}"
            print(message)
            raise RuntimeError(message) from exc

    @staticmethod
    def _resolve_model_dtype(value: str) -> torch.dtype:
        normalized = str(value).strip().lower()
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError("model_dtype must be bfloat16/bf16, float16/fp16, or float32/fp32")
        return mapping[normalized]

    @classmethod
    def _resolve_size_key(
        cls, size_key: Optional[str], model_id: Optional[str] = None
    ) -> str:
        key = (
            size_key
            or cls._size_key_for_model_id(model_id)
            or os.getenv("QWEN3_VL_SIZE_KEY")
            or cls.DEFAULT_SIZE_KEY
        )
        normalized = cls.SIZE_KEY_ALIASES.get(
            key.strip().lower(), key.strip().lower()
        )
        if normalized not in cls.SIZE_KEYS:
            valid = ", ".join(cls.SIZE_KEYS)
            raise ValueError(
                f"Unsupported Qwen3-VL embedding size key {key!r}. "
                f"Expected one of: {valid}"
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
    def _resolve_model_id(
        cls, model_id: Optional[str], size_key: Optional[str]
    ) -> str:
        if model_id:
            return model_id
        if size_key is None:
            env_model_id = os.getenv("QWEN3_VL_MODEL_ID")
            if env_model_id:
                return env_model_id
        return cls.MODEL_ID_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @classmethod
    def size_label(cls, size_key: str) -> str:
        return cls.SIZE_LABEL_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    def _resolve_dimension(self) -> int:
        configs = (self.model.config, getattr(self.model.config, "text_config", None))
        for config in configs:
            if config is None:
                continue
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is not None:
                return int(hidden_size)
        raise ValueError(
            f"Could not determine embedding dimension for {self.model_id}"
        )

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Apply only an explicitly configured pre-Qwen geometry override."""
        prepared = image if image.mode == "RGB" else image.convert("RGB")
        if self.geometry is None:
            return prepared
        target = (
            (self.geometry.max_width, self.geometry.max_width)
            if self.geometry.mode == RESIZE_AND_PAD and self.geometry.max_width
            else None
        )
        pad_color = pad_color_from_mean(tuple(self.processor.image_processor.image_mean))
        return self.geometry.apply(prepared, target, pad_color)

    def _format_conversation(self, image: Image.Image) -> list[dict]:
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.DEFAULT_INSTRUCTION}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": self.MIN_PIXELS,
                        "max_pixels": self.MAX_PIXELS,
                    }
                ],
            },
        ]

    def _preprocess_batch(self, images: List[Image.Image]) -> dict[str, torch.Tensor]:
        conversations = [
            self._format_conversation(self._prepare_image(image)) for image in images
        ]
        # For the longest-side mode our geometry only caps the larger dimension; let Qwen's
        # native smart-resize snap the result to its 32px patch grid. Every other path keeps
        # our geometry as the sole authority (do_resize disabled).
        do_resize = (
            self.geometry is not None and self.geometry.mode == RESIZE_LONGEST_SIDE
        )
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        processed_images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=self.IMAGE_BASE_FACTOR,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        return self.processor(
            text=text,
            images=processed_images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.MAX_LENGTH,
            padding=True,
            do_resize=do_resize,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _pooling_last(
        hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def _normalize_pooling(self, pooling: Sequence[str]) -> tuple[str, ...]:
        return normalize_pooling_modes(pooling, self.SUPPORTED_POOLINGS)

    def _embedding_map(
        self, embedding: torch.Tensor, pooling: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        modes = self._normalize_pooling(pooling)
        return {mode: embedding for mode in modes}

    def _to_numpy_batch(
        self, pooled: dict[str, torch.Tensor],
    ) -> dict[str, np.ndarray]:
        self.descriptor_tensor_dtypes = sorted({str(value.dtype).removeprefix("torch.") for value in pooled.values()})
        return {
            mode: value.float().cpu().numpy() for mode, value in pooled.items()
        }

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
        inputs = self._preprocess_batch(images)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._pooling_last(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            pooled = self._embedding_map(embeddings, pooling)

        return self._to_numpy_batch(pooled)

    def get_model_name(self) -> str:
        return self.model_id.replace("/", "_").replace("-", "_")

    def get_dimension(self) -> int:
        return self._dimension
