"""OpenCLIP image embedding adapter for LAION CLIP checkpoints."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from ..embedding_geometry import RESIZE_AND_PAD, ImageGeometry, pad_color_from_mean
from ..embedding_pooling import TOKEN_POOLING_MODES, aggregate_tokens, normalize_pooling_modes

from .base import BaseEmbeddingModel

_OPEN_CLIP_GEOMETRY_TRANSFORMS = frozenset(
    {"Resize", "ResizeKeepRatio", "CenterCrop", "CenterCropOrPad"}
)

_OPEN_CLIP_MODEL_SPECS = (
    ("l14-laion2b-s32b-b82k", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "ViT-L/14 LAION-2B s32B b82K", 768),
    ("l14-datacomp-xl-s13b-b90k", "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "ViT-L/14 DataComp-XL s13B b90K", 768),
    ("h14-laion2b-s32b-b79k", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "ViT-H/14 LAION-2B s32B b79K", 1024),
    ("g14-laion2b-s12b-b42k", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K", "ViT-g/14 LAION-2B s12B b42K", 1024),
    ("g14-laion2b-s34b-b88k", "laion/CLIP-ViT-g-14-laion2B-s34B-b88K", "ViT-g/14 LAION-2B s34B b88K", 1024),
    ("bigg14-laion2b-39b-b160k", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "ViT-bigG/14 LAION-2B 39B b160k", 1280),
)


class OpenClipAdapter(BaseEmbeddingModel):
    """Generate raw projected image descriptors with open_clip models."""

    SIZE_KEYS = [spec[0] for spec in _OPEN_CLIP_MODEL_SPECS]
    MODEL_ID_BY_SIZE_KEY = {spec[0]: spec[1] for spec in _OPEN_CLIP_MODEL_SPECS}
    SIZE_LABEL_BY_SIZE_KEY = {spec[0]: spec[2] for spec in _OPEN_CLIP_MODEL_SPECS}
    DIMENSION_BY_SIZE_KEY = {spec[0]: spec[3] for spec in _OPEN_CLIP_MODEL_SPECS}
    INPUT_SIZE_BY_SIZE_KEY = {key: 224 for key in SIZE_KEYS}
    DEFAULT_SIZE_KEY = "l14-datacomp-xl-s13b-b90k"
    DEFAULT_POOLING = "default"
    SUPPORTED_POOLINGS = frozenset({"default", "cls"}) | TOKEN_POOLING_MODES
    PAD_COLOR = pad_color_from_mean((0.48145466, 0.4578275, 0.40821073))

    def __init__(
        self,
        use_compile: bool = True,
        hf_token: str | None = None,
        geometry: ImageGeometry | None = None,
        model_id: str | None = None,
        size_key: str | None = None,
        revision: str | None = None,
        force_image_size: int | None = None,
    ):
        resolved_model_id = self._resolve_model_id(model_id, size_key)
        self.model_id, model_id_revision = self._split_model_id_revision(
            resolved_model_id
        )
        self.revision = revision or model_id_revision
        self.size_key = self._resolve_size_key(size_key, self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.geometry = geometry
        self.force_image_size = self._validate_force_image_size(force_image_size)
        self._configure_hf_token()
        print(f"Loading {self.model_id} with open_clip on {self.device}...")

        open_clip = self._import_open_clip()
        self.model, self.preprocess = self._load_model(open_clip)
        self._letterbox_preprocess = (
            self._without_geometry_transforms(self.preprocess)
            if self.geometry is not None and self.geometry.mode == RESIZE_AND_PAD
            else None
        )
        self.model.eval()
        self.input_size = self._resolve_input_size()
        self._dimension = self._resolve_dimension()

        if use_compile and self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"  Applied torch.compile() to {self.model_id}")
            except Exception as exc:
                print(f"  torch.compile() not available: {exc}")

    @classmethod
    def _resolve_size_key(cls, size_key: str | None, model_id: str | None = None) -> str:
        key = size_key or model_id or os.getenv("OPEN_CLIP_SIZE_KEY") or cls.DEFAULT_SIZE_KEY
        normalized = cls._normalize_size_key(key)
        if normalized not in cls.SIZE_KEYS:
            valid = ", ".join(cls.SIZE_KEYS)
            raise ValueError(
                f"Unsupported OpenCLIP size key {key!r}. Expected one of: {valid}"
            )
        return normalized

    @classmethod
    def _normalize_size_key(cls, key: str) -> str:
        normalized = key.strip().lower().replace("_", "-")
        if normalized.startswith("hf-hub:"):
            normalized = normalized[len("hf-hub:") :]
        normalized, _ = cls._split_model_id_revision(normalized)
        for size_key, model_id in cls.MODEL_ID_BY_SIZE_KEY.items():
            model_key = model_id.lower()
            repo_name = model_key.split("/", 1)[1]
            if normalized in {size_key, model_key, repo_name}:
                return size_key
        return normalized

    @classmethod
    def _resolve_model_id(cls, model_id: str | None, size_key: str | None) -> str:
        if model_id:
            return cls._strip_hf_hub_prefix(model_id)
        if size_key is None:
            env_model_id = os.getenv("OPEN_CLIP_MODEL_ID")
            if env_model_id:
                return cls._strip_hf_hub_prefix(env_model_id)
        return cls.MODEL_ID_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @staticmethod
    def _strip_hf_hub_prefix(model_id: str) -> str:
        model_id = model_id.strip()
        if model_id.lower().startswith("hf-hub:"):
            return model_id[len("hf-hub:") :]
        return model_id

    @staticmethod
    def _split_model_id_revision(model_id: str) -> tuple[str, str | None]:
        """Split the ``repo@revision`` shorthand accepted by our configs.

        Some open_clip releases pass the whole string to ``hf_hub_download`` as a
        repo id, which Hugging Face rejects.  Keep the public shorthand support
        here, but pass the revision to HF separately.
        """
        model_id = model_id.strip()
        repo_id, sep, revision = model_id.rpartition("@")
        if not sep or not repo_id or not revision:
            return model_id, None
        return repo_id, revision

    @classmethod
    def size_label(cls, size_key: str) -> str:
        return cls.SIZE_LABEL_BY_SIZE_KEY[cls._resolve_size_key(size_key)]

    @staticmethod
    def _validate_force_image_size(value: int | None) -> int | None:
        if value is None:
            return None
        size = int(value)
        if size <= 0:
            raise ValueError("force_image_size must be a positive integer or None")
        return size

    @staticmethod
    def _import_open_clip():
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "OpenClipAdapter requires open_clip_torch. "
                "Install experiments/requirements.txt first."
            ) from exc
        return open_clip

    def _configure_hf_token(self) -> None:
        if not self.hf_token:
            return
        os.environ["HF_TOKEN"] = self.hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self.hf_token)

    def _load_model(self, open_clip):
        self._patch_open_clip_hf_revision(open_clip)
        revision_suffix = f"@{self.revision}" if self.revision else ""
        model_name = f"hf-hub:{self.model_id}{revision_suffix}"
        load_kwargs = {}
        if self.force_image_size is not None:
            # open_clip constructs the visual tower at this size and resizes the
            # checkpoint positional embedding before strict state-dict loading.
            load_kwargs["force_image_size"] = self.force_image_size
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, device=self.device, **load_kwargs
            )
        except TypeError:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, **load_kwargs
            )
            model = model.to(self.device)
        return model, preprocess

    @classmethod
    def _patch_open_clip_hf_revision(cls, open_clip) -> None:
        """Teach older open_clip versions to handle ``hf-hub:repo@revision``.

        Newer experiment configs pin HF revisions.  Older open_clip releases do
        not split the ``@revision`` suffix before calling ``hf_hub_download``;
        with recent huggingface_hub this raises HFValidationError.  Patch both
        the source module and the factory module reference once per process.
        """
        import sys

        pretrained_mod = (
            getattr(open_clip, "pretrained", None) or sys.modules.get("open_clip.pretrained")
        )
        factory_mod = getattr(open_clip, "factory", None) or sys.modules.get(
            "open_clip.factory"
        )
        original = getattr(
            pretrained_mod, "download_pretrained_from_hf", None
        ) or getattr(factory_mod, "download_pretrained_from_hf", None)
        if original is None or getattr(original, "_bt_revision_patch", False):
            return

        def patched(model_id, filename="open_clip_pytorch_model.bin", *args, **kwargs):
            repo_id, embedded_revision = cls._split_model_id_revision(str(model_id))
            if embedded_revision and "revision" not in kwargs:
                kwargs["revision"] = embedded_revision
            try:
                return original(repo_id, filename=filename, *args, **kwargs)
            except TypeError:
                # Older open_clip versions do not accept revision=.  Reproduce
                # their HF download path for pinned revisions.
                revision = kwargs.pop("revision", None)
                if not revision:
                    raise
                cache_dir = kwargs.pop("cache_dir", None)
                if args and cache_dir is None and len(args) == 1:
                    cache_dir = args[0]
                    args = ()
                if args or kwargs:
                    raise
                return cls._download_open_clip_hf_file(
                    repo_id, filename=filename, revision=revision, cache_dir=cache_dir
                )

        patched._bt_revision_patch = True  # type: ignore[attr-defined]
        if pretrained_mod is not None:
            setattr(pretrained_mod, "download_pretrained_from_hf", patched)
        if factory_mod is not None:
            setattr(factory_mod, "download_pretrained_from_hf", patched)

    @staticmethod
    def _download_open_clip_hf_file(
        repo_id: str,
        filename: str | None,
        revision: str,
        cache_dir: str | None = None,
    ) -> str:
        from huggingface_hub import hf_hub_download

        filename = filename or "open_clip_pytorch_model.bin"
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to download file ({filename}) for {repo_id}@{revision}. "
                f"Last error: {exc}"
            ) from exc

    def _resolve_input_size(self) -> int:
        visual = getattr(self.model, "visual", None)
        size = self._coerce_input_size(getattr(visual, "image_size", None))
        if size is not None:
            return size
        for transform in getattr(self.preprocess, "transforms", []):
            if transform.__class__.__name__.lower() == "centercrop":
                size = self._coerce_input_size(getattr(transform, "size", None))
                if size is not None:
                    return size
        return self.INPUT_SIZE_BY_SIZE_KEY[self.size_key]

    @staticmethod
    def _coerce_input_size(size) -> int | None:
        if size is None:
            return None
        if isinstance(size, dict):
            size = size.get("height") or size.get("width") or size.get("shortest_edge")
        if isinstance(size, (tuple, list)):
            size = size[0] if size else None
        try:
            return int(size)
        except (TypeError, ValueError):
            return None

    def _resolve_dimension(self) -> int:
        visual = getattr(self.model, "visual", None)
        output_dim = getattr(visual, "output_dim", None) or getattr(
            self.model, "output_dim", None
        )
        if output_dim is not None:
            return int(output_dim)
        text_projection = getattr(self.model, "text_projection", None)
        shape = getattr(text_projection, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[-1])
        return self.DIMENSION_BY_SIZE_KEY[self.size_key]

    @staticmethod
    def _without_geometry_transforms(preprocess):
        transforms = getattr(preprocess, "transforms", None)
        if not transforms:
            raise RuntimeError(
                "OpenCLIP letterbox preprocessing requires a composed inference transform"
            )
        retained = [
            transform
            for transform in transforms
            if transform.__class__.__name__ not in _OPEN_CLIP_GEOMETRY_TRANSFORMS
        ]
        if len(retained) == len(transforms):
            raise RuntimeError(
                "OpenCLIP letterbox preprocessing found no recognized resize/crop transforms"
            )
        retained_names = {transform.__class__.__name__ for transform in retained}
        if "Normalize" not in retained_names or not retained_names.intersection(
            {"MaybeToTensor", "ToTensor", "PILToTensor"}
        ):
            raise RuntimeError(
                "OpenCLIP letterbox preprocessing must retain tensor conversion "
                "and normalization transforms"
            )
        from torchvision.transforms import Compose

        return Compose(retained)

    def _prepare_images(self, images: list[Image.Image]) -> list[Image.Image]:
        prepared = [
            image if image.mode == "RGB" else image.convert("RGB") for image in images
        ]
        if self.geometry is None:
            return prepared
        target = (self.input_size, self.input_size)
        return [self.geometry.apply(image, target, self.PAD_COLOR) for image in prepared]

    def _prepare_tensor(self, images: list[Image.Image]) -> torch.Tensor:
        preprocess = (
            self._letterbox_preprocess
            if self.geometry is not None and self.geometry.mode == RESIZE_AND_PAD
            else self.preprocess
        )
        tensors = [preprocess(image) for image in self._prepare_images(images)]
        return torch.stack(tensors).to(self.device)

    def _normalize_pooling(self, pooling: Sequence[str]) -> tuple[str, ...]:
        return normalize_pooling_modes(pooling, self.SUPPORTED_POOLINGS)

    def _model_for_pooling(self):
        return getattr(self.model, "_orig_mod", self.model)

    def _visual_for_token_pooling(self):
        visual = getattr(self._model_for_pooling(), "visual", None)
        required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
        )
        missing = [name for name in required if not hasattr(visual, name)]
        if missing:
            raise ValueError(
                "OpenCLIP token pooling is only supported for validated ViT-like "
                f"visual backbones. Missing: {', '.join(missing)}"
            )
        return visual

    @staticmethod
    def _add_positional_embedding(visual, tokens: torch.Tensor) -> torch.Tensor:
        positional = visual.positional_embedding.to(
            device=tokens.device, dtype=tokens.dtype
        )
        if positional.ndim == 2:
            positional = positional.unsqueeze(0)
        if positional.shape[1] != tokens.shape[1]:
            raise ValueError(
                "OpenCLIP positional embedding shape does not match prepared image "
                f"tokens: {tuple(positional.shape)} vs {tuple(tokens.shape)}"
            )
        return tokens + positional

    def _visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        visual = self._visual_for_token_pooling()
        tokens = visual.conv1(pixel_values)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        class_embedding = visual.class_embedding.to(
            device=tokens.device, dtype=tokens.dtype
        )
        class_tokens = class_embedding.reshape(1, 1, -1).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([class_tokens, tokens], dim=1)
        tokens = self._add_positional_embedding(visual, tokens)
        patch_dropout = getattr(visual, "patch_dropout", None)
        if patch_dropout is not None:
            tokens = patch_dropout(tokens)
        tokens = visual.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = visual.transformer(tokens)
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        return tokens.permute(1, 0, 2)

    def _project_visual_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        visual = self._visual_for_token_pooling()
        embedding = visual.ln_post(embedding)
        projection = getattr(visual, "proj", None)
        if projection is not None:
            embedding = embedding @ projection
        return embedding

    def _image_features(
        self, pixel_values: torch.Tensor, pooling: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        modes = self._normalize_pooling(pooling)
        if modes == ("default",):
            features = self.model.encode_image(pixel_values, normalize=False)
            return {"default": features}
        hidden_state = self._visual_tokens(pixel_values)
        patch_tokens = None
        pooled: dict[str, torch.Tensor] = {}
        for mode in modes:
            if mode in {"default", "cls"}:
                embedding = hidden_state[:, 0, :]
            else:
                if patch_tokens is None:
                    patch_tokens = hidden_state[:, 1:, :]
                embedding = aggregate_tokens(patch_tokens, mode)
            embedding = self._project_visual_embedding(embedding)
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
        self, image_paths: list[str], pooling: Sequence[str]
    ) -> dict[str, np.ndarray]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return self.generate_embeddings_batch_from_pil(images, pooling)

    def generate_embeddings_batch_from_pil(
        self, images: list[Image.Image], pooling: Sequence[str]
    ) -> dict[str, np.ndarray]:
        pixel_values = self._prepare_tensor(images)
        with torch.no_grad():
            pooled = self._image_features(pixel_values, pooling)
        return self._to_numpy_batch(pooled)

    def get_model_name(self) -> str:
        return self.model_id.replace("/", "_").replace("-", "_").replace(".", "_")

    def get_dimension(self) -> int:
        return self._dimension
