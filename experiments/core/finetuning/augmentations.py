"""Image augmentation builders used by finetuning datasets."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


_AUGMENTATION_KEYS = {
    "backend",
    "random_crop",
    "random_resized_crop",
    "horizontal_flip",
    "affine",
    "color_jitter",
    "grayscale",
    "image_compression",
    "gaussian_blur",
    "coarse_dropout",
    "perspective",
    "desaturation",
    "square_mask",
    "center_crop_in",
    "gaussian_noise",
}

HISTORIC_ROLES = frozenset(
    {
        "historic_archival",
        "historic_print",
        "historic_cropped_record",
        "historic_framed_photo",
    }
)
MODERN_ROLES = frozenset({"modern_original", "modern_generated"})


def _image_size(config: dict[str, Any]) -> int:
    preprocessing = dict(config.get("preprocessing") or {})
    resize = dict(preprocessing.get("resize") or {})
    return int(config.get("image_size") or resize.get("size") or resize.get("max_width") or 224)


def _resize_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(dict(config.get("preprocessing") or {}).get("resize") or {})


def _pil_bicubic():
    from PIL import Image

    return getattr(Image, "Resampling", Image).BICUBIC


def _shortest_side_size(height: int, width: int, resize_size: int, crop_size: int) -> tuple[int, int]:
    if crop_size > resize_size:
        raise ValueError("center_crop size must not exceed resize_size")
    if width <= height:
        return int(resize_size * height / width), resize_size
    return resize_size, int(resize_size * width / height)


def _crop_offset(length: int, crop_size: int, offsets: str) -> int:
    if offsets == "floor":
        return int((length - crop_size) / 2.0)
    return int(round((length - crop_size) / 2.0))


def shortest_side_center_crop(
    image: np.ndarray, resize_size: int, crop_size: int, offsets: str = "round"
) -> np.ndarray:
    """PIL bicubic resize of the shorter side to ``resize_size``, then a centered square crop.

    torchvision's size and offset rounding, as in the OpenCLIP ``model_default`` transform.
    """
    from PIL import Image

    new_height, new_width = _shortest_side_size(*image.shape[:2], resize_size, crop_size)
    resized = Image.fromarray(image).resize((new_width, new_height), _pil_bicubic())
    top = _crop_offset(new_height, crop_size, offsets)
    left = _crop_offset(new_width, crop_size, offsets)
    return np.asarray(resized.crop((left, top, left + crop_size, top + crop_size)))


def tensor_center_crop(
    image: np.ndarray, resize_size: int, crop_size: int, offsets: str = "round"
) -> np.ndarray:
    """Antialiased bicubic resize of the uint8 tensor, then a centered square crop.

    The geometry of the Hugging Face fast image processors (CLIP: floor offsets;
    ConvNeXt, used by ResNet-50: torchvision's rounded offsets).
    """
    import torch
    from torchvision.transforms.v2 import functional as F

    new_height, new_width = _shortest_side_size(*image.shape[:2], resize_size, crop_size)
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).contiguous()
    resized = F.resize(tensor, [new_height, new_width], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    top = _crop_offset(new_height, crop_size, offsets)
    left = _crop_offset(new_width, crop_size, offsets)
    return resized[:, top : top + crop_size, left : left + crop_size].permute(1, 2, 0).contiguous().numpy()


def pil_letterbox(image: np.ndarray, size: int, pad_color: tuple[int, int, int]) -> np.ndarray:
    """Letterbox exactly as the frozen evaluation adapters do (ImageGeometry letterbox).

    Wide images are first scaled to ``size`` pixels wide, then the result is fit
    into the ``size`` square and padded.
    """
    from PIL import Image

    resized = Image.fromarray(image)
    if resized.width > size:
        resized = resized.resize((size, max(1, int(round(resized.height * size / float(resized.width))))), _pil_bicubic())
    width, height = resized.size
    scale = min(size / float(width), size / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = resized.resize((new_width, new_height), _pil_bicubic())
    if (new_width, new_height) == (size, size):
        return np.asarray(resized)
    canvas = Image.new("RGB", (size, size), pad_color)
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return np.asarray(canvas)


def _norm(config: dict[str, Any]) -> tuple[list[float], list[float]]:
    norm = dict(config.get("normalization") or {})
    mean = [float(value) for value in norm.get("mean", [0.485, 0.456, 0.406])]
    std = [float(value) for value in norm.get("std", [0.229, 0.224, 0.225])]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("normalization.mean and normalization.std must contain three values")
    return mean, std


def _pad_color_from_mean(mean: list[float]) -> tuple[int, int, int]:
    """Return RGB pad color that normalizes to approximately zero."""
    color = []
    for value in mean:
        scaled = value * 255.0 if 0.0 <= value <= 1.0 else value
        color.append(int(max(0, min(255, round(scaled)))))
    return tuple(color)  # type: ignore[return-value]


def _pad_if_needed(A, *, size: int, mean: list[float], cv2):
    """Build PadIfNeeded with mean-color constant padding across albumentations versions."""
    pad_color = _pad_color_from_mean(mean)
    kwargs = {
        "min_height": size,
        "min_width": size,
        "border_mode": cv2.BORDER_CONSTANT,
    }
    try:
        return A.PadIfNeeded(**kwargs, fill=pad_color, fill_mask=0)
    except TypeError:  # albumentations<2
        return A.PadIfNeeded(**kwargs, value=pad_color, mask_value=0)


def resolve_augmentation_roles(value: Any) -> frozenset[str] | None:
    """Expand a YAML role list. ``historic`` and ``modern`` are groups.

    ``None`` means every training image.
    """
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not value:
        raise ValueError("augmentations roles must be a non-empty list")
    resolved: list[str] = []
    known = HISTORIC_ROLES | MODERN_ROLES
    for item in value:
        token = str(item).strip()
        if token == "historic":
            resolved.extend(sorted(HISTORIC_ROLES))
        elif token == "modern":
            resolved.extend(sorted(MODERN_ROLES))
        elif token in known:
            resolved.append(token)
        else:
            raise ValueError(f"Unknown augmentation role {token!r}")
    return frozenset(resolved)


def desaturate_rgb(image: np.ndarray, keep: float) -> np.ndarray:
    """Move RGB pixels toward luma, keeping ``keep`` of the original color."""
    if keep < 0.0 or keep > 1.0:
        raise ValueError("desaturation keep must be between 0 and 1")
    pixels = image.astype(np.float32)
    luma = (
        0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]
    )[..., None]
    mixed = luma + (pixels - luma) * float(keep)
    return np.clip(np.rint(mixed), 0, 255).astype(image.dtype)


def center_crop_in_image(image: np.ndarray, fraction: float) -> np.ndarray:
    """Crop equally from every edge, removing ``fraction`` of each side length."""
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError("center crop-in fraction must be in [0, 1)")
    if fraction == 0.0:
        return image
    height, width = image.shape[:2]
    keep = 1.0 - float(fraction)
    crop_h = max(1, int(round(height * keep)))
    crop_w = max(1, int(round(width * keep)))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    return image[top : top + crop_h, left : left + crop_w]


def square_mask_side(height: int, width: int, area_fraction: float) -> int:
    """Side length of one square that covers ``area_fraction`` of the image."""
    if area_fraction <= 0.0 or area_fraction > 1.0:
        raise ValueError("square mask area_fraction must be in (0, 1]")
    side = int(round(math.sqrt(float(area_fraction) * height * width)))
    return max(1, min(side, height, width))


def apply_square_mask(
    image: np.ndarray,
    area_fraction: float,
    *,
    top: int,
    left: int,
) -> np.ndarray:
    """Cover one square with black. ``top`` and ``left`` are the square's origin."""
    height, width = image.shape[:2]
    side = square_mask_side(height, width, area_fraction)
    top = min(max(0, int(top)), height - side)
    left = min(max(0, int(left)), width - side)
    covered = np.array(image, copy=True)
    covered[top : top + side, left : left + side] = 0
    return covered


def _role_allowed(role: str | None, roles: frozenset[str] | None) -> bool:
    if roles is None:
        return True
    return role in roles


class _AlbumentationsStep:
    def __init__(self, transform: object, roles: frozenset[str] | None) -> None:
        self.transform = transform
        self.roles = roles

    def __call__(self, array: np.ndarray, role: str | None) -> np.ndarray:
        if not _role_allowed(role, self.roles):
            return array
        output = self.transform(image=array)["image"]
        return np.asarray(output)


class _CallableStep:
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        roles: frozenset[str] | None,
    ) -> None:
        self.function = function
        self.roles = roles

    def __call__(self, array: np.ndarray, role: str | None) -> np.ndarray:
        if not _role_allowed(role, self.roles):
            return array
        return self.function(array)


class _AlbumentationsTransform:
    def __init__(self, steps: list[object], final: object) -> None:
        self.steps = steps
        self.final = final

    def __call__(self, image, role: str | None = None):
        array = np.asarray(image.convert("RGB"))
        for step in self.steps:
            array = step(array, role)
        return self.final(image=array)["image"]


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"augmentations.{key} must be a mapping")
    return dict(value)


def _probability(block: dict[str, Any], key: str) -> float:
    if "p" not in block:
        raise ValueError(f"augmentations.{key}.p is required")
    p = float(block["p"])
    if p < 0.0 or p > 1.0:
        raise ValueError(f"augmentations.{key}.p must be between 0 and 1")
    return p


def _tuple(value: Any, *, length: int, key: str, cast=float) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain {length} values")
    return tuple(cast(item) for item in value)


def _image_compression(A, quality_range: tuple[int, int], p: float):
    try:
        return A.ImageCompression(quality_range=quality_range, p=p)
    except TypeError:  # albumentations<2
        return A.ImageCompression(quality_lower=quality_range[0], quality_upper=quality_range[1], p=p)


def _coarse_dropout(A, block: dict[str, Any], p: float):
    try:
        return A.CoarseDropout(
            num_holes_range=_tuple(block.get("num_holes_range"), length=2, key="augmentations.coarse_dropout.num_holes_range", cast=int),
            hole_height_range=_tuple(block.get("hole_height_range"), length=2, key="augmentations.coarse_dropout.hole_height_range"),
            hole_width_range=_tuple(block.get("hole_width_range"), length=2, key="augmentations.coarse_dropout.hole_width_range"),
            p=p,
        )
    except TypeError:  # albumentations<2
        return A.CoarseDropout(
            min_holes=int(block.get("num_holes_range")[0]),
            max_holes=int(block.get("num_holes_range")[1]),
            max_height=int(block.get("max_height", 45)),
            max_width=int(block.get("max_width", 45)),
            p=p,
        )


def _augmentation_config(config: dict[str, Any]) -> dict[str, Any]:
    if "augmentations" not in config or config.get("augmentations") is None:
        return {}
    aug = config.get("augmentations")
    if not isinstance(aug, dict):
        raise ValueError("augmentations must be a mapping")
    if not aug:
        return {}
    unknown = sorted(set(aug) - _AUGMENTATION_KEYS)
    if unknown:
        raise ValueError(f"Unsupported augmentation key(s): {', '.join(unknown)}")
    backend = str(aug.get("backend", "")).strip().lower()
    if backend != "albumentations":
        raise ValueError("Only augmentations.backend='albumentations' is supported")
    return dict(aug)


def _roles_of(block: dict[str, Any]) -> frozenset[str] | None:
    return resolve_augmentation_roles(block.get("roles"))


def _albumentations_step(transform: object, block: dict[str, Any]) -> _AlbumentationsStep:
    return _AlbumentationsStep(transform, _roles_of(block))


def _desaturation_step(block: dict[str, Any]) -> _CallableStep:
    probability = _probability(block, "desaturation")
    keep = _tuple(block.get("keep", [0.05, 0.40]), length=2, key="augmentations.desaturation.keep")
    low, high = float(keep[0]), float(keep[1])
    if low < 0.0 or high > 1.0 or low > high:
        raise ValueError("augmentations.desaturation.keep must be within 0 and 1, low then high")
    roles = _roles_of(block)

    def apply(array: np.ndarray) -> np.ndarray:
        if float(np.random.random()) > probability:
            return array
        factor = float(np.random.uniform(low, high))
        return desaturate_rgb(array, factor)

    return _CallableStep(apply, roles)


def _center_crop_in_step(block: dict[str, Any]) -> _CallableStep:
    probability = _probability(block, "center_crop_in")
    max_fraction = float(block.get("max_fraction", 0.30))
    if max_fraction <= 0.0 or max_fraction >= 1.0:
        raise ValueError("augmentations.center_crop_in.max_fraction must be in (0, 1)")
    roles = _roles_of(block)

    def apply(array: np.ndarray) -> np.ndarray:
        if float(np.random.random()) > probability:
            return array
        fraction = float(np.random.uniform(0.0, max_fraction))
        return center_crop_in_image(array, fraction)

    return _CallableStep(apply, roles)


def _square_mask_step(block: dict[str, Any]) -> _CallableStep:
    probability = _probability(block, "square_mask")
    area = _tuple(block.get("area_fraction", [0.05, 0.20]), length=2, key="augmentations.square_mask.area_fraction")
    low, high = float(area[0]), float(area[1])
    if low <= 0.0 or high > 1.0 or low > high:
        raise ValueError("augmentations.square_mask.area_fraction must be within (0, 1], low then high")
    roles = _roles_of(block)

    def apply(array: np.ndarray) -> np.ndarray:
        if float(np.random.random()) > probability:
            return array
        fraction = float(np.random.uniform(low, high))
        height, width = array.shape[:2]
        side = square_mask_side(height, width, fraction)
        top = int(np.random.randint(0, height - side + 1))
        left = int(np.random.randint(0, width - side + 1))
        return apply_square_mask(array, fraction, top=top, left=left)

    return _CallableStep(apply, roles)


def _random_resized_crop(A: object, cv2: object, size: int, scale: tuple[Any, ...], ratio: tuple[Any, ...]) -> object:
    """Albumentations RandomResizedCrop across the height/width and size APIs."""
    import inspect

    common = {
        "scale": (float(scale[0]), float(scale[1])),
        "ratio": (float(ratio[0]), float(ratio[1])),
        "interpolation": cv2.INTER_CUBIC,
        "p": 1.0,
    }
    parameters = inspect.signature(A.RandomResizedCrop).parameters
    if "size" in parameters:
        return A.RandomResizedCrop(size=(size, size), **common)
    return A.RandomResizedCrop(height=size, width=size, **common)


def _build_albumentations_transform(cfg: dict[str, Any], *, train: bool) -> object:
    try:
        import albumentations as A
        import cv2
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Finetuning image transforms require albumentations and opencv-python-headless") from exc

    aug = _augmentation_config(cfg)
    size = _image_size(cfg)
    mean, std = _norm(cfg)
    steps: list[object] = []

    if train and "center_crop_in" in aug:
        steps.append(_center_crop_in_step(_required_mapping(aug, "center_crop_in")))

    random_crop = train and "random_crop" in aug
    resized_crop = train and "random_resized_crop" in aug
    if random_crop and resized_crop:
        raise ValueError("augmentations.random_crop and random_resized_crop cannot both be set")
    if resized_crop:
        crop_cfg = _required_mapping(aug, "random_resized_crop")
        scale = _tuple(crop_cfg.get("scale", [0.7, 1.0]), length=2, key="augmentations.random_resized_crop.scale")
        ratio = _tuple(crop_cfg.get("ratio"), length=2, key="augmentations.random_resized_crop.ratio")
        if float(scale[0]) <= 0 or float(scale[0]) > float(scale[1]) or float(scale[1]) > 1:
            raise ValueError("augmentations.random_resized_crop.scale must be within (0, 1], low then high")
        if float(ratio[0]) <= 0 or float(ratio[0]) > float(ratio[1]):
            raise ValueError("augmentations.random_resized_crop.ratio must be positive, low then high")
        resize = [_random_resized_crop(A, cv2, size, scale, ratio)]
    elif random_crop:
        crop_cfg = _required_mapping(aug, "random_crop")
        max_size = int(crop_cfg.get("max_size", 0))
        if max_size < size:
            raise ValueError("augmentations.random_crop.max_size must be >= image_size")
        resize = [
            A.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_CUBIC),
            _pad_if_needed(A, size=size, mean=mean, cv2=cv2),
            A.RandomCrop(height=size, width=size),
        ]
    else:
        resize_cfg = _resize_config(cfg)
        mode = str(resize_cfg.get("mode", "letterbox")).strip().lower()
        backend = str(resize_cfg.get("backend", "albumentations")).strip().lower()
        if backend not in {"albumentations", "pil", "torch"}:
            raise ValueError("preprocessing.resize.backend must be 'albumentations', 'pil', or 'torch'")
        if backend == "torch" and mode != "center_crop":
            raise ValueError("preprocessing.resize.backend 'torch' requires mode 'center_crop'")
        if mode == "center_crop":
            resize_size = int(resize_cfg.get("resize_size", size))
            offsets = str(resize_cfg.get("offsets", "round")).strip().lower()
            if offsets not in {"round", "floor"}:
                raise ValueError("preprocessing.resize.offsets must be 'round' or 'floor'")
            crop = tensor_center_crop if backend == "torch" else shortest_side_center_crop
            resize = [_CallableStep(lambda array: crop(array, resize_size, size, offsets), None)]
        elif backend == "pil":
            pad_color = _pad_color_from_mean(mean)
            resize = [_CallableStep(lambda array: pil_letterbox(array, size, pad_color), None)]
        else:
            resize = [
                A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC),
                _pad_if_needed(A, size=size, mean=mean, cv2=cv2),
                A.CenterCrop(height=size, width=size),
            ]
    steps.extend(step if isinstance(step, _CallableStep) else _AlbumentationsStep(step, None) for step in resize)

    if train:
        if "horizontal_flip" in aug:
            block = _required_mapping(aug, "horizontal_flip")
            steps.append(_albumentations_step(A.HorizontalFlip(p=_probability(block, "horizontal_flip")), block))
        if "affine" in aug:
            block = _required_mapping(aug, "affine")
            steps.append(
                _albumentations_step(
                    A.Affine(
                        scale=_tuple(block.get("scale"), length=2, key="augmentations.affine.scale"),
                        translate_percent=_tuple(
                            block.get("translate_percent"), length=2, key="augmentations.affine.translate_percent"
                        ),
                        rotate=_tuple(block.get("rotate"), length=2, key="augmentations.affine.rotate"),
                        shear=_tuple(block.get("shear"), length=2, key="augmentations.affine.shear"),
                        p=_probability(block, "affine"),
                    ),
                    block,
                )
            )
        if "color_jitter" in aug:
            block = _required_mapping(aug, "color_jitter")
            steps.append(
                _albumentations_step(
                    A.ColorJitter(
                        brightness=float(block["brightness"]),
                        contrast=float(block["contrast"]),
                        saturation=float(block["saturation"]),
                        hue=float(block["hue"]),
                        p=_probability(block, "color_jitter"),
                    ),
                    block,
                )
            )
        if "desaturation" in aug:
            steps.append(_desaturation_step(_required_mapping(aug, "desaturation")))
        if "grayscale" in aug:
            block = _required_mapping(aug, "grayscale")
            steps.append(_albumentations_step(A.ToGray(p=_probability(block, "grayscale")), block))
        if "image_compression" in aug:
            block = _required_mapping(aug, "image_compression")
            steps.append(
                _albumentations_step(
                    _image_compression(
                        A,
                        _tuple(
                            block.get("quality_range"),
                            length=2,
                            key="augmentations.image_compression.quality_range",
                            cast=int,
                        ),
                        p=_probability(block, "image_compression"),
                    ),
                    block,
                )
            )
        if "gaussian_blur" in aug:
            block = _required_mapping(aug, "gaussian_blur")
            steps.append(
                _albumentations_step(
                    A.GaussianBlur(
                        blur_limit=_tuple(
                            block.get("blur_limit"),
                            length=2,
                            key="augmentations.gaussian_blur.blur_limit",
                            cast=int,
                        ),
                        p=_probability(block, "gaussian_blur"),
                    ),
                    block,
                )
            )
        if "gaussian_noise" in aug:
            block = _required_mapping(aug, "gaussian_noise")
            steps.append(
                _albumentations_step(
                    A.GaussNoise(
                        std_range=_tuple(
                            block.get("std_range"),
                            length=2,
                            key="augmentations.gaussian_noise.std_range",
                        ),
                        mean_range=(0.0, 0.0),
                        per_channel=True,
                        p=_probability(block, "gaussian_noise"),
                    ),
                    block,
                )
            )
        if "coarse_dropout" in aug:
            block = _required_mapping(aug, "coarse_dropout")
            steps.append(_albumentations_step(_coarse_dropout(A, block, p=_probability(block, "coarse_dropout")), block))
        if "square_mask" in aug:
            steps.append(_square_mask_step(_required_mapping(aug, "square_mask")))
        if "perspective" in aug:
            block = _required_mapping(aug, "perspective")
            steps.append(
                _albumentations_step(
                    A.Perspective(
                        scale=_tuple(block.get("scale"), length=2, key="augmentations.perspective.scale"),
                        keep_size=bool(block.get("keep_size", True)),
                        p=_probability(block, "perspective"),
                    ),
                    block,
                )
            )

    final = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return _AlbumentationsTransform(steps, final)


def build_image_transform(config: dict[str, Any] | None = None, *, train: bool) -> object:
    """Build an image transform from a YAML-friendly config.

    Stochastic augmentations are opt-in: if ``augmentations`` is absent, the
    returned transform performs only deterministic resize/pad/crop,
    normalization, and tensor conversion. The only supported augmentation
    backend is albumentations.
    """
    return _build_albumentations_transform(dict(config or {}), train=train)


def preprocessing_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the checkpoint-serializable preprocessing block."""
    cfg = dict(config or {})
    return {"image_size": _image_size(cfg), "preprocessing": dict(cfg.get("preprocessing") or {})}


def normalization_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    mean, std = _norm(dict(config or {}))
    return {"mean": mean, "std": std}
