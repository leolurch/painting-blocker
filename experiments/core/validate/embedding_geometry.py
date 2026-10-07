"""Shared image-geometry helpers for embedding adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from PIL import Image

RESIZE = "RESIZE"
RESIZE_AND_PAD = "RESIZE_AND_PAD"
RESIZE_LONGEST_SIDE = "RESIZE_LONGEST_SIDE"
_VALID_MODES = {RESIZE, RESIZE_AND_PAD, RESIZE_LONGEST_SIDE}

try:  # Pillow >= 9
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover
    _BICUBIC = Image.BICUBIC


@dataclass(frozen=True)
class ImageGeometry:
    mode: str = RESIZE
    max_width: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            valid = ", ".join(sorted(_VALID_MODES))
            raise ValueError(
                f"Unsupported image geometry mode: {self.mode!r}. Expected one of: {valid}"
            )
        if self.max_width is not None and self.max_width <= 0:
            raise ValueError(
                "ImageGeometry.max_width must be a positive integer or None"
            )

    def describe(self) -> str:
        width = self.max_width if self.max_width is not None else "<none>"
        return f"mode={self.mode}, max_width={width}"

    def to_config(self) -> dict[str, int | str | None]:
        return {"mode": self.mode, "max_width": self.max_width}

    def apply(
        self,
        image: Image.Image,
        target_size: tuple[int, int] | None = None,
        pad_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        prepared = image if image.mode == "RGB" else image.convert("RGB")
        if self.mode == RESIZE_LONGEST_SIDE:
            # Cap the longer side to max_width, preserving aspect ratio. No pad, no crop.
            return self._apply_max_side(prepared)
        prepared = self._apply_max_width(prepared)
        if target_size is None:
            return prepared
        width, height = target_size
        if width <= 0 or height <= 0:
            raise ValueError("target_size values must be positive integers")
        if self.mode == RESIZE:
            return prepared.resize((width, height), _BICUBIC)
        return _resize_and_pad(prepared, width, height, pad_color)

    def _apply_max_width(self, image: Image.Image) -> Image.Image:
        if self.max_width is None or image.width <= self.max_width:
            return image
        scale = self.max_width / float(image.width)
        height = max(1, int(round(image.height * scale)))
        return image.resize((self.max_width, height), _BICUBIC)

    def _apply_max_side(self, image: Image.Image) -> Image.Image:
        longest = max(image.width, image.height)
        if self.max_width is None or longest <= self.max_width:
            return image
        scale = self.max_width / float(longest)
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        return image.resize((width, height), _BICUBIC)


def geometry_map_to_config(
    geometries: Mapping[str, ImageGeometry] | None,
) -> dict[str, dict[str, int | str | None]]:
    if not geometries:
        return {}
    return {key: geometry.to_config() for key, geometry in geometries.items()}


def pad_color_from_mean(mean: tuple[float, float, float]) -> tuple[int, int, int]:
    color = []
    for value in mean:
        scaled = value * 255 if 0.0 <= value <= 1.0 else value
        color.append(int(max(0, min(255, round(scaled)))))
    return tuple(color)


def resize_to_max_pixels(image: Image.Image, max_pixels: int | None) -> Image.Image:
    if max_pixels is None:
        return image
    if max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer or None")
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = (max_pixels / float(pixels)) ** 0.5
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    while width * height > max_pixels:
        if width >= height and width > 1:
            width -= 1
        elif height > 1:
            height -= 1
        else:  # pragma: no cover
            break
    return image.resize((width, height), _BICUBIC)


def _resize_and_pad(
    image: Image.Image,
    width: int,
    height: int,
    pad_color: tuple[int, int, int],
) -> Image.Image:
    scale = min(width / float(image.width), height / float(image.height))
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), _BICUBIC)
    if (resized_width, resized_height) == (width, height):
        return resized
    canvas = Image.new("RGB", (width, height), pad_color)
    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    canvas.paste(resized, (x_offset, y_offset))
    return canvas
