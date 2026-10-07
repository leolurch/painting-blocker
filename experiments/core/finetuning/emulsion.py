"""Historic photographic emulsions applied online during training.

The three grayscale conversions follow the film models in Luo et al.,
Time-Travel Rephotography (SIGGRAPH Asia 2021): blue-sensitive plates use the
blue channel, orthochromatic plates average green and blue, and panchromatic
film uses luma. A fourth draw keeps color but fades the cyan dyes of a
mid-century color print. Grain, a response curve, and occasional resolution
loss sit on top of that, in the spirit of the old-photo degradation split of
Wan et al. (CVPR 2020).
"""

from __future__ import annotations

import numpy as np

_BLUE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
_ORTHO = np.array([0.0, 0.5, 0.5], dtype=np.float32)
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _gray(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    tone = image @ weights
    return np.stack((tone, tone, tone), axis=-1)


def historic_emulsion(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    kind: int | None = None,
) -> np.ndarray:
    """Return an uint8 RGB image degraded as one historic photograph.

    ``rng`` decides the emulsion, so a caller that seeds it gets the same
    photograph back. The image is H×W×3.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"historic emulsion expects H×W×3, got {image.shape}")
    pixels = image.astype(np.float32)
    kind = int(rng.integers(0, 4) if kind is None else kind)
    if kind not in {0, 1, 2, 3}:
        raise ValueError("emulsion kind must be 0, 1, 2, or 3")
    if kind == 0:
        degraded = _gray(pixels, _BLUE)
    elif kind == 1:
        degraded = _gray(pixels, _ORTHO)
    elif kind == 2:
        degraded = _gray(pixels, _LUMA)
    else:
        fade = float(rng.uniform(0.35, 0.75))
        degraded = pixels.copy()
        degraded[..., 1] *= fade
        degraded[..., 2] *= fade * float(rng.uniform(0.5, 0.9))
    gamma = float(rng.uniform(0.7, 1.5))
    scale = float(rng.uniform(0.85, 1.15))
    unit = np.clip(degraded, 0.0, 255.0) / 255.0
    degraded = scale * np.power(unit, gamma) * 255.0
    noise = float(rng.uniform(4.0, 14.0) if kind < 3 else rng.uniform(2.0, 8.0))
    if kind < 3:
        grain = rng.normal(0.0, noise, size=degraded.shape[:2]).astype(np.float32)
        degraded += grain[..., None]
    else:
        degraded += rng.normal(0.0, noise, size=degraded.shape)
    if float(rng.random()) < 0.5:
        factor = int(rng.integers(2, 5))
        small = degraded[::factor, ::factor]
        restored = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
        degraded = restored[: pixels.shape[0], : pixels.shape[1]]
    return np.clip(degraded, 0.0, 255.0).astype(np.uint8)
