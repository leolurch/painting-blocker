"""Canonical IEEE-FP32 descriptor validation and post-processing."""

from __future__ import annotations

import numpy as np


DESCRIPTOR_POSTPROCESSOR_VERSION = 3
DESCRIPTOR_DTYPE = np.dtype("float32")
# Below this magnitude, an unscaled FP32 sum-of-squares starts entering the
# subnormal range. Treat such descriptors as numerically degenerate instead of
# manufacturing a direction from almost no signal.
MIN_DESCRIPTOR_NORM = np.float32(np.sqrt(np.finfo(np.float32).tiny))
UNIT_NORM_RTOL = 1e-4
UNIT_NORM_ATOL = 1e-5


def validate_descriptors(
    embeddings: np.ndarray,
    *,
    name: str = "descriptor",
    require_unit_norm: bool,
) -> np.ndarray:
    """Validate the canonical descriptor matrix contract without mutating it."""
    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise ValueError(f"Expected {name} matrix with shape [n, d], got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Expected non-empty {name} matrix, got {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} dtype must be floating, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} matrix contains NaN or infinity")
    if require_unit_norm:
        if values.dtype != DESCRIPTOR_DTYPE:
            raise TypeError(
                f"Normalized {name} values must be float32, got {values.dtype}"
            )
        norms = _stable_fp32_norm(values)
        if not np.allclose(
            norms,
            np.float32(1.0),
            rtol=UNIT_NORM_RTOL,
            atol=UNIT_NORM_ATOL,
        ):
            raise ValueError(f"Normalized {name} values are not unit length")
    return values


def postprocess_descriptors(
    embeddings: np.ndarray,
    *,
    normalize: bool = True,
    dtype: np.dtype = DESCRIPTOR_DTYPE,
) -> np.ndarray:
    """Copy descriptors, cast to FP32, validate, and optionally L2-normalize.

    FP32 is the only supported post-processing dtype. L2 normalization uses a
    scale-first algorithm so finite large descriptors cannot overflow while
    squaring. The input is never mutated.
    """
    requested_dtype = np.dtype(dtype)
    if requested_dtype != DESCRIPTOR_DTYPE:
        raise TypeError(
            "Descriptor post-processing is fixed to float32; "
            f"got requested dtype {requested_dtype}"
        )

    source = np.asarray(embeddings)
    validate_descriptors(source, name="descriptor", require_unit_norm=False)
    output = np.array(source, dtype=DESCRIPTOR_DTYPE, order="C", copy=True)
    if not np.isfinite(output).all():
        raise ValueError("Descriptor matrix became non-finite when cast to float32")

    if normalize:
        scales = np.max(np.abs(output), axis=1, keepdims=True)
        zero = scales[:, 0] == 0
        possible_near_zero = scales[:, 0] <= MIN_DESCRIPTOR_NORM

        scaled = np.empty_like(output)
        np.divide(output, scales, out=scaled, where=~zero[:, None])
        scaled[zero] = 0
        scaled_norms = np.sqrt(
            np.sum(scaled * scaled, axis=1, keepdims=True, dtype=np.float32)
        )

        near_zero = zero.copy()
        candidate_indices = np.flatnonzero(possible_near_zero & ~zero)
        if candidate_indices.size:
            candidate_norms = (
                scales[candidate_indices, 0]
                * scaled_norms[candidate_indices, 0]
            )
            near_zero[candidate_indices] = candidate_norms <= MIN_DESCRIPTOR_NORM
        if np.any(near_zero):
            indices = np.flatnonzero(near_zero)
            raise ValueError(
                f"Found {len(indices)} zero or near-zero descriptors "
                f"at indices {indices[:10].tolist()}"
            )

        output = scaled
        output /= scaled_norms
        validate_descriptors(
            output,
            name="post-processed descriptor",
            require_unit_norm=True,
        )
    return output


def _stable_fp32_norm(values: np.ndarray) -> np.ndarray:
    """Return row norms using only overflow-safe FP32 arithmetic."""
    scales = np.max(np.abs(values), axis=1, keepdims=True)
    zero = scales[:, 0] == 0
    scaled = np.empty_like(values, dtype=np.float32)
    np.divide(values, scales, out=scaled, where=~zero[:, None])
    scaled[zero] = 0
    norms = scales[:, 0] * np.sqrt(
        np.sum(scaled * scaled, axis=1, dtype=np.float32)
    )
    return norms.astype(np.float32, copy=False)
