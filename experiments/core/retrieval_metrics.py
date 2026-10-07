"""Shared cosine-similarity, F-score, and query-coverage helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from experiments.core.descriptor_postprocessing import (
    DESCRIPTOR_DTYPE,
    postprocess_descriptors,
    validate_descriptors,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight environments
    torch = None


def resolve_device(device_preference: str = "auto") -> str:
    if torch is None:
        return "cpu"
    if device_preference == "cpu":
        return "cpu"
    if device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device_preference='cuda' but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_descriptor_matrix(
    embeddings: np.ndarray,
    name: str,
    *,
    inputs_normalized: bool,
) -> np.ndarray:
    if inputs_normalized:
        return validate_descriptors(
            embeddings,
            name=f"{name} descriptor",
            require_unit_norm=True,
        )
    return postprocess_descriptors(embeddings)


@contextmanager
def _cuda_fp32_matmul_mode(
    policy: str,
    *,
    enabled: bool,
) -> Iterator[dict[str, Any]]:
    if policy != "ieee":
        raise ValueError("fp32_matmul_precision must be 'ieee'")
    metadata: dict[str, Any] = {
        "fp32_matmul_precision": policy,
        "tf32_enabled": False,
        "setting_api": "not_applicable",
    }
    if not enabled or torch is None or not torch.cuda.is_available():
        yield metadata
        return

    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        previous = matmul.fp32_precision
        try:
            matmul.fp32_precision = "ieee"
            if matmul.fp32_precision != "ieee":
                raise RuntimeError("Failed to enforce IEEE FP32 CUDA matmul precision")
            metadata["setting_api"] = "torch.backends.cuda.matmul.fp32_precision"
            yield metadata
        finally:
            matmul.fp32_precision = previous
        return

    previous = bool(matmul.allow_tf32)
    try:
        matmul.allow_tf32 = False
        if bool(matmul.allow_tf32):
            raise RuntimeError("Failed to disable TF32 for CUDA matmul")
        metadata["setting_api"] = "torch.backends.cuda.matmul.allow_tf32"
        yield metadata
    finally:
        matmul.allow_tf32 = previous


def cosine_similarity_matrix(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    device_preference: str = "auto",
    gpu_dtype: str = "float32",
    *,
    inputs_normalized: bool = False,
    fp32_matmul_precision: str = "ieee",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Compute cosine similarity through normalized FP32 direct dot products."""
    if gpu_dtype != "float32":
        raise ValueError(
            "FP32 similarity is mandatory; gpu_dtype must be 'float32'"
        )
    query_values = _prepare_descriptor_matrix(
        query_embeddings, "query", inputs_normalized=inputs_normalized
    )
    candidate_values = _prepare_descriptor_matrix(
        candidate_embeddings, "candidate", inputs_normalized=inputs_normalized
    )
    if query_values.shape[1] != candidate_values.shape[1]:
        raise ValueError("Query and candidate descriptor dimensions differ")

    device = resolve_device(device_preference)
    with _cuda_fp32_matmul_mode(
        fp32_matmul_precision,
        enabled=device == "cuda",
    ) as matmul_meta:
        if device == "cuda":
            query = torch.as_tensor(
                query_values, dtype=torch.float32, device=device
            )
            candidate = torch.as_tensor(
                candidate_values, dtype=torch.float32, device=device
            )
            similarities = (query @ candidate.T).cpu().numpy()
        else:
            query = np.asarray(query_values, dtype=DESCRIPTOR_DTYPE)
            candidate = np.asarray(candidate_values, dtype=DESCRIPTOR_DTYPE)
            similarities = np.dot(query, candidate.T).astype(
                DESCRIPTOR_DTYPE, copy=False
            )

    metadata = {
        "device": device,
        "inputs_normalized": bool(inputs_normalized),
        "normalization_applied_by_similarity": not bool(inputs_normalized),
        "similarity_input_dtype": "float32",
        "similarity_output_dtype": str(similarities.dtype),
        **matmul_meta,
    }
    if return_metadata:
        return similarities, metadata
    return similarities


def query_coverage(query_has_candidates: np.ndarray) -> float:
    """Return the fraction of query rows with at least one selected candidate."""
    covered = np.asarray(query_has_candidates, dtype=bool)
    if covered.ndim != 1:
        raise ValueError("query_has_candidates must be a one-dimensional array")
    return float(np.mean(covered)) if covered.size else 0.0


def f_score_vector(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    out = np.zeros_like(precision, dtype=np.float64)
    nz = denom > 0
    out[nz] = (1 + beta_sq) * (precision[nz] * recall[nz]) / denom[nz]
    return out
