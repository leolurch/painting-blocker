"""Offline hard-negative mining for metric-learning finetuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from experiments.core.retrieval_metrics import cosine_similarity_matrix

from .augmentations import build_image_transform
from .checkpointing import load_model_from_checkpoint
from .data import load_subset_items
from .evaluate import _selector_from_experiment, embed_items


def mine_hard_negatives_from_embeddings(
    image_ids: Sequence[str],
    class_ids: Sequence[str],
    embeddings: np.ndarray,
    *,
    top_m: int = 50,
    source_model: str = "checkpoint",
    epoch: int = 0,
    source: str = "nearest_non_class",
) -> pd.DataFrame:
    """Return top non-class nearest neighbors for each anchor."""
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    if len(image_ids) != len(class_ids) or len(image_ids) != embeddings.shape[0]:
        raise ValueError("image_ids, class_ids, and embeddings must have matching lengths")
    sims = cosine_similarity_matrix(
        embeddings,
        embeddings,
        device_preference="cpu",
        gpu_dtype="float32",
        inputs_normalized=False,
        fp32_matmul_precision="ieee",
    )
    labels = np.asarray(list(class_ids))
    same_class = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_class, True)
    sims = sims.astype(np.float32, copy=True)
    sims[same_class] = -np.inf
    rows: list[dict[str, object]] = []
    for idx, anchor_id in enumerate(image_ids):
        valid = np.isfinite(sims[idx])
        if not np.any(valid):
            continue
        k = min(int(top_m), int(valid.sum()))
        candidate_idx = np.argpartition(-sims[idx], kth=k - 1)[:k]
        candidate_idx = candidate_idx[np.argsort(-sims[idx][candidate_idx])]
        for neg_idx in candidate_idx:
            rows.append(
                {
                    "anchor_image_id": str(anchor_id),
                    "negative_image_id": str(image_ids[int(neg_idx)]),
                    "anchor_class_id": str(class_ids[idx]),
                    "negative_class_id": str(class_ids[int(neg_idx)]),
                    "score": float(sims[idx, int(neg_idx)]),
                    "source_model": source_model,
                    "epoch": int(epoch),
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def write_hard_negatives(path: Path | str, rows: pd.DataFrame) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(target, index=False)
    return target


def mine_hard_negatives_for_checkpoint(
    experiment_config: Any,
    split: dict[str, Any],
    checkpoint_path: Path | str,
    *,
    output: Path,
    subset_name: str = "train",
    top_m: int = 50,
    epoch: int = 0,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location="cpu")
    transform = build_image_transform(
        {**dict(checkpoint.get("preprocessing_config") or {}), "normalization": checkpoint.get("normalization_config") or {}},
        train=False,
    )
    selector = _selector_from_experiment(experiment_config, subset_name)
    items = load_subset_items(experiment_config.dataset, split, selector)
    image_ids, class_ids, embeddings = embed_items(model, items, transform=transform, batch_size=batch_size, device=device)
    df = mine_hard_negatives_from_embeddings(
        image_ids,
        class_ids,
        embeddings,
        top_m=top_m,
        source_model=str(Path(checkpoint_path).expanduser()),
        epoch=epoch,
    )
    path = write_hard_negatives(output, df)
    return {"status": "ok", "output": str(path), "rows": int(len(df)), "subset": selector["subset"], "top_m": int(top_m)}
