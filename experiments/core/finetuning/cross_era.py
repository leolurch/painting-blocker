"""Modern-to-historic neighbor lists for cross-era batch construction.

One modern view and one historic view of each training painting are embedded
with the current model. A painting's confusers are the other paintings whose
historic view is nearest to its modern view. A confuser that outscores the
painting's own historic view is dropped: that pair is more likely a missed
duplicate than a hard negative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def probe_plan(
    labels: Sequence[int],
    roles: Sequence[str | None],
) -> list[tuple[int, int, int]]:
    """Pick one modern and one static historic index per label.

    ``modern_original`` wins over other modern roles. ``historic_archival``
    wins over the other static historic roles. Online historic views are
    skipped because their pixels change every draw.
    """
    if len(labels) != len(roles):
        raise ValueError("roles must have the same length as labels")
    modern: dict[int, tuple[int, int]] = {}
    historic: dict[int, tuple[int, int]] = {}
    for index, (label, role) in enumerate(zip(labels, roles)):
        if role is None:
            continue
        name = str(role)
        if name == "modern_original":
            rank = 0
        elif name == "modern" or name.startswith("modern_"):
            rank = 1
        elif name == "historic_online":
            continue
        elif name == "historic_archival":
            rank = 0
        elif name == "historic" or name.startswith("historic_"):
            rank = 1
        else:
            continue
        target = modern if name.startswith("modern") else historic
        current = target.get(int(label))
        if current is None or rank < current[0]:
            target[int(label)] = (rank, index)
    pairs: list[tuple[int, int, int]] = []
    for label in sorted(set(modern) & set(historic)):
        pairs.append((label, modern[label][1], historic[label][1]))
    return pairs


def neighbor_lists(
    modern: np.ndarray,
    historic: np.ndarray,
    class_ids: Sequence[int],
    *,
    pool: int,
) -> tuple[dict[int, list[int]], int]:
    """Return ranked confusers per class, and how many pairs the guard dropped.

    ``modern[i]`` and ``historic[i]`` are the two probe views of
    ``class_ids[i]``. Both matrices are L2-normalized.
    """
    if pool <= 0:
        raise ValueError("pool must be positive")
    if modern.shape != historic.shape or modern.ndim != 2:
        raise ValueError("modern and historic probe matrices must share shape [N, D]")
    if modern.shape[0] != len(class_ids):
        raise ValueError("class_ids must have one entry per probe row")
    if modern.shape[0] < 2:
        return {}, 0
    left = np.asarray(modern, dtype=np.float32)
    right = np.asarray(historic, dtype=np.float32)
    own = np.sum(left * right, axis=1)
    scores = left @ right.T
    np.fill_diagonal(scores, -np.inf)
    guarded = scores > own[:, None]
    guarded_pairs = int(np.sum(guarded))
    scores = np.where(guarded, -np.inf, scores)
    width = min(int(pool), scores.shape[1] - 1)
    chosen = np.argpartition(-scores, kth=width - 1, axis=1)[:, :width]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, chosen], axis=1)
    chosen = chosen[row, order]
    neighbors: dict[int, list[int]] = {}
    ids = [int(class_id) for class_id in class_ids]
    for row_index, class_id in enumerate(ids):
        ranked = [
            ids[int(column)]
            for column in chosen[row_index]
            if np.isfinite(scores[row_index, int(column)])
        ]
        neighbors[class_id] = ranked
    return neighbors, guarded_pairs
