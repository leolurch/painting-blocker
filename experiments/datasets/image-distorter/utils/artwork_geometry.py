"""Geometry and mask scoring for paintings placed on a wall.

The wall generator warps a whole scene with one perspective transform. These
helpers map the pre-warp artwork rectangle into the final crop and score a
predicted segmentation against that ground truth.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


def perspective_matrix(
    src: Sequence[Sequence[float]],
    dst: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return the 3x3 homography that maps ``src`` points onto ``dst``."""
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("A perspective matrix needs four source and four destination points")
    rows: list[list[float]] = []
    targets: list[float] = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        rows.append([x_src, y_src, 1.0, 0.0, 0.0, 0.0, -x_dst * x_src, -x_dst * y_src])
        rows.append([0.0, 0.0, 0.0, x_src, y_src, 1.0, -y_dst * x_src, -y_dst * y_src])
        targets.extend((x_dst, y_dst))
    solution = np.linalg.solve(
        np.asarray(rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )
    return np.array(
        [
            [solution[0], solution[1], solution[2]],
            [solution[3], solution[4], solution[5]],
            [solution[6], solution[7], 1.0],
        ],
        dtype=np.float64,
    )


def apply_homography(
    matrix: np.ndarray,
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points."""
    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got {coords.shape}")
    homogeneous = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return mapped[:, :2] / mapped[:, 2:3]


def rectangle_corners(origin: tuple[int, int], size: tuple[int, int]) -> list[list[float]]:
    """Return TL, TR, BR, BL corners of an axis-aligned rectangle."""
    x, y = origin
    width, height = size
    return [
        [float(x), float(y)],
        [float(x + width), float(y)],
        [float(x + width), float(y + height)],
        [float(x), float(y + height)],
    ]


def map_artwork_corners(
    corners: Sequence[Sequence[float]],
    homography: np.ndarray,
    *,
    scale: tuple[float, float] = (1.0, 1.0),
    crop_origin: tuple[float, float] = (0.0, 0.0),
) -> list[list[float]]:
    """Map pre-warp rectangle corners into the final cropped scene."""
    warped = apply_homography(homography, corners)
    scaled = warped * np.asarray(scale, dtype=np.float64)
    shifted = scaled - np.asarray(crop_origin, dtype=np.float64)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in shifted]


def rasterize_polygon(
    corners: Sequence[Sequence[float]],
    size: tuple[int, int],
) -> np.ndarray:
    """Rasterize a quadrilateral into a boolean mask of shape (height, width)."""
    width, height = size
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(
        [(float(x), float(y)) for x, y in corners],
        fill=255,
    )
    return np.asarray(canvas) > 0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return an exclusive PIL-style bbox, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def union_bbox(
    boxes: Sequence[tuple[int, int, int, int] | None],
) -> tuple[int, int, int, int] | None:
    present = [box for box in boxes if box is not None]
    if not present:
        return None
    return (
        min(box[0] for box in present),
        min(box[1] for box in present),
        max(box[2] for box in present),
        max(box[3] for box in present),
    )


def find_opaque_crop(
    mask: np.ndarray,
    crop_w: int,
    crop_h: int,
    required_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, int] | None:
    """Return the opaque crop closest to center that contains ``required_bbox``.

    ``mask`` is True where the warped wall is opaque. This is the same window
    search the single-painting placer uses, so a multi-painting scene can keep
    every artwork inside the final frame.
    """
    opaque = np.asarray(mask, dtype=bool)
    height, width = opaque.shape
    if crop_w > width or crop_h > height or crop_w <= 0 or crop_h <= 0:
        return None
    padded = np.pad(opaque.astype(np.uint32), ((1, 0), (1, 0)))
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    y2 = np.arange(crop_h, height + 1)
    x2 = np.arange(crop_w, width + 1)
    y1 = y2 - crop_h
    x1 = x2 - crop_w
    window_sums = (
        integral[np.ix_(y2, x2)]
        - integral[np.ix_(y1, x2)]
        - integral[np.ix_(y2, x1)]
        + integral[np.ix_(y1, x1)]
    )
    valid = window_sums == (crop_w * crop_h)
    if required_bbox is not None:
        min_x = required_bbox[2] - crop_w
        max_x = required_bbox[0]
        min_y = required_bbox[3] - crop_h
        max_y = required_bbox[1]
        if min_x > max_x or min_y > max_y:
            return None
        x_min = max(0, int(np.ceil(min_x)))
        x_max = min(int(np.floor(max_x)), width - crop_w)
        y_min = max(0, int(np.ceil(min_y)))
        y_max = min(int(np.floor(max_y)), height - crop_h)
        if x_min > x_max or y_min > y_max:
            return None
        valid_x = (x1 >= x_min) & (x1 <= x_max)
        valid_y = (y1 >= y_min) & (y1 <= y_max)
        valid = valid & valid_y[:, None] & valid_x[None, :]
    if not valid.any():
        return None
    ys, xs = np.nonzero(valid)
    center_x = (width - crop_w) / 2.0
    center_y = (height - crop_h) / 2.0
    distance = (xs - center_x) ** 2 + (ys - center_y) ** 2
    chosen = int(np.argmin(distance))
    return int(xs[chosen]), int(ys[chosen])


def layout_rectangles(
    rng,
    wall_size: tuple[int, int],
    item_sizes: Sequence[tuple[int, int]],
    *,
    gap: int = 8,
    attempts: int = 80,
) -> list[tuple[int, int]]:
    """Place rectangles inside a wall with a gap, falling back to a grid."""
    wall_w, wall_h = wall_size
    placed: list[tuple[int, int, int, int]] = []
    origins: list[tuple[int, int]] = []
    for width, height in item_sizes:
        if width > wall_w or height > wall_h:
            raise ValueError(
                f"Artwork size {(width, height)} does not fit the wall {wall_size}"
            )
        found = None
        x_hi = wall_w - width
        y_hi = wall_h - height
        for _ in range(attempts):
            x = rng.randint(0, x_hi) if x_hi else 0
            y = rng.randint(0, y_hi) if y_hi else 0
            box = (x, y, x + width, y + height)
            if all(_separated(box, other, gap) for other in placed):
                found = (x, y)
                break
        if found is None:
            return _grid_origins(wall_size, item_sizes, gap)
        origins.append(found)
        placed.append((found[0], found[1], found[0] + width, found[1] + height))
    return origins


def _separated(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    gap: int,
) -> bool:
    return (
        left[2] + gap <= right[0]
        or right[2] + gap <= left[0]
        or left[3] + gap <= right[1]
        or right[3] + gap <= left[1]
    )


def _grid_origins(
    wall_size: tuple[int, int],
    item_sizes: Sequence[tuple[int, int]],
    gap: int,
) -> list[tuple[int, int]]:
    count = len(item_sizes)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    wall_w, wall_h = wall_size
    cell_w = wall_w / cols
    cell_h = wall_h / rows
    origins: list[tuple[int, int]] = []
    for index, (width, height) in enumerate(item_sizes):
        col = index % cols
        row = index // cols
        x = int(col * cell_w + max(0.0, (cell_w - width) / 2.0))
        y = int(row * cell_h + max(0.0, (cell_h - height) / 2.0))
        x = min(max(0, x), max(0, wall_w - width))
        y = min(max(0, y), max(0, wall_h - height))
        origins.append((x, y))
    if rows * cols < count:
        raise AssertionError("Grid layout under-counted cells")
    _ = gap
    return origins


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    if left.shape != right.shape:
        raise ValueError(f"Mask shapes differ: {left.shape} vs {right.shape}")
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_masks(
    predicted: Sequence[np.ndarray],
    truth: Sequence[np.ndarray],
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matches ordered by descending IoU."""
    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for true_index, true in enumerate(truth):
            pairs.append((mask_iou(pred, true), pred_index, true_index))
    pairs.sort(reverse=True)
    used_pred: set[int] = set()
    used_true: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pred_index, true_index in pairs:
        if score <= 0.0 or pred_index in used_pred or true_index in used_true:
            continue
        used_pred.add(pred_index)
        used_true.add(true_index)
        matches.append((pred_index, true_index, score))
    return matches


def score_masks(
    predicted: Sequence[np.ndarray],
    truth: Sequence[np.ndarray],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Score one scene. Unmatched ground-truth masks contribute IoU 0."""
    matches = match_masks(predicted, truth)
    matched_true = {true_index: score for _, true_index, score in matches}
    matched_pred = {pred_index: score for pred_index, _, score in matches}
    true_ious = [matched_true.get(index, 0.0) for index in range(len(truth))]
    mean_iou = float(np.mean(true_ious)) if true_ious else 0.0
    true_hits = sum(score >= iou_threshold for score in matched_true.values())
    pred_hits = sum(score >= iou_threshold for score in matched_pred.values())
    recall = true_hits / len(truth) if truth else 0.0
    precision = pred_hits / len(predicted) if predicted else 0.0
    return {
        "mean_iou": mean_iou,
        "recall_at_50": float(recall),
        "precision_at_50": float(precision),
        "count_error": float(abs(len(predicted) - len(truth))),
        "matched": float(len(matches)),
    }


def letterbox_image(
    image: Image.Image,
    size: int,
    fill: tuple[int, ...],
) -> tuple[Image.Image, dict[str, int | float]]:
    """Fit ``image`` inside a square canvas without changing its aspect ratio."""
    width, height = image.size
    scale = min(size / width, size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = image.resize((resized_w, resized_h), Image.BICUBIC)
    canvas = Image.new(image.mode, (size, size), fill)
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "size": size,
    }


def letterbox_mask(mask: np.ndarray, meta: dict[str, int | float]) -> np.ndarray:
    """Apply the same letterbox as ``letterbox_image`` with nearest sampling."""
    image = Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L")
    resized = image.resize(
        (int(meta["resized_w"]), int(meta["resized_h"])),
        Image.NEAREST,
    )
    canvas = Image.new("L", (int(meta["size"]), int(meta["size"])), 0)
    canvas.paste(resized, (int(meta["pad_x"]), int(meta["pad_y"])))
    return np.asarray(canvas) > 0


def mask_on_original(mask: np.ndarray, meta: dict[str, int | float], original_size: tuple[int, int]) -> np.ndarray:
    """Map a letterboxed square mask back onto the original image.

    ``original_size`` is ``(width, height)``. ``mask`` covers the square canvas
    described by ``meta``, including the padding around the resized photo.
    """
    width, height = original_size
    canvas = Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L")
    canvas = canvas.resize((int(meta["size"]), int(meta["size"])), Image.NEAREST)
    pad_x = int(meta["pad_x"])
    pad_y = int(meta["pad_y"])
    content = canvas.crop((pad_x, pad_y, pad_x + int(meta["resized_w"]), pad_y + int(meta["resized_h"])))
    restored = content.resize((width, height), Image.NEAREST)
    return np.asarray(restored) > 0
