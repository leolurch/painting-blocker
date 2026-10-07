"""Common image processing utilities for the compositing pipeline."""

import random
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .worker_utils import get_worker_np_rng, get_worker_rng

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image_rgba(path: Union[str, Path]) -> Image.Image:
    """Load an image and convert to RGBA mode."""
    return Image.open(path).convert("RGBA")


def load_image_rgb(path: Union[str, Path]) -> Image.Image:
    """Load an image and convert to RGB mode."""
    return Image.open(path).convert("RGB")


def resize_to_max_dimension(
    image: Image.Image,
    max_dimension: int,
) -> tuple[Image.Image, dict]:
    """Cap an image's longest side without changing its aspect ratio.

    The returned metadata is deliberately JSON-serializable.  Images below the
    cap are never upscaled and the original object is returned unchanged.
    """
    if max_dimension < 1:
        raise ValueError("max_dimension must be >= 1")
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_dimension:
        return image, {
            "input_size": [width, height],
            "output_size": [width, height],
            "max_dimension": max_dimension,
            "resized": False,
            "scale": 1.0,
        }

    scale = max_dimension / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, {
        "input_size": [width, height],
        "output_size": [new_width, new_height],
        "max_dimension": max_dimension,
        "resized": True,
        "scale": round(scale, 6),
    }


def resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Backward-compatible image-only wrapper around longest-side resizing."""
    return resize_to_max_dimension(image, max_side)[0]


def minimal_crop_to_fit(photo: Image.Image, inner_w: int, inner_h: int) -> Image.Image:
    """Minimal proportional cropping to ensure the photo fills
    the target dimensions precisely, with minimal image loss.

    Args:
        photo: Source image to crop/resize
        inner_w: Target width
        inner_h: Target height

    Returns:
        Cropped and resized image matching target dimensions
    """
    photo_w, photo_h = photo.size
    ratio_w = inner_w / photo_w
    ratio_h = inner_h / photo_h

    if ratio_w > ratio_h:
        new_w = int(photo_w * ratio_w)
        new_h = int(photo_h * ratio_w)
    else:
        new_w = int(photo_w * ratio_h)
        new_h = int(photo_h * ratio_h)

    photo_resized = photo.resize((new_w, new_h), Image.LANCZOS)

    crop_x = (new_w - inner_w) // 2
    crop_y = (new_h - inner_h) // 2

    return photo_resized.crop((crop_x, crop_y, crop_x + inner_w, crop_y + inner_h))


def flood_fill_interior_mask(
    alpha_channel: np.ndarray,
    seed_point: Optional[Tuple[int, int]] = None,
    alpha_threshold: int = 128,
) -> np.ndarray:
    """Find the interior region of a frame using flood-fill from a seed point.

    This handles round frames, irregular shapes, and frames where the transparent
    area doesn't extend to the image borders.

    Args:
        alpha_channel: 2D numpy array of alpha values (0-255)
        seed_point: (x, y) starting point for flood fill. If None, uses center.
        alpha_threshold: Pixels with alpha below this are considered "interior"

    Returns:
        Boolean mask where True indicates the interior region
    """
    height, width = alpha_channel.shape

    # Default seed point is the center
    if seed_point is None:
        seed_point = (width // 2, height // 2)

    seed_x, seed_y = seed_point

    # Check if seed point is valid
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        seed_x, seed_y = width // 2, height // 2

    # Check if seed point is in transparent area
    if alpha_channel[seed_y, seed_x] >= alpha_threshold:
        # Seed point is not transparent, search for a transparent point nearby
        found = False
        for radius in range(1, max(width, height) // 2):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # Only check perimeter
                    nx, ny = seed_x + dx, seed_y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if alpha_channel[ny, nx] < alpha_threshold:
                            seed_x, seed_y = nx, ny
                            found = True
                            break
                if found:
                    break
            if found:
                break

        if not found:
            return np.zeros((height, width), dtype=bool)

    # Flood-fill using BFS
    mask = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)

    queue = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True

    # 4-connected neighbors
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        x, y = queue.popleft()

        if alpha_channel[y, x] < alpha_threshold:
            mask[y, x] = True

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    return mask


def get_mask_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get the bounding box of a boolean mask.

    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)

    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def compose_image_with_frame(
    photo: Image.Image,
    frame: Image.Image,
    rotate_portrait: bool = True,
    alpha_threshold: int = 128,
    light_pos: Optional[Tuple[float, float, float]] = None,
    max_frame_dimension: Optional[int] = None,
) -> Image.Image:
    """Compose a photo with a frame overlay using flood-fill mask detection.

    This improved version:
    1. Finds the interior of the frame using flood-fill from the center
    2. Properly handles round frames and irregular shapes
    3. Clips the photo to the mask shape and applies proper alpha

    Args:
        photo: The source photo (will be converted to RGBA)
        frame: The frame image with transparent center (RGBA)
        rotate_portrait: If True, rotate portrait photos to landscape orientation
        alpha_threshold: Alpha value below which pixels are considered "interior"
        light_pos: Optional pre-sampled position shared by frame lighting effects
        max_frame_dimension: Optional cap for frame masking and shading work

    Returns:
        Composited image with photo inside frame, transparent outside
    """
    photo = photo.convert("RGBA")
    frame = frame.convert("RGBA")
    if max_frame_dimension is not None:
        if max_frame_dimension < 1:
            raise ValueError("max_frame_dimension must be >= 1")
        frame, _ = resize_to_max_dimension(frame, max_frame_dimension)

    # Rotate portrait photos to match horizontal frame orientation
    if rotate_portrait and photo.width < photo.height:
        photo = photo.rotate(-90, expand=True)

    # Get the alpha channel
    frame_alpha = np.array(frame.split()[-1])

    # Find interior mask using flood-fill from center
    interior_mask = flood_fill_interior_mask(
        frame_alpha, alpha_threshold=alpha_threshold
    )

    if not interior_mask.any():
        # No interior found, return frame as-is
        return frame

    # Get bounding box of the interior mask
    min_x, min_y, max_x, max_y = get_mask_bounding_box(interior_mask)

    inner_w = max_x - min_x + 1
    inner_h = max_y - min_y + 1

    if inner_w <= 0 or inner_h <= 0:
        return frame

    # Adjust frame aspect ratio to match photo based on interior mask
    target_aspect = photo.width / photo.height
    interior_aspect = inner_w / inner_h
    if abs(interior_aspect - target_aspect) > 1e-3:
        scale_x = target_aspect / interior_aspect
        # Pathological source aspect ratios can otherwise stretch a frame past
        # libjpeg's 65,500-pixel dimension ceiling and allocate several GiB.
        new_w = min(8_192, max(1, int(round(frame.width * scale_x))))
        new_h = frame.height
        if (
            max_frame_dimension is not None
            and max(new_w, new_h) > max_frame_dimension
        ):
            scale = max_frame_dimension / max(new_w, new_h)
            new_w = max(1, int(round(new_w * scale)))
            new_h = max(1, int(round(new_h * scale)))
        frame = frame.resize((new_w, new_h), Image.LANCZOS)

        frame_alpha = np.array(frame.split()[-1])
        interior_mask = flood_fill_interior_mask(
            frame_alpha, alpha_threshold=alpha_threshold
        )
        if not interior_mask.any():
            return frame

        min_x, min_y, max_x, max_y = get_mask_bounding_box(interior_mask)
        inner_w = max_x - min_x + 1
        inner_h = max_y - min_y + 1
        if inner_w <= 0 or inner_h <= 0:
            return frame

    # Apply lighting-based shading to the frame area only (not the painting interior)
    rng = get_worker_rng()
    if light_pos is None:
        light_pos = generate_frame_light_position(rng)
    frame = apply_frame_light_shading(
        frame, interior_mask, rng, alpha_threshold, light_pos=light_pos
    )

    # Crop photo to fit the frame's inner region bounding box
    photo_cropped = minimal_crop_to_fit(photo, inner_w, inner_h)

    # Extract the mask region for this bounding box
    mask_region = interior_mask[min_y : max_y + 1, min_x : max_x + 1]

    # Apply the mask to the photo's alpha channel
    # This clips the photo to the exact shape of the frame interior
    photo_array = np.array(photo_cropped)

    # Feather the inner edge slightly for smoother blending
    mask_img = Image.fromarray((mask_region * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    mask_alpha = np.array(mask_img, dtype=np.float32) / 255.0

    # Blend alpha using the softened mask
    photo_alpha = photo_array[:, :, 3].astype(np.float32) / 255.0
    photo_array[:, :, 3] = np.clip(photo_alpha * mask_alpha * 255.0, 0, 255).astype(
        np.uint8
    )

    photo_masked = Image.fromarray(photo_array, mode="RGBA")

    # Apply frame shadow onto the painting interior (aligned with light source)
    photo_masked = apply_frame_shadow_to_painting(
        photo_masked,
        frame_alpha,
        interior_mask,
        rng,
        alpha_threshold=alpha_threshold,
        light_pos=light_pos,
    )

    # Create transparent background canvas
    background = Image.new("RGBA", frame.size, (0, 0, 0, 0))

    # Use alpha_composite to properly handle transparency when placing the masked photo
    photo_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    photo_layer.paste(photo_masked, (min_x, min_y), photo_masked)  # Use mask parameter!

    # Composite: photo layer first, then frame on top
    result = Image.alpha_composite(background, photo_layer)
    result = Image.alpha_composite(result, frame)

    # CRITICAL: Apply the frame's alpha to make exterior corners transparent
    # The final alpha should be: opaque where frame is opaque OR where interior mask is true
    # But transparent where frame is transparent AND outside interior (exterior corners)
    result_array = np.array(result)

    # Exterior = transparent in frame AND not in interior mask
    # We want exterior to be fully transparent
    exterior_mask = (frame_alpha < alpha_threshold) & (~interior_mask)
    result_array[:, :, 3] = np.where(exterior_mask, 0, result_array[:, :, 3])

    return Image.fromarray(result_array, mode="RGBA")


def apply_frame_light_shading(
    frame: Image.Image,
    interior_mask: np.ndarray,
    rng: random.Random,
    alpha_threshold: int = 128,
    strength_range: Tuple[float, float] = (0.30, 0.60),
    xy_range: Tuple[float, float] = (-0.6, 1.6),
    z_range: Tuple[float, float] = (0.5, 1.4),
    light_pos: Optional[Tuple[float, float, float]] = None,
) -> Image.Image:
    """Apply a random light-source shading gradient to the frame only.

    Args:
        frame: Frame image (RGBA).
        interior_mask: Boolean mask of the painting interior (True = interior).
        rng: Random generator for light source placement.
        alpha_threshold: Alpha threshold to consider frame pixels as opaque.
        strength_range: Range for shading strength (0-1).
        xy_range: Light source x/y position range in normalized coords.
        z_range: Light source depth range in normalized coords (front of image).

    Returns:
        Frame image with randomized lighting-based shading applied.
    """
    frame = frame.convert("RGBA")
    arr = np.array(frame, dtype=np.float32)
    h, w = arr.shape[:2]

    if interior_mask.shape != (h, w):
        return frame

    frame_alpha = arr[:, :, 3]
    frame_mask = (frame_alpha >= alpha_threshold) & (~interior_mask)
    if not frame_mask.any():
        return frame

    # Random light source position in front of the image plane
    if light_pos is None:
        light_x = rng.uniform(xy_range[0], xy_range[1])
        light_y = rng.uniform(xy_range[0], xy_range[1])
        light_z = rng.uniform(z_range[0], z_range[1])
    else:
        light_x, light_y, light_z = light_pos

    # Normalized coordinate grid [0,1]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)

    dx = x_grid - light_x
    dy = y_grid - light_y
    dz = light_z

    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    dist = np.maximum(dist, 1e-6)

    # Lambertian-style falloff with gentle distance attenuation
    cos_term = dz / dist
    attenuation = 1.0 / (1.0 + dist * dist)
    brightness = np.clip(cos_term * attenuation, 0.0, 1.0)

    # Normalize brightness within frame pixels to expand contrast
    frame_vals = brightness[frame_mask]
    min_val = float(frame_vals.min())
    max_val = float(frame_vals.max())
    if max_val - min_val < 1e-6:
        return frame

    norm = (brightness - min_val) / (max_val - min_val)
    strength = rng.uniform(strength_range[0], strength_range[1])
    factor = 1.0 + (norm - 0.5) * 2.0 * strength
    factor = np.clip(factor, 0.3, 1.7)

    # Apply shading to RGB channels on frame pixels only
    for c in range(3):
        channel = arr[:, :, c]
        channel[frame_mask] = np.clip(channel[frame_mask] * factor[frame_mask], 0, 255)
        arr[:, :, c] = channel

    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def apply_frame_shadow_to_painting(
    painting: Image.Image,
    frame_alpha: np.ndarray,
    interior_mask: np.ndarray,
    rng: random.Random,
    alpha_threshold: int = 128,
    light_pos: Optional[Tuple[float, float, float]] = None,
    strength_range: Tuple[float, float] = (0.35, 0.60),
    blur_ratio: float = 0.004,
    offset_ratio: float = 0.020,
) -> Image.Image:
    """Apply the frame's shadow onto the painting interior, aligned to light."""
    painting = painting.convert("RGBA")
    arr = np.array(painting, dtype=np.float32)
    h, w = arr.shape[:2]

    if interior_mask.shape != (h, w) or frame_alpha.shape != (h, w):
        return painting

    # Frame mask (opaque frame pixels)
    frame_mask = frame_alpha >= alpha_threshold
    if not frame_mask.any():
        return painting

    # Determine shadow direction opposite to light source
    if light_pos is None:
        light_pos = generate_frame_light_position(rng)
    light_x, light_y, _ = light_pos

    dir_x = 0.5 - light_x
    dir_y = 0.5 - light_y
    norm = np.hypot(dir_x, dir_y)
    if norm < 1e-6:
        dir_x, dir_y = 0.0, 1.0
        norm = 1.0
    dir_x /= norm
    dir_y /= norm

    offset_mag = max(1, int(round(min(w, h) * offset_ratio)))
    offset_x = int(round(dir_x * offset_mag))
    offset_y = int(round(dir_y * offset_mag))

    # Shift frame mask to create the shadow mask
    shadow_mask = _shift_mask(frame_mask, offset_x, offset_y).astype(np.float32)

    # Blur the shadow for softness
    blur_radius = max(1.0, min(w, h) * blur_ratio)
    shadow_img = Image.fromarray((shadow_mask * 255).astype(np.uint8), mode="L")
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shadow = np.array(shadow_img, dtype=np.float32) / 255.0

    # Apply shadow only inside the painting interior
    shadow = shadow * interior_mask.astype(np.float32)
    if shadow.max() <= 0:
        return painting

    strength = rng.uniform(strength_range[0], strength_range[1])
    shade = 1.0 - shadow * strength
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * shade, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a boolean mask with zero padding."""
    h, w = mask.shape
    result = np.zeros_like(mask)

    if dx >= 0:
        src_x0, src_x1 = 0, w - dx
        dst_x0, dst_x1 = dx, w
    else:
        src_x0, src_x1 = -dx, w
        dst_x0, dst_x1 = 0, w + dx

    if dy >= 0:
        src_y0, src_y1 = 0, h - dy
        dst_y0, dst_y1 = dy, h
    else:
        src_y0, src_y1 = -dy, h
        dst_y0, dst_y1 = 0, h + dy

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return result

    result[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return result


def generate_frame_light_position(
    rng: random.Random,
    xy_range: Tuple[float, float] = (-0.6, 1.6),
    z_range: Tuple[float, float] = (0.5, 1.4),
) -> Tuple[float, float, float]:
    """Generate a random light position, biased to cover all sides."""
    mode = rng.choice(
        [
            "left",
            "right",
            "top",
            "bottom",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]
    )

    low, high = xy_range
    left_band = (low, 0.2)
    right_band = (0.8, high)
    mid_band = (0.0, 1.0)
    top_band = (low, 0.2)
    bottom_band = (0.8, high)

    if mode == "left":
        light_x = rng.uniform(*left_band)
        light_y = rng.uniform(*mid_band)
    elif mode == "right":
        light_x = rng.uniform(*right_band)
        light_y = rng.uniform(*mid_band)
    elif mode == "top":
        light_x = rng.uniform(*mid_band)
        light_y = rng.uniform(*top_band)
    elif mode == "bottom":
        light_x = rng.uniform(*mid_band)
        light_y = rng.uniform(*bottom_band)
    elif mode == "top_left":
        light_x = rng.uniform(*left_band)
        light_y = rng.uniform(*top_band)
    elif mode == "top_right":
        light_x = rng.uniform(*right_band)
        light_y = rng.uniform(*top_band)
    elif mode == "bottom_left":
        light_x = rng.uniform(*left_band)
        light_y = rng.uniform(*bottom_band)
    else:
        light_x = rng.uniform(*right_band)
        light_y = rng.uniform(*bottom_band)

    light_z = rng.uniform(z_range[0], z_range[1])
    return light_x, light_y, light_z


def apply_overlay_to_image(
    photo: Image.Image,
    overlay: Image.Image,
    opacity: float = 0.5,
) -> Image.Image:
    """Apply a semi-transparent overlay on top of an image.

    Args:
        photo: Base image (will be converted to RGBA)
        overlay: Overlay image with transparency (will be resized to match)
        opacity: Opacity multiplier for the overlay (0.0-1.0)

    Returns:
        Composited image with overlay applied
    """
    photo = photo.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Resize overlay to match photo dimensions
    overlay = overlay.resize(photo.size, Image.LANCZOS)

    # Adjust overlay alpha channel based on opacity
    overlay = overlay.copy()
    alpha_channel = overlay.split()[-1]
    opacity = max(0.0, min(1.0, opacity))
    overlay_alpha = alpha_channel.point(lambda a: int(a * opacity))
    overlay.putalpha(overlay_alpha)

    return Image.alpha_composite(photo, overlay)


def random_content_crop(
    image: Image.Image,
    rng: random.Random,
    max_area_frac: float = 0.15,
    *,
    retained_area_range: Optional[Tuple[float, float]] = None,
    return_metadata: bool = False,
):
    """Apply an off-centre crop and optionally return its exact geometry.

    ``retained_area_range`` is preferred by versioned recipes; the legacy
    ``max_area_frac`` argument remains available for old callers.
    """
    w, h = image.size
    metadata = {
        "applied": False,
        "box": [0, 0, w, h],
        "input_size": [w, h],
        "retained_area_fraction": 1.0,
    }
    if w < 2 or h < 2:
        return (image, metadata) if return_metadata else image
    if retained_area_range is None:
        if max_area_frac <= 0:
            return (image, metadata) if return_metadata else image
        low, high = max(0.0, 1.0 - max_area_frac), 1.0
    else:
        low, high = retained_area_range
        if not (0.0 < low <= high <= 1.0):
            raise ValueError("retained_area_range must satisfy 0 < low <= high <= 1")

    keep = rng.uniform(low, high)
    keep_w = rng.uniform(keep, 1.0)
    keep_h = min(1.0, max(keep, keep / keep_w))
    new_w = max(1, min(w, int(round(w * keep_w))))
    new_h = max(1, min(h, int(round(h * keep_h))))
    rem_w, rem_h = w - new_w, h - new_h
    # Independent side removal produces naturally off-centre records.
    left = rng.randint(0, rem_w)
    top = rng.randint(0, rem_h)
    box = (left, top, left + new_w, top + new_h)
    result = image.crop(box)
    metadata = {
        "applied": result.size != image.size,
        "box": list(box),
        "input_size": [w, h],
        "retained_area_fraction": round((new_w * new_h) / float(w * h), 6),
    }
    return (result, metadata) if return_metadata else result


def apply_sepia(
    image: Image.Image,
    strength: float,
    shadow_color: Tuple[int, int, int],
    highlight_color: Tuple[int, int, int],
) -> Image.Image:
    """Apply a parameterized sepia tint while preserving transparency."""
    strength = max(0.0, min(1.0, strength))
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[-1] if has_alpha else None
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    t = (luminance / 255.0)[..., None]
    shadows = np.asarray(shadow_color, dtype=np.float32)
    highlights = np.asarray(highlight_color, dtype=np.float32)
    sepia = shadows + t * (highlights - shadows)
    result_array = np.clip(rgb * (1.0 - strength) + sepia * strength, 0, 255)
    result = Image.fromarray(result_array.astype(np.uint8), mode="RGB")
    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    return result


def place_print_on_paper_document(
    image: Image.Image, rng: random.Random
) -> tuple[Image.Image, dict]:
    """Place a mildly reflective photographic print on a blank sheet of paper.

    Sheet orientation, dimensions, margins, rotations, paper tone/texture,
    reflection, and both cast shadows are randomized for each invocation.
    """
    orientation = rng.choice(("portrait", "landscape"))
    long_side = rng.randint(820, 1120)
    aspect = rng.uniform(1.28, 1.58)
    if orientation == "portrait":
        paper_h, paper_w = long_side, int(round(long_side / aspect))
    else:
        paper_w, paper_h = long_side, int(round(long_side / aspect))

    border_frac = rng.uniform(0.08, 0.16)
    canvas_w = int(round(paper_w * (1.0 + 2.0 * border_frac)))
    canvas_h = int(round(paper_h * (1.0 + 2.0 * border_frac)))
    backdrop_level = rng.randint(190, 225)
    backdrop = Image.new(
        "RGBA", (canvas_w, canvas_h),
        (backdrop_level, backdrop_level + rng.randint(-3, 3), backdrop_level + rng.randint(-4, 4), 255),
    )

    # Blank, slightly textured off-white document paper.
    paper_level = rng.randint(236, 252)
    paper_arr = np.empty((paper_h, paper_w, 4), dtype=np.uint8)
    paper_noise_std = rng.uniform(0.8, 2.4)
    paper_noise = get_worker_np_rng().normal(
        0.0, paper_noise_std, size=(paper_h, paper_w)
    ).astype(np.float32)
    for channel, shift in enumerate((0, rng.randint(-3, 1), rng.randint(-8, -2))):
        paper_arr[..., channel] = np.clip(paper_level + shift + paper_noise, 0, 255)
    paper_arr[..., 3] = 255
    paper = Image.fromarray(paper_arr, mode="RGBA")
    paper_angle = rng.uniform(-4.5, 4.5)
    paper = paper.rotate(paper_angle, Image.Resampling.BICUBIC, expand=True)

    paper_x = (canvas_w - paper.width) // 2 + rng.randint(-8, 8)
    paper_y = (canvas_h - paper.height) // 2 + rng.randint(-8, 8)
    paper_shadow_opacity = rng.uniform(0.18, 0.34)
    paper_shadow_alpha = paper.getchannel("A").filter(
        ImageFilter.GaussianBlur(rng.uniform(8.0, 20.0))
    ).point(lambda value: int(value * paper_shadow_opacity))
    paper_shadow = Image.new("RGBA", paper.size, (20, 18, 15, 0))
    paper_shadow.putalpha(paper_shadow_alpha)
    shadow_dx, shadow_dy = rng.randint(5, 18), rng.randint(7, 22)
    backdrop.alpha_composite(paper_shadow, (paper_x + shadow_dx, paper_y + shadow_dy))
    backdrop.alpha_composite(paper, (paper_x, paper_y))

    # Fit the photographic print inside the sheet with independently randomized margins.
    max_print_w = int(paper_w * rng.uniform(0.58, 0.78))
    max_print_h = int(paper_h * rng.uniform(0.50, 0.74))
    photo = image.convert("RGBA")
    scale = min(max_print_w / photo.width, max_print_h / photo.height)
    photo = photo.resize(
        (max(1, int(round(photo.width * scale))), max(1, int(round(photo.height * scale)))),
        Image.Resampling.LANCZOS,
    )

    # A broad, low-opacity highlight gives the print a subtle glossy surface.
    reflection_opacity = rng.randint(18, 52)
    reflection = Image.new("RGBA", photo.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(reflection)
    band_width = max(8, int(photo.width * rng.uniform(0.10, 0.24)))
    start_x = rng.randint(-photo.width // 3, photo.width)
    slope = rng.uniform(-0.35, 0.35)
    offset = int(photo.height * slope)
    draw.polygon(
        [(start_x, 0), (start_x + band_width, 0),
         (start_x + band_width + offset, photo.height), (start_x + offset, photo.height)],
        fill=(255, 255, 255, reflection_opacity),
    )
    reflection = reflection.filter(ImageFilter.GaussianBlur(rng.uniform(6.0, 18.0)))
    photo = Image.alpha_composite(photo, reflection)
    photo_angle = rng.uniform(-3.5, 3.5)
    photo = photo.rotate(photo_angle, Image.Resampling.BICUBIC, expand=True)

    # Position relative to the unrotated sheet but keep generous blank margins.
    center_x = paper_x + paper.width // 2 + rng.randint(-paper_w // 12, paper_w // 12)
    center_y = paper_y + paper.height // 2 + rng.randint(-paper_h // 12, paper_h // 12)
    photo_x, photo_y = center_x - photo.width // 2, center_y - photo.height // 2
    print_shadow_opacity = rng.uniform(0.22, 0.42)
    print_shadow_alpha = photo.getchannel("A").filter(
        ImageFilter.GaussianBlur(rng.uniform(3.0, 9.0))
    ).point(lambda value: int(value * print_shadow_opacity))
    print_shadow = Image.new("RGBA", photo.size, (15, 13, 10, 0))
    print_shadow.putalpha(print_shadow_alpha)
    print_dx, print_dy = rng.randint(3, 10), rng.randint(4, 12)
    backdrop.alpha_composite(print_shadow, (photo_x + print_dx, photo_y + print_dy))
    backdrop.alpha_composite(photo, (photo_x, photo_y))

    visible_w = max(0, min(canvas_w, photo_x + photo.width) - max(0, photo_x))
    visible_h = max(0, min(canvas_h, photo_y + photo.height) - max(0, photo_y))
    visible_fraction = (visible_w * visible_h) / float(max(1, photo.width * photo.height))
    return backdrop, {
        "orientation": orientation,
        "canvas_size": [canvas_w, canvas_h],
        "paper_size": [paper_w, paper_h],
        "paper_position": [paper_x, paper_y],
        "print_box": [photo_x, photo_y, photo_x + photo.width, photo_y + photo.height],
        "visible_fraction": round(visible_fraction, 6),
        "paper_noise_std": round(paper_noise_std, 6),
        "paper_angle_deg": round(paper_angle, 3),
        "print_angle_deg": round(photo_angle, 3),
        "reflection_opacity": reflection_opacity,
        "paper_shadow_offset": [shadow_dx, shadow_dy],
        "print_shadow_offset": [print_dx, print_dy],
    }


def apply_white_balance(
    image: Image.Image,
    np_rng: np.random.Generator,
    max_shift: float = 0.12,
    *,
    return_metadata: bool = False,
):
    """Apply a random warm/cool colour-temperature cast via per-channel gains."""
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[-1] if has_alpha else None

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    gains = 1.0 + np_rng.uniform(-max_shift, max_shift, size=3).astype(np.float32)
    arr *= gains.reshape(1, 1, 3)
    arr = np.clip(arr, 0, 255)
    result = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    metadata = {"applied": True, "channel_gains": [round(float(value), 6) for value in gains]}
    return (result, metadata) if return_metadata else result


def apply_vignette(
    image: Image.Image,
    np_rng: np.random.Generator,
    strength_range: Tuple[float, float] = (0.2, 0.5),
    *,
    return_metadata: bool = False,
):
    """Darken the frame towards the corners with a radial falloff mask."""
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[-1] if has_alpha else None

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    cx = 0.5 + float(np_rng.uniform(-0.1, 0.1))
    cy = 0.5 + float(np_rng.uniform(-0.1, 0.1))
    ys, xs = np.ogrid[:h, :w]
    nx = xs / max(1, w - 1) - cx
    ny = ys / max(1, h - 1) - cy
    dist = np.sqrt(nx * nx + ny * ny)
    max_dist = float(dist.max()) or 1.0
    dist /= max_dist
    strength = float(np_rng.uniform(strength_range[0], strength_range[1]))
    mask = np.clip(1.0 - strength * (dist**2), 0.0, 1.0).astype(np.float32)
    arr *= mask[..., None]
    arr = np.clip(arr, 0, 255)
    result = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    metadata = {
        "applied": True,
        "strength": round(strength, 6),
        "center": [round(cx, 6), round(cy, 6)],
    }
    return (result, metadata) if return_metadata else result


def apply_defocus(
    image: Image.Image,
    rng: random.Random,
    radius_range: Tuple[float, float] = (0.4, 1.2),
    *,
    return_metadata: bool = False,
):
    """Apply a mild Gaussian blur to mimic handheld/out-of-focus capture."""
    radius = rng.uniform(radius_range[0], radius_range[1])
    if radius <= 0:
        metadata = {"applied": False, "radius": round(radius, 6)}
        return (image, metadata) if return_metadata else image
    result = image.filter(ImageFilter.GaussianBlur(radius=radius))
    metadata = {"applied": True, "radius": round(radius, 6)}
    return (result, metadata) if return_metadata else result


def apply_downscale_upscale(
    image: Image.Image,
    rng: random.Random,
    factor_range: Tuple[float, float] = (0.4, 0.8),
    *,
    longest_side_range: Optional[Tuple[int, int]] = None,
    return_metadata: bool = False,
):
    """Destroy resolution using either a legacy factor or an absolute target."""
    w, h = image.size
    if longest_side_range is not None:
        low, high = longest_side_range
        if low < 1 or high < low:
            raise ValueError("longest_side_range must satisfy 1 <= low <= high")
        current_longest = max(w, h)
        sampled_target = rng.randint(low, high)
        target = min(max(1, current_longest - 1), sampled_target)
        factor = target / float(current_longest)
    else:
        factor = rng.uniform(factor_range[0], factor_range[1])
        target = max(1, int(round(max(w, h) * factor)))
    sw = max(1, int(round(w * factor)))
    sh = max(1, int(round(h * factor)))
    small = image.resize((sw, sh), Image.Resampling.BILINEAR)
    result = small.resize((w, h), Image.Resampling.BILINEAR)
    metadata = {
        "applied": (sw, sh) != (w, h),
        "downsampled_size": [sw, sh],
        "restored_size": [w, h],
        "longest_side_pixels": max(sw, sh),
        "interpolation": "bilinear",
    }
    return (result, metadata) if return_metadata else result


def apply_in_plane_rotation(
    image: Image.Image,
    rng: random.Random,
    max_deg: float = 10.0,
) -> Image.Image:
    """Rotate slightly in-plane (camera not level), filling corners with the
    mean border colour so no black/transparent wedges appear."""
    angle = rng.uniform(-max_deg, max_deg)
    if abs(angle) < 1e-3:
        return image

    has_alpha = image.mode == "RGBA"
    alpha = image.split()[-1] if has_alpha else None

    base = image.convert("RGB")
    arr = np.asarray(base)
    border = np.concatenate(
        [
            arr[0].reshape(-1, 3),
            arr[-1].reshape(-1, 3),
            arr[:, 0].reshape(-1, 3),
            arr[:, -1].reshape(-1, 3),
        ]
    )
    fill = tuple(int(c) for c in border.mean(axis=0))
    rotated = base.rotate(
        angle, resample=Image.BICUBIC, expand=False, fillcolor=fill
    )

    if has_alpha:
        rotated = rotated.convert("RGBA")
        rot_alpha = alpha.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
        rotated.putalpha(rot_alpha)
    return rotated


def apply_chromatic_aberration(
    image: Image.Image,
    rng: random.Random,
    max_shift_px: int = 2,
    *,
    return_metadata: bool = False,
):
    """Shift the red and blue channels a few pixels apart to fringe edges."""
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[-1] if has_alpha else None

    arr = np.array(image.convert("RGB"))
    shift = rng.randint(1, max(1, max_shift_px))
    r = np.roll(arr[:, :, 0], shift, axis=1)
    b = np.roll(arr[:, :, 2], -shift, axis=1)
    out = np.stack([r, arr[:, :, 1], b], axis=2)
    result = Image.fromarray(out.astype(np.uint8), mode="RGB")

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    metadata = {"applied": True, "red_shift_px": shift, "blue_shift_px": -shift}
    return (result, metadata) if return_metadata else result


def get_image_files(directory: Union[str, Path]) -> list[Path]:
    """Get all image files from a directory.

    Args:
        directory: Path to directory to scan

    Returns:
        List of Path objects for valid image files
    """
    directory = Path(directory)
    return [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
