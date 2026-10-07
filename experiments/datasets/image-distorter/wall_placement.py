"""Core wall-placement and augmentation helpers for DB-backed dataset builds."""

import hashlib
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from utils import (
    IMAGE_EXTENSIONS,
    apply_archival_annotations,
    apply_archival_occlusion,
    apply_chromatic_aberration,
    apply_defocus,
    apply_downscale_upscale,
    apply_in_plane_rotation,
    apply_overlay_to_image,
    apply_sepia,
    apply_vignette,
    apply_white_balance,
    compose_image_with_frame,
    generate_frame_light_position,
    get_worker_np_rng,
    get_worker_rng,
    load_image_rgba,
    place_print_on_paper_document,
    random_content_crop,
    resize_to_max_dimension,
    resize_to_max_side,
)
from utils.wall_generator import WallConfig, WallGenerator

_WORKER_WALL_GENERATOR: WallGenerator | None = None


@dataclass
class PostProcessConfig:
    """Configuration for post-processing effects."""

    # Noise: standard deviation of Gaussian noise (0 = disabled)
    noise_std: float = 0.0

    # Saturation: range for random saturation multiplier (min, max)
    # 1.0 = no change, <1 = desaturate, >1 = oversaturate
    saturation_range: Optional[Tuple[float, float]] = None

    # Reflection: opacity for reflection overlay on painting (0 = disabled)
    reflection_opacity: float = 0.0

    # Glare: procedural glare effect on painting (simulates glass/light reflection)
    glare_intensity: float = 0.0  # Max opacity of glare spots (0 = disabled)
    glare_num_spots: Tuple[int, int] = (1, 3)  # Range for number of glare spots
    glare_size_range: Tuple[float, float] = (0.1, 0.4)  # Spot size as fraction of image

    # Vintage mode: simulate old 1910s-1940s grayscale photo scans
    vintage_mode: bool = False
    vintage_blur_radius: float = 0.8  # Soft blur radius
    vintage_noise_std: float = 0.04  # Extra noise for vintage look

    # --- Probabilistic realism effects (used by the archetype pipeline). ---
    # All default to 0.0/None so legacy callers get identical behaviour.
    crop_max_area: float = 0.0  # Max fraction of area removed by content crop
    flip_prob: float = 0.0  # Chance of a horizontal flip
    overlay_prob: float = 1.0  # Chance of a distortion/reflection overlay
    vintage_prob: float = 0.0  # Chance a variant gets the vintage look
    # Randomized desaturation range for vintage (None => full grayscale)
    vintage_saturation_range: Optional[Tuple[float, float]] = None
    # Sepia is independent of vintage, allowing both clean and aged tinted scans.
    sepia_prob: float = 0.0
    sepia_strength_range: Tuple[float, float] = (0.55, 0.95)
    # Chance to place the reproduction as a glossy print on a blank document.
    paper_document_prob: float = 0.0
    white_balance_prob: float = 0.0
    white_balance_max_shift: float = 0.12
    defocus_prob: float = 0.0
    defocus_radius_range: Tuple[float, float] = (0.4, 1.2)
    vignette_prob: float = 0.0
    vignette_strength_range: Tuple[float, float] = (0.2, 0.5)
    downscale_prob: float = 0.0
    downscale_factor_range: Tuple[float, float] = (0.4, 0.8)
    rotation_prob: float = 0.0
    rotation_max_deg: float = 10.0
    chroma_prob: float = 0.0
    chroma_max_shift_px: int = 2
    # JPEG save (consumed by the archetype worker, not by process_single_image)
    jpeg_prob: float = 0.80
    jpeg_quality_range: Tuple[int, int] = (65, 100)
    # Longest-side cap applied to the source before augmentation (0 => no cap)
    max_source_side: int = 0
    # Versioned hard-profile controls. ``None`` preserves legacy behaviour.
    crop_retained_area_range: Optional[Tuple[float, float]] = None
    downsample_longest_side_range: Optional[Tuple[int, int]] = None
    occlusion_prob: float = 0.0
    occlusion_fraction_range: Tuple[float, float] = (0.04, 0.15)
    annotation_prob: float = 0.0
    annotation_count_range: Tuple[int, int] = (1, 3)


@dataclass(frozen=True)
class NamedAsset:
    """An immutable loaded asset with content-addressed identity."""

    name: str
    sha256: str
    image: Image.Image


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def generate_glare_spot(
    size: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
    intensity: float,
) -> np.ndarray:
    """Generate a single radial gradient glare spot.

    Args:
        size: (width, height) of the output array
        center: (x, y) center position as fractions (0-1)
        radius: Radius of the glare spot as fraction of image diagonal
        intensity: Peak intensity at center (0-1)

    Returns:
        2D numpy array with glare values (0-1)
    """
    w, h = size
    cx, cy = center[0] * w, center[1] * h
    diagonal = np.sqrt(w**2 + h**2)
    r = radius * diagonal

    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Radial gradient with soft falloff
    glare = np.clip(1 - (dist / r), 0, 1)
    # Apply smooth falloff (quadratic)
    glare = glare**2 * intensity

    return glare


def generate_glare_pattern(
    size: Tuple[int, int],
    num_spots: int,
    intensity: float,
    size_range: Tuple[float, float],
    rng: random.Random,
) -> Image.Image:
    """Generate a procedural glare pattern with multiple spots.

    Args:
        size: (width, height) of the output image
        num_spots: Number of glare spots to generate
        intensity: Maximum intensity of glare (0-1)
        size_range: (min, max) range for spot size as fraction of image
        rng: Random number generator

    Returns:
        RGBA image with white glare on transparent background
    """
    w, h = size
    glare_array = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_spots):
        # Random position (biased toward upper portions for realistic light)
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.05, 0.7)  # Glare often comes from above

        # Random size and intensity
        spot_size = rng.uniform(size_range[0], size_range[1])
        spot_intensity = intensity * rng.uniform(0.5, 1.0)

        # Generate and add spot
        spot = generate_glare_spot(size, (cx, cy), spot_size, spot_intensity)
        glare_array = np.maximum(glare_array, spot)  # Use max for overlapping

    # Convert to RGBA image (white color with glare as alpha)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255  # R
    rgba[:, :, 1] = 255  # G
    rgba[:, :, 2] = 255  # B
    rgba[:, :, 3] = (glare_array * 255).astype(np.uint8)  # Alpha

    return Image.fromarray(rgba, mode="RGBA")


def apply_glare_to_painting(
    painting: Image.Image,
    intensity: float,
    num_spots_range: Tuple[int, int],
    size_range: Tuple[float, float],
    rng: random.Random,
) -> Image.Image:
    """Apply procedural glare effect to a painting.

    This simulates light reflection/glare on glass-covered artwork.
    The glare is applied BEFORE framing, so it naturally clips to the painting.

    Args:
        painting: The painting image (RGBA)
        intensity: Maximum glare intensity (0-1)
        num_spots_range: (min, max) number of glare spots
        size_range: (min, max) spot size as fraction of image
        rng: Random number generator

    Returns:
        Painting with glare applied
    """
    if intensity <= 0:
        return painting

    painting = painting.convert("RGBA")

    # Randomize number of spots
    num_spots = rng.randint(num_spots_range[0], num_spots_range[1])

    # Generate glare pattern
    glare = generate_glare_pattern(painting.size, num_spots, intensity, size_range, rng)

    # Apply glare using alpha composite (additive-like blending)
    result = Image.alpha_composite(painting, glare)

    return result


def apply_vintage_effect(
    image: Image.Image,
    blur_radius: float,
    noise_std: float,
    rng: random.Random,
    saturation_range: Optional[Tuple[float, float]] = None,
    *,
    return_metadata: bool = False,
):
    """Apply vintage 1910s-1940s photo scan effect.

    This simulates old photo scans with soft blur, film grain, lowered
    contrast, and either full grayscale or a randomized desaturation.

    Args:
        image: Input image (RGB or RGBA)
        blur_radius: Gaussian blur radius for soft focus effect
        noise_std: Standard deviation of grain noise (0-1)
        rng: Random number generator
        saturation_range: If None, convert to full grayscale (legacy behaviour).
            Otherwise sample a saturation multiplier from this (min, max) range
            so the result keeps muted colour rather than pure black-and-white.

    Returns:
        Image with vintage effect applied
    """
    # Preserve alpha if present
    has_alpha = image.mode == "RGBA"
    if has_alpha:
        alpha = image.split()[-1]

    saturation = 0.0
    contrast = rng.uniform(0.85, 1.0)
    brightness = rng.uniform(0.95, 1.05)
    if saturation_range is None:
        work = image.convert("L")
        if blur_radius > 0:
            work = work.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        work = ImageEnhance.Contrast(work).enhance(contrast)
        work = ImageEnhance.Brightness(work).enhance(brightness)
        result = work.convert("RGB")
    else:
        saturation = rng.uniform(saturation_range[0], saturation_range[1])
        work = ImageEnhance.Color(image.convert("RGB")).enhance(saturation)
        if blur_radius > 0:
            work = work.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        work = ImageEnhance.Contrast(work).enhance(contrast)
        work = ImageEnhance.Brightness(work).enhance(brightness)
        result = work

    # Add film grain noise
    actual_std = 0.0
    if noise_std > 0:
        arr = np.array(result, dtype=np.float32)
        actual_std = noise_std * rng.uniform(0.7, 1.3)
        noise = get_worker_np_rng().normal(0, actual_std * 255, arr.shape[:2])
        # Apply same noise to all channels (grayscale noise)
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] + noise, 0, 255)
        result = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    # Restore alpha if present
    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)

    metadata = {
        "applied": True,
        "blur_radius": blur_radius,
        "grain_std": round(actual_std, 6),
        "saturation": round(saturation, 6),
        "contrast": round(contrast, 6),
        "brightness": round(brightness, 6),
    }
    return (result, metadata) if return_metadata else result


def apply_post_noise(image: Image.Image, std: float, rng: random.Random) -> Image.Image:
    """Add Gaussian noise to the image.

    Args:
        image: Input image (RGB or RGBA)
        std: Standard deviation of noise (0-1 range, e.g., 0.02 = 2%)
        rng: Random number generator

    Returns:
        Image with noise added
    """
    if std <= 0:
        return image

    arr = np.array(image, dtype=np.float32)

    # Generate noise
    noise = get_worker_np_rng().normal(0, std * 255, arr.shape[:2] + (3,))

    # Add noise to RGB channels only
    if arr.shape[2] == 4:
        arr[:, :, :3] = np.clip(arr[:, :, :3] + noise, 0, 255)
    else:
        arr = np.clip(arr + noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), mode=image.mode)


def apply_post_saturation(
    image: Image.Image,
    sat_range: Tuple[float, float],
    rng: random.Random,
) -> Image.Image:
    """Randomly adjust saturation within a range.

    Args:
        image: Input image
        sat_range: (min, max) saturation multiplier range
        rng: Random number generator

    Returns:
        Image with adjusted saturation
    """
    if sat_range is None:
        return image

    factor = rng.uniform(sat_range[0], sat_range[1])

    # Convert to RGB if needed for saturation adjustment
    if image.mode == "RGBA":
        rgb = image.convert("RGB")
        alpha = image.split()[-1]
        enhanced = ImageEnhance.Color(rgb).enhance(factor)
        enhanced = enhanced.convert("RGBA")
        enhanced.putalpha(alpha)
        return enhanced
    else:
        return ImageEnhance.Color(image).enhance(factor)


def _rotate_known_angle(image: Image.Image, angle: float) -> Image.Image:
    """Rotate with a sampled angle while avoiding synthetic black corners."""
    if abs(angle) < 1e-6:
        return image
    base = image.convert("RGB")
    arr = np.asarray(base)
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]), axis=0)
    fill = tuple(int(value) for value in border.mean(axis=0))
    return base.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )


def flatten_on_background(
    image: Image.Image,
    background: Tuple[int, int, int, int] = (245, 245, 245, 255),
) -> Image.Image:
    """Composite an RGBA image onto a solid background for no-wall variants."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    base = Image.new("RGBA", image.size, background)
    return Image.alpha_composite(base, image)


def apply_reflection_to_painting(
    framed_image: Image.Image,
    overlays: List[NamedAsset],
    opacity: float,
    rng: random.Random,
) -> Image.Image:
    """Apply a reflection overlay to the framed painting.

    The reflection is applied ONLY to the non-transparent areas of the framed image,
    simulating glass reflection on the painting surface.

    Args:
        framed_image: The framed painting (RGBA with transparency for frame exterior)
        overlays: List of reflection overlay images
        opacity: Maximum opacity for the reflection (actual will be randomized)
        rng: Random number generator

    Returns:
        Framed image with reflection applied to painting area
    """
    if not overlays or opacity <= 0:
        return framed_image

    framed_image = framed_image.convert("RGBA")

    # Select random overlay
    overlay = rng.choice(overlays).image.copy().convert("RGBA")

    # Resize overlay to match framed image
    overlay = overlay.resize(framed_image.size, Image.LANCZOS)

    # Get the alpha channel of the framed image (used to restore transparency later)
    framed_alpha = np.array(framed_image.split()[-1])

    # Random opacity within range
    actual_opacity = rng.uniform(opacity * 0.5, opacity)

    # Apply opacity to overlay
    overlay_alpha = overlay.split()[-1]
    new_alpha = overlay_alpha.point(lambda a: int(a * actual_opacity))
    overlay.putalpha(new_alpha)

    # Composite the overlay
    result = Image.alpha_composite(framed_image, overlay)

    # Restore original alpha (keep frame transparency)
    result_array = np.array(result)
    result_array[:, :, 3] = framed_alpha

    return Image.fromarray(result_array, mode="RGBA")


# Difficulty presets - images take 80-95% of wall area
DIFFICULTY_PRESETS = {
    "easy": {
        "min_angle": 3.0,
        "max_angle": 15.0,
        "min_image_scale": 0.85,
        "max_image_scale": 0.95,
        "position_jitter": 0.05,
        "shadow_opacity": 0.10,
    },
    "medium": {
        "min_angle": 8.0,
        "max_angle": 25.0,
        "min_image_scale": 0.80,
        "max_image_scale": 0.92,
        "position_jitter": 0.08,
        "shadow_opacity": 0.15,
    },
    "hard": {
        "min_angle": 15.0,
        "max_angle": 35.0,
        "min_image_scale": 0.75,
        "max_image_scale": 0.88,
        "position_jitter": 0.12,
        "shadow_opacity": 0.20,
    },
}


def get_worker_wall_generator(
    config: WallConfig, textures: List[Tuple[str, Optional[Image.Image]]] = None
) -> WallGenerator:
    """Get or create the process-local WallGenerator."""
    global _WORKER_WALL_GENERATOR
    rng = get_worker_rng()
    if _WORKER_WALL_GENERATOR is None or _WORKER_WALL_GENERATOR.config != config:
        _WORKER_WALL_GENERATOR = WallGenerator(
            config=config, rng=rng, texture_images=textures or []
        )
    else:
        # A process can handle multiple independently seeded source tasks.
        _WORKER_WALL_GENERATOR.rng = rng
    return _WORKER_WALL_GENERATOR


def load_assets(
    frames_dir: Optional[Path],
    overlays_dir: Optional[Path],
    textures_dir: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> Tuple[
    List[NamedAsset],
    List[NamedAsset],
    List[Tuple[str, Optional[Image.Image]]],
]:
    """Pre-load frame, overlay, and texture assets for pipeline mode."""
    frames = []
    overlays = []
    textures = []

    if frames_dir and frames_dir.exists():
        frame_files = sorted(
            [f for f in frames_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if verbose:
            print(f"Loading {len(frame_files)} frames...")
        for f in frame_files:
            image = load_image_rgba(f)
            image.info["asset_sha256"] = _sha256_file(f)
            frames.append(NamedAsset(f.stem, image.info["asset_sha256"], image))

    if overlays_dir and overlays_dir.exists():
        overlay_files = sorted(
            [f for f in overlays_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if verbose:
            print(f"Loading {len(overlay_files)} overlays...")
        for f in overlay_files:
            image = load_image_rgba(f)
            image.info["asset_sha256"] = _sha256_file(f)
            overlays.append(NamedAsset(f.stem, image.info["asset_sha256"], image))

    if textures_dir and textures_dir.exists():
        texture_files = sorted(
            [f for f in textures_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if verbose:
            print(f"Loading {len(texture_files)} wall textures...")
        for f in texture_files:
            image = load_image_rgba(f)
            image.info["asset_sha256"] = _sha256_file(f)
            textures.append((f.stem, image))
        textures.append(("rand-001", None))
        textures.append(("rand-002", None))

    return frames, overlays, textures


def process_single_image(
    image: Image.Image,
    wall_config: WallConfig,
    frames: List[NamedAsset],
    overlays: List[NamedAsset],
    textures: List[Tuple[str, Optional[Image.Image]]],
    apply_pipeline: bool,
    post_config: PostProcessConfig,
    apply_wall_composition: bool = True,
    apply_strong_perspective: bool = True,
) -> Tuple[Image.Image, str, str]:
    """Process a single image through the augmentation pipeline.

    Args:
        image: Source image
        wall_config: Wall generation configuration
        frames: List of pre-loaded frame images (for pipeline mode)
        overlays: List of pre-loaded overlay images (for pipeline mode)
        textures: List of pre-loaded wall texture images
        apply_pipeline: If True, apply overlay→frame/reflection effects.
            If False, process the source image directly.
        post_config: Configuration for post-processing effects
        apply_wall_composition: If True, place the processed image on a wall.
            If False, return the processed image on a plain opaque background
            without wall textures or perspective.
        apply_strong_perspective: If True, use the configured perspective/depth
            wall warp. If False, still place on a wall but with flat/no strong
            perspective shift.

    Returns:
        Tuple of (final image, frame name, wall texture name)
    """
    rng = get_worker_rng()
    frame_name = "no_frame"
    target_size = (960, 720)
    width_scale = 1.0 - rng.uniform(0.0, 0.20)
    height_scale = 1.0 - rng.uniform(0.0, 0.20)
    target_w = max(1, int(round(target_size[0] * width_scale)))
    target_h = max(1, int(round(target_size[1] * height_scale)))
    wall_config = replace(wall_config, width=target_w, height=target_h)

    processed = image.convert("RGBA")
    light_pos = generate_frame_light_position(rng)

    if apply_pipeline:
        # Step 1: Apply random overlay to the painting (before framing)
        if overlays:
            overlay_asset = rng.choice(overlays)
            opacity = rng.uniform(0.3, 0.8)
            processed = apply_overlay_to_image(processed, overlay_asset.image, opacity)

        # Step 2: Apply glare to the painting (simulates glass/light reflection)
        if post_config.glare_intensity > 0:
            processed = apply_glare_to_painting(
                processed,
                post_config.glare_intensity,
                post_config.glare_num_spots,
                post_config.glare_size_range,
                rng,
            )

        # Step 3: Compose with random frame
        if frames:
            is_portrait = processed.height > processed.width
            frame_asset = rng.choice(frames)
            frame_name, frame = frame_asset.name, frame_asset.image
            if is_portrait:
                frame = frame.rotate(90, expand=True)
            processed = compose_image_with_frame(
                processed,
                frame,
                rotate_portrait=not is_portrait,
                light_pos=light_pos,
            )

        # Step 4: Apply reflection to the framed painting (simulates glass)
        if post_config.reflection_opacity > 0 and overlays:
            processed = apply_reflection_to_painting(
                processed, overlays, post_config.reflection_opacity, rng
            )

    if apply_wall_composition:
        # Step 4: Place on wall. For flat-wall variants, keep wall textures,
        # placement, lighting, and shadows, but disable the strong perspective
        # and depth warp controlled by the difficulty preset.
        effective_wall_config = wall_config
        if not apply_strong_perspective:
            effective_wall_config = replace(
                wall_config,
                min_angle=0.0,
                max_angle=0.0,
                add_depth_shading=False,
                depth_shading_strength=0.0,
            )
        wall_gen = get_worker_wall_generator(effective_wall_config, textures)
        result = wall_gen.place_on_wall(processed, light_pos=light_pos)
        wall_name = wall_gen.last_texture_name or "no_texture"
    else:
        result = flatten_on_background(processed)
        wall_name = "no_wall"

    # === POST-PROCESSING (applied to final image) ===

    # Step 5: Apply vintage effect (grayscale + blur + grain)
    if post_config.vintage_mode:
        result = apply_vintage_effect(
            result,
            post_config.vintage_blur_radius,
            post_config.vintage_noise_std,
            rng,
        )

    # Step 6: Apply saturation adjustment (skipped if vintage mode is on)
    if post_config.saturation_range and not post_config.vintage_mode:
        result = apply_post_saturation(result, post_config.saturation_range, rng)

    # Step 7: Apply additional noise
    if post_config.noise_std > 0:
        result = apply_post_noise(result, post_config.noise_std, rng)

    return result, frame_name, wall_name


def process_single_image_archetype(
    image: Image.Image,
    wall_config: WallConfig,
    frames: List[NamedAsset],
    overlays: List[NamedAsset],
    textures: List[Tuple[str, Optional[Image.Image]]],
    post_config: PostProcessConfig,
    *,
    apply_frame: bool,
    apply_perspective: bool,
    apply_wall: bool,
    source_max_dimension: int | None = None,
) -> Tuple[Image.Image, str, str, dict]:
    """Run one source image through the refined archetype augmentation pipeline.

    Composition is controlled by the three toggles (frame / perspective / wall);
    the remaining effects are global per-variant probabilistic degradations
    configured on ``post_config``. All randomness comes from the seeded
    worker RNGs, so output is deterministic for a fixed seed.

    Returns:
        Tuple of (final RGBA/RGB image, frame name, wall name, applied-effects dict).
    """
    rng = get_worker_rng()
    np_rng = get_worker_np_rng()
    frame_name = "no_frame"
    wall_name = "no_wall"
    applied: dict = {
        "apply_frame": apply_frame,
        "apply_perspective": apply_perspective,
        "apply_wall": apply_wall,
    }

    processed = image.convert("RGBA")
    source_caps = [value for value in (post_config.max_source_side, source_max_dimension) if value and value > 0]
    if source_caps:
        processed, source_resize = resize_to_max_dimension(processed, min(source_caps))
        applied["source_resize"] = source_resize

    # 1. Content crop with exact retained-area provenance.
    if post_config.crop_max_area > 0 or post_config.crop_retained_area_range is not None:
        processed, crop_metadata = random_content_crop(
            processed,
            rng,
            post_config.crop_max_area,
            retained_area_range=post_config.crop_retained_area_range,
            return_metadata=True,
        )
        applied["crop"] = crop_metadata

    # 2. Horizontal flip.
    if rng.random() < post_config.flip_prob:
        processed = ImageOps.mirror(processed)
        applied["hflip"] = True

    light_pos = generate_frame_light_position(rng)

    # 3. Optional distortion overlay on the painting.
    if overlays and rng.random() < post_config.overlay_prob:
        overlay_asset = rng.choice(overlays)
        opacity = rng.uniform(0.3, 0.8)
        processed = apply_overlay_to_image(processed, overlay_asset.image, opacity)
        applied["overlay"] = {
            "applied": True, "name": overlay_asset.name,
            "sha256": overlay_asset.sha256, "opacity": round(opacity, 6),
        }

    # 4. Glare (glass reflection) on the painting.
    if post_config.glare_intensity > 0:
        processed = apply_glare_to_painting(
            processed,
            post_config.glare_intensity,
            post_config.glare_num_spots,
            post_config.glare_size_range,
            rng,
        )
        applied["glare"] = {
            "applied": True,
            "maximum_intensity": post_config.glare_intensity,
            "spot_count_range": list(post_config.glare_num_spots),
            "size_range": list(post_config.glare_size_range),
        }

    # 5. Frame.
    if apply_frame and frames:
        is_portrait = processed.height > processed.width
        frame_asset = rng.choice(frames)
        frame_name, frame = frame_asset.name, frame_asset.image
        applied["frame"] = {"name": frame_name, "sha256": frame_asset.sha256}
        if is_portrait:
            frame = frame.rotate(90, expand=True)
        # Frame assets can be much larger than the <=1K benchmark outputs.
        # Bound the working frame before NumPy mask/shading operations; 2x the
        # painting side preserves border detail without multi-hundred-MB arrays.
        frame_working_cap = max(1024, 2 * max(processed.size))
        processed = compose_image_with_frame(
            processed,
            frame,
            rotate_portrait=not is_portrait,
            light_pos=light_pos,
            max_frame_dimension=frame_working_cap,
        )

    # 6. Reflection overlay on the (framed) painting.
    if post_config.reflection_opacity > 0 and overlays:
        processed = apply_reflection_to_painting(
            processed, overlays, post_config.reflection_opacity, rng
        )

    # 7. Composition: wall, standalone perspective, or flat background.
    if apply_wall:
        target_size = (960, 720)
        width_scale = 1.0 - rng.uniform(0.0, 0.20)
        height_scale = 1.0 - rng.uniform(0.0, 0.20)
        target_w = max(1, int(round(target_size[0] * width_scale)))
        target_h = max(1, int(round(target_size[1] * height_scale)))
        effective_wall_config = replace(wall_config, width=target_w, height=target_h)
        wall_gen = get_worker_wall_generator(effective_wall_config, textures)
        result = wall_gen.place_on_wall(processed, light_pos=light_pos)
        wall_name = wall_gen.last_texture_name or "no_texture"
        if wall_name != "no_texture":
            texture = next((item for name, item in textures if name == wall_name), None)
            applied["wall_texture"] = {
                "name": wall_name,
                "sha256": texture.info.get("asset_sha256") if texture is not None else None,
            }
        applied["perspective"] = dict(wall_gen.last_perspective or {})
    elif apply_perspective:
        wall_gen = get_worker_wall_generator(wall_config, textures)
        warped = wall_gen.apply_perspective_standalone(processed)
        result = flatten_on_background(warped)
        applied["perspective"] = dict(wall_gen.last_perspective or {})
    else:
        result = flatten_on_background(processed)

    # 8. Small in-plane rotation (camera not level).
    if rng.random() < post_config.rotation_prob:
        before = result.size
        # Sample here so provenance records the value actually applied.
        angle = rng.uniform(-post_config.rotation_max_deg, post_config.rotation_max_deg)
        result = _rotate_known_angle(result, angle)
        applied["rotation"] = {"applied": True, "angle_deg": round(angle, 6), "input_size": list(before)}

    # 9. Vintage (randomized desaturation) XOR saturation jitter; non-vintage
    #    variants may pick up a mild defocus instead.
    if rng.random() < post_config.vintage_prob:
        result, vintage_metadata = apply_vintage_effect(
            result,
            post_config.vintage_blur_radius,
            post_config.vintage_noise_std,
            rng,
            saturation_range=post_config.vintage_saturation_range,
            return_metadata=True,
        )
        applied["vintage"] = vintage_metadata
    else:
        if post_config.saturation_range:
            result = apply_post_saturation(result, post_config.saturation_range, rng)
        if rng.random() < post_config.defocus_prob:
            result, defocus_metadata = apply_defocus(
                result, rng, post_config.defocus_radius_range, return_metadata=True
            )
            applied["defocus"] = defocus_metadata

    # 10. Optional explicit sepia toning. Tint and strength vary per image.
    if rng.random() < post_config.sepia_prob:
        sepia_strength = rng.uniform(*post_config.sepia_strength_range)
        shadow_color = (
            rng.randint(38, 62), rng.randint(24, 43), rng.randint(12, 27)
        )
        highlight_color = (
            rng.randint(228, 247), rng.randint(205, 232), rng.randint(160, 202)
        )
        result = apply_sepia(result, sepia_strength, shadow_color, highlight_color)
        applied["sepia"] = {
            "strength": round(sepia_strength, 4),
            "shadow_color": shadow_color,
            "highlight_color": highlight_color,
        }

    # 11. Optional physical-document composition. It is applied after photo
    # colour processing so the surrounding sheet remains recognizably blank.
    if rng.random() < post_config.paper_document_prob:
        result, paper_metadata = place_print_on_paper_document(result, rng)
        applied["paper_document"] = {"applied": True, **paper_metadata}

    # Archival context effects are bounded and retain recoverable content.
    if rng.random() < post_config.annotation_prob:
        result, annotation_metadata = apply_archival_annotations(
            result, rng, post_config.annotation_count_range,
            surface="document" if "paper_document" in applied else "painting",
        )
        applied["annotations"] = annotation_metadata
    if rng.random() < post_config.occlusion_prob:
        result, occlusion_metadata = apply_archival_occlusion(
            result, rng, post_config.occlusion_fraction_range
        )
        applied["occlusion"] = occlusion_metadata

    # 12. Colour-temperature cast, chromatic aberration, vignette.
    if rng.random() < post_config.white_balance_prob:
        result, white_balance_metadata = apply_white_balance(
            result, np_rng, post_config.white_balance_max_shift, return_metadata=True
        )
        applied["white_balance"] = white_balance_metadata
    if rng.random() < post_config.chroma_prob:
        result, chroma_metadata = apply_chromatic_aberration(
            result, rng, post_config.chroma_max_shift_px, return_metadata=True
        )
        applied["chroma"] = chroma_metadata
    if rng.random() < post_config.vignette_prob:
        result, vignette_metadata = apply_vignette(
            result, np_rng, post_config.vignette_strength_range, return_metadata=True
        )
        applied["vignette"] = vignette_metadata

    # 13. Sensor noise.
    if post_config.noise_std > 0:
        result = apply_post_noise(result, post_config.noise_std, rng)
        applied["noise"] = {"applied": True, "std": post_config.noise_std}

    # 14. Downscale->upscale (compounds with the later JPEG compression).
    if rng.random() < post_config.downscale_prob:
        result, resolution_metadata = apply_downscale_upscale(
            result,
            rng,
            post_config.downscale_factor_range,
            longest_side_range=post_config.downsample_longest_side_range,
            return_metadata=True,
        )
        applied["resolution_loss"] = resolution_metadata

    return result, frame_name, wall_name, applied


def generate_light_source_position(
    rng: random.Random,
    xy_range: Tuple[float, float] = (-0.6, 1.6),
    z_range: Tuple[float, float] = (0.5, 1.4),
) -> Tuple[float, float, float]:
    """Deprecated: use generate_frame_light_position from utils."""
    return generate_frame_light_position(rng, xy_range=xy_range, z_range=z_range)
