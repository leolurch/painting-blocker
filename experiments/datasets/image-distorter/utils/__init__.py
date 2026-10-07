"""Utility modules for image distortion and compositing pipeline."""

from .image_utils import (
    IMAGE_EXTENSIONS,
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
    load_image_rgb,
    load_image_rgba,
    minimal_crop_to_fit,
    place_print_on_paper_document,
    random_content_crop,
    resize_to_max_dimension,
    resize_to_max_side,
)
from .annotations import apply_archival_annotations
from .occlusion import apply_archival_occlusion
from .worker_utils import (
    ProgressTracker,
    get_worker_np_rng,
    get_worker_rng,
    get_worker_rng_seed,
    set_rng_base_seed,
    set_worker_rng_index,
)
from .wall_generator import (
    WALL_PALETTES,
    WallConfig,
    WallGenerator,
)

__all__ = [
    # Worker utilities
    "ProgressTracker",
    "get_worker_np_rng",
    "get_worker_rng",
    "get_worker_rng_seed",
    "set_rng_base_seed",
    "set_worker_rng_index",
    # Image processing
    "IMAGE_EXTENSIONS",
    "apply_archival_annotations",
    "apply_archival_occlusion",
    "apply_chromatic_aberration",
    "apply_defocus",
    "apply_downscale_upscale",
    "apply_in_plane_rotation",
    "apply_overlay_to_image",
    "apply_sepia",
    "apply_vignette",
    "apply_white_balance",
    "compose_image_with_frame",
    "generate_frame_light_position",
    "minimal_crop_to_fit",
    "place_print_on_paper_document",
    "random_content_crop",
    "load_image_rgba",
    "load_image_rgb",
    "resize_to_max_dimension",
    "resize_to_max_side",
    # Wall generation
    "WallGenerator",
    "WallConfig",
    "WALL_PALETTES",
]
