#!/usr/bin/env python3
"""Generate role-aware synthetic painting views for sim-to-real transfer.

Each source painting is retained as a modern image and expanded into configurable
modern and historic children. Modern profiles approximate catalogue and auction
photography; historic profiles approximate archival scans, printed reproductions,
cropped records, and (less frequently) framed photographs. Profiles reuse the
shared composition and pixel-effect implementation in :mod:`wall_placement`.

The default is three generated modern views and four historic views.
The modern photograph keeps at most 6% of its area cropped and uses JPEG quality
75 to 100. An archival view is aged in 90% of cases and sepia-toned in 45%.
A print is placed on a paper document in 75% of cases. A cropped record removes
at most 25% of the area. ``--recipe legacy`` reproduces an older four-archetype pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

# One process per --worker is the outer parallelism layer. Prevent BLAS/OpenMP
# libraries imported by NumPy from multiplying 64 workers into thousands of
# native threads. Override with SYNTH_NATIVE_THREADS only for profiling.
_NATIVE_THREADS_PER_WORKER = os.environ.get("SYNTH_NATIVE_THREADS", "1")
if not _NATIVE_THREADS_PER_WORKER.isdigit() or int(_NATIVE_THREADS_PER_WORKER) < 1:
    raise RuntimeError("SYNTH_NATIVE_THREADS must be a positive integer")
for _thread_env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env_var] = _NATIVE_THREADS_PER_WORKER

from PIL import Image, ImageFile

from wall_placement import (
    DIFFICULTY_PRESETS,
    PostProcessConfig,
    WallConfig,
    load_assets,
    process_single_image_archetype,
)
from utils import (
    get_worker_rng,
    resize_to_max_dimension,
    set_rng_base_seed,
    set_worker_rng_index,
)

# Reuse the DB schema, plumbing, and helpers from the original builder.
from place_on_wall_dataset import (
    DEFAULT_FRAMES_DIR,
    DEFAULT_OVERLAYS_DIR,
    DEFAULT_WALL_TEXTURES_DIR,
    ROLE_TAGS,
    SCHEMA_PATCH,
    create_image_dir_dataset,
    empty_sql_batch,
    ensure_output_root,
    fetch_sources,
    flat_image_files,
    jsonable,
    parse_range,
    parse_wall_size,
    read_summary_json,
    resolve_source_kind,
)

DEFAULT_SEED = 42

ROLE_AWARE_RECIPE = "role-aware-v1"
HARD_SYNTH_RECIPE = "hard-synth-v1"
LEGACY_RECIPE = "legacy"
DIMENSION_JITTER_FRACTION = 0.15
PROVENANCE_SCHEMA_VERSION = 3
HISTORIC_PROFILE_COUNTS = {
    "archival": 2,
    "print": 2,
    "cropped_record": 2,
    "framed_photo": 2,
}
HISTORIC_PROFILES = tuple(HISTORIC_PROFILE_COUNTS)
DEFAULT_RECIPE_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "synthetic" / "hard_synth_medium_o1m3h4_v1.yml"
)

RESUME_SCHEMA = """
CREATE TABLE IF NOT EXISTS synthetic_skipped_sources(
    synthetic_run_id INTEGER NOT NULL REFERENCES synthetic_runs(synthetic_run_id) ON DELETE CASCADE,
    parent_file_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL,
    error_message TEXT,
    PRIMARY KEY(synthetic_run_id, parent_file_id)
);
"""

# Toggle map for the reproducible legacy recipe.
ARCHETYPE_TOGGLES = {
    0: (False, False),
    1: (True, False),
    2: (True, True),
    3: (False, True),
}

# Worker-local immutable state (set per process by init_worker_state).
_WORKER_OUT_ROOT: Path | None = None
_WORKER_WALL_CFG: WallConfig | None = None
_WORKER_POST_CFG: PostProcessConfig | None = None
_WORKER_FRAMES = None
_WORKER_OVERLAYS = None
_WORKER_TEXTURES = None
_WORKER_RUN_ID: int | None = None
_WORKER_NOW: str | None = None
_WORKER_THREADPOOL_LIMITER = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a refined synthetic painting dataset (archetype split + "
            "probabilistic real-world degradations) from a dataset DB or a flat "
            "image directory."
        )
    )
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument(
        "--source-kind",
        choices=("auto", "dataset", "image-dir"),
        default="auto",
        help="Input type. auto uses dataset.db when present, else a flat image dir.",
    )
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument(
        "--source-split", default="all_files",
        help="Split containing one source image per painting class (default: all_files).",
    )
    ap.add_argument("--name", default="generate_synth")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest matching run in an existing --output-root.",
    )
    ap.add_argument(
        "--recipe",
        choices=(ROLE_AWARE_RECIPE, HARD_SYNTH_RECIPE, LEGACY_RECIPE),
        default=HARD_SYNTH_RECIPE,
        help="Generation recipe. hard-synth-v1 with the packaged recipe file is the paper setting.",
    )
    ap.add_argument(
        "--modern-variants",
        type=int,
        default=3,
        help="Generated modern variants per source (the retained source image is additional).",
    )
    ap.add_argument(
        "--historic-variants",
        type=int,
        default=4,
        help="Generated historic variants per source.",
    )
    ap.add_argument(
        "--num-variants",
        type=int,
        default=None,
        help="Deprecated legacy alias for --historic-variants. Cannot be combined with that option.",
    )
    ap.add_argument("--workers", type=int, default=4)
    start_methods = mp.get_all_start_methods()
    ap.add_argument(
        "--process-start-method",
        choices=start_methods,
        default="fork" if "fork" in start_methods else start_methods[0],
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed (worker seeds are base_seed * one_based_worker_index).",
    )
    ap.add_argument("--limit", type=int, default=0)

    # Assets.
    ap.add_argument("--frames-horizontal", type=Path, default=DEFAULT_FRAMES_DIR)
    ap.add_argument("--overlays", type=Path, default=DEFAULT_OVERLAYS_DIR)
    ap.add_argument("--wall-textures", type=Path, default=DEFAULT_WALL_TEXTURES_DIR)
    ap.add_argument("--wall-texture-chance", type=float, default=0.7)
    ap.add_argument("--wall-texture-opacity", type=float, default=0.4)
    ap.add_argument("--difficulty", choices=sorted(DIFFICULTY_PRESETS), default="medium")
    ap.add_argument("--wall-size", type=parse_wall_size, default=(1920, 1080))
    ap.add_argument(
        "--max-output-dimension",
        type=int,
        default=1024,
        help=(
            "Maximum final width or height before fixed 15%% per-image jitter. "
            "Images are never upscaled."
        ),
    )
    ap.add_argument(
        "--recipe-config",
        type=Path,
        default=DEFAULT_RECIPE_PATH,
        help="Versioned recipe used by --recipe hard-synth-v1.",
    )
    ap.add_argument(
        "--generation-plan",
        type=Path,
        default=None,
        help="Immutable class/asset/severity holdout plan (required for hard-synth-v1).",
    )
    ap.add_argument(
        "--max-input-dimension",
        type=int,
        default=15_000,
        help="Maximum width or height accepted from a source image (default: 15000).",
    )
    ap.add_argument(
        "--max-input-aspect-ratio",
        type=float,
        default=20.0,
        help="Skip pathological panoramas/slivers above this long/short-side ratio.",
    )
    ap.add_argument(
        "--max-source-side",
        type=int,
        default=1024,
        help="Cap the longest side of each source before augmenting (0 = no cap).",
    )
    # Always-on effects (100%).
    ap.add_argument("--crop-max-area", type=float, default=0.15)
    ap.add_argument("--post-noise", type=float, default=0.02)

    # Probabilistic effects.
    ap.add_argument("--flip-prob", type=float, default=0.10)
    ap.add_argument("--jpeg-prob", type=float, default=0.80)
    ap.add_argument(
        "--jpeg-quality", type=lambda s: parse_range(s, int), default=(65, 100)
    )
    ap.add_argument("--vintage-prob", type=float, default=0.50)
    ap.add_argument(
        "--vintage-saturation", type=lambda s: parse_range(s, float), default=(0.0, 0.35)
    )
    ap.add_argument("--vintage-blur", type=float, default=0.8)
    ap.add_argument("--vintage-noise", type=float, default=0.04)
    ap.add_argument("--white-balance-prob", type=float, default=0.60)
    ap.add_argument("--white-balance-shift", type=float, default=0.12)
    ap.add_argument("--defocus-prob", type=float, default=0.30)
    ap.add_argument(
        "--defocus-radius", type=lambda s: parse_range(s, float), default=(0.4, 1.2)
    )
    ap.add_argument("--vignette-prob", type=float, default=0.50)
    ap.add_argument(
        "--vignette-strength", type=lambda s: parse_range(s, float), default=(0.2, 0.5)
    )
    ap.add_argument("--downscale-prob", type=float, default=0.40)
    ap.add_argument(
        "--downscale-factor", type=lambda s: parse_range(s, float), default=(0.4, 0.8)
    )
    ap.add_argument("--rotation-prob", type=float, default=0.40)
    ap.add_argument("--rotation-max-deg", type=float, default=10.0)
    ap.add_argument("--chroma-prob", type=float, default=0.30)
    ap.add_argument("--chroma-shift", type=int, default=2)

    # Optional extras (off by default, kept for parity with the old builder).
    ap.add_argument(
        "--post-saturation", type=lambda s: parse_range(s, float), default=None
    )
    ap.add_argument("--post-reflection", type=float, default=0.0)
    ap.add_argument("--glare", type=float, default=0.0)
    ap.add_argument("--glare-spots", type=lambda s: parse_range(s, int), default=(1, 3))
    ap.add_argument(
        "--glare-size", type=lambda s: parse_range(s, float), default=(0.1, 0.4)
    )

    return ap.parse_args()


def make_wall_config(args: argparse.Namespace) -> tuple[WallConfig, PostProcessConfig]:
    preset = DIFFICULTY_PRESETS[args.difficulty]
    wall = WallConfig(
        width=args.wall_size[0],
        height=args.wall_size[1],
        min_angle=preset["min_angle"],
        max_angle=preset["max_angle"],
        min_image_scale=preset["min_image_scale"],
        max_image_scale=preset["max_image_scale"],
        position_jitter=preset["position_jitter"],
        min_visibility=0.70,
        add_texture=True,
        texture_chance=args.wall_texture_chance,
        texture_opacity=args.wall_texture_opacity,
        add_shadow=True,
        shadow_opacity=preset["shadow_opacity"],
        add_depth_shading=True,
        depth_shading_strength=0.18,
    )
    post = PostProcessConfig(
        noise_std=args.post_noise,
        saturation_range=args.post_saturation,
        reflection_opacity=args.post_reflection,
        glare_intensity=args.glare,
        glare_num_spots=args.glare_spots,
        glare_size_range=args.glare_size,
        vintage_blur_radius=args.vintage_blur,
        vintage_noise_std=args.vintage_noise,
        crop_max_area=args.crop_max_area,
        flip_prob=args.flip_prob,
        vintage_prob=args.vintage_prob,
        vintage_saturation_range=args.vintage_saturation,
        white_balance_prob=args.white_balance_prob,
        white_balance_max_shift=args.white_balance_shift,
        defocus_prob=args.defocus_prob,
        defocus_radius_range=args.defocus_radius,
        vignette_prob=args.vignette_prob,
        vignette_strength_range=args.vignette_strength,
        downscale_prob=args.downscale_prob,
        downscale_factor_range=args.downscale_factor,
        rotation_prob=args.rotation_prob,
        rotation_max_deg=args.rotation_max_deg,
        chroma_prob=args.chroma_prob,
        chroma_max_shift_px=args.chroma_shift,
        jpeg_prob=args.jpeg_prob,
        jpeg_quality_range=args.jpeg_quality,
        max_source_side=args.max_source_side,
    )
    return wall, post


def _set_worker_state(
    out_root: Path,
    wall_cfg: WallConfig,
    post_cfg: PostProcessConfig,
    frames,
    overlays,
    textures,
    run_id: int,
    now: str,
) -> None:
    global _WORKER_OUT_ROOT, _WORKER_WALL_CFG, _WORKER_POST_CFG
    global _WORKER_FRAMES, _WORKER_OVERLAYS, _WORKER_TEXTURES
    global _WORKER_RUN_ID, _WORKER_NOW
    _WORKER_OUT_ROOT = out_root
    _WORKER_WALL_CFG = wall_cfg
    _WORKER_POST_CFG = post_cfg
    _WORKER_FRAMES = frames
    _WORKER_OVERLAYS = overlays
    _WORKER_TEXTURES = textures
    _WORKER_RUN_ID = run_id
    _WORKER_NOW = now


def init_worker_state(
    out_root: Path,
    wall_cfg: WallConfig,
    post_cfg: PostProcessConfig,
    frames_dir: Path,
    overlays_dir: Path,
    textures_dir: Path,
    run_id: int,
    now: str,
    base_seed: int,
    max_input_dimension: int,
) -> None:
    """Initialize process state, reusing copy-on-write assets after a fork."""
    global _WORKER_THREADPOOL_LIMITER
    set_rng_base_seed(base_seed)
    set_worker_rng_index(1)
    Image.MAX_IMAGE_PIXELS = max(15_000, max_input_dimension) ** 2

    # NumPy's BLAS may otherwise create a native thread pool in every process.
    # The augmentation work is parallelized at the process level, and its few
    # tiny linear solves do not benefit from nested BLAS parallelism.
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _WORKER_THREADPOOL_LIMITER = None

    # Under fork, main() preloads these immutable PIL assets once and children
    # inherit their memory copy-on-write. Spawn/forkserver workers load locally.
    inherited = (
        _WORKER_OUT_ROOT == out_root
        and _WORKER_FRAMES is not None
        and _WORKER_OVERLAYS is not None
        and _WORKER_TEXTURES is not None
    )
    if inherited:
        frames, overlays, textures = (
            _WORKER_FRAMES, _WORKER_OVERLAYS, _WORKER_TEXTURES
        )
    else:
        frames, overlays, textures = load_assets(
            frames_dir, overlays_dir, textures_dir, verbose=False
        )
    _set_worker_state(
        out_root, wall_cfg, post_cfg, frames, overlays, textures, run_id, now
    )


def _named_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def realized_dimension_cap(
    maximum: int,
    *,
    seed: int,
    parent_file_id: str,
    variant_key: str,
) -> int:
    """Return a scheduling- and transform-independent fixed-jitter cap."""
    if maximum < 1:
        raise ValueError("maximum must be >= 1")
    rng = random.Random(_named_seed(seed, parent_file_id, variant_key, "dimension"))
    factor = 1.0 - rng.uniform(0.0, DIMENSION_JITTER_FRACTION)
    return max(1, round(maximum * factor))


def expand_profile_counts(counts: dict[str, int]) -> list[tuple[str, int]]:
    """Expand exact profile quotas into stable ``(profile, local_index)`` pairs."""
    unknown = sorted(set(counts) - set(HISTORIC_PROFILES))
    if unknown:
        raise ValueError(f"Unknown historic profiles: {', '.join(unknown)}")
    expanded: list[tuple[str, int]] = []
    for profile in HISTORIC_PROFILES:
        count = counts.get(profile, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Historic profile count for {profile!r} must be a non-negative integer")
        expanded.extend((profile, index) for index in range(count))
    return expanded


def role_profile_schedule(role: str, count: int) -> list[str]:
    """Return deterministic profile allocation without weighted historic bias."""
    if count <= 0:
        return []
    if role == "modern":
        profiles = ["catalogue", "framed_catalogue", "wall_photo", "perspective_photo"]
        return [profiles[i % len(profiles)] for i in range(count)]
    if role != "historic":
        raise ValueError(f"Unsupported role: {role}")
    profiles = list(HISTORIC_PROFILES)
    return [profiles[i % len(profiles)] for i in range(count)]


def role_profile_config(
    base: PostProcessConfig,
    role: str,
    profile: str,
    severity: str | None = None,
) -> tuple[PostProcessConfig, bool, bool, bool]:
    """Resolve a coherent profile into shared post-processing and composition toggles."""
    common = dict(flip_prob=0.0, max_source_side=base.max_source_side)
    if role in {"modern", "modern_original", "modern_generated"}:
        cfg = replace(
            base, **common, crop_max_area=0.06, overlay_prob=0.15,
            noise_std=0.004, vintage_prob=0.0, white_balance_prob=0.35,
            defocus_prob=0.10, vignette_prob=0.12, downscale_prob=0.20,
            downscale_factor_range=(0.65, 0.9), rotation_prob=0.18,
            rotation_max_deg=4.0, chroma_prob=0.04, jpeg_prob=0.60,
            jpeg_quality_range=(75, 100), glare_intensity=0.0,
            reflection_opacity=0.0, sepia_prob=0.0, paper_document_prob=0.0,
        )
        toggles = {
            "modern_query": (False, False, False),
            "catalogue": (False, False, False),
            "framed_catalogue": (True, False, False),
            "wall_photo": (True, True, True),
            "perspective_photo": (False, True, False),
        }
        if profile == "wall_photo":
            cfg = replace(cfg, glare_intensity=0.10, overlay_prob=0.25, vignette_prob=0.20)
        return cfg, *toggles[profile]

    if profile == "archival":
        cfg = replace(
            base, **common, crop_max_area=0.08, overlay_prob=0.25,
            noise_std=0.025, vintage_prob=0.90, vintage_saturation_range=(0.0, 0.15),
            white_balance_prob=0.10, defocus_prob=0.25, vignette_prob=0.35,
            downscale_prob=0.70, downscale_factor_range=(0.30, 0.70),
            rotation_prob=0.35, rotation_max_deg=4.0, chroma_prob=0.02,
            jpeg_prob=0.85, jpeg_quality_range=(45, 90), glare_intensity=0.0,
            reflection_opacity=0.0, sepia_prob=0.45,
            sepia_strength_range=(0.50, 0.95), paper_document_prob=0.15,
        )
        return cfg, False, False, False
    if profile == "print":
        cfg = replace(
            base, **common, crop_max_area=0.10, overlay_prob=0.15,
            noise_std=0.016, vintage_prob=0.70, vintage_saturation_range=(0.0, 0.25),
            white_balance_prob=0.08, defocus_prob=0.20, vignette_prob=0.15,
            downscale_prob=0.75, downscale_factor_range=(0.25, 0.65),
            rotation_prob=0.35, rotation_max_deg=3.0, chroma_prob=0.01,
            jpeg_prob=0.90, jpeg_quality_range=(40, 85), glare_intensity=0.0,
            reflection_opacity=0.0, sepia_prob=0.35,
            sepia_strength_range=(0.45, 0.90), paper_document_prob=0.75,
        )
        return cfg, False, False, False
    if profile == "cropped_record":
        cfg = replace(
            base, **common, crop_max_area=0.25, overlay_prob=0.18,
            noise_std=0.014, vintage_prob=0.50, vintage_saturation_range=(0.0, 0.30),
            white_balance_prob=0.15, defocus_prob=0.25, vignette_prob=0.20,
            downscale_prob=0.55, downscale_factor_range=(0.40, 0.75),
            rotation_prob=0.30, rotation_max_deg=5.0, chroma_prob=0.02,
            jpeg_prob=0.85, jpeg_quality_range=(50, 90), glare_intensity=0.0,
            reflection_opacity=0.0, sepia_prob=0.25,
            sepia_strength_range=(0.40, 0.85), paper_document_prob=0.30,
        )
        return cfg, False, True, False
    if profile == "framed_photo":
        cfg = replace(
            base, **common, crop_max_area=0.08, overlay_prob=0.25,
            noise_std=0.008, vintage_prob=0.25, vintage_saturation_range=(0.05, 0.40),
            white_balance_prob=0.45, defocus_prob=0.20, vignette_prob=0.25,
            downscale_prob=0.40, downscale_factor_range=(0.50, 0.80),
            rotation_prob=0.25, rotation_max_deg=4.0, chroma_prob=0.05,
            jpeg_prob=0.75, jpeg_quality_range=(60, 95), glare_intensity=0.08,
            reflection_opacity=0.0, sepia_prob=0.12,
            sepia_strength_range=(0.35, 0.75), paper_document_prob=0.05,
        )
        return cfg, True, True, True
    raise ValueError(f"Unsupported {role} profile: {profile}")


def hard_profile_config(
    base: PostProcessConfig,
    profile: str,
    severity: str,
) -> tuple[PostProcessConfig, bool, bool, bool]:
    """Apply versioned hard/compound-OOD requirements to a historic profile."""
    cfg, frame, perspective, wall = role_profile_config(base, "historic", profile)
    if severity not in {"mild", "medium", "hard", "extreme"}:
        raise ValueError(f"Unknown severity preset: {severity}")
    if severity in {"mild", "medium"}:
        return cfg, frame, perspective, wall
    extreme = severity == "extreme"
    crop_range = (0.35, 0.60) if extreme else (0.55, 0.75)
    resolution_range = (32, 96) if extreme else (96, 160)
    cfg = replace(
        cfg,
        crop_retained_area_range=crop_range,
        downscale_prob=1.0,
        downsample_longest_side_range=resolution_range,
        jpeg_prob=1.0,
        jpeg_quality_range=(28, 58) if extreme else (40, 72),
        annotation_prob=1.0 if profile in {"print", "cropped_record"} else 0.45,
        occlusion_prob=1.0 if profile == "framed_photo" else (0.55 if extreme else 0.25),
        occlusion_fraction_range=(0.08, 0.20) if extreme else (0.04, 0.12),
    )
    if profile == "archival":
        cfg = replace(cfg, vintage_prob=1.0, sepia_prob=max(cfg.sepia_prob, 0.65))
        perspective = True
    elif profile == "print":
        cfg = replace(cfg, paper_document_prob=1.0)
    elif profile == "cropped_record":
        perspective = True
    elif profile == "framed_photo":
        frame, perspective, wall = True, True, True
        cfg = replace(cfg, glare_intensity=max(0.12, cfg.glare_intensity), overlay_prob=1.0)
    return cfg, frame, perspective, wall


def severity_from_transforms(transforms: dict) -> tuple[str, float]:
    """Classify actual sampled effects into a stable coarse severity score."""
    score = 0.0
    crop = transforms.get("crop") or {}
    score += max(0.0, 1.0 - float(crop.get("retained_area_fraction", 1.0))) * 0.40
    resolution = transforms.get("resolution_loss") or {}
    longest = int(resolution.get("longest_side_pixels", 1024))
    score += max(0.0, min(1.0, (192 - longest) / 160.0)) * 0.45
    score += 0.10 if transforms.get("perspective", {}).get("applied") else 0.0
    score += min(0.10, float((transforms.get("occlusion") or {}).get("coverage", 0.0)))
    score += 0.05 if transforms.get("annotations", {}).get("applied") else 0.0
    score += 0.05 if transforms.get("paper_document", {}).get("applied") else 0.0
    score += 0.08 if int((transforms.get("jpeg") or {}).get("quality", 100)) <= 72 else 0.0
    score = round(min(1.0, score), 6)
    if score >= 0.72:
        bucket = "extreme"
    elif score >= 0.50:
        bucket = "hard"
    elif score >= 0.25:
        bucket = "medium"
    else:
        bucket = "mild"
    return bucket, score


def generated_tags(parent_tags, role: str, profile: str, applied: dict) -> list[str]:
    tags = {tag.strip() for tag in parent_tags if tag.strip() and tag.strip() not in ROLE_TAGS}
    tags.update((role, f"profile_{profile}", "synthetic", "synthetic_derivative"))
    if role == "modern_original":
        tags.update(("modern", "retained_original", "source_original"))
    elif role == "modern_generated":
        tags.add("modern")
    elif role.startswith("historic_"):
        tags.add("historic")
    if applied.get("apply_frame"):
        tags.add("frame_included")
    if applied.get("apply_perspective"):
        tags.add("perspective")
    if applied.get("apply_wall"):
        tags.add("on_wall")
    if applied.get("sepia"):
        tags.add("sepia")
    if applied.get("paper_document"):
        tags.add("paper_document")
        tags.add("reflective_print")
    if role.startswith("historic"):
        tags.add("poor_quality")
    return sorted(tags)


def archetype_tags(parent_tags, archetype: int, applied: dict) -> list[str]:
    tags = set()
    for tag in parent_tags:
        clean = tag.strip()
        if clean and clean not in ROLE_TAGS:
            tags.add(clean)
    tags.add("historic")
    tags.add(f"archetype_{archetype}")
    frame, perspective = ARCHETYPE_TOGGLES[archetype]
    if frame:
        tags.add("frame_included")
    if perspective:
        tags.add("perspective")
    if applied.get("apply_wall"):
        tags.add("on_wall")
    tags.add("poor_quality")
    return sorted(tags)


def stable_task_seed(base_seed: int, file_id: str) -> int:
    """Derive a worker/scheduling-independent 64-bit seed for one source."""
    digest = hashlib.blake2b(
        f"{base_seed}\0{file_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def source_working_cap(post_cfg: PostProcessConfig, output_cap: int) -> int:
    """Largest source raster needed by any variant in a source task."""
    caps = [output_cap]
    if post_cfg.max_source_side > 0:
        caps.append(post_cfg.max_source_side)
    return min(caps)


def prepare_source_raster(
    image: Image.Image,
    source_size: tuple[int, int],
    working_cap: int,
) -> tuple[Image.Image, dict]:
    """Decode and cap a source once, before its variants are generated.

    JPEG ``draft`` decoding uses libjpeg's native 1/2, 1/4, or 1/8 scaling and
    avoids materializing a potentially 15k-by-15k raster. Other formats are
    decoded normally, but their full-size buffer is released before RGBA
    conversion. The returned image is therefore bounded by ``working_cap``.
    """
    decoder_size_before = image.size
    decoder_format = image.format
    if max(source_size) > working_cap:
        image.draft("RGB", (working_cap, working_cap))
    decoder_size = image.size
    image.load()

    prepared, resize_metadata = resize_to_max_dimension(image, working_cap)
    if prepared is not image:
        # The resized raster owns its pixels; release the large decoded source
        # before allocating the RGBA working copy.
        image.close()
    try:
        base = prepared.convert("RGBA")
    finally:
        prepared.close()

    metadata = {
        "input_size": list(source_size),
        "decoder_format": decoder_format,
        "decoder_size_before_draft": list(decoder_size_before),
        "decoder_size": list(decoder_size),
        "decoder_draft_applied": decoder_size != decoder_size_before,
        "max_dimension": working_cap,
        "output_size": list(base.size),
        "resized": max(source_size) > working_cap,
        "resize_from_decoder": resize_metadata,
    }
    return base, metadata


def worker(task: dict) -> dict[str, object]:
    """Generate all variants for one source using a source-stable RNG stream."""
    if (
        _WORKER_OUT_ROOT is None
        or _WORKER_WALL_CFG is None
        or _WORKER_POST_CFG is None
        or _WORKER_FRAMES is None
        or _WORKER_OVERLAYS is None
        or _WORKER_TEXTURES is None
        or _WORKER_RUN_ID is None
        or _WORKER_NOW is None
    ):
        raise RuntimeError("Worker state was not initialized")

    # Reset both Python and NumPy generators for every source. Results therefore
    # do not depend on process count, task order, or worker scheduling.
    set_rng_base_seed(stable_task_seed(task["seed"], task["parent_file_id"]))
    set_worker_rng_index(1)
    base_cfg = _WORKER_POST_CFG
    rng = get_worker_rng()

    src_path = _WORKER_OUT_ROOT / task["src_rel_path"]
    source_update = None
    source_truncated_recovery = False
    working_cap = source_working_cap(_WORKER_POST_CFG, task["max_output_dimension"])
    try:
        # Inspect the header without Pillow's area check so dimensions above the
        # configured cap can be skipped before allocating/decompressing pixels.
        previous_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            image_context = Image.open(src_path)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_max_pixels
        source_width, source_height = image_context.size
        source_aspect = max(source_width, source_height) / max(
            1, min(source_width, source_height)
        )
        skip_reason = None
        if max(source_width, source_height) > task["max_input_dimension"]:
            skip_reason = "dimension_limit"
        elif source_aspect > task["max_input_aspect_ratio"]:
            skip_reason = "aspect_ratio_limit"
        if skip_reason is not None:
            image_context.close()
            # This is a copied output-dataset file, never the immutable source
            # dataset. Remove it so skipped oversized files cannot violate the
            # benchmark's strict local dimension bound.
            src_path.unlink(missing_ok=True)
            batch = empty_sql_batch()
            batch["source_update"] = None
            batch["retire_source"] = True
            batch["remove_source_file"] = True
            batch["source_was_resized"] = False
            batch["source_was_repaired"] = False
            batch["warnings"] = [
                f"skipped source image ({source_width}x{source_height}, "
                f"aspect {source_aspect:.2f}, reason {skip_reason}): "
                f"{task['src_rel_path']}"
            ]
            batch["skipped_source"] = (
                _WORKER_RUN_ID, task["parent_file_id"], skip_reason,
                source_width, source_height, _WORKER_NOW, None,
            )
            return batch
        base, source_prepare = prepare_source_raster(
            image_context, (source_width, source_height), working_cap
        )
    except OSError as exc:
        try:
            image_context.close()
        except (NameError, AttributeError):
            pass
        if "image file is truncated" not in str(exc).lower():
            raise
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            image_context = Image.open(src_path)
            source_width, source_height = image_context.size
            base, source_prepare = prepare_source_raster(
                image_context, (source_width, source_height), working_cap
            )
            source_truncated_recovery = True
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous

    # The copied source remains an immutable regeneration input. Every
    # benchmark/query image, including modern_original, is a derivative.
    source_input_size = [source_width, source_height]
    # The copied source file itself remains immutable; only the in-memory
    # augmentation raster was resized.
    should_resize = False

    parent_tags = (task["tags_csv"] or "").split(",")
    specifications: list[tuple[str, str, int, str, PostProcessConfig, bool, bool, bool]] = []
    if task["recipe"] == LEGACY_RECIPE:
        count = task["historic_variants"]
        for variant_idx in range(count):
            archetype = (variant_idx * 4) // count if count else 0
            apply_frame, apply_perspective = ARCHETYPE_TOGGLES[archetype]
            apply_wall = apply_perspective and (rng.random() < 0.5)
            specifications.append(
                ("historic", f"archetype_{archetype}", variant_idx, "legacy", base_cfg,
                 apply_frame, apply_perspective, apply_wall)
            )
    elif task["recipe"] == HARD_SYNTH_RECIPE:
        query_cfg, query_frame, query_perspective, query_wall = role_profile_config(
            base_cfg, "modern_original", "modern_query"
        )
        specifications.append(
            ("modern_original", "modern_query", 0, "mild", query_cfg,
             query_frame, query_perspective, query_wall)
        )
        for variant_idx, profile in enumerate(
            role_profile_schedule("modern", task["modern_variants"])
        ):
            cfg, apply_frame, apply_perspective, apply_wall = role_profile_config(
                base_cfg, "modern_generated", profile
            )
            specifications.append(
                ("modern_generated", profile, variant_idx, "medium", cfg,
                 apply_frame, apply_perspective, apply_wall)
            )
        profile_counts = task["historic_profile_counts"]
        for profile, variant_idx in expand_profile_counts(profile_counts):
            policy = task.get("severity_policy") or ["hard", "extreme"]
            severity = policy[variant_idx % len(policy)]
            cfg, apply_frame, apply_perspective, apply_wall = hard_profile_config(
                base_cfg, profile, severity
            )
            specifications.append(
                (f"historic_{profile}", profile, variant_idx, severity, cfg,
                 apply_frame, apply_perspective, apply_wall)
            )
    else:
        for role, count in (
            ("modern", task["modern_variants"]),
            ("historic", task["historic_variants"]),
        ):
            for variant_idx, profile in enumerate(role_profile_schedule(role, count)):
                cfg, apply_frame, apply_perspective, apply_wall = role_profile_config(
                    base_cfg, role, profile
                )
                specifications.append(
                    (role, profile, variant_idx, "legacy", cfg,
                     apply_frame, apply_perspective, apply_wall)
                )

    batch = empty_sql_batch()
    batch["source_update"] = source_update
    batch["skipped_source"] = None
    batch["retire_source"] = task["recipe"] == HARD_SYNTH_RECIPE
    batch["remove_source_file"] = False
    batch["source_was_resized"] = should_resize
    batch["source_was_repaired"] = source_truncated_recovery
    batch["warnings"] = (
        [f"recovered truncated source image: {task['src_rel_path']}"]
        if source_truncated_recovery else []
    )
    allowed_assets = set(task.get("allowed_asset_hashes") or [])
    frames = [asset for asset in _WORKER_FRAMES if not allowed_assets or asset.sha256 in allowed_assets]
    overlays = [asset for asset in _WORKER_OVERLAYS if not allowed_assets or asset.sha256 in allowed_assets]
    textures = [
        (name, image) for name, image in _WORKER_TEXTURES
        if image is None or not allowed_assets or image.info.get("asset_sha256") in allowed_assets
    ]
    for role, profile, variant_idx, severity_preset, cfg, apply_frame, apply_perspective, apply_wall in specifications:
        variant_key = f"{role}:{profile}:{variant_idx}"
        realized_cap = realized_dimension_cap(
            task["max_output_dimension"],
            seed=task["seed"],
            parent_file_id=task["parent_file_id"],
            variant_key=variant_key,
        )
        result, frame_name, wall_name, applied = process_single_image_archetype(
            base,
            _WORKER_WALL_CFG,
            frames,
            overlays,
            textures,
            cfg,
            apply_frame=apply_frame,
            apply_perspective=apply_perspective,
            apply_wall=apply_wall,
            source_max_dimension=realized_cap,
        )
        applied["source_prepare"] = source_prepare
        if source_truncated_recovery:
            applied["source_truncated_recovery"] = True

        quality = (
            rng.randint(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1])
            if rng.random() < cfg.jpeg_prob else 100
        )
        safe_profile = profile.replace("_", "-")
        safe_role = role.replace("_", "-")
        child_file_id = (
            f"{task['parent_file_id']}_{safe_role}-{safe_profile}"
            f"_{frame_name}_{wall_name}-v{variant_idx + 1:03d}"
        )
        rel_path = f"images/{child_file_id}.jpg"
        dst = _WORKER_OUT_ROOT / rel_path
        pre_save_size = list(result.size)
        result, output_resize = resize_to_max_dimension(result, realized_cap)
        result.convert("RGB").save(dst, "JPEG", quality=quality, optimize=True)
        result_width, result_height = result.size
        dimensions = {
            "configured_max_output_dimension": task["max_output_dimension"],
            "jitter_fraction": DIMENSION_JITTER_FRACTION,
            "realized_cap": realized_cap,
            "source_input_size": source_input_size,
            "source_working_size": source_prepare["output_size"],
            "pre_transform_size": (applied.get("source_resize") or {}).get(
                "output_size", source_prepare["output_size"]
            ),
            "pre_save_size": pre_save_size,
            "final_size": [result_width, result_height],
            "resized_at_output": bool(output_resize["resized"]),
            "output_resize": output_resize,
        }

        if task["recipe"] == LEGACY_RECIPE:
            archetype = int(profile.rsplit("_", 1)[1])
            tags = archetype_tags(parent_tags, archetype, applied)
        else:
            tags = generated_tags(parent_tags, role, profile, applied)
        applied["jpeg"] = {"quality": quality}
        if applied.get("annotations", {}).get("applied"):
            font_hash = "sha256:" + hashlib.sha256(
                f"Pillow.load_default:v1:{task.get('subset', 'unassigned')}".encode()
            ).hexdigest()
            applied["annotations"]["font"] = font_hash
        actual_bucket, severity_score = severity_from_transforms(applied)
        assets = {
            key: value.get("sha256")
            for key, value in {
                "frame": applied.get("frame") or {},
                "overlay": applied.get("overlay") or {},
                "wall_texture": applied.get("wall_texture") or {},
            }.items()
            if value.get("sha256")
        }
        if applied.get("annotations", {}).get("font"):
            assets["font"] = applied["annotations"]["font"]
        batch["file_rows"].append(
            (
                child_file_id, ".jpg", f"{child_file_id}.jpg", None, None, None,
                rel_path, "generated", "image/jpeg", result_width, result_height,
                dst.stat().st_size, None, None,
                "modern" if role.startswith("modern") else "historic",
                task["english_description"],
                task["description_source"], None, None, None, None, None, None,
                _WORKER_NOW, _WORKER_NOW,
            )
        )
        batch["rel_rows"].append(
            (child_file_id, task["class_id"], "same_painting", task["reviewed_at"], _WORKER_NOW)
        )
        batch["deriv_rows"].append(
            (
                child_file_id, task["parent_file_id"], _WORKER_RUN_ID,
                f"generate_synth_{task['recipe']}",
                json.dumps(
                    {
                        "schema_version": PROVENANCE_SCHEMA_VERSION,
                        "recipe": task["recipe"],
                        "role": role,
                        "origin": "source_original" if role == "modern_original" else "synthetic",
                        "is_synthetic_derivative": True,
                        "profile": profile,
                        "profile_version": 1,
                        "variant_index": variant_idx + 1,
                        "parent_file_id": task["parent_file_id"],
                        "subset": task.get("subset"),
                        "severity": {
                            "preset": severity_preset,
                            "bucket": actual_bucket,
                            "score": severity_score,
                        },
                        "dimensions": dimensions,
                        "assets": assets,
                        "transforms": applied,
                    },
                    sort_keys=True,
                ),
            )
        )
        for tag in tags:
            batch["tag_specs"].append((child_file_id, task["class_id"], tag))
        batch["split_rows"].extend(
            [
                ("all_files", child_file_id), ("generated", child_file_id),
                ("synthetic_only", child_file_id), ("augmented_all", child_file_id),
                ("same_painting", child_file_id), (role, child_file_id),
                (f"profile_{profile}", child_file_id),
                (f"subset_{task.get('subset', 'unassigned')}", child_file_id),
            ]
        )
        if role in {"modern", "modern_original"} and task["in_query"]:
            batch["split_rows"].extend(
                [("query_synthetic", child_file_id), ("query_augmented", child_file_id)]
            )
        if role.startswith("historic") and task["in_candidate"]:
            batch["split_rows"].extend(
                [("candidate_synthetic", child_file_id), ("candidate_augmented", child_file_id)]
            )
        # Release large Pillow buffers before starting the next variant.
        result.close()
    base.close()
    return batch


def fetch_source_classes(
    conn: sqlite3.Connection,
    split_name: str,
    limit: int,
    synthetic_run_id: int | None = None,
) -> list[sqlite3.Row]:
    """Select stable source parents, including retired parents during resume."""
    if synthetic_run_id is None:
        rows = fetch_sources(conn, split_name, 0)
    else:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT f.file_id, f.file_ext, f.local_rel_path, f.english_description,
                   f.description_source, ifc.class_id, ifc.reviewed_at,
                   EXISTS(SELECT 1 FROM splits s WHERE s.file_id=f.file_id AND s.split_name='query_default') AS in_query,
                   EXISTS(SELECT 1 FROM splits s WHERE s.file_id=f.file_id AND s.split_name='candidate_default') AS in_candidate,
                   COALESCE((SELECT GROUP_CONCAT(tag, ',') FROM image_file_class_tags t
                             WHERE t.image_file_class_id=ifc.image_file_class_id), '') AS tags_csv
            FROM image_files f
            JOIN image_file_classes ifc ON ifc.file_id=f.file_id
            WHERE ifc.primary_label='same_painting' AND (
                (EXISTS(SELECT 1 FROM splits s WHERE s.file_id=f.file_id AND s.split_name=?)
                 AND NOT EXISTS(SELECT 1 FROM image_file_derivations d
                                WHERE d.child_file_id=f.file_id AND d.synthetic_run_id=?))
                OR EXISTS(SELECT 1 FROM splits s WHERE s.file_id=f.file_id AND s.split_name='source_inputs')
                OR EXISTS(SELECT 1 FROM synthetic_skipped_sources ss
                          WHERE ss.parent_file_id=f.file_id AND ss.synthetic_run_id=?)
            )
            ORDER BY f.file_id
            """,
            (split_name, synthetic_run_id, synthetic_run_id),
        ).fetchall()
    selected, seen = [], set()
    for row in rows:
        if row["class_id"] in seen:
            continue
        seen.add(row["class_id"])
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressLogger:
    """Throttled stage progress with counts, percentage, rate, and ETA."""

    def __init__(
        self, stage: str, total: int, interval: float = 5.0, initial_done: int = 0
    ) -> None:
        self.stage = stage
        self.total = total
        self.initial_done = initial_done
        self.done = initial_done
        self.started = time.monotonic()
        self.last_log = 0.0
        self.interval = interval
        self.log(force=True)

    def increment(self, count: int = 1) -> None:
        self.done += count
        self.log(force=self.done >= self.total)

    def log(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_log < self.interval:
            return
        elapsed = max(now - self.started, 1e-9)
        session_done = self.done - self.initial_done
        rate = session_done / elapsed
        missing = max(0, self.total - self.done)
        eta = missing / rate if rate > 0 else None
        percentage = 100.0 if self.total == 0 else 100.0 * self.done / self.total
        print(
            f"[{self.stage}] {percentage:6.2f}% | done {self.done:,}/{self.total:,} "
            f"| missing {missing:,} | {rate:.2f}/s | ETA {format_eta(eta)}",
            flush=True,
        )
        self.last_log = now


def parallel_copy_paths(
    src_root: Path, out_root: Path, rel_paths: list[Path], workers: int
) -> None:
    """Copy source files concurrently; useful for high-latency source storage."""
    progress = ProgressLogger("copy", len(rel_paths))

    def copy_one(rel_path: Path) -> None:
        src = src_root / rel_path
        dst = out_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 32))) as pool:
        for _ in pool.map(copy_one, rel_paths):
            progress.increment()


def copy_limited_dataset_root(
    src_root: Path, out_root: Path, source_split: str, limit: int, workers: int
) -> None:
    """Copy a DB dataset in parallel, optionally pruning it by source class."""
    shutil.copy2(src_root / "dataset.db", out_root / "dataset.db")
    if (src_root / "summary.json").exists():
        shutil.copy2(src_root / "summary.json", out_root / "summary.json")
    conn = sqlite3.connect(out_root / "dataset.db")
    conn.execute("PRAGMA foreign_keys = ON")
    if limit > 0:
        selected = fetch_source_classes(conn, source_split, limit)
        if not selected:
            conn.close()
            raise SystemExit(f"No source classes found in split: {source_split}")
        selected_ids = [row["file_id"] for row in selected]
        placeholders = ",".join("?" for _ in selected_ids)
        conn.execute(
            f"DELETE FROM image_files WHERE file_id NOT IN ({placeholders})", selected_ids
        )
        conn.execute(
            "DELETE FROM classes WHERE class_id NOT IN "
            "(SELECT DISTINCT class_id FROM image_file_classes)"
        )
        conn.commit()
    rel_paths = [
        Path(row[0])
        for row in conn.execute(
            "SELECT DISTINCT local_rel_path FROM image_files "
            "WHERE local_rel_path IS NOT NULL ORDER BY local_rel_path"
        )
    ]
    conn.close()
    parallel_copy_paths(src_root, out_root, rel_paths, workers)


def insert_sql_batch(
    conn: sqlite3.Connection, batch: dict[str, object]
) -> tuple[int, int, int, int]:
    """Insert one completed source batch immediately, keeping parent memory bounded."""
    for warning in batch.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr, flush=True)
    skipped_source = batch.get("skipped_source")
    if skipped_source is not None:
        conn.execute(
            "INSERT OR REPLACE INTO synthetic_skipped_sources(synthetic_run_id, parent_file_id, reason, width, height, created_at, error_message) VALUES(?, ?, ?, ?, ?, ?, ?)",
            skipped_source,
        )
        parent_file_id = skipped_source[1]
        conn.execute(
            "INSERT OR IGNORE INTO splits(split_name, file_id) VALUES('skipped_sources', ?)",
            (parent_file_id,),
        )
        if skipped_source[2] in {"dimension_limit", "aspect_ratio_limit"}:
            conn.execute(
                "INSERT OR IGNORE INTO splits(split_name, file_id) "
                "VALUES('skipped_oversized', ?)",
                (parent_file_id,),
            )
        conn.execute("DELETE FROM splits WHERE file_id=? AND split_name!='skipped_sources'", (parent_file_id,))
        conn.execute(
            "UPDATE image_files SET download_status='skipped_source', local_rel_path=NULL WHERE file_id=?",
            (parent_file_id,),
        )
        return 0, 0, 0, 1

    if batch.get("retire_source"):
        # Keep the immutable bytes as regeneration input, but make the copied
        # source explicitly unavailable to benchmark selectors.
        parent_file_id = batch["deriv_rows"][0][1] if batch["deriv_rows"] else None
        if parent_file_id:
            conn.execute("DELETE FROM splits WHERE file_id=?", (parent_file_id,))
            conn.execute(
                "INSERT OR IGNORE INTO splits(split_name, file_id) VALUES('source_inputs', ?)",
                (parent_file_id,),
            )
            conn.execute(
                "UPDATE image_files SET download_status='source_input' WHERE file_id=?",
                (parent_file_id,),
            )

    source_update = batch.get("source_update")
    if source_update is not None:
        conn.execute(
            "UPDATE image_files SET width=?, height=?, bytes_on_disk=?, updated_at=? WHERE file_id=?",
            source_update,
        )

    file_rows = batch["file_rows"]
    if not file_rows:
        return (
            0, int(bool(batch.get("source_was_resized"))),
            int(bool(batch.get("source_was_repaired"))), 0,
        )
    conn.executemany(
        "INSERT INTO image_files(file_id, file_ext, canonical_title, file_page_url, source_url, download_url, local_rel_path, download_status, mime_type, width, height, bytes_on_disk, year, epoch_bucket, source_role, english_description, description_source, has_p6243, is_photo, p180_count, pair_count, historical_count, modern_count, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        file_rows,
    )
    conn.executemany(
        "INSERT INTO image_file_classes(file_id, class_id, primary_label, reviewed_at, updated_at) VALUES(?, ?, ?, ?, ?)",
        batch["rel_rows"],
    )
    file_ids = [row[0] for row in file_rows]
    placeholders = ",".join("?" for _ in file_ids)
    rel_map = {
        (file_id, class_id): rel_id
        for rel_id, file_id, class_id in conn.execute(
            f"SELECT image_file_class_id, file_id, class_id FROM image_file_classes "
            f"WHERE file_id IN ({placeholders})",
            file_ids,
        )
    }
    conn.executemany(
        "INSERT INTO image_file_class_tags(image_file_class_id, tag) VALUES(?, ?)",
        [
            (rel_map[(file_id, class_id)], tag)
            for file_id, class_id, tag in batch["tag_specs"]
        ],
    )
    conn.executemany(
        "INSERT INTO image_file_derivations(child_file_id, parent_file_id, synthetic_run_id, transform_family, transform_params_json) VALUES(?, ?, ?, ?, ?)",
        batch["deriv_rows"],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO splits(split_name, file_id) VALUES(?, ?)",
        batch["split_rows"],
    )
    return (
        len(file_rows), int(bool(batch.get("source_was_resized"))),
        int(bool(batch.get("source_was_repaired"))), 0,
    )


def failed_source_batch(task: dict, run_id: int, now: str, exc: Exception) -> dict:
    """Convert an isolated source failure into a persistent skip record."""
    message = f"{type(exc).__name__}: {exc}"
    batch = empty_sql_batch()
    batch["source_update"] = None
    batch["retire_source"] = True
    batch["remove_source_file"] = False
    batch["source_was_resized"] = False
    batch["source_was_repaired"] = False
    batch["warnings"] = [
        f"skipped failed source image {task['src_rel_path']}: {message}"
    ]
    batch["skipped_source"] = (
        run_id, task["parent_file_id"], "processing_error", None, None, now, message,
    )
    return batch


def remove_partial_source_outputs(out_root: Path, parent_file_id: str) -> None:
    """Remove files written by a source task that failed before returning metadata."""
    for path in (out_root / "images").glob(f"{parent_file_id}_*.jpg"):
        path.unlink(missing_ok=True)


def write_summary(
    out_root: Path,
    args: argparse.Namespace,
    source_kind: str,
    dataset_name: str,
    run_id: int,
    now: str,
    sources_count: int,
    generated_count: int,
    wall_cfg: WallConfig,
    post_cfg: PostProcessConfig,
    source_summary: dict | None,
    skipped_sources_count: int,
    resumed_sources: int,
    skipped_sources: list[dict],
) -> None:
    summary = {
        "dataset_name": dataset_name,
        "output_root": str(out_root),
        "source_root": str(args.dataset_root),
        "source_kind": source_kind,
        "synthetic_run_id": run_id,
        "run_name": args.name,
        "tool": "generate_synth",
        "created_at_utc": now,
        "source_files": sources_count,
        "generated_files": generated_count,
        "skipped_sources_count": skipped_sources_count,
        "skipped_oversized_sources": sum(
            item["reason"] in {"dimension_limit", "aspect_ratio_limit"}
            for item in skipped_sources
        ),
        "skipped_processing_errors": sum(
            item["reason"] == "processing_error" for item in skipped_sources
        ),
        "skipped_sources": skipped_sources,
        "resumed_sources": resumed_sources,
        "recipe": args.recipe,
        "modern_variants_per_source": args.modern_variants,
        "historic_variants_per_source": args.historic_variants,
        "total_variants_requested": sources_count * (
            args.modern_variants + args.historic_variants
            + (1 if args.recipe == HARD_SYNTH_RECIPE else 0)
        ),
        "role_profiles": {
            "modern": ["catalogue", "framed_catalogue", "wall_photo", "perspective_photo"],
            "historic": ["archival", "print", "cropped_record", "framed_photo"],
        },
        "argv": sys.argv[1:],
        "args": jsonable(vars(args)),
        "wall_config": jsonable(asdict(wall_cfg)),
        "post_config": jsonable(asdict(post_cfg)),
        "runtime": {
            "workers": args.workers,
            "process_start_method": args.process_start_method,
            "native_threads_per_worker": int(_NATIVE_THREADS_PER_WORKER),
        },
        "reproducibility": {
            "generator_commit": _git_commit(),
            "generator_version": PROVENANCE_SCHEMA_VERSION,
            "random_seed": args.seed,
            "dimension_jitter_fraction": DIMENSION_JITTER_FRACTION,
            "max_output_dimension": args.max_output_dimension,
            "source_dataset_sha256": _sha256_path(args.dataset_root / "dataset.db")
            if source_kind == "dataset"
            else _image_directory_source_hash(args.dataset_root),
            "recipe_sha256": _sha256_path(args.recipe_config)
            if args.recipe == HARD_SYNTH_RECIPE else None,
            "generation_plan_sha256": _sha256_path(args.generation_plan)
            if args.generation_plan else None,
        },
    }
    if args.recipe == HARD_SYNTH_RECIPE:
        summary["resolved_recipe"] = _load_hard_recipe(args.recipe_config)
        plan = json.loads(args.generation_plan.read_text(encoding="utf-8"))
        registry = plan.get("asset_registry") or _asset_registry(
            args.frames_horizontal, args.overlays, args.wall_textures
        )
        summary["asset_registry"] = registry
        summary["reproducibility"]["asset_registry_sha256"] = plan.get(
            "asset_registry_hash"
        ) or "sha256:" + hashlib.sha256(
            json.dumps(registry, sort_keys=True).encode("utf-8")
        ).hexdigest()
    if source_summary is not None:
        summary["source_summary"] = source_summary
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_hard_recipe(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is in experiments requirements
        raise SystemExit("PyYAML is required for --recipe hard-synth-v1") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("recipe_id") != "hard_synth_v1":
        raise SystemExit(f"Invalid hard-synth recipe: {path}")
    counts = {
        profile: int((data.get("historic_profiles", {}).get(profile) or {}).get("variants_per_test_class", -1))
        for profile in HISTORIC_PROFILES
    }
    try:
        expand_profile_counts(counts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if any(count < 1 for count in counts.values()):
        raise SystemExit("hard-synth-v1 requires at least one variant of every historic profile")
    return data


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def _asset_registry(*directories: Path) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for directory in directories:
        asset_type = directory.name
        if not directory.exists():
            continue
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            digest = _sha256_path(path)
            registry[digest] = {"type": asset_type, "name": path.name}
    return registry


def _image_directory_source_hash(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from experiments.core.generation_plan import image_directory_hash

    return image_directory_hash(path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _paper_generation_plan(args: argparse.Namespace) -> dict:
    """Plan the paper recipe: medium severity on every subset, four historic profiles."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from experiments.core.generation_plan import make_image_directory_generation_plan

    source_kind = resolve_source_kind(args.dataset_root, args.source_kind)
    if source_kind != "image-dir":
        raise SystemExit(
            "--generation-plan is required when the source is a dataset database; "
            "an image directory uses the paper plan"
        )
    plan = make_image_directory_generation_plan(
        args.dataset_root, args.recipe_config, seed=args.seed
    )
    import csv

    plan["plan_id"] = "medium_o1m3h4_v1"
    plan["severity_policy"] = {subset: ["medium"] for subset in ("train", "val", "test")}
    split_csv = repo_root / "data/syn/class_split.csv"
    if split_csv.is_file():
        assigned = {
            row["parent_file_id"]: row["subset"]
            for row in csv.DictReader(split_csv.open(encoding="utf-8"))
        }
        if set(plan["class_assignments"]) <= set(assigned):
            plan["class_assignments"] = {
                key: assigned[key] for key in plan["class_assignments"]
            }
    return plan


def main() -> None:
    args = parse_args()
    if args.num_variants is not None:
        if args.historic_variants is not None:
            raise SystemExit("--num-variants cannot be combined with --historic-variants")
        args.historic_variants = args.num_variants
    elif args.historic_variants is None:
        args.historic_variants = 8
    if args.modern_variants < 0 or args.historic_variants < 0:
        raise SystemExit("--modern-variants and --historic-variants must be >= 0")
    if args.modern_variants + args.historic_variants < 1:
        raise SystemExit("At least one generated variant per source is required")
    if args.recipe == LEGACY_RECIPE:
        if args.modern_variants:
            print("warning: --modern-variants is ignored by the legacy recipe", file=sys.stderr)
        args.modern_variants = 0
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.max_output_dimension < 1:
        raise SystemExit("--max-output-dimension must be >= 1")
    if args.max_source_side < 0:
        raise SystemExit("--max-source-side must be >= 0")
    if args.max_input_dimension < 1:
        raise SystemExit("--max-input-dimension must be >= 1")
    if args.max_input_aspect_ratio < 1.0:
        raise SystemExit("--max-input-aspect-ratio must be >= 1")
    # Treat a 15000x15000 source as trusted. Larger dimensions are inspected by
    # header and skipped before pixel decoding in each worker.
    Image.MAX_IMAGE_PIXELS = max(15_000, args.max_input_dimension) ** 2
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.process_start_method != "fork" and args.workers > 16:
        print(
            "warning: spawn/forkserver cannot share preloaded image assets; "
            "prefer --process-start-method fork for large worker counts on Linux.",
            file=sys.stderr,
        )
    if not 0.0 <= args.wall_texture_chance <= 1.0:
        raise SystemExit("--wall-texture-chance must be between 0 and 1")
    if not 0.0 <= args.wall_texture_opacity <= 1.0:
        raise SystemExit("--wall-texture-opacity must be between 0 and 1")
    resolved_recipe = None
    generation_plan = None
    recipe_counts = dict(HISTORIC_PROFILE_COUNTS)
    if args.recipe == HARD_SYNTH_RECIPE:
        resolved_recipe = _load_hard_recipe(args.recipe_config)
        # Perspective bounds are part of the versioned benchmark recipe rather
        # than a user-selectable probability branch.
        args.difficulty = "hard"
        recipe_counts = {
            profile: int((resolved_recipe.get("historic_profiles", {}).get(profile) or {}).get("variants_per_test_class", -1))
            for profile in HISTORIC_PROFILES
        }
        if args.historic_variants != sum(recipe_counts.values()):
            raise SystemExit(
                "hard-synth-v1 --historic-variants must equal the recipe total "
                f"({sum(recipe_counts.values())})"
            )
        if args.generation_plan is None:
            generation_plan = _paper_generation_plan(args)
            import tempfile
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8"
            )
            json.dump(generation_plan, handle)
            handle.close()
            args.generation_plan = Path(handle.name)
        else:
            generation_plan = json.loads(args.generation_plan.read_text(encoding="utf-8"))
        if int(generation_plan.get("seed", -1)) != args.seed:
            raise SystemExit("generation plan seed does not match --seed")
    if args.recipe == LEGACY_RECIPE and args.historic_variants % 4 != 0:
        print(
            "warning: the historic variant count is not a multiple of 4; legacy "
            "archetypes will not be split exactly 25% each.", file=sys.stderr,
        )

    source_kind = resolve_source_kind(args.dataset_root, args.source_kind)
    if generation_plan is not None:
        actual_source_hash = (
            _sha256_path(args.dataset_root / "dataset.db")
            if source_kind == "dataset"
            else _image_directory_source_hash(args.dataset_root)
        )
        if generation_plan.get("source_dataset_hash") != actual_source_hash:
            raise SystemExit(
                "generation plan source hash does not match the supplied source dataset/directory"
            )
        expected_assignment_key = "class_id" if source_kind == "dataset" else "parent_file_id"
        if generation_plan.get("assignment_key", "class_id") != expected_assignment_key:
            raise SystemExit(
                f"generation plan expects {generation_plan.get('assignment_key')}, "
                f"but source kind {source_kind!r} requires {expected_assignment_key}"
            )
    if args.resume:
        if not (args.output_root / "dataset.db").is_file():
            raise SystemExit("--resume requires an existing output dataset.db")
        print(f"Resuming existing output: {args.output_root}", flush=True)
    else:
        ensure_output_root(args.output_root)
        if source_kind == "dataset":
            copy_limited_dataset_root(
                args.dataset_root, args.output_root, args.source_split, args.limit,
                args.workers,
            )
        else:
            dataset_name = args.dataset_name or args.dataset_root.resolve().name
            selected_count = len(flat_image_files(args.dataset_root, args.limit))
            copy_progress = ProgressLogger("copy", selected_count)
            create_image_dir_dataset(
                args.dataset_root,
                args.output_root,
                dataset_name,
                args.limit,
                copy_workers=args.workers,
                progress_callback=copy_progress.increment,
                max_input_dimension=args.max_input_dimension,
            )

    set_rng_base_seed(args.seed)
    source_summary = read_summary_json(args.output_root / "summary.json")
    wall_cfg, post_cfg = make_wall_config(args)
    # Fork workers can share preloaded, read-only assets copy-on-write. Other
    # start methods must load assets independently inside each process.
    if args.process_start_method == "fork":
        frames, overlays, textures = load_assets(
            args.frames_horizontal, args.overlays, args.wall_textures
        )
    else:
        frames = overlays = textures = None

    conn = sqlite3.connect(args.output_root / "dataset.db")
    conn.executescript(SCHEMA_PATCH)
    conn.executescript(RESUME_SCHEMA)
    skipped_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(synthetic_skipped_sources)")
    }
    if "error_message" not in skipped_columns:
        conn.execute(
            "ALTER TABLE synthetic_skipped_sources ADD COLUMN error_message TEXT"
        )
    dataset_name = conn.execute(
        "SELECT value FROM dataset_meta WHERE key='dataset_name'"
    ).fetchone()[0]
    config = vars(args) | {
        "generator_version": PROVENANCE_SCHEMA_VERSION,
        "resolved_source_kind": source_kind,
        "wall_config": asdict(wall_cfg),
        "post_config": asdict(post_cfg),
    }
    now = datetime.now(timezone.utc).isoformat()
    if args.resume:
        run_row = conn.execute(
            "SELECT synthetic_run_id, config_json FROM synthetic_runs "
            "WHERE tool='generate_synth' AND name=? ORDER BY synthetic_run_id DESC LIMIT 1",
            (args.name,),
        ).fetchone()
        if run_row is None:
            raise SystemExit(f"No resumable synthetic run named {args.name!r} was found")
        run_id = int(run_row[0])
        previous_config = json.loads(run_row[1])
        if previous_config.get("generator_version") != PROVENANCE_SCHEMA_VERSION:
            raise SystemExit(
                "Cannot resume a run created by a different generator version; "
                "create a new versioned output dataset instead"
            )
        for key in (
            "recipe", "modern_variants", "historic_variants", "seed", "max_output_dimension",
            "max_input_dimension", "max_input_aspect_ratio", "source_split", "limit",
            "recipe_config", "generation_plan", "wall_config", "post_config",
        ):
            raw_current_value = (
                config.get(key)
                if key in {"wall_config", "post_config"}
                else getattr(args, key)
            )
            # Stored configs have passed through JSON, which converts tuples
            # such as shadow offsets and ranges into lists.
            current_value = jsonable(raw_current_value)
            if key in previous_config and previous_config.get(key) != current_value:
                raise SystemExit(
                    f"Cannot resume: {key.replace('_', '-')} differs from run "
                    f"({current_value!r} != {previous_config.get(key)!r})"
                )
    else:
        run_id = conn.execute(
            "INSERT INTO synthetic_runs(name, tool, config_json, source_dataset_name, created_at) VALUES(?, ?, ?, ?, ?)",
            (
                args.name,
                "generate_synth",
                json.dumps(config, default=str, sort_keys=True),
                dataset_name,
                now,
            ),
        ).lastrowid
    sources = fetch_source_classes(conn, args.source_split, args.limit, run_id)
    class_assignments = (generation_plan or {}).get("class_assignments", {})
    assignment_key = (generation_plan or {}).get("assignment_key", "class_id")

    def assignment_id(row: sqlite3.Row) -> str:
        return str(row["file_id"] if assignment_key == "parent_file_id" else row["class_id"])

    def assignment_for(row: sqlite3.Row) -> str:
        return class_assignments.get(assignment_id(row), "unassigned")

    asset_assignments = (generation_plan or {}).get("asset_assignments", {})
    severity_policy = (generation_plan or {}).get("severity_policy", {})
    profile_policy = (generation_plan or {}).get("profile_policy", {})
    if generation_plan is not None:
        missing_classes = sorted(
            assignment_id(row) for row in sources
            if assignment_id(row) not in class_assignments
        )
        if missing_classes:
            raise SystemExit(f"generation plan misses source classes: {missing_classes[:10]}")
    derivation_counts = {
        str(parent): int(count)
        for parent, count in conn.execute(
            "SELECT parent_file_id, COUNT(*) FROM image_file_derivations "
            "WHERE synthetic_run_id=? GROUP BY parent_file_id",
            (run_id,),
        )
    }
    completed_parent_ids = set()
    for row in sources:
        expected_variants = args.modern_variants + args.historic_variants
        if args.recipe == HARD_SYNTH_RECIPE:
            subset = class_assignments[assignment_id(row)]
            allowed_profiles = profile_policy.get(subset, list(HISTORIC_PROFILES))
            expected_variants = 1 + args.modern_variants + sum(
                recipe_counts[profile] for profile in allowed_profiles
            )
        if derivation_counts.get(str(row["file_id"]), 0) >= expected_variants:
            completed_parent_ids.add(str(row["file_id"]))
    skipped_parent_ids = {
        row[0]
        for row in conn.execute(
            "SELECT parent_file_id FROM synthetic_skipped_sources WHERE synthetic_run_id=?",
            (run_id,),
        )
    }
    already_done_ids = completed_parent_ids | skipped_parent_ids
    existing_generated_count = conn.execute(
        "SELECT COUNT(*) FROM image_file_derivations WHERE synthetic_run_id=?",
        (run_id,),
    ).fetchone()[0]
    conn.commit()
    conn.close()

    tasks = [
        {
            "parent_file_id": row["file_id"],
            "src_rel_path": row["local_rel_path"],
            "class_id": row["class_id"],
            "reviewed_at": row["reviewed_at"],
            "in_query": bool(row["in_query"]),
            "in_candidate": bool(row["in_candidate"]),
            "tags_csv": row["tags_csv"],
            "english_description": row["english_description"],
            "description_source": row["description_source"],
            "recipe": args.recipe,
            "modern_variants": args.modern_variants,
            "historic_variants": args.historic_variants,
            "max_output_dimension": args.max_output_dimension,
            "max_input_dimension": args.max_input_dimension,
            "max_input_aspect_ratio": args.max_input_aspect_ratio,
            "seed": args.seed,
            "historic_profile_counts": {
                profile: count
                for profile, count in recipe_counts.items()
                if profile in profile_policy.get(
                    assignment_for(row), list(HISTORIC_PROFILES)
                )
            },
            "subset": assignment_for(row),
            "allowed_asset_hashes": [
                digest for digest, subset in asset_assignments.items()
                if subset == assignment_for(row)
            ],
            "severity_policy": severity_policy.get(
                assignment_for(row), ["hard", "extreme"]
            ),
        }
        for row in sources
        if row["file_id"] not in already_done_ids
    ]

    if args.process_start_method == "fork":
        _set_worker_state(
            args.output_root, wall_cfg, post_cfg, frames, overlays, textures, run_id, now
        )

    generated_count = int(existing_generated_count)
    resized_originals = 0
    repaired_originals = 0
    skipped_sources_count = len(skipped_parent_ids)
    completed_sources = 0
    initial_done = len(already_done_ids)
    progress = ProgressLogger(
        "generate", len(sources), initial_done=initial_done
    )
    conn = sqlite3.connect(args.output_root / "dataset.db")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        if tasks:
            ctx = mp.get_context(args.process_start_method)
            initargs = (
                args.output_root,
                wall_cfg,
                post_cfg,
                args.frames_horizontal,
                args.overlays,
                args.wall_textures,
                run_id,
                now,
                args.seed,
                args.max_input_dimension,
            )
            # Keep only a small window of source jobs and returned metadata in
            # memory. Completed batches are inserted into SQLite immediately.
            max_pending = max(args.workers, args.workers * 2)
            task_iterator = iter(tasks)
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=ctx,
                initializer=init_worker_state,
                initargs=initargs,
            ) as pool:
                pending: dict = {}
                for _ in range(min(max_pending, len(tasks))):
                    task = next(task_iterator)
                    pending[pool.submit(worker, task)] = task

                while pending:
                    finished, _ = wait(
                        set(pending), timeout=progress.interval,
                        return_when=FIRST_COMPLETED,
                    )
                    if not finished:
                        progress.log()
                        continue
                    for future in finished:
                        task = pending.pop(future)
                        try:
                            batch = future.result()
                        except Exception as exc:
                            remove_partial_source_outputs(
                                args.output_root, task["parent_file_id"]
                            )
                            batch = failed_source_batch(task, run_id, now, exc)
                        generated, resized, repaired, skipped = insert_sql_batch(conn, batch)
                        generated_count += generated
                        resized_originals += resized
                        repaired_originals += repaired
                        skipped_sources_count += skipped
                        completed_sources += 1
                        progress.increment()
                        if completed_sources % 100 == 0:
                            conn.commit()
                        try:
                            task = next(task_iterator)
                            pending[pool.submit(worker, task)] = task
                        except StopIteration:
                            pass
        conn.commit()
        skipped_sources = [
            {
                "file_id": row[0], "local_rel_path": row[1], "reason": row[2],
                "width": row[3], "height": row[4], "error": row[5],
            }
            for row in conn.execute(
                "SELECT ss.parent_file_id, f.local_rel_path, ss.reason, ss.width, "
                "ss.height, ss.error_message FROM synthetic_skipped_sources ss "
                "LEFT JOIN image_files f ON f.file_id=ss.parent_file_id "
                "WHERE ss.synthetic_run_id=? ORDER BY ss.parent_file_id",
                (run_id,),
            )
        ]
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    write_summary(
        args.output_root,
        args,
        source_kind,
        dataset_name,
        run_id,
        now,
        len(sources),
        generated_count,
        wall_cfg,
        post_cfg,
        source_summary,
        skipped_sources_count,
        initial_done,
        skipped_sources,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "source_kind": source_kind,
                "synthetic_run_id": run_id,
                "source_files": len(sources),
                "generated_files": generated_count,
                "resized_originals": resized_originals,
                "repaired_truncated_originals": repaired_originals,
                "skipped_sources": skipped_sources_count,
                "skipped_oversized_sources": sum(
                    item["reason"] in {"dimension_limit", "aspect_ratio_limit"}
                    for item in skipped_sources
                ),
                "skipped_processing_errors": sum(
                    item["reason"] == "processing_error" for item in skipped_sources
                ),
                "resumed_sources": initial_done,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
