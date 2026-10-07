#!/usr/bin/env python3
"""Create or extend filename-keyed dataset DBs with wall-placement variants."""
from __future__ import annotations

import argparse
import json
import mimetypes
import multiprocessing as mp
import re
import shutil
import sqlite3
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageFile
from wall_placement import (
    DIFFICULTY_PRESETS,
    PostProcessConfig,
    WallConfig,
    load_assets,
    process_single_image,
)
from utils import IMAGE_EXTENSIONS, set_rng_base_seed, set_worker_rng_index
from utils.worker_utils import DEFAULT_STATIC_SEED

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FRAMES_DIR = BASE_DIR / "horizontal_frames"
DEFAULT_OVERLAYS_DIR = BASE_DIR / "reflection_overlay"
DEFAULT_WALL_TEXTURES_DIR = BASE_DIR / "wall_textures"

DATASET_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE dataset_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE classes(class_id INTEGER PRIMARY KEY, qid TEXT NOT NULL UNIQUE, label TEXT, painting_inception_year INTEGER, sitelinks INTEGER);
CREATE TABLE image_files(file_id TEXT PRIMARY KEY, file_ext TEXT NOT NULL, canonical_title TEXT, file_page_url TEXT, source_url TEXT, download_url TEXT, local_rel_path TEXT, download_status TEXT NOT NULL, mime_type TEXT, width INTEGER, height INTEGER, bytes_on_disk INTEGER NOT NULL DEFAULT 0, year INTEGER, epoch_bucket TEXT, source_role TEXT, english_description TEXT, description_source TEXT, has_p6243 INTEGER, is_photo INTEGER, p180_count INTEGER, pair_count INTEGER, historical_count INTEGER, modern_count INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE image_file_classes(image_file_class_id INTEGER PRIMARY KEY AUTOINCREMENT, file_id TEXT NOT NULL REFERENCES image_files(file_id) ON DELETE CASCADE, class_id INTEGER NOT NULL REFERENCES classes(class_id), primary_label TEXT, reviewed_at TEXT, updated_at TEXT NOT NULL, UNIQUE(file_id, class_id));
CREATE TABLE image_file_class_tags(image_file_class_id INTEGER NOT NULL REFERENCES image_file_classes(image_file_class_id) ON DELETE CASCADE, tag TEXT NOT NULL, PRIMARY KEY(image_file_class_id, tag));
CREATE TABLE splits(split_name TEXT NOT NULL, file_id TEXT NOT NULL REFERENCES image_files(file_id) ON DELETE CASCADE, PRIMARY KEY(split_name, file_id));
CREATE TABLE synthetic_runs(synthetic_run_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, tool TEXT NOT NULL, config_json TEXT NOT NULL, source_dataset_name TEXT, created_at TEXT NOT NULL);
CREATE TABLE image_file_derivations(child_file_id TEXT PRIMARY KEY REFERENCES image_files(file_id) ON DELETE CASCADE, parent_file_id TEXT NOT NULL REFERENCES image_files(file_id), synthetic_run_id INTEGER NOT NULL REFERENCES synthetic_runs(synthetic_run_id) ON DELETE CASCADE, transform_family TEXT NOT NULL, transform_params_json TEXT NOT NULL);
CREATE UNIQUE INDEX uq_one_same_painting_per_file ON image_file_classes(file_id) WHERE primary_label='same_painting';
CREATE INDEX idx_ifc_class_id ON image_file_classes(class_id);
CREATE INDEX idx_ifc_primary_label ON image_file_classes(primary_label);
CREATE INDEX idx_ifct_tag ON image_file_class_tags(tag);
CREATE INDEX idx_splits_name ON splits(split_name);
CREATE INDEX idx_deriv_parent ON image_file_derivations(parent_file_id);
CREATE INDEX idx_deriv_run ON image_file_derivations(synthetic_run_id);
"""

SCHEMA_PATCH = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS synthetic_runs(synthetic_run_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, tool TEXT NOT NULL, config_json TEXT NOT NULL, source_dataset_name TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS image_file_derivations(child_file_id TEXT PRIMARY KEY REFERENCES image_files(file_id) ON DELETE CASCADE, parent_file_id TEXT NOT NULL REFERENCES image_files(file_id), synthetic_run_id INTEGER NOT NULL REFERENCES synthetic_runs(synthetic_run_id) ON DELETE CASCADE, transform_family TEXT NOT NULL, transform_params_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_deriv_parent ON image_file_derivations(parent_file_id);
CREATE INDEX IF NOT EXISTS idx_deriv_run ON image_file_derivations(synthetic_run_id);
"""

FILE_ID_CLEANUP = re.compile(r"[^A-Za-z0-9._-]+")
ROLE_TAGS = {"historic", "historical", "modern"}

_WORKER_OUT_ROOT: Path | None = None
_WORKER_WALL_CFG: WallConfig | None = None
_WORKER_POST_CFG: PostProcessConfig | None = None
_WORKER_FRAMES = None
_WORKER_OVERLAYS = None
_WORKER_TEXTURES = None
_WORKER_RUN_ID: int | None = None
_WORKER_NOW: str | None = None


def parse_wall_size(value: str) -> tuple[int, int]:
    w, h = value.lower().split("x")
    return int(w), int(h)


def parse_range(value: str, cast=float) -> tuple:
    a, b = value.split(",")
    return cast(a), cast(b)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate DB-backed synthetic wall variants from a dataset DB or flat image directory"
    )
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument(
        "--source-kind",
        choices=("auto", "dataset", "image-dir"),
        default="auto",
        help="Input type. auto uses dataset.db when present, otherwise a flat image directory.",
    )
    ap.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name to write when bootstrapping from --source-kind image-dir.",
    )
    ap.add_argument("--source-split", default="candidate_default")
    ap.add_argument("--name", default="place_on_wall")
    ap.add_argument("--num-variants", type=int, default=2)
    ap.add_argument(
        "--composition-num-variants",
        type=int,
        default=None,
        help=(
            "Number of variants per source that receive the strong perspective "
            "and depth wall warp. All variants still receive wall composition. "
            "Defaults to --num-variants."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of process workers.",
    )
    process_start_methods = mp.get_all_start_methods()
    ap.add_argument(
        "--process-start-method",
        choices=process_start_methods,
        default="fork" if "fork" in process_start_methods else process_start_methods[0],
        help="Multiprocessing start method for process workers.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_STATIC_SEED,
        help="Base RNG seed. Worker seeds are base_seed * one_based_worker_index.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of source images to use. For image-dir sources, "
            "images are sorted alphabetically and only the first N are copied "
            "into the output DB/dataset. 0 means no limit."
        ),
    )
    ap.add_argument(
        "--frames-horizontal",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
    )
    ap.add_argument(
        "--overlays",
        type=Path,
        default=DEFAULT_OVERLAYS_DIR,
    )
    ap.add_argument(
        "--wall-textures",
        type=Path,
        default=DEFAULT_WALL_TEXTURES_DIR,
    )
    ap.add_argument(
        "--wall-texture-chance",
        type=float,
        default=0.7,
        help=(
            "Probability of applying an image-based wall texture when wall "
            "textures are loaded."
        ),
    )
    ap.add_argument(
        "--wall-texture-opacity",
        type=float,
        default=0.4,
        help=(
            "Opacity/strength for image-based wall textures"
        ),
    )
    ap.add_argument(
        "--difficulty", choices=sorted(DIFFICULTY_PRESETS), default="medium"
    )
    ap.add_argument("--wall-size", type=parse_wall_size, default=(1920, 1080))
    ap.add_argument("--post-noise", type=float, default=0.0)
    ap.add_argument("--post-reflection", type=float, default=0.0)
    ap.add_argument(
        "--post-saturation", type=lambda s: parse_range(s, float), default=None
    )
    ap.add_argument("--vintage", action="store_true")
    ap.add_argument("--vintage-blur", type=float, default=0.8)
    ap.add_argument("--vintage-noise", type=float, default=0.04)
    ap.add_argument("--glare", type=float, default=0.0)
    ap.add_argument("--glare-spots", type=lambda s: parse_range(s, int), default=(1, 3))
    ap.add_argument(
        "--glare-size", type=lambda s: parse_range(s, float), default=(0.1, 0.4)
    )
    args = ap.parse_args()
    if args.composition_num_variants is None:
        args.composition_num_variants = args.num_variants
    return args


def resolve_source_kind(src_root: Path, requested: str) -> str:
    if not src_root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {src_root}")
    has_db = (src_root / "dataset.db").is_file()
    if requested == "auto":
        return "dataset" if has_db else "image-dir"
    if requested == "dataset" and not has_db:
        raise SystemExit(f"Dataset source requires dataset.db: {src_root / 'dataset.db'}")
    return requested


def ensure_output_root(out_root: Path) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output root already exists and is not empty: {out_root}")
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "embeddings").mkdir(parents=True, exist_ok=True)


def copy_dataset_root(src_root: Path, out_root: Path) -> None:
    shutil.copy2(src_root / "dataset.db", out_root / "dataset.db")
    if (src_root / "summary.json").exists():
        shutil.copy2(src_root / "summary.json", out_root / "summary.json")
    for src in (src_root / "images").glob("*"):
        dst = out_root / "images" / src.name
        if src.is_file():
            shutil.copy2(src, dst)


def read_summary_json(path: Path) -> dict | None:
    """Read an existing summary.json, raising clearly if it is invalid JSON."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in existing summary file {path}: {exc}") from exc


def jsonable(value):
    """Convert argparse/config values into JSON-safe summary values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value)]
    return value


def write_dataset_summary(
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
) -> None:
    """Write the output dataset summary, including all resolved CLI args."""
    summary = {
        "dataset_name": dataset_name,
        "output_root": str(out_root),
        "source_root": str(args.dataset_root),
        "source_kind": source_kind,
        "synthetic_run_id": run_id,
        "run_name": args.name,
        "tool": "place_on_wall",
        "created_at_utc": now,
        "source_files": sources_count,
        "generated_files": generated_count,
        "total_variants_requested": sources_count * args.num_variants,
        "composition_variants_requested": sources_count * args.num_variants,
        "strong_perspective_variants_requested": sources_count
        * args.composition_num_variants,
        "flat_wall_variants_requested": sources_count
        * (args.num_variants - args.composition_num_variants),
        "argv": sys.argv[1:],
        "args": jsonable(vars(args)),
        "wall_config": jsonable(asdict(wall_cfg)),
        "post_config": jsonable(asdict(post_cfg)),
    }
    if source_summary is not None:
        summary["source_summary"] = source_summary
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def flat_image_files(src_root: Path, limit: int = 0) -> list[Path]:
    files = sorted(
        (
            path
            for path in src_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not files:
        exts = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise SystemExit(f"No supported image files found in {src_root} ({exts})")
    return files[:limit] if limit > 0 else files


def safe_file_id(path: Path, used: set[str]) -> str:
    base = FILE_ID_CLEANUP.sub("_", path.stem.strip()).strip("._-") or "image"
    candidate = base
    idx = 2
    while candidate in used:
        candidate = f"{base}_{idx:03d}"
        idx += 1
    used.add(candidate)
    return candidate


_TRUNCATED_IMAGE_LOCK = threading.Lock()


def read_image_size(
    path: Path, max_dimension: int | None = None
) -> tuple[int, int]:
    """Read dimensions, avoiding pixel decode above an optional dimension cap."""
    if max_dimension is not None:
        # Pillow runs its area bomb check during Image.open(), before callers can
        # inspect dimensions. Disable it only for header inspection; oversized
        # files are returned without decoding any pixels.
        with _TRUNCATED_IMAGE_LOCK:
            previous = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            try:
                with Image.open(path) as image:
                    size = image.size
            finally:
                Image.MAX_IMAGE_PIXELS = previous
        if max(size) > max_dimension:
            return size
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except OSError as exc:
        if "image file is truncated" not in str(exc).lower():
            raise SystemExit(f"Could not read image file {path}: {exc}") from exc

    # LOAD_TRUNCATED_IMAGES is process-global, so serialize the temporary change
    # while image-directory preprocessing uses an I/O thread pool.
    with _TRUNCATED_IMAGE_LOCK:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as image:
                image.load()
                size = image.size
        except Exception as exc:
            raise SystemExit(
                f"Could not recover truncated image file {path}: {exc}"
            ) from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
    print(
        f"warning: recovered truncated image while reading dimensions: {path}",
        file=sys.stderr,
        flush=True,
    )
    return size


def create_image_dir_dataset(
    src_root: Path,
    out_root: Path,
    dataset_name: str,
    limit: int = 0,
    copy_workers: int = 1,
    progress_callback: Callable[[], None] | None = None,
    max_input_dimension: int | None = None,
) -> None:
    images = flat_image_files(src_root, limit)
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(out_root / "dataset.db")
    conn.executescript(DATASET_SCHEMA)
    meta = {
        "dataset_name": dataset_name,
        "created_at_utc": now,
        "builder": "experiments/datasets/image-distorter/place_on_wall_dataset.py",
        "source_kind": "image-dir",
        "source_dir": str(src_root),
        "image_root": "images",
        "embeddings_root": "embeddings",
        "file_id_policy": "sanitized_filename_stem",
        "class_policy": "one_class_per_source_image",
        "source_file_limit": str(limit) if limit > 0 else "unlimited",
        "source_files_selected": str(len(images)),
    }
    conn.executemany("INSERT INTO dataset_meta(key, value) VALUES(?, ?)", meta.items())

    used_ids: set[str] = set()
    specs = []
    for class_id, src in enumerate(images, start=1):
        file_id = safe_file_id(src, used_ids)
        file_ext = src.suffix.lower()
        rel_path = f"images/{file_id}{file_ext}"
        specs.append((class_id, src, file_id, file_ext, rel_path))

    def inspect_and_copy(spec):
        _, src, _, _, rel_path = spec
        width, height = read_image_size(src, max_input_dimension)
        dst = out_root / rel_path
        shutil.copy2(src, dst)
        return width, height, dst.stat().st_size

    materialized = []
    with ThreadPoolExecutor(max_workers=max(1, min(copy_workers, 32))) as pool:
        for result in pool.map(inspect_and_copy, specs):
            materialized.append(result)
            if progress_callback is not None:
                progress_callback()

    class_rows, file_rows, rel_rows, split_rows = [], [], [], set()
    for spec, (width, height, bytes_on_disk) in zip(specs, materialized):
        class_id, src, file_id, file_ext, rel_path = spec
        class_rows.append((class_id, f"LOCAL-{class_id:06d}", src.stem, None, None))
        file_rows.append(
            (
                file_id,
                file_ext,
                src.stem,
                None,
                None,
                None,
                rel_path,
                "downloaded",
                mimetypes.guess_type(src.name)[0],
                width,
                height,
                bytes_on_disk,
                None,
                None,
                "modern",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                now,
                now,
            )
        )
        rel_rows.append((file_id, class_id, "same_painting", now, now))
        split_rows.update(
            {
                ("all_files", file_id),
                ("downloaded", file_id),
                ("same_painting", file_id),
                ("query_default", file_id),
                ("candidate_default", file_id),
                ("modern", file_id),
            }
        )

    conn.executemany(
        "INSERT INTO classes(class_id, qid, label, painting_inception_year, sitelinks) VALUES(?, ?, ?, ?, ?)",
        class_rows,
    )
    conn.executemany(
        "INSERT INTO image_files(file_id, file_ext, canonical_title, file_page_url, source_url, download_url, local_rel_path, download_status, mime_type, width, height, bytes_on_disk, year, epoch_bucket, source_role, english_description, description_source, has_p6243, is_photo, p180_count, pair_count, historical_count, modern_count, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        file_rows,
    )
    conn.executemany(
        "INSERT INTO image_file_classes(file_id, class_id, primary_label, reviewed_at, updated_at) VALUES(?, ?, ?, ?, ?)",
        rel_rows,
    )
    rel_ids = [
        row[0]
        for row in conn.execute("SELECT image_file_class_id FROM image_file_classes")
    ]
    conn.executemany(
        "INSERT INTO image_file_class_tags(image_file_class_id, tag) VALUES(?, ?)",
        [(rel_id, "modern") for rel_id in rel_ids],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO splits(split_name, file_id) VALUES(?, ?)",
        sorted(split_rows),
    )
    conn.commit()
    conn.close()


def fetch_sources(
    conn: sqlite3.Connection, split_name: str, limit: int
) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT f.file_id, f.file_ext, f.local_rel_path, f.english_description, f.description_source, ifc.class_id, ifc.reviewed_at,
               EXISTS(SELECT 1 FROM splits s WHERE s.file_id=f.file_id AND s.split_name='query_default') AS in_query,
               EXISTS(SELECT 1 FROM splits s WHERE s.file_id=f.file_id AND s.split_name='candidate_default') AS in_candidate,
               COALESCE((SELECT GROUP_CONCAT(tag, ',') FROM image_file_class_tags t WHERE t.image_file_class_id=ifc.image_file_class_id), '') AS tags_csv
        FROM image_files f
        JOIN image_file_classes ifc ON ifc.file_id=f.file_id
        JOIN splits s0 ON s0.file_id=f.file_id AND s0.split_name=?
        WHERE f.download_status IN ('downloaded','generated') AND f.local_rel_path IS NOT NULL AND ifc.primary_label='same_painting'
        ORDER BY f.file_id
        """,
        (split_name,),
    ).fetchall()
    return rows[:limit] if limit > 0 else rows


def derived_tags(
    parent_tags: Iterable[str],
    args: argparse.Namespace,
    has_frames: bool | None = None,
    has_perspective: bool = True,
) -> list[str]:
    tags = set()
    for tag in parent_tags:
        clean = tag.strip()
        if clean and clean not in ROLE_TAGS:
            tags.add(clean)
    tags.add("historic")
    include_frame_tag = bool(args.frames_horizontal) if has_frames is None else has_frames
    if include_frame_tag:
        tags.add("frame_included")
    if has_perspective:
        tags.add("perspective")
    if args.vintage or args.post_noise > 0:
        tags.add("poor_quality")
    return sorted(tags)


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
        vintage_mode=args.vintage,
        vintage_blur_radius=args.vintage_blur,
        vintage_noise_std=args.vintage_noise,
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
    """Store immutable worker inputs globally to avoid per-task asset pickling."""
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


def current_process_worker_index() -> int | None:
    """Return the one-based multiprocessing worker index when available."""
    process = mp.current_process()
    identity = getattr(process, "_identity", ())
    if identity:
        return int(identity[0])
    match = re.search(r"-(\d+)$", process.name)
    return int(match.group(1)) if match else None


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
) -> None:
    """Initialize process workers with deterministic RNG and shared state."""
    set_rng_base_seed(base_seed)
    worker_index = current_process_worker_index()
    if worker_index is not None:
        set_worker_rng_index(worker_index)

    frames = _WORKER_FRAMES
    overlays = _WORKER_OVERLAYS
    textures = _WORKER_TEXTURES
    if frames is None or overlays is None or textures is None:
        frames, overlays, textures = load_assets(
            frames_dir, overlays_dir, textures_dir, verbose=False
        )
    _set_worker_state(
        out_root, wall_cfg, post_cfg, frames, overlays, textures, run_id, now
    )


def empty_sql_batch() -> dict[str, list]:
    """Return a container for SQLite rows accumulated by one worker."""
    return {
        "file_rows": [],
        "rel_rows": [],
        "tag_specs": [],
        "deriv_rows": [],
        "split_rows": [],
    }


def merge_sql_batch(target: dict[str, list], source: dict[str, list]) -> None:
    """Merge one worker's accumulated SQLite rows into the final batch."""
    for key, rows in source.items():
        target[key].extend(rows)


def chunks(items: list, size: int) -> Iterable[list]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def worker(task: dict) -> dict[str, list]:
    """Generate image variants and return this worker's SQLite row batch."""
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

    src_path = _WORKER_OUT_ROOT / task["src_rel_path"]
    with Image.open(src_path) as image:
        base = image.convert("RGBA")

    batch = empty_sql_batch()
    for variant_idx in range(task["num_variants"]):
        strong_perspective = variant_idx < task["strong_perspective_num_variants"]
        result, frame_name, wall_name = process_single_image(
            base,
            _WORKER_WALL_CFG,
            _WORKER_FRAMES,
            _WORKER_OVERLAYS,
            _WORKER_TEXTURES,
            True,
            _WORKER_POST_CFG,
            apply_wall_composition=True,
            apply_strong_perspective=strong_perspective,
        )
        child_file_id = (
            f"{task['parent_file_id']}_{frame_name}_{wall_name}-v{variant_idx + 1:03d}"
        )
        rel_path = f"images/{child_file_id}.jpg"
        dst = _WORKER_OUT_ROOT / rel_path
        result.convert("RGB").save(dst, "JPEG", quality=92)
        tags = (
            task["strong_perspective_tags"]
            if strong_perspective
            else task["flat_wall_tags"]
        )
        batch["file_rows"].append(
            (
                child_file_id,
                ".jpg",
                f"{child_file_id}.jpg",
                None,
                None,
                None,
                rel_path,
                "generated",
                "image/jpeg",
                result.width,
                result.height,
                dst.stat().st_size,
                None,
                None,
                "historic",
                task["english_description"],
                task["description_source"],
                None,
                None,
                None,
                None,
                None,
                None,
                _WORKER_NOW,
                _WORKER_NOW,
            )
        )
        batch["rel_rows"].append(
            (
                child_file_id,
                task["class_id"],
                "same_painting",
                task["reviewed_at"],
                _WORKER_NOW,
            )
        )
        batch["deriv_rows"].append(
            (
                child_file_id,
                task["parent_file_id"],
                _WORKER_RUN_ID,
                "place_on_wall",
                json.dumps(
                    {
                        "frame": frame_name,
                        "wall": wall_name,
                        "wall_composition": True,
                        "strong_perspective": strong_perspective,
                    },
                    sort_keys=True,
                ),
            )
        )
        for tag in tags:
            batch["tag_specs"].append((child_file_id, task["class_id"], tag))
        batch["split_rows"].extend(
            [
                ("all_files", child_file_id),
                ("generated", child_file_id),
                ("synthetic_only", child_file_id),
                ("augmented_all", child_file_id),
                ("same_painting", child_file_id),
                ("historic", child_file_id),
            ]
        )
        if task["in_query"]:
            batch["split_rows"].extend(
                [("query_synthetic", child_file_id), ("query_augmented", child_file_id)]
            )
        if task["in_candidate"]:
            batch["split_rows"].extend(
                [
                    ("candidate_synthetic", child_file_id),
                    ("candidate_augmented", child_file_id),
                ]
            )
    return batch


def main() -> None:
    args = parse_args()
    if args.num_variants < 0:
        raise SystemExit("--num-variants must be >= 0")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.composition_num_variants < 0:
        raise SystemExit("--composition-num-variants must be >= 0")
    if args.composition_num_variants > args.num_variants:
        raise SystemExit("--composition-num-variants must be <= --num-variants")
    if not 0.0 <= args.wall_texture_chance <= 1.0:
        raise SystemExit("--wall-texture-chance must be between 0 and 1")
    if not 0.0 <= args.wall_texture_opacity <= 1.0:
        raise SystemExit("--wall-texture-opacity must be between 0 and 1")
    source_kind = resolve_source_kind(args.dataset_root, args.source_kind)
    ensure_output_root(args.output_root)
    if source_kind == "dataset":
        copy_dataset_root(args.dataset_root, args.output_root)
    else:
        dataset_name = args.dataset_name or args.dataset_root.resolve().name
        create_image_dir_dataset(
            args.dataset_root, args.output_root, dataset_name, args.limit
        )
    set_rng_base_seed(args.seed)
    source_summary = read_summary_json(args.output_root / "summary.json")
    wall_cfg, post_cfg = make_wall_config(args)
    frames, overlays, textures = load_assets(
        args.frames_horizontal, args.overlays, args.wall_textures
    )

    conn = sqlite3.connect(args.output_root / "dataset.db")
    conn.executescript(SCHEMA_PATCH)
    dataset_name = conn.execute(
        "SELECT value FROM dataset_meta WHERE key='dataset_name'"
    ).fetchone()[0]
    config = vars(args) | {
        "resolved_source_kind": source_kind,
        "wall_config": asdict(wall_cfg),
        "post_config": asdict(post_cfg),
    }
    now = datetime.now(timezone.utc).isoformat()
    run_id = conn.execute(
        "INSERT INTO synthetic_runs(name, tool, config_json, source_dataset_name, created_at) VALUES(?, ?, ?, ?, ?)",
        (
            args.name,
            "place_on_wall",
            json.dumps(config, default=str, sort_keys=True),
            dataset_name,
            now,
        ),
    ).lastrowid
    sources = fetch_sources(conn, args.source_split, args.limit)
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
            "strong_perspective_tags": derived_tags(
                (row["tags_csv"] or "").split(","),
                args,
                bool(frames),
                has_perspective=True,
            ),
            "flat_wall_tags": derived_tags(
                (row["tags_csv"] or "").split(","),
                args,
                bool(frames),
                has_perspective=False,
            ),
            "strong_perspective_num_variants": args.composition_num_variants,
            "english_description": row["english_description"],
            "description_source": row["description_source"],
            "num_variants": args.num_variants,
        }
        for row in sources
    ]

    _set_worker_state(
        args.output_root, wall_cfg, post_cfg, frames, overlays, textures, run_id, now
    )
    sql_batch = empty_sql_batch()
    worker_count = args.workers
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
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=init_worker_state,
            initargs=initargs,
        ) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for future in as_completed(futures):
                merge_sql_batch(sql_batch, future.result())

    file_rows = sorted(sql_batch["file_rows"], key=lambda row: row[0])
    rel_rows = sorted(sql_batch["rel_rows"], key=lambda row: row[0])
    tag_specs = sorted(sql_batch["tag_specs"], key=lambda row: (row[0], row[1], row[2]))
    deriv_rows = sorted(sql_batch["deriv_rows"], key=lambda row: row[0])
    split_rows = sorted(sql_batch["split_rows"], key=lambda row: (row[1], row[0]))

    conn = sqlite3.connect(args.output_root / "dataset.db")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executemany(
            "INSERT INTO image_files(file_id, file_ext, canonical_title, file_page_url, source_url, download_url, local_rel_path, download_status, mime_type, width, height, bytes_on_disk, year, epoch_bucket, source_role, english_description, description_source, has_p6243, is_photo, p180_count, pair_count, historical_count, modern_count, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            file_rows,
        )
        conn.executemany(
            "INSERT INTO image_file_classes(file_id, class_id, primary_label, reviewed_at, updated_at) VALUES(?, ?, ?, ?, ?)",
            rel_rows,
        )
        rel_map = {}
        file_ids = [row[0] for row in file_rows]
        for file_id_chunk in chunks(file_ids, 900):
            placeholders = ",".join("?" for _ in file_id_chunk)
            rel_map.update(
                {
                    (file_id, class_id): rel_id
                    for rel_id, file_id, class_id in conn.execute(
                        f"SELECT image_file_class_id, file_id, class_id FROM image_file_classes WHERE file_id IN ({placeholders})",
                        file_id_chunk,
                    )
                }
            )
        tag_rows = [
            (rel_map[(file_id, class_id)], tag)
            for file_id, class_id, tag in tag_specs
        ]
        conn.executemany(
            "INSERT INTO image_file_class_tags(image_file_class_id, tag) VALUES(?, ?)",
            tag_rows,
        )
        conn.executemany(
            "INSERT INTO image_file_derivations(child_file_id, parent_file_id, synthetic_run_id, transform_family, transform_params_json) VALUES(?, ?, ?, ?, ?)",
            deriv_rows,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO splits(split_name, file_id) VALUES(?, ?)", split_rows
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    write_dataset_summary(
        args.output_root,
        args,
        source_kind,
        dataset_name,
        run_id,
        now,
        len(sources),
        len(file_rows),
        wall_cfg,
        post_cfg,
        source_summary,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "source_kind": source_kind,
                "synthetic_run_id": run_id,
                "source_files": len(sources),
                "generated_files": len(file_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
