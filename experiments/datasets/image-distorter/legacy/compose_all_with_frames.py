import argparse
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# Static seed for deterministic frame selection
DEFAULT_STATIC_SEED = 1337

# Thread-safe progress counter
_progress_lock = threading.Lock()
_progress_counter = 0
_progress_total = 0
_progress_start_time = 0

# Usage:
# python compose_all_with_frames.py <frames_folder> <photos_folder>
# python compose_all_with_frames.py <frames_folder> <photos_folder> --output-dir <output_folder>
# python compose_all_with_frames.py <frames_folder> <photos_folder> --threads 64

# Thread-local storage for per-thread random generators
_thread_local = threading.local()


def get_thread_rng():
    """Get or create a thread-local random generator."""
    if not hasattr(_thread_local, "rng"):
        # Seed with thread ID + random value for uniqueness
        _thread_local.rng = random.Random()
    return _thread_local.rng


def minimal_crop_to_fit(photo, inner_w, inner_h):
    """Minimal proportional cropping to ensure the photo fills
    the frame window precisely, with minimal image loss."""

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


def process_pair(frame_path, photo_path, output_path):
    """Process a single frame+photo pair. Thread-safe."""
    frame = Image.open(frame_path).convert("RGBA")
    photo = Image.open(photo_path).convert("RGBA")

    # Frame remains horizontal, rotate photo if it is vertical
    if photo.width < photo.height:
        photo = photo.rotate(-90, expand=True)

    alpha = np.array(frame.split()[-1])
    transparent_mask = alpha < 255
    ys, xs = np.where(transparent_mask)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    inner_w = max_x - min_x
    inner_h = max_y - min_y

    photo_cropped = minimal_crop_to_fit(photo, inner_w, inner_h)

    background = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    background.paste(photo_cropped, (min_x, min_y))

    composite = Image.alpha_composite(background, frame)
    composite.save(output_path)
    print(f"Saved: {output_path}")


def process_task(task_args):
    """Worker function for parallel execution."""
    global _progress_counter
    frames_dir, photos_dir, output_dir, frame_file, photo_file = task_args

    frame_path = os.path.join(frames_dir, frame_file)
    photo_path = os.path.join(photos_dir, photo_file)

    photo_name = os.path.splitext(photo_file)[0]
    frame_id = os.path.splitext(frame_file)[0]

    output_filename = f"{photo_name}_{frame_id}.png"
    output_path = os.path.join(output_dir, output_filename)

    process_pair(frame_path, photo_path, output_path)

    with _progress_lock:
        _progress_counter += 1
        elapsed = time.time() - _progress_start_time
        throughput = _progress_counter / elapsed if elapsed > 0 else 0
        print(
            f"[{_progress_counter}/{_progress_total}] Saved: {output_filename} ({throughput:.1f}/s)"
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compose all photos with random frames."
    )
    parser.add_argument("frames_folder", help="Folder with frame images")
    parser.add_argument("photos_folder", help="Folder with photos to put into frames")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="composed_paintings",
        help="Output folder for composed images (default: composed_paintings)",
    )

    parser.add_argument(
        "--variation-count",
        "-n",
        type=int,
        default=1,
        help="Number of variations (frames) to generate per photo. The same frames are used for all photos.",
    )

    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of parallel threads for processing (default: 1)",
    )

    args = parser.parse_args()

    frames_dir = args.frames_folder
    photos_dir = args.photos_folder
    output_dir = args.output_dir
    num_threads = args.threads

    os.makedirs(output_dir, exist_ok=True)

    frame_files = [
        f
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    photo_files = [
        f
        for f in os.listdir(photos_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not frame_files:
        print("No frames found!")
        sys.exit(1)
    if not photo_files:
        print("No photos found!")
        sys.exit(1)

    if args.variation_count > len(frame_files):
        print(
            f"Error: Requested {args.variation_count} variations, but only {len(frame_files)} frames are available."
        )
        sys.exit(1)

    # Build task list with pre-selected frames (using main thread RNG for reproducibility)
    tasks = []
    print(
        f"Randomly assigning {args.variation_count} unique frames to each of the {len(photo_files)} photos."
    )

    rng = random.Random(DEFAULT_STATIC_SEED)
    for photo_file in photo_files:
        selected_frames = rng.sample(frame_files, args.variation_count)
        for frame_file in selected_frames:
            tasks.append((frames_dir, photos_dir, output_dir, frame_file, photo_file))

    global _progress_counter, _progress_total, _progress_start_time
    _progress_counter = 0
    _progress_total = len(tasks)
    _progress_start_time = time.time()

    print(f"Processing {len(tasks)} tasks with {num_threads} threads...")

    if num_threads == 1:
        # Sequential processing
        for task in tasks:
            process_task(task)
    else:
        # Parallel processing with fixed block distribution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(process_task, tasks))

    print(f"Done! Processed {len(tasks)} images.")


if __name__ == "__main__":
    main()
