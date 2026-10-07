import argparse
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

# Thread-safe progress counter
_progress_lock = threading.Lock()
_progress_counter = 0
_progress_total = 0
_progress_start_time = 0

# Usage: python compose_with_reflections.py reflection_overlay/ samples/
# Usage: python compose_with_reflections.py <reflections_folder> <photos_folder> --output-dir <output_folder>
# Usage: python compose_with_reflections.py <reflections_folder> <photos_folder> --threads 64

# Thread-local storage for per-thread random generators
_thread_local = threading.local()


def get_thread_rng():
    """Get or create a thread-local random generator."""
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = random.Random()
    return _thread_local.rng


def apply_overlay(photo_path, overlay_path, output_path, opacity=None):
    """Apply an overlay on top of the photo with the given (optional) opacity. Thread-safe."""

    photo = Image.open(photo_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # Resize overlay to match the photo size
    overlay = overlay.resize(photo.size, Image.LANCZOS)

    # Random opacity if not provided (range 0.3–1.0 for visible effect)
    # Uses thread-local RNG for thread safety
    if opacity is None:
        rng = get_thread_rng()
        opacity = rng.uniform(0.3, 1.0)

    # Adjust overlay alpha channel based on chosen opacity
    overlay = overlay.copy()
    alpha_channel = overlay.split()[-1]
    overlay_alpha = alpha_channel.point(lambda a: int(a * max(0.0, min(1.0, opacity))))
    overlay.putalpha(overlay_alpha)

    # Composite overlay on top of the base photo
    composite = Image.alpha_composite(photo, overlay)
    composite.save(output_path)

    print(f"Saved: {output_path}  (opacity={opacity:.2f})")


def process_task(task_args):
    """Worker function for parallel execution."""
    global _progress_counter
    overlays_dir, photos_dir, output_dir, overlay_file, photo_file = task_args

    photo_path = os.path.join(photos_dir, photo_file)
    overlay_path = os.path.join(overlays_dir, overlay_file)

    photo_name = os.path.splitext(photo_file)[0]
    overlay_id = os.path.splitext(overlay_file)[0]

    output_filename = f"{photo_name}_{overlay_id}.png"
    output_path = os.path.join(output_dir, output_filename)

    apply_overlay(photo_path, overlay_path, output_path)

    with _progress_lock:
        _progress_counter += 1
        elapsed = time.time() - _progress_start_time
        throughput = _progress_counter / elapsed if elapsed > 0 else 0
        print(
            f"[{_progress_counter}/{_progress_total}] Saved: {output_filename} ({throughput:.1f}/s)"
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Apply random overlays to photos.")

    parser.add_argument(
        "overlays_folder",
        help="Folder containing overlay images (preferably PNG with transparency).",
    )

    parser.add_argument(
        "photos_folder", help="Folder containing photos that will receive overlays."
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="overlayed_photos",
        help="Output folder for processed images (default: overlayed_photos).",
    )

    parser.add_argument(
        "--variation-count",
        "-n",
        type=int,
        default=1,
        help="Number of variations (overlays) to generate per photo. The same overlays are used for all photos.",
    )

    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of parallel threads for processing (default: 1)",
    )

    args = parser.parse_args()

    overlays_dir = args.overlays_folder
    photos_dir = args.photos_folder
    output_dir = args.output_dir
    num_threads = args.threads

    os.makedirs(output_dir, exist_ok=True)

    overlay_files = [
        f
        for f in os.listdir(overlays_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    photo_files = [
        f
        for f in os.listdir(photos_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not overlay_files:
        print("No overlay images found!")
        sys.exit(1)

    if not photo_files:
        print("No photos found!")
        sys.exit(1)

    if args.variation_count > len(overlay_files):
        print(
            f"Error: Requested {args.variation_count} variations, but only {len(overlay_files)} overlays are available."
        )
        sys.exit(1)

    # Build task list with pre-selected overlays (using main thread RNG for reproducibility)
    tasks = []
    print(
        f"Randomly assigning {args.variation_count} unique overlays to each of the {len(photo_files)} photos."
    )

    for photo_file in photo_files:
        selected_overlays = random.sample(overlay_files, args.variation_count)
        for overlay_file in selected_overlays:
            tasks.append(
                (overlays_dir, photos_dir, output_dir, overlay_file, photo_file)
            )

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
