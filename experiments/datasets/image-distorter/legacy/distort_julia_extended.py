import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

# Thread-safe progress counter
_progress_lock = threading.Lock()
_progress_counter = 0
_progress_total = 0
_progress_start_time = 0

# Usage: python distort_julia_extended.py <photos_folder>
# default output folder: variants
# default num variants per image: 6
# Usage: python distort_julia_extended.py <photos_folder> --output-dir <output_folder> --num <num_variants>
# Usage: python distort_julia_extended.py <photos_folder> --threads 64

# Thread-local storage for per-thread random state
_thread_local = threading.local()


def get_thread_rng_seed():
    """Get or create a thread-local random seed and set numpy/random state."""
    if not hasattr(_thread_local, "initialized"):
        # Create unique seed per thread using thread ID and random value
        seed = hash((threading.current_thread().ident, random.random())) & 0xFFFFFFFF
        np.random.seed(seed)
        random.seed(seed)
        _thread_local.initialized = True
        _thread_local.seed = seed
    return _thread_local.seed


# All possible transformations:: https://albumentations.ai/docs/reference/supported-targets-by-transform/


def create_transform():
    """Create the albumentations transform pipeline."""
    return A.Compose(
        [
            A.OneOf(
                [
                    A.AdditiveNoise(
                        noise_type="gaussian",
                        spatial_mode="shared",
                        noise_params={"mean_range": [0, 0], "std_range": [0.05, 0.15]},
                        approximation=1,
                        p=1.0,
                    ),
                    A.ChannelDropout(
                        channel_drop_range=[1, 1],
                        fill=0,
                        p=1.0,
                    ),
                    A.PlasmaShadow(
                        shadow_intensity_range=[0.3, 0.7],
                        plasma_size=256,
                        roughness=3,
                        p=1.0,
                    ),
                    A.Posterize(
                        num_bits=4,
                        p=1.0,
                    ),
                    A.ISONoise(
                        color_shift=[0.01, 0.05],
                        intensity=[0.1, 0.5],
                        p=1.0,
                    ),
                    A.RandomGravel(
                        gravel_roi=[0.1, 0.6, 0.9, 0.9],
                        number_of_patches=30,
                        p=1.0,
                    ),
                    A.RandomShadow(
                        shadow_roi=[0, 0.5, 1, 1],
                        num_shadows_limit=[2, 3],
                        shadow_dimension=4,
                        shadow_intensity_range=[0.2, 0.7],
                        p=1.0,
                    ),
                    A.ChromaticAberration(
                        primary_distortion_limit=[-0.3, 0.3],
                        secondary_distortion_limit=[-0.3, 0.3],
                        mode="random",
                        interpolation=cv2.INTER_LINEAR,
                        p=1.0,
                    ),
                    A.Emboss(
                        alpha=[0.5, 0.5],
                        strength=[0.6, 0.9],
                        p=1.0,
                    ),
                    A.RandomSunFlare(
                        flare_roi=[0, 0, 1, 1],
                        src_radius=300,
                        src_color=[255, 255, 255],
                        angle_range=[0, 0.5],
                        num_flare_circles_range=[6, 10],
                        method="physics_based",
                        p=1.0,
                    ),
                    A.MultiplicativeNoise(
                        multiplier=[0.7, 1.3],
                        per_channel=False,
                        elementwise=True,
                        p=1.0,
                    ),
                    A.SaltAndPepper(
                        amount=[0.01, 0.06], salt_vs_pepper=[0.4, 0.6], p=1.0
                    ),
                    A.Downscale(
                        scale_range=[0.2, 0.4],
                        interpolation_pair={"upscale": 0, "downscale": 0},
                        p=1.0,
                    ),
                    A.GaussianBlur(blur_limit=0, sigma_limit=[0.5, 3], p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=[-0.2, 0.2],
                        contrast_limit=[-0.4, 0.4],
                        brightness_by_max=True,
                        ensure_safe_range=False,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=[-20, 20],
                        sat_shift_limit=[-30, 30],
                        val_shift_limit=[-20, 20],
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=[40, 160], p=1.0),
                    A.ToGray(method="weighted_average", p=1.0),
                    A.Posterize(num_bits=4, p=1.0),
                    A.ToSepia(p=1.0),
                    A.Blur(
                        blur_limit=[3, 7],
                        p=1.0,
                    ),
                    A.Sharpen(
                        alpha=[0.2, 0.5],
                        lightness=[0.5, 1],
                        method="kernel",
                        kernel_size=5,
                        sigma=1,
                        p=1.0,
                    ),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.Rotate(
                        limit=[-90, 90],
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        rotate_method="ellipse",
                        crop_border=False,
                        mask_interpolation=cv2.INTER_NEAREST,
                        fill=0,
                        fill_mask=0,
                        p=1.0,
                    ),
                    A.Perspective(
                        scale=[0.05, 0.1],
                        keep_size=True,
                        fit_output=True,
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=1.0,
                    ),
                    A.RandomResizedCrop(
                        size=[512, 512],
                        scale=[0.7, 1],
                        ratio=[0.75, 1.3333333333333333],
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        p=1.0,
                    ),
                    A.RandomRain(
                        slant_range=[-15, 15],
                        drop_length=50,
                        drop_width=1,
                        drop_color=[200, 200, 200],
                        blur_value=7,
                        brightness_coefficient=0.7,
                        rain_type="default",
                        p=1.0,
                    ),
                    A.Pad(
                        padding=[40, 40, 40, 50],
                        fill=50,
                        fill_mask=50,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1.0,
                    ),
                    A.Pad(
                        padding=[30, 30, 40, 50],
                        fill=0,
                        fill_mask=0,
                        border_mode=cv2.BORDER_WRAP,
                        p=1.0,
                    ),
                ],
                p=1,
            ),
        ],
        p=1.0,
    )


def process_task(task_args):
    """Worker function for parallel execution."""
    global _progress_counter
    img_path, variants_dir, variant_idx = task_args

    # Initialize thread-local random state
    get_thread_rng_seed()

    # Create thread-local transform (albumentations uses internal random state)
    transform = create_transform()

    image = cv2.imread(str(img_path))
    if image is None:
        with _progress_lock:
            _progress_counter += 1
            elapsed = time.time() - _progress_start_time
            throughput = _progress_counter / elapsed if elapsed > 0 else 0
            print(
                f"[{_progress_counter}/{_progress_total}] Skipping non-image: {img_path.name} ({throughput:.1f}/s)"
            )
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image)
    aug_img = augmented["image"]

    out_path = variants_dir / f"{img_path.stem}_distort-{variant_idx + 1:03d}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

    with _progress_lock:
        _progress_counter += 1
        elapsed = time.time() - _progress_start_time
        throughput = _progress_counter / elapsed if elapsed > 0 else 0
        print(
            f"[{_progress_counter}/{_progress_total}] Saved: {out_path.name} ({throughput:.1f}/s)"
        )

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Apply augmentation distortions to images."
    )

    parser.add_argument("photos_folder", help="Folder containing photos to distort")

    parser.add_argument(
        "--output-dir",
        "-o",
        default="variants",
        help="Output directory for generated variants (default: variants)",
    )

    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=6,
        help="Number of distortion variants per image (default: 6)",
    )

    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of parallel threads for processing (default: 1)",
    )

    args = parser.parse_args()

    samples_dir = Path(args.photos_folder)
    variants_dir = Path(args.output_dir)
    variants_dir.mkdir(exist_ok=True)

    num_variants = args.num
    num_threads = args.threads

    # Build task list: (img_path, variants_dir, variant_index)
    tasks = []
    image_paths = [p for p in samples_dir.glob("*") if p.is_file()]

    for img_path in image_paths:
        for j in range(num_variants):
            tasks.append((img_path, variants_dir, j))

    global _progress_counter, _progress_total, _progress_start_time
    _progress_counter = 0
    _progress_total = len(tasks)
    _progress_start_time = time.time()

    print(
        f"Processing {len(tasks)} tasks ({len(image_paths)} images x {num_variants} variants) with {num_threads} threads..."
    )

    if num_threads == 1:
        # Sequential processing
        for task in tasks:
            process_task(task)
    else:
        # Parallel processing with fixed block distribution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(process_task, tasks))

    print(f"Done! Processed {len(tasks)} distortion variants.")


if __name__ == "__main__":
    main()
