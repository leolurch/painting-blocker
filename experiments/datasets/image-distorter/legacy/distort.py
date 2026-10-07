#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError as exc:  # pragma: no cover - Pillow is required
    raise SystemExit("Pillow must be installed to run this script.") from exc

try:  # Pillow 10+
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - Pillow < 10
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_LANCZOS = Image.LANCZOS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class VariantSpec:
    """Represents a logical variant made up of one or more transformations."""

    name: str
    generator: Callable[[Image.Image, random.Random], Iterable[Image.Image]]

    def generate(self, image: Image.Image, rng: random.Random) -> Iterator[Image.Image]:
        for variant in self.generator(image, rng):
            yield variant


def quarter_resolution_variant(
    image: Image.Image, _rng: random.Random
) -> Iterable[Image.Image]:
    """Return a downscaled image with quarter pixel count."""
    width, height = image.size
    new_size = (max(1, width // 2), max(1, height // 2))
    yield image.resize(new_size, RESAMPLE_LANCZOS)


def grayscale_variant(image: Image.Image, _rng: random.Random) -> Iterable[Image.Image]:
    """Convert the image to grayscale while keeping three channels."""
    yield ImageOps.grayscale(image).convert("RGB")


def low_saturation_variant(
    image: Image.Image, _rng: random.Random
) -> Iterable[Image.Image]:
    """Greatly reduce image saturation while preserving brightness."""
    enhancer = ImageEnhance.Color(image)
    yield enhancer.enhance(0.15)


def rotate_point(
    x: float, y: float, z: float, angle_x: float, angle_y: float
) -> Tuple[float, float, float]:
    """Rotate a 3D point around the X and Y axes."""
    cos_x = math.cos(angle_x)
    sin_x = math.sin(angle_x)
    cos_y = math.cos(angle_y)
    sin_y = math.sin(angle_y)

    # rotation around X axis (pitch)
    y1 = y * cos_x - z * sin_x
    z1 = y * sin_x + z * cos_x

    # rotation around Y axis (yaw)
    x2 = x * cos_y + z1 * sin_y
    z2 = -x * sin_y + z1 * cos_y

    return x2, y1, z2


def find_perspective_coeffs(
    src_points: Sequence[Tuple[float, float]],
    dst_points: Sequence[Tuple[float, float]],
) -> List[float]:
    """Solve coefficients for Pillow's perspective transform."""
    if len(src_points) != 4 or len(dst_points) != 4:
        raise ValueError("src_points and dst_points must contain 4 points each.")

    matrix = []
    target = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src_points, dst_points):
        matrix.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
        matrix.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])
        target.append(x_src)
        target.append(y_src)

    # Solve linear system with Gaussian elimination (numpy-less implementation).
    return _solve_linear_system(matrix, target)


def _solve_linear_system(matrix: List[List[float]], target: List[float]) -> List[float]:
    """Simple Gaussian elimination with partial pivoting."""
    size = len(target)
    # Augmented matrix
    aug = [row[:] + [target_val] for row, target_val in zip(matrix, target)]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be solved.")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, size + 1):
            aug[col][j] /= pivot

        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, size + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][size] for i in range(size)]


def perspective_variant_factory(
    count: int = 5,
    max_angle_deg: float = 30.0,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> Callable[[Image.Image, random.Random], Iterable[Image.Image]]:
    """Generate perspective variants with random pitch/yaw within bounds."""
    max_angle_rad = math.radians(max_angle_deg)

    def generate(image: Image.Image, rng: random.Random) -> Iterable[Image.Image]:
        width, height = image.size
        distance = max(width, height)
        src_corners = [
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), float(height)),
            (0.0, float(height)),
        ]

        for _ in range(count):
            angle_x = rng.uniform(-max_angle_rad, max_angle_rad)
            angle_y = rng.uniform(-max_angle_rad, max_angle_rad)

            projected = []
            for x_src, y_src in src_corners:
                x = x_src - width / 2.0
                y = y_src - height / 2.0
                x_rot, y_rot, z_rot = rotate_point(x, y, 0.0, angle_x, angle_y)
                z_shifted = z_rot + distance
                if abs(z_shifted) < 1e-6:
                    z_shifted = 1e-6 if z_shifted >= 0 else -1e-6
                scale = distance / z_shifted
                proj_x = x_rot * scale + width / 2.0
                proj_y = y_rot * scale + height / 2.0
                projected.append((proj_x, proj_y))

            min_x = min(x for x, _ in projected)
            max_x = max(x for x, _ in projected)
            min_y = min(y for _, y in projected)
            max_y = max(y for _, y in projected)

            out_width = max(1, int(math.ceil(max_x - min_x)))
            out_height = max(1, int(math.ceil(max_y - min_y)))

            dest_points = [(x - min_x, y - min_y) for x, y in projected]
            coeffs = find_perspective_coeffs(src_corners, dest_points)

            transformed = image.transform(
                (out_width, out_height),
                Image.PERSPECTIVE,
                coeffs,
                resample=RESAMPLE_BICUBIC,
                fillcolor=background,
            )
            yield transformed

    return generate


def build_default_variants(max_angle_deg: float) -> List[VariantSpec]:
    return [
        VariantSpec("quarter_res", quarter_resolution_variant),
        VariantSpec("grayscale", grayscale_variant),
        VariantSpec("low_saturation", low_saturation_variant),
        VariantSpec(
            "perspective",
            perspective_variant_factory(count=5, max_angle_deg=max_angle_deg),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distorted variants for images."
    )
    default_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--samples",
        type=Path,
        default=default_root / "samples",
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=default_root / "variants",
        help="Directory to write generated variants.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base RNG seed to keep perspective variants deterministic.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing files in the variants directory before generating.",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=30.0,
        help="Maximum absolute tilt angle in degrees for perspective variants.",
    )
    return parser.parse_args()


def collect_samples(samples_dir: Path) -> List[Path]:
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")

    images = sorted(
        (
            path
            for path in samples_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name.lower(),
    )
    return images


def ensure_variants_dir(path: Path, clean: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if clean:
        for existing in path.glob("*"):
            if existing.is_file():
                existing.unlink()


def main() -> None:
    args = parse_args()
    variants = build_default_variants(args.max_angle)
    samples = collect_samples(args.samples)
    if not samples:
        print(
            f"No images with extensions {sorted(IMAGE_EXTENSIONS)} found in {args.samples}."
        )
        return

    ensure_variants_dir(args.variants, clean=args.clean)
    print(f"Generating variants into {args.variants}")

    for image_index, image_path in enumerate(samples, start=1):
        base_seed = f"{args.seed}-{image_index}"
        with Image.open(image_path) as original:
            base_image = original.convert("RGB")

        original_path = args.variants / f"{image_index}_0.jpg"
        base_image.save(original_path, "JPEG", quality=95)
        print(f"[{image_index}] -> {original_path.name} (original)")

        variant_counter = 1
        for spec in variants:
            spec_seed = f"{base_seed}-{spec.name}"
            rng = random.Random(spec_seed)
            for variant_image in spec.generate(base_image, rng):
                output_path = args.variants / f"{image_index}_{variant_counter}.jpg"
                variant_image.convert("RGB").save(output_path, "JPEG", quality=95)
                print(f"[{image_index}] -> {output_path.name} ({spec.name})")
                variant_counter += 1


if __name__ == "__main__":
    main()
