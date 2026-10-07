import argparse
from pathlib import Path

import albumentations as A
import cv2

# Usage: python distort_julia.py <photos_folder>
# default output folder: variants
# default num variants per image: 6
# Usage: python distort_julia.py <photos_folder> --output-dir <output_folder> --num <num_variants>


# All possible transformations:: https://albumentations.ai/docs/reference/supported-targets-by-transform/

transform = A.Compose(
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
                A.MultiplicativeNoise(
                    multiplier=[0.7, 1.3], per_channel=False, elementwise=True, p=1.0
                ),
                A.SaltAndPepper(amount=[0.01, 0.06], salt_vs_pepper=[0.4, 0.6], p=1.0),
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

    args = parser.parse_args()

    samples_dir = Path(args.photos_folder)
    variants_dir = Path(args.output_dir)
    variants_dir.mkdir(exist_ok=True)

    num_variants = args.num

    for i, img_path in enumerate(samples_dir.glob("*")):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipping non-image file: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate requested number of variants
        for j in range(num_variants):
            augmented = transform(image=image)
            aug_img = augmented["image"]

            out_path = variants_dir / f"{img_path.stem}_{j}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
