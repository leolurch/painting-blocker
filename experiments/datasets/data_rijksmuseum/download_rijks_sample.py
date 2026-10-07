#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
import tarfile
from urllib.parse import urlparse

import requests

# python download_rijks_sample.py image.tar.gz \
# Führe diesen Befehl aus, um eine Stichprobe von IIIF-Bild-URLs zu extrahieren herunterzuladen:
#     --fraction 0.01 \
#     --limit-members 5000 \
#     --csv sample_test.csv \
#     --download-dir test_images


# Regex to match IIIF image URLs like:
# https://iiif.micr.io/wOrfE/full/max/0/default.jpg
IIIF_PATTERN = re.compile(
    r"https://iiif\.micr\.io/[A-Za-z0-9]+/full/max/0/default\.jpg"
)


def extract_iiif_urls_from_member(tar, member):
    """
    Extract IIIF image URLs from a single file inside the tar archive.
    """
    urls = set()
    f = tar.extractfile(member)
    if f is None:
        return urls

    for raw_line in f:
        try:
            line = raw_line.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            continue

        for match in IIIF_PATTERN.findall(line):
            urls.add(match)

    return urls


def normalize_service_id(iiif_url):
    """
    Extract the IIIF service ID from a full image URL.

    Example:
        https://iiif.micr.io/wOrfE/full/max/0/default.jpg
        -> "wOrfE"
    """
    parsed = urlparse(iiif_url)
    parts = parsed.path.strip("/").split("/")
    if not parts:
        return None
    return parts[0]


def build_full_image_url(service_id, fmt="jpg"):
    """
    Build the full IIIF image URL from a service ID.
    """
    return f"https://iiif.micr.io/{service_id}/full/max/0/default.{fmt}"


def process_archive(
    archive_path,
    output_csv,
    download_dir=None,
    fraction=1.0,
    max_images=None,
    limit_members=None,
    random_seed=42,
):
    """
    Process the Rijksmuseum image.tar.gz archive.

    Steps:
      1. Iterate over all files in the archive.
      2. Extract IIIF image URLs.
      3. Normalize to unique service IDs.
      4. Randomly sample a fraction (e.g. 0.01 = 1%).
      5. Write CSV with sampled service_id and image URL.
      6. Optionally download only the sampled images.
    """
    service_ids = set()

    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        print(f"Found {len(members)} files in the archive.")

        for idx, member in enumerate(members, start=1):
            if limit_members is not None and idx > limit_members:
                print(f"Limit of {limit_members} members reached, stopping early.")
                break

            urls = extract_iiif_urls_from_member(tar, member)
            for url in urls:
                sid = normalize_service_id(url)
                if sid:
                    service_ids.add(sid)

            if idx % 1000 == 0:
                print(
                    f"Processed {idx} files, "
                    f"unique service IDs so far: {len(service_ids)}"
                )

    total = len(service_ids)
    print(f"Total unique IIIF service IDs found: {total}")

    if total == 0:
        print("No IIIF image URLs found. Exiting.")
        return

    # --- sampling ---
    random.seed(random_seed)
    service_ids_list = sorted(service_ids)

    # how many images to sample based on fraction
    k = int(round(total * fraction))
    if k < 1:
        k = 1

    if max_images is not None:
        k = min(k, max_images)

    print(
        f"Sampling {k} images out of {total} "
        f"({fraction * 100:.2f}% requested, max_images={max_images})"
    )

    sampled_ids = set(random.sample(service_ids_list, k))

    # --- write CSV with sampled images ---
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["service_id", "iiif_image_url"])
        for sid in sorted(sampled_ids):
            img_url = build_full_image_url(sid)
            writer.writerow([sid, img_url])

    print(f"Saved sampled URL list to: {output_csv}")

    # --- optionally download images ---
    if download_dir:
        os.makedirs(download_dir, exist_ok=True)
        session = requests.Session()

        for i, sid in enumerate(sorted(sampled_ids), start=1):
            img_url = build_full_image_url(sid)
            out_path = os.path.join(download_dir, f"{sid}.jpg")

            if os.path.exists(out_path):
                continue

            try:
                resp = session.get(img_url, timeout=30)
                resp.raise_for_status()
                with open(out_path, "wb") as img_f:
                    img_f.write(resp.content)
            except Exception as e:
                print(f"❗ Error downloading {img_url}: {e}")
                continue

            if i % 50 == 0:
                print(f"Downloaded {i} images...")

        print(f"Finished downloading sampled images to: {download_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample a fraction of IIIF images from Rijksmuseum image.tar.gz "
            "and optionally download them."
        )
    )
    parser.add_argument("archive", help="Path to image.tar.gz")
    parser.add_argument(
        "--csv",
        default="rijks_sample_urls.csv",
        help="Path to output CSV file (default: rijks_sample_urls.csv)",
    )
    parser.add_argument(
        "--download-dir",
        help="If provided, directory where sampled images will be downloaded.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.01,
        help="Fraction of images to sample, e.g. 0.01 = 1%%, 0.05 = 5%% "
        "(default: 0.01)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional hard cap on number of downloaded images.",
    )
    parser.add_argument(
        "--limit-members",
        type=int,
        help="Optional limit of files from the tar archive to process (for testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )

    args = parser.parse_args()

    process_archive(
        archive_path=args.archive,
        output_csv=args.csv,
        download_dir=args.download_dir,
        fraction=args.fraction,
        max_images=args.max_images,
        limit_members=args.limit_members,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
