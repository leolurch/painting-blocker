#!/usr/bin/env python3
import argparse
import csv
import os
import re
import tarfile
from urllib.parse import urlparse

import requests

# Um alle Bilder runterzuladen führe aus: python download_rijks_iiif.py image.tar.gz --csv rijks_urls.csv --download-dir rijks_images
# ABER das sind über 400 000 bis 500 000 Bilder, also viel Speicherplatz nötig!

# test einer Stichprobe: python download_rijks_iiif.py image.tar.gz --csv rijks_urls_test.csv --limit-members 1000
# schaue dir mal an: rijks_urls_test.csv

# Regex to match IIIF image URLs like:
# https://iiif.micr.io/wOrfE/full/max/0/default.jpg
IIIF_PATTERN = re.compile(
    r"https://iiif\.micr\.io/[A-Za-z0-9]+/full/max/0/default\.jpg"
)


def extract_iiif_urls_from_member(tar, member):
    """
    Extracts IIIF image URLs from a single file inside the tar archive.

    Parameters
    ----------
    tar : tarfile.TarFile
        Open tar archive.
    member : tarfile.TarInfo
        A member (file) inside the archive.

    Returns
    -------
    set[str]
        Set of IIIF image URLs found in this member.
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
    Extracts the IIIF service ID from a full image URL.

    Example:
        https://iiif.micr.io/wOrfE/full/max/0/default.jpg
        -> service_id = "wOrfE"

    Parameters
    ----------
    iiif_url : str
        Full IIIF image URL.

    Returns
    -------
    str | None
        Service ID (first path segment) or None if parsing fails.
    """
    parsed = urlparse(iiif_url)
    parts = parsed.path.strip("/").split("/")
    if not parts:
        return None
    return parts[0]


def build_full_image_url(service_id, fmt="jpg"):
    """
    Builds the full IIIF image URL from a service ID.

    Parameters
    ----------
    service_id : str
        IIIF service ID (e.g. "wOrfE").
    fmt : str
        File format, default "jpg".

    Returns
    -------
    str
        Full IIIF image URL.
    """
    return f"https://iiif.micr.io/{service_id}/full/max/0/default.{fmt}"


def process_archive(archive_path, output_csv, download_dir=None, limit_members=None):
    """
    Processes the image.tar.gz archive from Rijksmuseum.

    Steps:
      1. Iterate over all files in the archive.
      2. Extract IIIF image URLs.
      3. Normalize to unique service IDs.
      4. Write CSV with service_id and full image URL.
      5. Optionally download all images.

    Parameters
    ----------
    archive_path : str
        Path to image.tar.gz.
    output_csv : str
        Path to CSV file where URLs will be saved.
    download_dir : str | None
        If not None, directory where images will be downloaded.
    limit_members : int | None
        Optional limit on the number of files (members) processed
        from the tar archive (useful for testing).
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

    print(f"Total unique IIIF service IDs found: {len(service_ids)}")

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["service_id", "iiif_image_url"])
        for sid in sorted(service_ids):
            img_url = build_full_image_url(sid)
            writer.writerow([sid, img_url])

    print(f"Saved URL list to: {output_csv}")

    # Optional: download images
    if download_dir and service_ids:
        os.makedirs(download_dir, exist_ok=True)
        session = requests.Session()

        for i, sid in enumerate(sorted(service_ids), start=1):
            img_url = build_full_image_url(sid)
            out_path = os.path.join(download_dir, f"{sid}.jpg")

            if os.path.exists(out_path):
                # Skip if already downloaded
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

        print(f"Finished downloading images to: {download_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract IIIF image URLs from Rijksmuseum image.tar.gz "
            "and optionally download all images."
        )
    )
    parser.add_argument("archive", help="Path to image.tar.gz")
    parser.add_argument(
        "--csv",
        default="rijks_urls.csv",
        help="Path to output CSV file (default: rijks_urls.csv)",
    )
    parser.add_argument(
        "--download-dir",
        help="If provided, directory where images will be downloaded.",
    )
    parser.add_argument(
        "--limit-members",
        type=int,
        help="Optional limit of files from the tar archive to process (for testing).",
    )

    args = parser.parse_args()

    process_archive(
        archive_path=args.archive,
        output_csv=args.csv,
        download_dir=args.download_dir,
        limit_members=args.limit_members,
    )


if __name__ == "__main__":
    main()
