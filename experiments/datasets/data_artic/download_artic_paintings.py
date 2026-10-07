#!/usr/bin/env python3
"""Download open-access Art Institute of Chicago (ArtIC) painting images.

ArtIC exposes a keyless REST API plus its own IIIF Image server, so discovery is
an Elasticsearch query rather than a CSV join or a paged Linked-Art crawl -- see
``count_artic_paintings.py`` for the query, the id-range paging trick, and the
``iter_painting_images`` candidate iterator this script reuses. Per candidate:

    search(is_public_domain=true, artwork_type=Painting)  (paged by id range)
        -> image_id + native width/height (thumbnail.*) + IIIF base URL
        -> download (<= 2048px wide) to {out-dir}/{objectid}.jpg

Rights: ArtIC releases every public-domain image under CC0, so the single
``is_public_domain`` flag is the whole rights filter (already applied in the
iterator); there is no per-image license URL to resolve.

Resize policy (<= 2048px wide, ``MAX_WIDTH``): the size is chosen from the known
native width, not by probing, because the ArtIC IIIF server *refuses to upscale*
-- a ``sizeByW`` request above native returns HTTP 403 -- and its ``full`` size
is capped at 3000px. So:

  * native width >  2048  -> request ``/full/2048,/0/default.jpg`` (server
                             downscales -- huge bandwidth saving)
  * native width <= 2048  -> request ``/full/full/0/default.jpg`` (returns the
                             native image, which is <= 2048 here, never upscaled)

If the chosen request fails, we fall back to the other size, and a final Pillow
pass enforces the <= 2048px invariant without ever upscaling.

Downloads are idempotent/resumable (non-empty targets are left untouched) and
concurrency is bounded two ways at once, exactly like the NGA and Rijks
downloaders:

  * ``--workers N``        parallel download workers (thread pool)
  * ``--max-req-per-s R``  one global token-bucket rate limiter shared by every
                           HTTP request, so N workers never exceed R req/s.

Usage
-----
    # Smoke test: 20 images, 4 workers, <= 8 requests/second
    python download_artic_paintings.py --limit 20 --workers 4 --max-req-per-s 8

    # Full open-access paintings pull
    python download_artic_paintings.py --workers 8 --max-req-per-s 10 \
        --out-dir artic_paintings --metadata-csv artic_paintings.csv
"""
import argparse
import csv
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from PIL import Image

from count_artic_paintings import iter_painting_images, make_session, OPEN_ACCESS_LICENSE

# Cap every saved image at this width (px). Height scales proportionally.
MAX_WIDTH = 2048


class RateLimiter:
    """Thread-safe global token bucket: at most ``max_per_s`` acquires/second.

    Scheduling is serialized under a lock but the actual sleep happens outside
    it, so many worker threads share one true global rate rather than each
    sleeping independently.
    """

    def __init__(self, max_per_s):
        self.min_interval = 1.0 / max_per_s if max_per_s and max_per_s > 0 else 0.0
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if self._next <= now:
                self._next = now + self.min_interval
                wait = 0.0
            else:
                wait = self._next - now
                self._next += self.min_interval
        if wait > 0:
            time.sleep(wait)


def iiif_image_url(iiif_base, size):
    """Build a IIIF Image API request URL for ``size`` (e.g. ``2048,`` or ``full``)."""
    return f"{iiif_base.rstrip('/')}/full/{size}/0/default.jpg"


def choose_urls(candidate):
    """Return (preferred_url, fallback_url) for a candidate image.

    The preferred request downscales wide sources server-side; narrow sources
    are fetched at full size so they are never upscaled (ArtIC 403s an upscale).
    The fallback is the other size, used only if the preferred request fails.
    """
    base = candidate["iiif_url"]
    width = candidate.get("width")
    downscaled = iiif_image_url(base, f"{MAX_WIDTH},")
    original = iiif_image_url(base, "full")
    if width is not None and width <= MAX_WIDTH:
        return original, downscaled
    return downscaled, original


def ensure_max_width(path, max_width=MAX_WIDTH):
    """Safety net: downscale ``path`` in place if wider than ``max_width``.

    Never upscales. Normally a no-op because the size was chosen from the known
    native width, but it enforces the invariant if a fallback returned something
    wider (e.g. ArtIC's ``full`` can be up to 3000px).
    """
    with Image.open(path) as im:
        width, height = im.size
        if width <= max_width:
            return
        new_height = round(height * max_width / width)
        resized = im.convert("RGB").resize((max_width, new_height), Image.LANCZOS)
        resized.save(path, format="JPEG", quality=90)


def download_image(session, limiter, image_url, out_path, retries=4):
    """Stream an image to out_path atomically. Returns True on success.

    Non-429 client errors (4xx) are not retried, so a bad size request (e.g. an
    upscale ArtIC rejects with 403) fails fast into the caller's fallback.
    """
    tmp_path = out_path + ".part"
    delay = 1.0
    for attempt in range(retries + 1):
        limiter.acquire()
        try:
            with session.get(image_url, timeout=60, stream=True) as resp:
                if resp.status_code == 429:
                    time.sleep(float(resp.headers.get("Retry-After", delay)))
                    delay *= 1.5
                    continue
                if 400 <= resp.status_code < 500:
                    return False
                resp.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
            if os.path.getsize(tmp_path) == 0:
                raise IOError("empty download")
            os.replace(tmp_path, out_path)
            return True
        except (requests.RequestException, IOError):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt >= retries:
                return False
            time.sleep(delay)
            delay *= 1.5
    return False


def process_candidate(session, limiter, candidate, out_dir):
    """Full per-object job: skip/download. Returns a result dict."""
    stem = candidate["objectid"]
    out_path = os.path.join(out_dir, f"{stem}.jpg")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return {"id": stem, "status": "exists", "path": out_path}

    preferred_url, fallback_url = choose_urls(candidate)
    used_url = preferred_url
    ok = download_image(session, limiter, preferred_url, out_path)
    if not ok and fallback_url != preferred_url:
        used_url = fallback_url
        ok = download_image(session, limiter, fallback_url, out_path)

    if not ok:
        return {"id": stem, "status": "failed", "image_url": preferred_url}

    try:
        ensure_max_width(out_path)
    except Exception as exc:  # undecodable/corrupt image: fail this one, keep going
        return {"id": stem, "status": "failed", "image_url": used_url,
                "reason": f"decode:{exc}"}
    return {
        "id": stem,
        "status": "downloaded",
        "image_id": candidate["image_id"],
        "title": candidate.get("title", ""),
        "image_url": used_url,
        "license": candidate["license"],
        "path": out_path,
    }


def scan_existing_stems(out_dir):
    """Filename stems (sans .jpg) of the non-empty images already in out_dir.

    Lets a restarted run resume: images already on disk are skipped rather than
    re-downloaded.
    """
    stems = set()
    try:
        names = os.listdir(out_dir)
    except OSError:
        return stems
    for name in names:
        if name.endswith(".jpg"):
            try:
                if os.path.getsize(os.path.join(out_dir, name)) > 0:
                    stems.add(name[:-4])
            except OSError:
                pass
    return stems


def _script_provenance():
    """(repo-relative script path, short commit hash, dirty?) for the manifest."""
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.abspath(__file__)

    def git(*args):
        return subprocess.run(
            ["git", "-C", here, *args],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()

    try:
        root = git("rev-parse", "--show-toplevel")
        rel = os.path.relpath(script, root) if root else os.path.basename(script)
        return rel, (git("rev-parse", "--short", "HEAD") or None), bool(
            git("status", "--porcelain", "--", script))
    except Exception:
        return os.path.basename(script), None, None


def write_manifest(source, license_, args, stats):
    """Write a compact manifest.json into the image dir, alongside the images."""
    rel_path, commit, dirty = _script_provenance()
    manifest = {
        "source": source,
        "script_path": rel_path,
        "script_commit": commit,
        "script_dirty": dirty,
        "run_finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir": args.out_dir,
        "metadata_csv": args.metadata_csv,
        "max_width": MAX_WIDTH,
        "license": license_,
        "images_in_dir": len(scan_existing_stems(args.out_dir)),
        "stats": stats,
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Download open-access ArtIC paintings (concurrent, rate-limited)."
    )
    parser.add_argument("--out-dir", default="artic_paintings",
                        help="Directory for downloaded images (default: artic_paintings).")
    parser.add_argument("--metadata-csv", default="artic_paintings.csv",
                        help="Append-mode CSV log of downloads (default: artic_paintings.csv).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download workers (default: 8).")
    parser.add_argument("--max-req-per-s", type=float, default=10.0,
                        help="Global cap on HTTP requests per second across all workers (default: 10).")
    parser.add_argument("--limit", type=int,
                        help="Only process the first N objects (for testing/partial runs).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    limiter = RateLimiter(args.max_req_per_s)
    session = make_session()

    print("Listing painting objects (is_public_domain=true, artwork_type=Painting)"
          + (f", limit={args.limit}" if args.limit else "") + " ...")
    candidates = []
    for cand in iter_painting_images(session=session):
        candidates.append(cand)
        if args.limit is not None and len(candidates) >= args.limit:
            break

    # Resume: skip paintings whose image is already on disk from an earlier run.
    existing = scan_existing_stems(args.out_dir)
    pending = [c for c in candidates if c["objectid"] not in existing]
    already = len(candidates) - len(pending)
    print(f"Got {len(candidates)} candidate paintings; {len(existing)} images "
          f"already in {args.out_dir} ({already} of these present). "
          f"Downloading {len(pending)} with {args.workers} workers, "
          f"<= {args.max_req_per_s} req/s.")

    csv_exists = os.path.exists(args.metadata_csv)
    csv_lock = threading.Lock()
    stats = {"downloaded": 0, "exists": already, "skipped": 0, "failed": 0}

    with open(args.metadata_csv, "a", newline="", encoding="utf-8") as meta_f:
        writer = csv.DictWriter(
            meta_f,
            fieldnames=["objectid", "image_id", "title", "image_url", "license", "path"],
        )
        if not csv_exists:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_candidate, session, limiter, cand, args.out_dir): cand
                for cand in pending
            }
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                res = future.result()
                stats[res["status"]] += 1
                if res["status"] == "downloaded":
                    with csv_lock:
                        writer.writerow({
                            "objectid": res["id"],
                            "image_id": res.get("image_id", ""),
                            "title": res.get("title", ""),
                            "image_url": res.get("image_url", ""),
                            "license": res.get("license", ""),
                            "path": res.get("path", ""),
                        })
                        meta_f.flush()
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  [{done}/{total}] "
                          f"downloaded={stats['downloaded']} exists={stats['exists']} "
                          f"skipped={stats['skipped']} failed={stats['failed']}")

    manifest_path = write_manifest(
        "Art Institute of Chicago", OPEN_ACCESS_LICENSE, args, stats)

    print("\nDone.")
    for k in ("downloaded", "exists", "skipped", "failed"):
        print(f"  {k:<11}: {stats[k]}")
    print(f"  images dir : {args.out_dir}")
    print(f"  metadata   : {args.metadata_csv}")
    print(f"  manifest   : {manifest_path}")


if __name__ == "__main__":
    main()
