#!/usr/bin/env python3
"""Download reusable Rijksmuseum painting images via the public Data Services API.

Pipeline per object (all through the keyless Search + Linked-Art API):

    search(type=painting, imageAvailable=true)   -> object IDs (paged)
    object  -> shows[0]                           -> VisualItem
    VisualItem.subject_to -> Right.classified_as  -> reuse license (filter)
    VisualItem.digitally_shown_by[0]              -> DigitalObject
    DigitalObject.access_point[0].id              -> IIIF image URL
    download (resized to <= 2048px wide)           -> {out-dir}/{lod_id}.jpg

Images are capped at 2048px width (``MAX_WIDTH``) automatically. This uses the
IIIF Image API size parameter server-side -- the ``.../full/max/...`` URL is
rewritten to ``.../full/2048,/...`` so the museum returns an already-downscaled
JPEG (no full-res transfer). Sources narrower than 2048px would make that
request 400, so we fall back to the original, and a final Pillow pass enforces
the <= 2048px invariant either way (never upscaling).

Objects whose image license is not reuse-permitting (see ``REUSE_LICENSES`` in
count_rijks_paintings.py) are skipped. Downloads are idempotent/resumable: a
non-empty target file is left untouched, so re-running finishes an interrupted
run.

Concurrency is bounded two ways at once:

  * ``--workers N``        parallel download workers (thread pool)
  * ``--max-req-per-s R``  a single global token-bucket rate limiter shared by
                           every HTTP request (metadata + image), so N workers
                           together never exceed R requests/second.

Usage
-----
    # Smoke test: 20 images, 4 workers, <=8 requests/second
    python download_rijks_paintings.py --limit 20 --workers 4 --max-req-per-s 8

    # Full reusable-paintings pull
    python download_rijks_paintings.py --workers 8 --max-req-per-s 10 \
        --out-dir rijks_paintings --metadata-csv rijks_paintings.csv
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

from count_rijks_paintings import REUSE_LICENSES, SEARCH_URL

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


def get_json(session, url, limiter, params=None, retries=4):
    """Rate-limited GET + JSON parse.

    Retries 429 (rate limit) and transient failures (5xx, connection/JSON
    errors) with backoff. Non-429 client errors (4xx, e.g. a 404 for a withdrawn
    record) are not retried and raise immediately, so the caller can skip that
    object rather than aborting the whole run.
    """
    delay = 1.0
    for attempt in range(retries + 1):
        limiter.acquire()
        try:
            resp = session.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", delay)))
                delay *= 1.5
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code is not None and 400 <= code < 500:
                raise  # client error: not transient, not rate-limit -> caller skips
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= 1.5
        except (requests.RequestException, ValueError):
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= 1.5
    return None


def iter_object_ids(session, limiter, limit=None):
    """Yield painting object IDs (with an image), paging via ``next.id``."""
    url = SEARCH_URL
    params = {"type": "painting", "imageAvailable": "true"}
    yielded = 0
    while url:
        data = get_json(session, url, limiter, params=params)
        params = None  # pageToken is embedded in subsequent next.id URLs
        for item in data.get("orderedItems", []):
            yield item["id"]
            yielded += 1
            if limit is not None and yielded >= limit:
                return
        url = (data.get("next") or {}).get("id")


def resolve_image(session, limiter, object_id):
    """Resolve an object to (image_url, license_url).

    Returns (None, reason) when there is no reusable image to fetch.
    """
    obj = get_json(session, object_id, limiter)
    shows = obj.get("shows") or []
    if not shows:
        return None, "no-visualitem"
    visual = get_json(session, shows[0]["id"], limiter)

    license_url = None
    for right in visual.get("subject_to") or []:
        for cls in right.get("classified_as") or []:
            if cls.get("id"):
                license_url = cls["id"]
                break
        if license_url:
            break
    if license_url not in REUSE_LICENSES:
        return None, f"license:{license_url}"

    digital = visual.get("digitally_shown_by") or []
    if not digital:
        return None, "no-digitalobject"
    dobj = get_json(session, digital[0]["id"], limiter)
    for ap in dobj.get("access_point") or []:
        if ap.get("id"):
            return ap["id"], license_url
    return None, "no-access-point"


def resized_iiif_url(image_url, max_width=MAX_WIDTH):
    """Rewrite a ``.../full/max/...`` IIIF URL to request width ``max_width``.

    Uses the IIIF Image API ``sizeByW`` feature so the server downscales; leaves
    non-matching URLs untouched.
    """
    marker = "/full/max/"
    if marker in image_url:
        return image_url.replace(marker, f"/full/{max_width},/")
    return image_url


def ensure_max_width(path, max_width=MAX_WIDTH):
    """Safety net: downscale ``path`` in place if wider than ``max_width``.

    Never upscales. Normally a no-op because IIIF already returned <= max_width,
    but it enforces the invariant when we had to fall back to the original.
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

    Non-429 client errors (4xx) are not retried, so a too-large ``sizeByW``
    request (source narrower than the cap) fails fast into the caller's
    original-image fallback.
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


def lod_id(object_id):
    """Last path segment of a LOD URL, used as a stable filename stem."""
    return object_id.rstrip("/").rsplit("/", 1)[-1]


def process_object(session, limiter, object_id, out_dir):
    """Full per-object job: resolve -> skip/download. Returns a result dict.

    Never raises: a non-rate-limit HTTP error (e.g. a 404 for a withdrawn
    record) becomes a skipped result and a corrupt/undecodable download becomes
    a failed result, so one bad object cannot abort the whole run.
    """
    stem = lod_id(object_id)
    out_path = os.path.join(out_dir, f"{stem}.jpg")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return {"id": object_id, "status": "exists", "path": out_path}

    try:
        image_url, info = resolve_image(session, limiter, object_id)
    except requests.RequestException as exc:
        return {"id": object_id, "status": "skipped", "reason": f"http-error:{exc}"}
    if image_url is None:
        return {"id": object_id, "status": "skipped", "reason": info}

    # Prefer the server-side resized image; fall back to the original when the
    # source is narrower than the cap (resized request 400s).
    resized_url = resized_iiif_url(image_url)
    used_url = resized_url
    ok = download_image(session, limiter, resized_url, out_path)
    if not ok and resized_url != image_url:
        used_url = image_url
        ok = download_image(session, limiter, image_url, out_path)

    if not ok:
        return {"id": object_id, "status": "failed", "image_url": image_url}

    try:
        ensure_max_width(out_path)
    except Exception as exc:  # undecodable/corrupt image: fail this one, keep going
        return {"id": object_id, "status": "failed", "image_url": used_url,
                "reason": f"decode:{exc}"}
    return {
        "id": object_id,
        "status": "downloaded",
        "image_url": used_url,
        "license": info,
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
        description="Download reusable Rijksmuseum paintings (concurrent, rate-limited)."
    )
    parser.add_argument("--out-dir", default="rijks_paintings",
                        help="Directory for downloaded images (default: rijks_paintings).")
    parser.add_argument("--metadata-csv", default="rijks_paintings.csv",
                        help="Append-mode CSV log of downloads (default: rijks_paintings.csv).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download workers (default: 8).")
    parser.add_argument("--max-req-per-s", type=float, default=10.0,
                        help="Global cap on HTTP requests per second across all workers (default: 10).")
    parser.add_argument("--limit", type=int,
                        help="Only process the first N objects (for testing/partial runs).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    limiter = RateLimiter(args.max_req_per_s)
    session = requests.Session()

    print(f"Listing painting objects (type=painting, imageAvailable=true)"
          + (f", limit={args.limit}" if args.limit else "") + " ...")
    object_ids = list(iter_object_ids(session, limiter, limit=args.limit))

    # Resume: skip objects whose image is already on disk from an earlier run.
    existing = scan_existing_stems(args.out_dir)
    pending = [oid for oid in object_ids if lod_id(oid) not in existing]
    already = len(object_ids) - len(pending)
    print(f"Got {len(object_ids)} object IDs; {len(existing)} images already in "
          f"{args.out_dir} ({already} of these objects present). "
          f"Downloading {len(pending)} with {args.workers} workers, "
          f"<= {args.max_req_per_s} req/s.")

    csv_exists = os.path.exists(args.metadata_csv)
    csv_lock = threading.Lock()
    stats = {"downloaded": 0, "exists": already, "skipped": 0, "failed": 0}

    with open(args.metadata_csv, "a", newline="", encoding="utf-8") as meta_f:
        writer = csv.DictWriter(
            meta_f, fieldnames=["lod_id", "object_url", "image_url", "license", "path"]
        )
        if not csv_exists:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_object, session, limiter, oid, args.out_dir): oid
                for oid in pending
            }
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                res = future.result()
                stats[res["status"]] += 1
                if res["status"] == "downloaded":
                    with csv_lock:
                        writer.writerow({
                            "lod_id": lod_id(res["id"]),
                            "object_url": res["id"],
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
        "Rijksmuseum", sorted(REUSE_LICENSES), args, stats)

    print("\nDone.")
    for k in ("downloaded", "exists", "skipped", "failed"):
        print(f"  {k:<11}: {stats[k]}")
    print(f"  images dir : {args.out_dir}")
    print(f"  metadata   : {args.metadata_csv}")
    print(f"  manifest   : {manifest_path}")


if __name__ == "__main__":
    main()
