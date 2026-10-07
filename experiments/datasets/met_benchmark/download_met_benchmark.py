#!/usr/bin/env python3
"""Download the official Met benchmark archives into a target directory.

Designed to run on the cluster (e.g. data/datasets/met_benchmark). Does not
download anything when invoked without --execute; use --execute to start.

With --execute, missing archives are downloaded in parallel, then extracted
sequentially (shared images/ tree).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "http://ptak.felk.cvut.cz/met/dataset"

ARCHIVES = {
    "ground_truth": {
        "url": f"{BASE_URL}/ground_truth.tar.gz",
        "extract_to": "ground_truth",
        "done": "ground_truth/.done",
        "required": True,
    },
    "test_met": {
        "url": f"{BASE_URL}/test_met.tar.gz",
        "extract_to": "images",
        "done": "images/test_met/.done",
        "required": True,
    },
    "test_other": {
        "url": f"{BASE_URL}/test_other.tar.gz",
        "extract_to": "images",
        "done": "images/test_other/.done",
        "required": True,
    },
    "test_noart": {
        "url": f"{BASE_URL}/test_noart.tar.gz",
        "extract_to": "images",
        "done": "images/test_noart/.done",
        "required": True,
    },
    "MET": {
        "url": f"{BASE_URL}/MET.tar.gz",
        "extract_to": "images",
        "done": "images/MET/.done",
        "required": True,
    },
    "small_MET": {
        "url": f"{BASE_URL}/small_MET.tar.gz",
        "extract_to": "images",
        "done": "images/.small_MET.done",
        "required": False,
    },
}


def _selected_names(*, include_mini: bool, skip_full_train: bool) -> list[str]:
    names: list[str] = []
    for name in ARCHIVES:
        if name == "small_MET" and not include_mini:
            continue
        if name == "MET" and skip_full_train:
            continue
        names.append(name)
    return names


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "-o",
        str(dest),
        url,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive} -> {dest}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)


def _mark_done(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def download_met_benchmark(
    root: Path,
    *,
    include_mini: bool = True,
    execute: bool = False,
    skip_full_train: bool = False,
    parallel_downloads: int = 4,
) -> dict[str, object]:
    root = Path(root).expanduser().resolve()
    names = _selected_names(include_mini=include_mini, skip_full_train=skip_full_train)
    plan = []
    for name in names:
        meta = ARCHIVES[name]
        done = root / meta["done"]
        plan.append(
            {
                "name": name,
                "url": meta["url"],
                "done": str(done),
                "already_done": done.exists(),
                "required": meta["required"],
            }
        )

    if not execute:
        return {"status": "dry_run", "root": str(root), "plan": plan}

    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    pending_extract: list[str] = []

    for name in names:
        done = root / ARCHIVES[name]["done"]
        if done.exists():
            print(f"skip {name}: already done ({done})", flush=True)
            completed.append(name)
        else:
            pending_extract.append(name)

    # Parallel download of all missing archives.
    workers = max(1, min(int(parallel_downloads), len(pending_extract) or 1))

    def _fetch(name: str) -> str:
        meta = ARCHIVES[name]
        archive = root / f"{name}.tar.gz"
        print(f"[download] start {name}", flush=True)
        _download(meta["url"], archive)
        print(f"[download] done {name}", flush=True)
        return name

    if pending_extract:
        print(
            f"downloading {len(pending_extract)} archives in parallel (workers={workers}): "
            f"{pending_extract}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch, name): name for name in pending_extract}
            for future in as_completed(futures):
                name = futures[future]
                future.result()  # raise on failure

    # Sequential extract to avoid concurrent writes into images/.
    for name in pending_extract:
        meta = ARCHIVES[name]
        archive = root / f"{name}.tar.gz"
        _extract(archive, root / meta["extract_to"])
        _mark_done(root, meta["done"])
        completed.append(name)
        print(f"[extract] done {name}", flush=True)

    return {
        "status": "ok",
        "root": str(root),
        "completed": completed,
        "parallel_downloads": workers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.path.expanduser("data/datasets/met_benchmark")),
        help="Destination directory on the cluster",
    )
    parser.add_argument("--execute", action="store_true", help="Actually download/extract")
    parser.add_argument("--no-mini", action="store_true", help="Skip small_MET archive")
    parser.add_argument(
        "--skip-full-train",
        action="store_true",
        help="Skip the 28GB MET.tar.gz (useful for query-only smoke)",
    )
    parser.add_argument(
        "--parallel-downloads",
        type=int,
        default=4,
        help="Max concurrent archive downloads",
    )
    args = parser.parse_args(argv)
    result = download_met_benchmark(
        args.root,
        include_mini=not args.no_mini,
        execute=args.execute,
        skip_full_train=args.skip_full_train,
        parallel_downloads=args.parallel_downloads,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
