"""Stage 1 — ingest raw source directories into the shared image directory (TECHNICAL_PLAN 9).

Raw source files are never mutated. Every supported raw image is copied into ``work/images/`` with
a source-prefixed flat filename, byte/sha verified, and validated with Pillow before any downstream
stage runs. A complete collision preflight runs before anything is copied.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import file_io, manifest
from .config import Config

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path


def parse_source_arg(value: str) -> SourceSpec:
    """Parse a ``--source-dir`` value: ``PATH`` or ``NAME=PATH``."""
    whole_path = Path(value).expanduser()
    if "=" in value and not whole_path.exists():
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"Invalid NAME=PATH source specification: {value!r}")
        root = Path(path).expanduser().resolve()
    else:
        root = whole_path.resolve()
        name = root.name
    return SourceSpec(name=safe_source_name(name), root=root)


def safe_source_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name.strip()).strip("_")
    if not cleaned:
        raise ValueError(f"Source name {name!r} is not filesystem-safe")
    return cleaned


@dataclass(frozen=True)
class RawFile:
    source_name: str
    raw_relative_path: str
    original_filename: str
    abs_path: Path

    @property
    def image_id(self) -> str:
        return manifest.make_image_id(self.source_name, self.raw_relative_path)

    @property
    def shared_filename(self) -> str:
        return f"{self.source_name}_{self.original_filename}"


def enumerate_source(spec: SourceSpec, extensions: tuple[str, ...]) -> list[RawFile]:
    if not spec.root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {spec.root}")
    exts = {e.lower() for e in extensions}
    files: list[RawFile] = []
    for path in spec.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(spec.root).as_posix()
        files.append(
            RawFile(
                source_name=spec.name,
                raw_relative_path=rel,
                original_filename=path.name,
                abs_path=path,
            )
        )
    files.sort(key=lambda f: (f.source_name, f.raw_relative_path))
    return files


def collision_report(raw_files: list[RawFile], images_dir: Path) -> list[str]:
    """Return a complete list of collision problems; empty means safe to copy."""
    problems: list[str] = []
    by_target: dict[str, list[RawFile]] = {}
    for rf in raw_files:
        by_target.setdefault(rf.shared_filename, []).append(rf)
    for target, group in sorted(by_target.items()):
        if len(group) > 1:
            identities = sorted((rf.source_name, rf.raw_relative_path) for rf in group)
            joined = "; ".join(f"{s}:{p}" for s, p in identities)
            problems.append(f"target {target!r} produced by multiple raw files -> {joined}")
    return problems


def _pillow_validate(path: Path) -> tuple[bool, str, int, int, str]:
    """Verify + reopen + decode. Returns (ok, mime, width, height, error)."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mime = Image.MIME.get(img.format or "", "") or ""
        return True, mime, int(width), int(height), ""
    except Exception as exc:  # PIL raises many exception classes for bad files.
        return False, "", 0, 0, repr(exc)


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def ingest(config: Config, source_specs: list[SourceSpec]) -> dict[str, object]:
    """Run ingestion; returns a summary dict with manifest hashes."""
    images_dir = config.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    raw_files: list[RawFile] = []
    for spec in source_specs:
        raw_files.extend(enumerate_source(spec, config.extensions))
    raw_files.sort(key=lambda f: (f.source_name, f.raw_relative_path))

    # Complete collision preflight before copying anything.
    problems = collision_report(raw_files, images_dir)
    # Also flag existing targets with different bytes.
    for rf in raw_files:
        dst = images_dir / rf.shared_filename
        if dst.exists():
            if file_io.sha256_file(dst) != file_io.sha256_file(rf.abs_path):
                problems.append(
                    f"target {rf.shared_filename!r} already exists with different bytes "
                    f"(raw {rf.source_name}:{rf.raw_relative_path})"
                )
    if problems:
        raise ValueError("Ingestion collision preflight failed:\n  - " + "\n  - ".join(problems))

    source_rows: list[dict] = []
    image_rows: list[dict] = []
    invalid_rows: list[dict] = []

    for rf in raw_files:
        dst = images_dir / rf.shared_filename
        shared_rel = file_io.rel_to_project(dst)
        raw_sha = file_io.sha256_file(rf.abs_path)
        raw_bytes = rf.abs_path.stat().st_size

        if dst.exists() and file_io.sha256_file(dst) == raw_sha:
            pass  # restart-safe: identical shared copy already present
        else:
            _copy_atomic(rf.abs_path, dst)

        copied_bytes = dst.stat().st_size
        copied_sha = file_io.sha256_file(dst)
        if copied_bytes != raw_bytes or copied_sha != raw_sha:
            raise RuntimeError(
                f"Copy verification failed for {rf.shared_filename}: "
                f"bytes {copied_bytes}!={raw_bytes} or sha mismatch"
            )

        source_rows.append(
            {
                "image_id": rf.image_id,
                "source_name": rf.source_name,
                "raw_relative_path": rf.raw_relative_path,
                "original_filename": rf.original_filename,
                "shared_filename": rf.shared_filename,
                "shared_rel_path": shared_rel,
                "bytes_on_disk": copied_bytes,
                "sha256": copied_sha,
            }
        )

        ok, mime, width, height, error = _pillow_validate(dst)
        if ok:
            image_rows.append(
                {
                    "image_id": rf.image_id,
                    "source_name": rf.source_name,
                    "raw_relative_path": rf.raw_relative_path,
                    "shared_filename": rf.shared_filename,
                    "shared_rel_path": shared_rel,
                    "mime_type": mime,
                    "width": width,
                    "height": height,
                    "bytes_on_disk": copied_bytes,
                    "sha256": copied_sha,
                }
            )
        else:
            invalid_rows.append(
                {
                    "image_id": rf.image_id,
                    "source_name": rf.source_name,
                    "raw_relative_path": rf.raw_relative_path,
                    "shared_filename": rf.shared_filename,
                    "shared_rel_path": shared_rel,
                    "bytes_on_disk": copied_bytes,
                    "sha256": copied_sha,
                    "error": error,
                }
            )

    work = file_io.work_dir()
    source_csv = work / "source_files.csv"
    images_csv = work / "images.csv"
    invalid_csv = work / "invalid_images.csv"
    file_io.atomic_write_csv(source_csv, manifest.SOURCE_FILES_FIELDS, source_rows)
    file_io.atomic_write_csv(images_csv, manifest.IMAGES_FIELDS, image_rows)
    file_io.atomic_write_csv(invalid_csv, manifest.INVALID_IMAGES_FIELDS, invalid_rows)

    orphans = _detect_orphans(images_dir, {rf.shared_filename for rf in raw_files})
    file_io.atomic_write_csv(
        work / "orphaned_shared_images.csv", manifest.ORPHANED_SHARED_IMAGES_FIELDS, orphans
    )

    return {
        "sources": [{"name": s.name} for s in source_specs],
        "num_raw_files": len(raw_files),
        "num_valid_images": len(image_rows),
        "num_invalid_images": len(invalid_rows),
        "num_orphaned_shared_images": len(orphans),
        "manifest_hashes": {
            "source_files.csv": file_io.sha256_file(source_csv),
            "images.csv": file_io.sha256_file(images_csv),
            "invalid_images.csv": file_io.sha256_file(invalid_csv),
        },
    }


def _detect_orphans(images_dir: Path, known_targets: set[str]) -> list[dict]:
    orphans: list[dict] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name in known_targets:
            continue
        orphans.append(
            {
                "shared_filename": path.name,
                "shared_rel_path": file_io.rel_to_project(path),
                "bytes_on_disk": path.stat().st_size,
                "sha256": file_io.sha256_file(path),
            }
        )
    return orphans
