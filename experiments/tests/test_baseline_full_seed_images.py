"""Manual integration test for baseline retrieval models on seed images.

This test is intentionally opt-in because it downloads/loads large embedding models.
Run it with:

    RUN_BLOCKING_BASELINE_SEED_TEST=1 \
    python -m unittest experiments.tests.test_baseline_full_seed_images

Optional overrides:
    BLOCKING_BASELINE_SEED_GLOB='db/images/*seed*'
    BLOCKING_BASELINE_SEED_WORK_DIR='local/blocking_baseline_seed_test'
"""

from __future__ import annotations

import glob
import os
import re
import sqlite3
import time
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from experiments.core.config_schema import DatasetConfig, load_experiment_config
from experiments.core.runner import run_experiment
from experiments.core.split_schema import materialize_baseline_split, write_split

RUN_ENV = "RUN_BLOCKING_BASELINE_SEED_TEST"
SEED_GLOB_ENV = "BLOCKING_BASELINE_SEED_GLOB"
WORK_DIR_ENV = "BLOCKING_BASELINE_SEED_WORK_DIR"
DEFAULT_SEED_GLOB = "db/images/*seed*"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SCHEMA = """
CREATE TABLE classes(class_id INTEGER PRIMARY KEY, qid TEXT NOT NULL UNIQUE, label TEXT);
CREATE TABLE image_files(file_id TEXT PRIMARY KEY, file_ext TEXT NOT NULL, local_rel_path TEXT, download_status TEXT NOT NULL, source_role TEXT);
CREATE TABLE image_file_classes(image_file_class_id INTEGER PRIMARY KEY AUTOINCREMENT, file_id TEXT NOT NULL, class_id INTEGER NOT NULL, primary_label TEXT);
CREATE TABLE image_file_class_tags(image_file_class_id INTEGER NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(image_file_class_id, tag));
CREATE TABLE splits(split_name TEXT NOT NULL, file_id TEXT NOT NULL, PRIMARY KEY(split_name, file_id));
"""


class BaselineFullSeedImagesIntegrationTest(unittest.TestCase):
    def test_full_baseline_models_on_seed_images(self) -> None:
        if os.getenv(RUN_ENV) != "1":
            self.skipTest(f"Set {RUN_ENV}=1 to run the heavyweight seed-image baseline test")

        repo_root = Path(__file__).resolve().parents[3]
        image_paths = _discover_seed_images(repo_root, os.getenv(SEED_GLOB_ENV, DEFAULT_SEED_GLOB))
        if not image_paths:
            self.skipTest(f"No seed images matched {os.getenv(SEED_GLOB_ENV, DEFAULT_SEED_GLOB)!r}")

        timestamp = f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        work_dir = Path(os.getenv(WORK_DIR_ENV, repo_root / "local" / "blocking_baseline_seed_test")).expanduser()
        if not work_dir.is_absolute():
            work_dir = repo_root / work_dir
        test_dir = work_dir / timestamp
        test_dir.mkdir(parents=True, exist_ok=False)

        experiment_path = _write_seed_experiment(repo_root, test_dir, image_paths)
        experiment = load_experiment_config(experiment_path)
        rows: list[dict[str, str]] = []
        for model in experiment.models:
            started = time.monotonic()
            run_id = f"{timestamp}_{_safe_token(model.model_id)}"
            try:
                run_dir = run_experiment(experiment_path, model_filter=model.model_id, run_id=run_id)
            except Exception as exc:  # keep trying remaining models and summarize all failures
                rows.append(
                    {
                        "status": "FAILED",
                        "model_id": model.model_id,
                        "seconds": f"{time.monotonic() - started:.1f}",
                        "detail": _one_line_error(exc),
                    }
                )
            else:
                rows.append(
                    {
                        "status": "WORKED",
                        "model_id": model.model_id,
                        "seconds": f"{time.monotonic() - started:.1f}",
                        "detail": _display_path(run_dir, repo_root),
                    }
                )

        print("\nFull baseline seed-image model summary")
        print(_format_table(rows))
        failed = [row for row in rows if row["status"] != "WORKED"]
        self.assertFalse(failed, "One or more baseline models failed; see summary table above")


def _discover_seed_images(repo_root: Path, pattern: str) -> list[Path]:
    raw_pattern = Path(pattern).expanduser()
    search_pattern = str(raw_pattern if raw_pattern.is_absolute() else repo_root / raw_pattern)
    found: list[Path] = []
    for match in sorted(Path(path) for path in glob.glob(search_pattern, recursive=True)):
        if match.is_dir():
            found.extend(path for path in sorted(match.rglob("*")) if _is_image(path))
        elif _is_image(match):
            found.append(match)
    return sorted(dict.fromkeys(path.resolve() for path in found))


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _write_seed_experiment(repo_root: Path, test_dir: Path, image_paths: list[Path]) -> Path:
    db_path = test_dir / "seed_dataset.db"
    class_keys = [_class_key(path) for path in image_paths]
    counts = Counter(class_keys)
    if not counts or max(counts.values()) < 2:
        raise unittest.SkipTest("Seed images must contain at least one inferred same-painting pair")
    _write_seed_dataset_db(db_path, image_paths, class_keys)

    dataset = DatasetConfig(
        path=test_dir / "seed_dataset.yml",
        dataset_id="seed-images",
        dataset_db=db_path,
        image_root=repo_root / "db" / "images",
        identity={"expected_num_classes": len(counts), "expected_num_images": len(image_paths)},
        raw={},
    )
    split = materialize_baseline_split(dataset, "baseline_full")
    _assert_split_has_positives(split)
    split_path = test_dir / "baseline_full_seed_split.json"
    write_split(split_path, split)

    dataset_yml = _dataset_config(dataset)
    dataset.path.write_text(yaml.safe_dump(dataset_yml, sort_keys=False), encoding="utf-8")

    baseline_path = repo_root / "experiments" / "configs" / "experiments" / "baseline_full.yml"
    baseline = load_experiment_config(baseline_path)
    experiment = dict(baseline.raw)
    experiment["experiment_id"] = "baseline_full_seed_images_v1"
    experiment["description"] = "Full baseline retrieval models on images matching db/images/*seed*."
    experiment["dataset"] = {"path": str(test_dir), "image_root": str(dataset.image_root)}
    experiment["split"] = {"file": str(split_path)}
    experiment["models"] = {
        "include": [
            {"config": str(model.path), "model_id": model.model_id}
            for model in baseline.models
        ]
    }
    experiment["embedding"] = {**dict(baseline.raw.get("embedding") or {}), "cache_dir": str(test_dir / "embedding_cache")}
    experiment["run"] = {**dict(baseline.raw.get("run") or {}), "output_dir": str(test_dir / "runs")}

    experiment_path = test_dir / "baseline_full_seed_experiment.yml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return experiment_path


def _write_seed_dataset_db(db_path: Path, image_paths: list[Path], class_keys: list[str]) -> None:
    class_id_by_key = {key: idx + 1 for idx, key in enumerate(sorted(set(class_keys)))}
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO classes(class_id, qid, label) VALUES (?, ?, ?)",
            [(class_id, key, key) for key, class_id in class_id_by_key.items()],
        )
        for path, class_key in zip(image_paths, class_keys, strict=True):
            file_id = _file_id(path)
            role = _source_role(path)
            tags = [role] if role else []
            if role == "historic":
                tags.append("historic")
            conn.execute(
                "INSERT INTO image_files(file_id, file_ext, local_rel_path, download_status, source_role) VALUES (?, ?, ?, ?, ?)",
                (file_id, path.suffix.lower(), str(path), "downloaded", role),
            )
            cursor = conn.execute(
                "INSERT INTO image_file_classes(file_id, class_id, primary_label) VALUES (?, ?, ?)",
                (file_id, class_id_by_key[class_key], "same_painting"),
            )
            image_file_class_id = int(cursor.lastrowid)
            for tag in tags:
                conn.execute(
                    "INSERT INTO image_file_class_tags(image_file_class_id, tag) VALUES (?, ?)",
                    (image_file_class_id, tag),
                )
            split_names = ["same_painting", "query_default", "candidate_default", *([role] if role else [])]
            for split_name in split_names:
                conn.execute("INSERT INTO splits(split_name, file_id) VALUES (?, ?)", (split_name, file_id))
        conn.commit()


def _dataset_config(dataset: DatasetConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_id": dataset.dataset_id,
        "paths": {"dataset_db": str(dataset.dataset_db), "image_root": str(dataset.image_root)},
        "identity": dataset.identity,
    }


def _assert_split_has_positives(split: dict[str, Any]) -> None:
    class_by_id = {file_id: record["class_id"] for file_id, record in split["images"].items()}
    roles = split["subsets"]["eval"]["roles"]
    positives = 0
    for query_id in roles["modern"]:
        for candidate_id in roles["historic"]:
            positives += int(class_by_id[query_id] == class_by_id[candidate_id])
    if positives == 0:
        raise unittest.SkipTest("Seed baseline split has no inferred modern-to-historic positive pairs")


def _class_key(path: Path) -> str:
    stem = path.stem
    ignored_tokens = {"seed", "historic", "modern", "auction", "lost"}
    tokens = [token for token in re.split(r"[_.-]+", stem) if token]
    filtered = [token for token in tokens if token.lower() not in ignored_tokens]
    return "_".join(filtered) or stem


def _source_role(path: Path) -> str | None:
    stem = path.stem.lower()
    if re.search(r"(^|[_.-])(historic|lost)([_.-]|$)", stem):
        return "historic"
    if re.search(r"(^|[_.-])(modern|auction)([_.-]|$)", stem):
        return "modern"
    return None


def _file_id(path: Path) -> str:
    return _safe_token(path.with_suffix("").as_posix())


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _one_line_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:240]


def _format_table(rows: list[dict[str, str]]) -> str:
    headers = ["status", "model_id", "seconds", "detail"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row.get(header, "")))
    separator = "| " + " | ".join("-" * widths[header] for header in headers) + " |"
    lines = ["| " + " | ".join(header.ljust(widths[header]) for header in headers) + " |", separator]
    lines.extend("| " + " | ".join(row.get(header, "").ljust(widths[header]) for header in headers) + " |" for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    unittest.main()
