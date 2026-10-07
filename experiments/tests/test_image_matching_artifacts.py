"""Tests for shared image-matching Parquet artifact helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pyarrow.parquet as pq

sys.modules.setdefault(
    "dotenv",
    types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False),
)

from experiments.core.env import env_image_files_parquet_path  # noqa: E402
from image_matching.utils.artifacts import (  # noqa: E402
    load_auction_to_lost_rankings_with_paths,
    read_image_files_parquet,
    summarize_auction_to_lost_rankings,
    write_auction_to_lost_rankings_parquet,
    write_image_files_parquet,
)


class ArtifactParquetTests(unittest.TestCase):
    def test_image_files_roundtrip_stores_relative_paths(self) -> None:
        with _temporary_dirs() as dirs:
            image_root = dirs.image_root
            write_image_files_parquet(
                "auction",
                [
                    {"file_id": "a", "file_path": image_root / "a.jpg"},
                    {"file_id": "b", "file_path": image_root / "nested" / "b.jpg"},
                ],
            )

            table = pq.read_table(env_image_files_parquet_path("auction"))
            self.assertEqual(table.column("file_path").to_pylist(), ["a.jpg", "nested/b.jpg"])
            self.assertEqual(
                read_image_files_parquet("auction"),
                {
                    "a": str((image_root / "a.jpg").resolve()),
                    "b": str((image_root / "nested" / "b.jpg").resolve()),
                },
            )

    def test_rankings_load_joined_paths_sorted_by_rank(self) -> None:
        with _temporary_dirs() as dirs:
            image_root = dirs.image_root
            write_image_files_parquet(
                "auction",
                [{"file_id": "a1", "file_path": image_root / "auction" / "a1.jpg"}],
            )
            write_image_files_parquet(
                "lost",
                [
                    {"file_id": "l1", "file_path": image_root / "lost" / "l1.jpg"},
                    {"file_id": "l2", "file_path": image_root / "lost" / "l2.jpg"},
                ],
            )
            write_auction_to_lost_rankings_parquet(
                "part-000000.parquet",
                auction_file_ids=["a1", "a1"],
                lost_file_ids=["l2", "l1"],
                ranks=[2, 1],
                blocking_scores=[0.25, 0.5],
            )

            self.assertEqual(
                summarize_auction_to_lost_rankings(),
                {"part_count": 1, "row_count": 2, "auction_file_count": 1},
            )
            self.assertEqual(
                list(load_auction_to_lost_rankings_with_paths(batch_size=1)),
                [
                    {
                        "auction_file_id": "a1",
                        "auction_file_path": str((image_root / "auction" / "a1.jpg").resolve()),
                        "match_candidates": [
                            {
                                "lost_file_id": "l1",
                                "lost_file_path": str((image_root / "lost" / "l1.jpg").resolve()),
                                "blocking_score": 0.5,
                            },
                            {
                                "lost_file_id": "l2",
                                "lost_file_path": str((image_root / "lost" / "l2.jpg").resolve()),
                                "blocking_score": 0.25,
                            },
                        ],
                    }
                ],
            )


class _TempDirs:
    def __init__(self, tmp_dir: tempfile.TemporaryDirectory[str]) -> None:
        self._tmp_dir = tmp_dir
        self.root = Path(tmp_dir.name)
        self.image_root = self.root / "images"


class _temporary_dirs:
    def __enter__(self) -> _TempDirs:
        self._previous_cache = os.environ.get("CACHE_DIR")
        self._previous_image_root = os.environ.get("SMARTMATCH_IMAGES_DIR")
        self._tmp = tempfile.TemporaryDirectory()
        dirs = _TempDirs(self._tmp)
        dirs.image_root.mkdir(parents=True)
        os.environ["CACHE_DIR"] = str((dirs.root / "cache").resolve())
        os.environ["SMARTMATCH_IMAGES_DIR"] = str(dirs.image_root.resolve())
        return dirs

    def __exit__(self, *_exc_info) -> None:
        self._tmp.cleanup()
        _restore_env("CACHE_DIR", self._previous_cache)
        _restore_env("SMARTMATCH_IMAGES_DIR", self._previous_image_root)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
