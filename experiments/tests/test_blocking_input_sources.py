"""Tests for DB-derived blocking input row resolution."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault(
    "dotenv",
    types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False),
)
sys.modules.setdefault(
    "psycopg",
    types.SimpleNamespace(connect=lambda *args, **kwargs: None),
)

from image_matching.blocking import pipeline  # noqa: E402
from image_matching.blocking.config import AUCTION_ROLE, LOST_ROLE, candidate_dir, repo_root  # noqa: E402
from image_matching.blocking.input_sources import ImageFileRow, _db_query, _fetch_db_rows  # noqa: E402


class BlockingInputSourceTests(unittest.TestCase):
    def test_db_rows_use_image_root_relative_file_path(self) -> None:
        with _temporary_image_root() as image_root:
            rows = _fetch_db_rows(
                _FakeConn([("28732", "28732.jpg")]),
                LOST_ROLE,
                None,
                False,
                False,
            )

            self.assertEqual(rows[0].file_path, image_root / "28732.jpg")

    def test_db_rows_use_repo_relative_file_path_under_image_root(self) -> None:
        image_root = repo_root() / "db" / "images"
        with _temporary_image_root(image_root):
            rows = _fetch_db_rows(
                _FakeConn([("57457", "db/images/auction.jpg")]),
                LOST_ROLE,
                None,
                False,
                False,
            )

            self.assertEqual(rows[0].file_path, image_root / "auction.jpg")

    def test_db_rows_reject_missing_file_path(self) -> None:
        with _temporary_image_root():
            with self.assertRaisesRegex(ValueError, "Missing image_file.file_path"):
                _fetch_db_rows(
                    _FakeConn([("missing-path", None)]),
                    LOST_ROLE,
                    None,
                    False,
                    False,
                )

    def test_db_rows_still_validate_files_for_blocking_runs(self) -> None:
        with _temporary_image_root() as image_root:
            existing = image_root / "existing.jpg"
            existing.write_bytes(b"image")

            rows = _fetch_db_rows(
                _FakeConn([("existing-image", "existing.jpg")]),
                LOST_ROLE,
                None,
                False,
                True,
            )
            self.assertEqual(rows[0].file_path, existing)

            with self.assertRaisesRegex(FileNotFoundError, "Image file not found"):
                _fetch_db_rows(
                    _FakeConn([("missing-image", "missing.jpg")]),
                    LOST_ROLE,
                    None,
                    False,
                    True,
                )

    def test_auction_limit_query_limits_artworks_not_images(self) -> None:
        sql = _db_query(AUCTION_ROLE, False, use_auction_artwork_limit=True)

        self.assertIn("WITH selected_auction_artwork", sql)
        self.assertIn("LIMIT %s", sql)
        self.assertIn("GROUP BY aaif.auction_artwork_id", sql)

    def test_blocking_skips_db_input_loading_when_no_unprocessed_rows(self) -> None:
        with _temporary_blocking_dirs():
            stale_part = candidate_dir() / "part-000000-stale.parquet"
            stale_part.parent.mkdir(parents=True, exist_ok=True)
            stale_part.write_bytes(b"stale")

            with mock.patch.object(
                pipeline,
                "has_unprocessed_auction_image_file_rows",
                return_value=False,
            ), mock.patch.object(
                pipeline,
                "_load_inputs",
                side_effect=AssertionError("DB inputs should not be loaded"),
            ), mock.patch.object(
                pipeline,
                "ensure_lost_embedding_cache",
                side_effect=AssertionError("lost cache should not be prepared"),
            ), mock.patch.object(
                pipeline,
                "_write_candidates",
                side_effect=AssertionError("candidates should not be generated"),
            ):
                result = pipeline.run_image_blocking()

            self.assertEqual(result.lost_image_count, 0)
            self.assertEqual(result.auction_image_count, 0)
            self.assertEqual(result.candidate_count, 0)
            self.assertEqual(result.generated_lost_embedding_count, 0)
            self.assertFalse(stale_part.exists())

    def test_blocking_skips_model_work_when_loaded_input_has_no_auction_rows(self) -> None:
        with _temporary_blocking_dirs() as dirs:
            lost_file = dirs.image_root / "lost.jpg"
            lost_file.write_bytes(b"lost image")
            stale_part = candidate_dir() / "part-000000-stale.parquet"
            stale_part.parent.mkdir(parents=True, exist_ok=True)
            stale_part.write_bytes(b"stale")

            with mock.patch.object(
                pipeline,
                "_load_inputs",
                return_value=([ImageFileRow("1", lost_file)], []),
            ), mock.patch.object(
                pipeline,
                "ensure_lost_embedding_cache",
                side_effect=AssertionError("lost cache should not be prepared"),
            ), mock.patch.object(
                pipeline,
                "_write_candidates",
                side_effect=AssertionError("candidates should not be generated"),
            ):
                result = pipeline.run_image_blocking(input_csv=Path("snapshot.csv"))

            self.assertEqual(result.lost_image_count, 1)
            self.assertEqual(result.auction_image_count, 0)
            self.assertEqual(result.candidate_count, 0)
            self.assertEqual(result.generated_lost_embedding_count, 0)
            self.assertFalse(stale_part.exists())


class _FakeConn:
    def __init__(self, rows: list[tuple[str, str | None]]) -> None:
        self.rows = rows

    def cursor(self) -> "_FakeCursor":
        return _FakeCursor(self.rows)


class _FakeCursor:
    def __init__(self, rows: list[tuple[str, str | None]]) -> None:
        self.rows = rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *_exc_info) -> None:
        return None

    def execute(self, _sql: str, _params: tuple[int, ...] = ()) -> None:
        return None

    def fetchall(self) -> list[tuple[str, str | None]]:
        return self.rows


class _TempBlockingDirs:
    def __init__(self, tmp_dir: tempfile.TemporaryDirectory[str]) -> None:
        self._tmp_dir = tmp_dir
        self.root = Path(tmp_dir.name).resolve()
        self.image_root = self.root / "images"
        self.cache_root = self.root / "cache"


class _temporary_blocking_dirs:
    def __enter__(self) -> _TempBlockingDirs:
        self._previous_cache = os.environ.get("CACHE_DIR")
        self._previous_image_root = os.environ.get("SMARTMATCH_IMAGES_DIR")
        self._tmp = tempfile.TemporaryDirectory()
        dirs = _TempBlockingDirs(self._tmp)
        dirs.image_root.mkdir(parents=True)
        dirs.cache_root.mkdir(parents=True)
        os.environ["CACHE_DIR"] = str(dirs.cache_root)
        os.environ["SMARTMATCH_IMAGES_DIR"] = str(dirs.image_root)
        return dirs

    def __exit__(self, *_exc_info) -> None:
        self._tmp.cleanup()
        _restore_env("CACHE_DIR", self._previous_cache)
        _restore_env("SMARTMATCH_IMAGES_DIR", self._previous_image_root)


class _temporary_image_root:
    def __init__(self, image_root: Path | None = None) -> None:
        self._requested_image_root = image_root

    def __enter__(self) -> Path:
        self._previous_image_root = os.environ.get("SMARTMATCH_IMAGES_DIR")
        self._tmp = tempfile.TemporaryDirectory()
        if self._requested_image_root is None:
            self.image_root = Path(self._tmp.name).resolve()
            self.image_root.mkdir(parents=True, exist_ok=True)
        else:
            self.image_root = self._requested_image_root.resolve()
        os.environ["SMARTMATCH_IMAGES_DIR"] = str(self.image_root)
        return self.image_root

    def __exit__(self, *_exc_info) -> None:
        self._tmp.cleanup()
        _restore_env("SMARTMATCH_IMAGES_DIR", self._previous_image_root)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
