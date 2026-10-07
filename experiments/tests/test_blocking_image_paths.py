"""Tests for blocking image-file path resolution."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault(
    "dotenv",
    types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False),
)

from image_matching.blocking.image_paths import (  # noqa: E402
    image_file_path,
    normalize_file_extension,
    require_image_file,
)


class BlockingImagePathTests(unittest.TestCase):
    def test_non_uuid_file_id_is_used_as_filename_stem(self) -> None:
        root = Path("/image-root")

        self.assertEqual(
            image_file_path(root, "auction-image-123", "JPG"),
            root / "auction-image-123.jpg",
        )

    def test_none_extension_sentinel_omits_dot_and_extension(self) -> None:
        root = Path("/image-root")

        self.assertEqual(
            image_file_path(root, "lost-image-123", "NONE"),
            root / "lost-image-123",
        )
        self.assertEqual(
            image_file_path(root, "lost-image-123", ".none"),
            root / "lost-image-123",
        )
        self.assertEqual(
            image_file_path(root, "lost-image-123", None),
            root / "lost-image-123",
        )

    def test_require_image_file_accepts_extensionless_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "extensionless-image"
            image_path.write_bytes(b"image")

            self.assertEqual(
                require_image_file(root, "extensionless-image", "NONE"),
                image_path,
            )

    def test_invalid_path_like_file_ids_are_rejected(self) -> None:
        root = Path("/image-root")

        with self.assertRaisesRegex(ValueError, "Invalid image file id"):
            image_file_path(root, "../outside", "jpg")
        with self.assertRaisesRegex(ValueError, "Invalid image file id"):
            image_file_path(root, "nested/image", "jpg")

    def test_invalid_extensions_are_rejected_except_none_sentinel(self) -> None:
        self.assertEqual(normalize_file_extension("NONE"), "")
        with self.assertRaisesRegex(ValueError, "Invalid image file extension"):
            normalize_file_extension("jpg/../../bad")


if __name__ == "__main__":
    unittest.main()
