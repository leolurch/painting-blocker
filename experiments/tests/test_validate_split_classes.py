import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from experiments.core.validate_split_classes import (
    class_sets_by_subset,
    main,
    validate_split_class_disjointness,
)


def _split_with_classes(train: list[int], val: list[int], test: list[int]) -> dict:
    images = {}
    subsets = {}
    for subset_name, class_ids in (("train", train), ("val", val), ("test", test)):
        role_ids = []
        for index, class_id in enumerate(class_ids, start=1):
            file_id = f"{subset_name}_{index}.jpg"
            images[file_id] = {"class_id": class_id}
            role_ids.append(file_id)
        subsets[subset_name] = {"roles": {"all": role_ids}}
    return {
        "schema_version": 2,
        "split_id": "toy_class_disjoint",
        "dataset": {"dataset_id": "toy", "image_root": "/tmp/images"},
        "subsets": subsets,
        "images": images,
    }


class ValidateSplitClassesTest(unittest.TestCase):
    def test_collects_class_sets_by_subset(self) -> None:
        split = _split_with_classes(train=[1, 2], val=[3], test=[4])

        class_sets, image_counts = class_sets_by_subset(split)

        self.assertEqual(class_sets["train"], {1, 2})
        self.assertEqual(class_sets["val"], {3})
        self.assertEqual(class_sets["test"], {4})
        self.assertEqual(image_counts, {"train": 2, "val": 1, "test": 1})

    def test_passes_when_train_and_val_are_disjoint_from_test(self) -> None:
        split = _split_with_classes(train=[1, 2], val=[3], test=[4, 5])

        result = validate_split_class_disjointness(split)

        self.assertTrue(result.ok)
        self.assertEqual(result.critical_overlaps, ())
        self.assertEqual(result.subset_class_counts["test"], 2)

    def test_fails_for_train_test_class_overlap(self) -> None:
        split = _split_with_classes(train=[1, 2], val=[3], test=[2, 4])

        result = validate_split_class_disjointness(split)

        self.assertFalse(result.ok)
        self.assertEqual(len(result.critical_overlaps), 1)
        self.assertEqual(result.critical_overlaps[0].left_subset, "train")
        self.assertEqual(result.critical_overlaps[0].right_subset, "test")
        self.assertEqual(result.critical_overlaps[0].class_ids, (2,))

    def test_fails_for_val_test_class_overlap(self) -> None:
        split = _split_with_classes(train=[1], val=[3, 4], test=[2, 4])

        result = validate_split_class_disjointness(split)

        self.assertFalse(result.ok)
        overlaps = {(item.left_subset, item.right_subset): item.class_ids for item in result.critical_overlaps}
        self.assertEqual(overlaps[("val", "test")], (4,))

    def test_cli_returns_failure_for_critical_overlap(self) -> None:
        split = _split_with_classes(train=[1], val=[2], test=[1])
        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                exit_code = main([str(split_path), "--json"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
