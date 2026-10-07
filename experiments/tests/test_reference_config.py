import unittest
from pathlib import Path

from experiments.core.config_schema import _validate_references


class ReferenceConfigValidationTest(unittest.TestCase):
    SOURCE = Path("experiment.yml")

    def _validate(self, references, *, known_sets=None, include_model_ids=None):
        finetune_eval = None if references is None else {"references": references}
        _validate_references(
            finetune_eval,
            self.SOURCE,
            known_sets=known_sets or {"synth_val", "wikidata_val"},
            include_model_ids=include_model_ids or ["facebook/dinov3-vith16plus-pretrain-lvd1689m"],
        )

    def test_valid_reference_resolves(self):
        self._validate(
            [
                {
                    "id": "frozen_backbone",
                    "kind": "frozen_backbone",
                    "label": "Frozen backbone",
                    "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                }
            ]
        )

    def test_no_references_is_valid(self):
        self._validate(None)
        # finetune_eval without a references key is also fine.
        _validate_references({}, self.SOURCE, known_sets=set(), include_model_ids=["m"])

    def test_unknown_model_id_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly one models.include"):
            self._validate(
                [{"id": "frozen_backbone", "kind": "frozen_backbone", "model_id": "unknown/model"}]
            )

    def test_duplicate_reference_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self._validate(
                [
                    {"id": "frozen_backbone", "kind": "frozen_backbone", "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m"},
                    {"id": "frozen_backbone", "kind": "frozen_backbone", "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m"},
                ]
            )

    def test_unsupported_kind_fails(self):
        with self.assertRaisesRegex(ValueError, "frozen_backbone"):
            self._validate(
                [{"id": "r", "kind": "trained_head", "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m"}]
            )

    def test_epoch_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "epoch"):
            self._validate(
                [
                    {
                        "id": "frozen_backbone",
                        "kind": "frozen_backbone",
                        "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                        "epoch": 0,
                    }
                ]
            )

    def test_malformed_validation_sets_filter_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            self._validate(
                [
                    {
                        "id": "frozen_backbone",
                        "kind": "frozen_backbone",
                        "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                        "validation_sets": ["does_not_exist"],
                    }
                ]
            )
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            self._validate(
                [
                    {
                        "id": "frozen_backbone",
                        "kind": "frozen_backbone",
                        "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                        "validation_sets": "synth_val",
                    }
                ]
            )

    def test_valid_validation_sets_filter(self):
        self._validate(
            [
                {
                    "id": "frozen_backbone",
                    "kind": "frozen_backbone",
                    "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                    "validation_sets": ["synth_val"],
                }
            ]
        )


if __name__ == "__main__":
    unittest.main()
