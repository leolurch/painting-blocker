import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.core.finetuning.reporting import write_training_reports


def _metrics(pc: float) -> dict:
    return {
        "calibrated_threshold": {
            "threshold": 0.5,
            "pair_completeness": pc,
            "pair_quality": 0.8,
            "reduction_ratio": 0.9,
            "query_coverage": 0.75,
            "candidate_pairs": 100,
        },
        "recall_at_k": [
            {"k": 1, "pair_completeness": pc, "pair_quality": 0.8, "query_coverage": 1.0},
            {"k": 10, "pair_completeness": pc + 0.05, "pair_quality": 0.7, "query_coverage": 1.0},
        ],
    }


def _history(epochs=3):
    return [
        {
            "epoch": e,
            "train_loss": 1.0 / e,
            "validation": {
                "primary": "synth_val",
                "sets": {"synth_val": _metrics(0.5 + e * 0.05), "wikidata_val": _metrics(0.4 + e * 0.03)},
            },
        }
        for e in range(1, epochs + 1)
    ]


def _references():
    return [
        {
            "id": "frozen_backbone",
            "kind": "frozen_backbone",
            "label": "Frozen backbone",
            "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "validation": {
                "primary": "synth_val",
                "sets": {"synth_val": _metrics(0.4), "wikidata_val": _metrics(0.35)},
            },
            "cache": {"sets": {}},
        }
    ]


class ReferenceReportingTest(unittest.TestCase):
    def _write(self):
        self._tmp = TemporaryDirectory()
        out = Path(self._tmp.name)
        paths = write_training_reports(
            _history(),
            out,
            references=_references(),
            best_epoch=3,
            last_epoch=3,
            best_validation_set="synth_val",
            checkpoint_selection={"validation_set": "synth_val", "metric": "calibrated_threshold"},
        )
        csv_rows = list(csv.DictReader(Path(paths["validation_metrics_csv"]).read_text().splitlines()))
        svg = Path(paths["training_metrics_svg"]).read_text()
        return csv_rows, svg

    def test_csv_contains_frozen_backbone_stage(self):
        rows, _ = self._write()
        frozen = [r for r in rows if r["stage"] == "frozen_backbone"]
        self.assertTrue(frozen)
        self.assertTrue(all(r["reference_id"] == "frozen_backbone" for r in frozen))
        self.assertTrue(all(r["model_id"] == "facebook/dinov3-vith16plus-pretrain-lvd1689m" for r in frozen))
        self.assertTrue(all(r["calibrated_query_coverage"] == "0.75" for r in frozen))
        self.assertTrue(all(r["qc_at_1"] == "1" and r["qc_at_10"] == "1" for r in frozen))

    def test_frozen_row_has_blank_epoch_and_loss(self):
        rows, _ = self._write()
        frozen = [r for r in rows if r["stage"] == "frozen_backbone"]
        self.assertTrue(all(r["epoch"] == "" for r in frozen))
        self.assertTrue(all(r["train_loss"] == "" for r in frozen))
        self.assertTrue(all(r["checkpoint_marker"] == "frozen_backbone" for r in frozen))
        self.assertTrue(all(r["is_best_checkpoint"] == "false" for r in frozen))
        self.assertTrue(all(r["is_last_checkpoint"] == "false" for r in frozen))

    def test_no_csv_cell_contains_epoch_zero(self):
        rows, _ = self._write()
        self.assertTrue(all(r["epoch"] != "0" for r in rows))

    def test_best_last_only_on_training_epochs(self):
        rows, _ = self._write()
        best = [r for r in rows if r["is_best_checkpoint"] == "true"]
        last = [r for r in rows if r["is_last_checkpoint"] == "true"]
        self.assertTrue(all(r["stage"] == "epoch" and r["epoch"] == "3" for r in best))
        self.assertTrue(all(r["stage"] == "epoch" and r["epoch"] == "3" for r in last))

    def test_svg_contains_frozen_backbone_and_no_epoch_zero(self):
        _, svg = self._write()
        self.assertIn("Frozen backbone", svg)
        self.assertNotIn("epoch 0", svg)
        self.assertNotIn("e0", svg)
        # Frozen marker connector to the first real epoch is present.
        self.assertIn("connector", svg)

    def test_reports_without_references_still_render(self):
        with TemporaryDirectory() as tmp:
            paths = write_training_reports(
                _history(),
                Path(tmp),
                best_epoch=3,
                last_epoch=3,
                best_validation_set="synth_val",
            )
            rows = list(csv.DictReader(Path(paths["validation_metrics_csv"]).read_text().splitlines()))
            self.assertFalse([r for r in rows if r["stage"] == "frozen_backbone"])
            svg = Path(paths["training_metrics_svg"]).read_text()
            self.assertNotIn("Frozen backbone", svg)


if __name__ == "__main__":
    unittest.main()
