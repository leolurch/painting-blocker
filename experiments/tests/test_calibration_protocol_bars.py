from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from experiments.core.calibration_protocol_bars import (
    CalibrationProtocolBarsError,
    parse_protocol_chart,
    prune_incomplete_union_rows,
    union_rows_for_batch,
    zero_margin_union_metrics,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class ZeroMarginUnionTest(unittest.TestCase):
    def test_only_the_zero_margin_cell_is_read(self):
        metrics = zero_margin_union_metrics(
            {
                "target_pc": 0.99,
                "k_margins": [0, 5],
                "threshold_margins": [0.0, 0.3],
                "pair_completeness": [[0.91, 0.99], [0.97, 1.0]],
                "reduction_ratio": [[0.5, 0.1], [0.2, 0.05]],
                "applied_k": [12, 17],
                "applied_threshold": [0.42, 0.12],
            }
        )

        self.assertEqual(metrics["pair_completeness"], 0.91)
        self.assertEqual(metrics["reduction_ratio"], 0.5)
        self.assertEqual(metrics["selected_k"], 12.0)
        self.assertEqual(metrics["threshold"], 0.42)

    def test_incomplete_union_coverage_is_dropped(self):
        rows = [
            {
                "configuration_id": "a",
                "split_seed": 6,
                "task_id": "test",
                "selection_mode": "calibrated_threshold",
                "calibration_target_pair_completeness": 0.99,
            },
            {
                "configuration_id": "b",
                "split_seed": 6,
                "task_id": "test",
                "selection_mode": "calibrated_threshold",
                "calibration_target_pair_completeness": 0.99,
            },
            {
                "configuration_id": "a",
                "split_seed": 6,
                "task_id": "test",
                "selection_mode": "calibrated_union",
                "calibration_target_pair_completeness": 0.99,
            },
        ]

        kept = prune_incomplete_union_rows(rows)

        self.assertFalse(any(row["selection_mode"] == "calibrated_union" for row in kept))

    def test_union_row_does_not_keep_the_threshold_pair_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            eval_path = model_dir / "eval.json"
            eval_path.write_text("{}", encoding="utf-8")
            (model_dir / "calibrated_union.json").write_text(
                json.dumps(
                    {
                        "target_pc": 0.99,
                        "k_margins": [0],
                        "threshold_margins": [0.0],
                        "pair_completeness": [[0.93]],
                        "reduction_ratio": [[0.44]],
                        "applied_k": [9],
                        "applied_threshold": [0.3],
                    }
                ),
                encoding="utf-8",
            )
            batch = [
                {
                    "selection_mode": "calibrated_threshold",
                    "calibration_target_pair_completeness": 0.99,
                    "pair_quality": 0.2,
                    "pair_completeness": 0.8,
                    "reduction_ratio": 0.7,
                    "calibration_reduction_ratio": 0.6,
                }
            ]

            rows = union_rows_for_batch(batch, eval_path)

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["pair_quality"])
        self.assertIsNone(rows[0]["calibration_reduction_ratio"])
        self.assertEqual(rows[0]["pair_completeness"], 0.93)
        self.assertEqual(rows[0]["selection_mode"], "calibrated_union")

    def test_safety_margins_are_rejected(self):
        with self.assertRaises(CalibrationProtocolBarsError):
            parse_protocol_chart(
                {
                    "charts": {
                        "calibration_protocol_bars": {
                            "enabled": True,
                            "safety_margins": True,
                            "protocols": ["union"],
                        }
                    }
                }
            )


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
class MultisplitProtocolDirectoryTest(unittest.TestCase):
    def test_each_protocol_directory_uses_the_threshold_transfer_names(self):
        from experiments.core.multisplit_charts import write_multisplit_charts

        def _row(mode: str, pc: float) -> dict:
            return {
                "configuration_id": "model-a",
                "task_id": "test",
                "selection_mode": mode,
                "calibration_target_pair_completeness": 0.99,
                "pair_completeness_mean": pc,
                "reduction_ratio_mean": 0.4,
                "pair_completeness_split_values": [
                    {"split_seed": 6, "mean_over_training_seeds": pc}
                ],
                "reduction_ratio_split_values": [
                    {"split_seed": 6, "mean_over_training_seeds": 0.4}
                ],
                "target_attainment_count": 1,
                "num_split_realizations": 1,
            }

        config = {
            "models": [
                {
                    "analysis_model_id": "model-a",
                    "include": True,
                    "presentation": {"label": "A", "color": "#111111"},
                }
            ],
            "charts": {
                "calibration_protocol_bars": {
                    "enabled": True,
                    "safety_margins": False,
                    "protocols": ["calibrated_k", "union"],
                    "task": "test",
                    "target_pc": 0.99,
                    "formats": ["pdf"],
                }
            },
            "multisplit": {"aggregation": {"attainment_tolerance": 0.005}},
        }
        artifact = {
            "summary_rows": [
                _row("calibrated_top_k", 0.96),
                _row("calibrated_union", 0.98),
            ],
            "expected_split_seeds": [6],
            "dataset_id": "toy",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            _paths, warnings = write_multisplit_charts(artifact, config, output)
            names = {
                "multisplit_calibration_transfer_rho0p99.pdf",
                "multisplit_calibration_transfer_rho0p99.json",
            }
            for protocol in ("calibrated_k", "union"):
                directory = output / "calibration_protocols" / protocol
                self.assertEqual(names, {path.name for path in directory.iterdir()})
            self.assertFalse(any("margin" in warning for warning in warnings))
