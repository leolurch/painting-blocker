from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.core.split_sensitivity import (
    SplitSensitivityError,
    _render_calibrated_table,
    _validate_static_splits,
    _write_charts,
    aggregate_split_sensitivity_rows,
    load_split_sensitivity_config,
)


SPLITS = [6, 7, 9]
TRAINING_SEEDS = [101, 202]


def _config() -> dict:
    return {
        "schema_version": 1,
        "analysis_id": "wikidata_sensitivity_test",
        "dataset_id": "wikidata-test",
        "expected_split_seeds": SPLITS,
        "protocol": {"frozen_before_evaluation": True},
        "runs": ["unused"],
        "static_splits": [
            {"seed": seed, "file": f"split-{seed}.json"} for seed in SPLITS
        ],
        "configurations": [
            {
                "configuration_id": "trained",
                "label": "Trained",
                "source_ids": ["trained"],
                "training_seeds": TRAINING_SEEDS,
            },
            {
                "configuration_id": "frozen",
                "label": "Frozen",
                "source_ids": ["frozen"],
            },
        ],
        "comparisons": [
            {
                "comparison_id": "trained-minus-frozen",
                "minuend": "trained",
                "subtrahend": "frozen",
            }
        ],
    }


def _row(configuration: str, split: int, training_seed: int | None, pc: float, pq: float) -> dict:
    return {
        "dataset_id": "wikidata-test",
        "dataset_source_db_sha256": "wikidata-db-sha",
        "evaluation_protocol_sha256": "evaluation-protocol-sha",
        "split_id": f"wikidata_split_seed_{split}",
        "split_seed": split,
        "split_sha256": f"sha-{split}",
        "configuration_id": configuration,
        "training_seed": training_seed,
        "model_id": f"{configuration}-{training_seed}",
        "model_storage_key": configuration,
        "display_name": configuration.title(),
        "task_id": "test_modern_to_historic",
        "selection_mode": "calibrated_threshold",
        "k": None,
        "calibration_target_pair_completeness": 0.95,
        "pair_completeness": pc,
        "pair_quality": pq,
        "reduction_ratio": 0.90 + pq / 10,
        "query_coverage": 1.0,
        "threshold": 0.5,
        "candidate_pairs": int(round(1000 * (1.0 - pq))),
        "num_queries": 100,
        "num_candidates": 150,
        "num_possible_pairs": 15000,
        "num_excluded_pairs": 0,
        "num_positive_pairs": 120,
        "run_id": f"{configuration}-{split}-{training_seed}",
        "eval_path": f"/{configuration}/{split}/{training_seed}/eval.json",
    }


def _rows() -> list[dict]:
    rows: list[dict] = []
    trained = {
        6: [(0.90, 0.10), (0.94, 0.14)],
        7: [(0.96, 0.16), (1.00, 0.20)],
        9: [(0.98, 0.18), (1.00, 0.22)],
    }
    frozen = {6: (0.91, 0.11), 7: (0.97, 0.17), 9: (0.98, 0.19)}
    for split, values in trained.items():
        for seed, (pc, pq) in zip(TRAINING_SEEDS, values):
            rows.append(_row("trained", split, seed, pc, pq))
    for split, (pc, pq) in frozen.items():
        rows.append(_row("frozen", split, None, pc, pq))
    return rows


class SplitSensitivityStatisticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = aggregate_split_sensitivity_rows(_rows(), _config())
        self.trained = next(
            row for row in self.artifact["summary_rows"] if row["configuration_id"] == "trained"
        )

    def test_training_repeats_are_averaged_before_split_statistics(self) -> None:
        split_means = [0.92, 0.98, 0.99]
        expected_mean = sum(split_means) / len(split_means)
        expected_variance = sum((value - expected_mean) ** 2 for value in split_means) / 2
        self.assertAlmostEqual(self.trained["pair_completeness_mean"], expected_mean)
        self.assertAlmostEqual(
            self.trained["pair_completeness_sample_variance_split"], expected_variance
        )
        self.assertAlmostEqual(
            self.trained["pair_completeness_sample_sd_split"], math.sqrt(expected_variance)
        )
        self.assertAlmostEqual(self.trained["pair_completeness_min"], 0.92)
        self.assertAlmostEqual(self.trained["pair_completeness_max"], 0.99)
        self.assertNotIn("pair_completeness", self.trained)

    def test_average_within_split_training_variance(self) -> None:
        # Sample variances: .0008, .0008, .0002; arithmetic average = .0006.
        self.assertAlmostEqual(
            self.trained["pair_completeness_average_within_split_training_variance"],
            0.0006,
        )
        self.assertAlmostEqual(
            self.trained["pair_completeness_average_within_split_training_sd"],
            math.sqrt(0.0006),
        )

    def test_target_attainment_uses_split_level_means(self) -> None:
        self.assertEqual(self.trained["target_attainment_count"], 2)
        self.assertAlmostEqual(self.trained["target_attainment_rate"], 2 / 3)

    def test_target_attainment_uses_configured_shortfall_tolerance(self) -> None:
        config = _config()
        config["aggregation"] = {"attainment_tolerance": 0.05}
        artifact = aggregate_split_sensitivity_rows(_rows(), config)
        trained = next(
            row
            for row in artifact["summary_rows"]
            if row["configuration_id"] == "trained"
        )

        self.assertEqual(trained["target_attainment_count"], 3)
        self.assertEqual(trained["target_attainment_tolerance"], 0.05)
        self.assertAlmostEqual(trained["target_attainment_threshold"], 0.90)
        self.assertEqual(artifact["aggregation"]["attainment_tolerance"], 0.05)

    def test_metric_denominators_and_retained_pairs_are_recorded_per_split(self) -> None:
        denominators = [
            row
            for row in self.artifact["split_denominators"]
            if row["configuration_id"] == "trained"
        ]
        self.assertEqual(len(denominators), 3)
        first = denominators[0]
        self.assertEqual(first["num_queries"], 100)
        self.assertEqual(first["num_candidates"], 150)
        self.assertEqual(first["num_positive_pairs"], 120)
        self.assertEqual(first["num_cartesian_pairs"], 15000)
        self.assertEqual(first["num_possible_pairs"], 15000)
        self.assertEqual(first["candidate_pairs_training_seed_values"], [900.0, 860.0])
        self.assertEqual(first["candidate_pairs_mean_over_training_seeds"], 880.0)

    def test_raw_and_split_level_values_are_retained(self) -> None:
        self.assertEqual(len(self.artifact["raw_rows"]), 9)
        trained_split_rows = [
            row for row in self.artifact["split_level_rows"] if row["configuration_id"] == "trained"
        ]
        self.assertEqual(len(trained_split_rows), 3)
        self.assertEqual(trained_split_rows[0]["training_seeds"], TRAINING_SEEDS)
        self.assertEqual(
            len(self.trained["pair_completeness_split_values"]),
            len(SPLITS),
        )

    def test_paired_differences_use_identical_splits(self) -> None:
        paired = self.artifact["paired_comparisons"][0]
        # Split differences after seed averaging: .01, .01, .01.
        self.assertAlmostEqual(paired["pair_completeness_difference_mean"], 0.01)
        self.assertAlmostEqual(paired["pair_completeness_difference_min"], 0.01)
        self.assertAlmostEqual(paired["reduction_ratio_difference_mean"], 0.001)
        self.assertEqual(
            [round(row["difference"], 6) for row in paired["reduction_ratio_difference_split_values"]],
            [0.001, 0.001, 0.001],
        )
        self.assertEqual(paired["pair_completeness_wins"], 3)
        self.assertEqual(paired["pair_completeness_ties"], 0)
        self.assertEqual(paired["pair_completeness_losses"], 0)

    def test_paired_differences_use_configured_tie_tolerance(self) -> None:
        config = _config()
        config["aggregation"] = {"tie_tolerance": 0.02}
        paired = aggregate_split_sensitivity_rows(_rows(), config)[
            "paired_comparisons"
        ][0]
        self.assertEqual(paired["tie_tolerance"], 0.02)
        self.assertEqual(paired["pair_completeness_wins"], 0)
        self.assertEqual(paired["pair_completeness_ties"], 3)
        self.assertEqual(paired["pair_completeness_losses"], 0)

    def test_hybrid_threshold_floors_remain_distinct_operating_points(self) -> None:
        rows = []
        config = _config()
        config["configurations"] = [config["configurations"][1]]
        config["comparisons"] = []
        for split in SPLITS:
            for floor in (5, 10):
                row = _row("frozen", split, None, 0.95 + floor / 1000, 0.2)
                row.update(
                    selection_mode="calibrated_threshold_top_k_floor",
                    calibration_target_pair_completeness=0.95,
                    minimum_top_k=floor,
                )
                rows.append(row)
        artifact = aggregate_split_sensitivity_rows(rows, config)
        self.assertEqual(
            sorted(row["minimum_top_k"] for row in artifact["summary_rows"]),
            [5, 10],
        )
        self.assertEqual(
            sorted({row["minimum_top_k"] for row in artifact["split_denominators"]}),
            [5, 10],
        )
        self.assertTrue(
            all(row["target_attainment_count"] == 3 for row in artifact["summary_rows"])
        )

    def test_calibrated_top_k_summarizes_selected_budget_and_test_attainment(self) -> None:
        rows = []
        config = _config()
        config["configurations"] = [config["configurations"][1]]
        config["comparisons"] = []
        for split, selected_k, pc in ((6, 20, 0.94), (7, 40, 0.96), (9, 40, 0.95)):
            row = _row("frozen", split, None, pc, 0.2)
            row.update(
                selection_mode="calibrated_top_k",
                calibration_target_pair_completeness=0.95,
                selected_k=selected_k,
                selection_attained=True,
            )
            rows.append(row)
        artifact = aggregate_split_sensitivity_rows(rows, config)
        summary = artifact["summary_rows"][0]
        self.assertAlmostEqual(summary["selected_k_mean"], 100 / 3)
        self.assertEqual(summary["validation_selection_attainment_count"], 3)
        self.assertEqual(summary["target_attainment_count"], 2)

    def test_interpretation_is_explicitly_descriptive(self) -> None:
        interpretation = self.artifact["interpretation"]
        self.assertFalse(interpretation["split_results_independent"])
        self.assertFalse(interpretation["ranges_are_confidence_intervals"])
        self.assertFalse(interpretation["counts_pooled_across_splits"])
        self.assertFalse(interpretation["inference_or_significance_claimed"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib not installed")
    def test_range_charts_render_without_reinterpreting_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, warnings = _write_charts(
                self.artifact["summary_rows"], Path(tmp), ("svg",)
            )
            self.assertEqual(warnings, [])
            self.assertTrue(paths)
            self.assertTrue(any("calibrated_pair_completeness" in path.name for path in paths))
            self.assertTrue(all(path.is_file() for path in paths))

    def test_calibrated_table_reports_range_and_attainment(self) -> None:
        rendered = _render_calibrated_table(
            self.artifact["summary_rows"], "wikidata_sensitivity_test"
        )
        self.assertIn("0.963 [0.920, 0.990]", rendered)
        self.assertIn("2/3", rendered)
        self.assertIn("not confidence intervals", rendered)
        self.assertIn(
            "\\label{tab:wikidata_sensitivity_test-calibrated-sensitivity}",
            rendered,
        )
        label_lines = [line for line in rendered.splitlines() if r"\label{" in line]
        self.assertTrue(all(r"\_" not in line for line in label_lines))


class SplitSensitivityValidationTest(unittest.TestCase):
    def test_missing_split_fails_instead_of_silently_averaging(self) -> None:
        rows = [row for row in _rows() if not (row["configuration_id"] == "trained" and row["split_seed"] == 9)]
        with self.assertRaisesRegex(SplitSensitivityError, "has split seeds"):
            aggregate_split_sensitivity_rows(rows, _config())

    def test_missing_training_seed_fails(self) -> None:
        rows = [
            row
            for row in _rows()
            if not (
                row["configuration_id"] == "trained"
                and row["split_seed"] == 7
                and row["training_seed"] == 202
            )
        ]
        with self.assertRaisesRegex(SplitSensitivityError, "has training seeds"):
            aggregate_split_sensitivity_rows(rows, _config())

    def test_population_mismatch_between_models_fails(self) -> None:
        rows = _rows()
        rows[-1]["num_queries"] = 99
        with self.assertRaisesRegex(SplitSensitivityError, "Incompatible task population"):
            aggregate_split_sensitivity_rows(rows, _config())

    def test_static_split_files_are_validated_as_one_class_disjoint_population(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = {
                f"{role}{class_id}.jpg": {"class_id": class_id}
                for class_id in range(4)
                for role in ("m", "h")
            }
            static = []
            for seed, val_classes in ((6, {0, 1}), (7, {0, 2})):
                test_classes = set(range(4)) - val_classes
                split = {
                    "schema_version": 2,
                    "split_id": f"wikidata-test-seed-{seed}",
                    "dataset": {
                        "dataset_id": "wikidata-test",
                        "image_root": "/unused/images",
                        "source_db_sha256": "db-sha",
                    },
                    "split_strategy": {
                        "type": "class_disjoint_role",
                        "seed": seed,
                        "ratios": {"train": 0.0, "val": 0.5, "test": 0.5},
                    },
                    "subsets": {
                        name: {
                            "roles": {
                                "modern": [f"m{class_id}.jpg" for class_id in sorted(classes)],
                                "historic": [f"h{class_id}.jpg" for class_id in sorted(classes)],
                            }
                        }
                        for name, classes in (("val", val_classes), ("test", test_classes))
                    },
                    "images": images,
                }
                path = root / f"split-{seed}.json"
                path.write_text(json.dumps(split), encoding="utf-8")
                static.append({"seed": seed, "file": path.name})
            config = {
                "dataset_id": "wikidata-test",
                "static_splits": static,
            }
            records = _validate_static_splits(root / "analysis.yml", config)
            self.assertEqual([record["split_seed"] for record in records], [6, 7])
            self.assertEqual({record["num_classes"] for record in records}, {4})
            self.assertEqual({record["num_val_classes"] for record in records}, {2})

    def test_config_requires_wikidata_and_frozen_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yml"
            config = _config()
            config["dataset_id"] = "synthetic"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(SplitSensitivityError, "restricted to Wikidata"):
                load_split_sensitivity_config(path)

            config["dataset_id"] = "wikidata-test"
            config["protocol"]["frozen_before_evaluation"] = False
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(SplitSensitivityError, "must not become a tuning set"):
                load_split_sensitivity_config(path)


if __name__ == "__main__":
    unittest.main()
