import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.core.aggregate_runs import rows_from_eval
from experiments.core.config_schema import (
    _validate_threshold_calibration,
    _validate_threshold_top_k_floor,
)
from experiments.core.eval_pipeline import (
    calibrate_task_thresholds,
    calibrate_task_top_k,
    calibrated_candidate_sizes,
    evaluate_task,
    evaluate_task_at_calibrated_thresholds,
    evaluate_task_at_calibrated_threshold_top_k_floors,
    evaluate_task_at_calibrated_top_k,
    summarize_fixed_threshold_with_top_k_floor,
    summarize_target_pc,
)
from experiments.core.runner import _write_calibration_protocol
from experiments.core.split_schema import SplitImage


def _records(classes: dict[str, int]) -> dict[str, SplitImage]:
    return {file_id: SplitImage(file_id, class_id) for file_id, class_id in classes.items()}


def _task(task_id: str, query: list[str], candidates: list[str]) -> dict:
    return {
        "task_id": task_id,
        "query_ids": query,
        "candidate_ids": candidates,
        "exclude_self": True,
        "positive_policy": "same_painting",
    }


CANONICAL_PC_TARGETS = [0.95, 0.99, 0.995]


class ThresholdCalibrationTest(unittest.TestCase):
    def _leakage_scenario(self):
        """Validation calibration picks 0.7 (PC=1); on test 0.7 only reaches PC=0.8."""
        classes = {
            "vm1": 1, "vh1": 1, "vm2": 2, "vh2": 2,
            "tm1": 3, "th1": 3, "tm2": 4, "th2": 4, "tm3": 5, "th3": 5,
            "tm4": 6, "th4": 6, "tm5": 7, "th5": 7,
        }
        image_ids = list(classes)
        records = _records(classes)
        idx = {file_id: i for i, file_id in enumerate(image_ids)}
        sims = np.full((len(image_ids), len(image_ids)), -1.0, dtype=np.float32)

        def put(a: str, b: str, value: float) -> None:
            sims[idx[a], idx[b]] = value
            sims[idx[b], idx[a]] = value

        # Validation positives both at 0.7 -> threshold 0.7 reaches PC=1.0.
        put("vm1", "vh1", 0.7)
        put("vm1", "vh2", 0.3)
        put("vm2", "vh2", 0.7)
        put("vm2", "vh1", 0.2)
        # Test positives: four at/above 0.7, one at 0.4. Applying 0.7 -> PC=0.8;
        # an oracle threshold selected on test labels would drop to 0.4 for PC=1.0.
        for modern, historic, score in [
            ("tm1", "th1", 0.9), ("tm2", "th2", 0.85), ("tm3", "th3", 0.8),
            ("tm4", "th4", 0.75), ("tm5", "th5", 0.4),
        ]:
            put(modern, historic, score)
        val_task = _task("cal", ["vm1", "vm2"], ["vh1", "vh2"])
        test_task = _task("test", ["tm1", "tm2", "tm3", "tm4", "tm5"], ["th1", "th2", "th3", "th4", "th5"])
        return image_ids, records, sims, val_task, test_task

    def test_calibrated_threshold_is_frozen_and_test_pc_not_relabeled(self):
        image_ids, records, sims, val_task, test_task = self._leakage_scenario()

        calibration_rows = calibrate_task_thresholds(
            val_task, image_ids, records, sims, CANONICAL_PC_TARGETS
        )
        self.assertEqual(
            [row["target_pair_completeness"] for row in calibration_rows],
            CANONICAL_PC_TARGETS,
        )
        self.assertTrue(all(abs(float(row["threshold"]) - 0.7) < 1e-5 for row in calibration_rows))
        self.assertTrue(all(row["pair_completeness"] == 1.0 for row in calibration_rows))
        self.assertTrue(all(row["query_coverage"] == 1.0 for row in calibration_rows))

        calibrated = evaluate_task_at_calibrated_thresholds(
            test_task, image_ids, records, sims, calibration_rows, "wikidata_val"
        )
        self.assertEqual(
            [row["calibration_target_pair_completeness"] for row in calibrated],
            CANONICAL_PC_TARGETS,
        )
        self.assertTrue(all(abs(float(row["threshold"]) - 0.7) < 1e-5 for row in calibrated))
        self.assertTrue(all(row["pair_completeness"] == 0.8 for row in calibrated))
        row = next(
            row
            for row in calibrated
            if row["calibration_target_pair_completeness"] == 0.99
        )
        # The threshold is applied unchanged (no re-search on test labels).
        self.assertAlmostEqual(row["threshold"], 0.7, places=5)
        # The achieved test PC (0.8) must be reported, not the 0.99 objective.
        self.assertAlmostEqual(row["pair_completeness"], 0.8)
        self.assertEqual(row["calibration_id"], "wikidata_val")
        self.assertAlmostEqual(row["calibration_target_pair_completeness"], 0.99)
        self.assertAlmostEqual(row["calibration_pair_completeness"], 1.0)
        # Four of five test queries have a candidate at the frozen threshold.
        self.assertAlmostEqual(row["query_coverage"], 0.8)
        self.assertAlmostEqual(row["calibration_query_coverage"], 1.0)

        # Oracle selection on the test labels would pick a strictly lower threshold.
        test_result, _ = evaluate_task(test_task, image_ids, records, sims, [1], True, [0.99])
        oracle = test_result["target_pc_metrics"][0]
        self.assertAlmostEqual(oracle["threshold"], 0.4, places=5)
        self.assertAlmostEqual(oracle["pair_completeness"], 1.0)
        self.assertGreater(row["threshold"], oracle["threshold"])

    def test_top_k_calibration_selects_smallest_qualifying_k_and_freezes_it(self):
        image_ids, records, sims, val_task, test_task = self._leakage_scenario()
        index = {file_id: position for position, file_id in enumerate(image_ids)}
        sims[index["tm5"], index["th1"]] = 0.6

        calibration = calibrate_task_top_k(
            val_task,
            image_ids,
            records,
            sims,
            [1, 2],
            [0.99],
        )
        self.assertEqual(calibration[0]["k"], 1)
        self.assertTrue(calibration[0]["target_attained"])

        tested = evaluate_task_at_calibrated_top_k(
            test_task,
            image_ids,
            records,
            sims,
            calibration,
            "wikidata_val_top_k",
        )
        self.assertEqual(tested[0]["k"], 1)
        self.assertAlmostEqual(tested[0]["pair_completeness"], 0.8)
        self.assertFalse(tested[0]["test_target_attained"])
        self.assertAlmostEqual(tested[0]["calibration_error"], -0.19)

    def test_top_k_calibration_reports_failure_without_silent_closest_selection(self):
        image_ids, records, sims, _val_task, test_task = self._leakage_scenario()
        index = {file_id: position for position, file_id in enumerate(image_ids)}
        sims[index["tm5"], index["th1"]] = 0.6
        # On this test population one positive sits below a wrong score,
        # so k=1 cannot attain the predeclared target.
        calibration = calibrate_task_top_k(
            test_task,
            image_ids,
            records,
            sims,
            [1],
            [0.99],
            on_unattained="report_only",
        )
        self.assertEqual(calibration[0]["selection_status"], "target_not_attained")
        self.assertIsNone(calibration[0]["k"])
        self.assertEqual(calibration[0]["fallback_policy"], "none")

        fallback = calibrate_task_top_k(
            test_task,
            image_ids,
            records,
            sims,
            [1],
            [0.99],
            on_unattained="least_restrictive",
        )
        self.assertEqual(fallback[0]["selection_status"], "target_not_attained_fallback")
        self.assertFalse(fallback[0]["target_attained"])
        self.assertEqual(fallback[0]["k"], 1)

    def test_frozen_threshold_top_k_floor_rescues_below_threshold_positive(self):
        image_ids, records, sims, val_task, test_task = self._leakage_scenario()
        calibration = calibrate_task_thresholds(
            val_task, image_ids, records, sims, [0.99]
        )

        rows = evaluate_task_at_calibrated_threshold_top_k_floors(
            test_task,
            image_ids,
            records,
            sims,
            calibration,
            "wikidata_val",
            [1, 2],
        )

        self.assertEqual([row["minimum_top_k"] for row in rows], [1, 2])
        self.assertTrue(all(abs(float(row["threshold"]) - 0.7) < 1e-5 for row in rows))
        # Top-1 restores the one low-scoring positive without changing the frozen threshold.
        self.assertAlmostEqual(rows[0]["pair_completeness"], 1.0)
        self.assertEqual(rows[0]["floor_rescued_true_positives"], 1)
        self.assertEqual(rows[0]["queries_requiring_floor"], 1)
        self.assertEqual(rows[0]["threshold_source"], "pure_calibrated_threshold")
        # A larger floor cannot reduce PC or increase RR.
        self.assertGreaterEqual(rows[1]["pair_completeness"], rows[0]["pair_completeness"])
        self.assertLessEqual(rows[1]["reduction_ratio"], rows[0]["reduction_ratio"])

    def test_threshold_top_k_floor_uses_stable_candidate_order_for_ties(self):
        sims = np.asarray([[0.5, 0.5, 0.1]], dtype=np.float32)
        positives = np.asarray([[False, True, False]])
        row = summarize_fixed_threshold_with_top_k_floor(sims, positives, 0.9, 1)
        # Equal scores retain column order, so the first candidate wins the floor tie.
        self.assertEqual(row["candidate_pairs"], 1)
        self.assertEqual(row["true_positives"], 0)

    def test_threshold_top_k_floor_clamps_to_finite_candidates(self):
        sims = np.asarray([[0.9, 0.2, -np.inf]], dtype=np.float32)
        positives = np.asarray([[True, False, False]])
        row = summarize_fixed_threshold_with_top_k_floor(sims, positives, 0.8, 5)
        self.assertEqual(row["candidate_pairs"], 2)
        self.assertEqual(row["floor_added_candidate_pairs"], 1)
        self.assertEqual(row["queries_requiring_floor"], 1)
        self.assertAlmostEqual(row["pair_completeness"], 1.0)

    def test_threshold_top_k_floor_schema_requires_pure_calibration(self):
        evaluation = {
            "retrieval_tasks": [],
            "threshold_top_k_floor": {
                "method": "frozen_calibrated_threshold_or_top_k_floor",
                "minimum_k": [5, 10, 20, 40],
            },
        }
        with self.assertRaisesRegex(ValueError, "requires evaluation.threshold_calibration"):
            _validate_threshold_top_k_floor(evaluation, Path("experiment.yml"))

    def test_hybrid_floor_rows_flatten_with_distinct_operating_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "run" / "models" / "model"
            model_dir.mkdir(parents=True)
            eval_path = model_dir / "eval.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "model_id": "model",
                        "model_storage_key": "model",
                        "tasks": [
                            {
                                "task_id": "test",
                                "calibrated_threshold_top_k_floor_metrics": [
                                    {
                                        "minimum_top_k": floor,
                                        "calibration_target_pair_completeness": target,
                                        "pair_completeness": target,
                                    }
                                    for target in (0.95, 0.99)
                                    for floor in (5, 10, 20, 40)
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows = rows_from_eval(eval_path, {"experiment_id": "hybrid", "split": {}})
        self.assertEqual(len(rows), 8)
        self.assertTrue(
            all(row["selection_mode"] == "calibrated_threshold_top_k_floor" for row in rows)
        )
        self.assertEqual(sorted({row["minimum_top_k"] for row in rows}), [5, 10, 20, 40])

    def test_calibrated_candidate_sizes_logs_per_query_counts(self):
        image_ids, records, sims, val_task, test_task = self._leakage_scenario()
        calibration_rows = calibrate_task_thresholds(
            val_task, image_ids, records, sims, [0.95, 0.99]
        )

        rows = calibrated_candidate_sizes(test_task, image_ids, records, sims, calibration_rows)

        self.assertEqual(
            [row["calibration_target_pair_completeness"] for row in rows],
            [0.95, 0.99],
        )
        for row in rows:
            self.assertAlmostEqual(row["threshold"], 0.7, places=5)
            # One candidate at/above 0.7 for four test queries; tm5 (0.4) is uncovered.
            self.assertEqual(row["candidates_per_query"], [1, 1, 1, 1, 0])
            self.assertAlmostEqual(row["query_coverage"], 0.8)

    def test_calibrated_candidate_sizes_skips_infeasible_operating_points(self):
        image_ids, records, sims, _val_task, test_task = self._leakage_scenario()
        rows = calibrated_candidate_sizes(
            test_task,
            image_ids,
            records,
            sims,
            [{"target_pair_completeness": 0.99, "threshold": None}],
        )
        self.assertEqual(rows, [])

    def test_canonical_targets_select_distinct_monotonic_operating_points(self):
        positive_scores = 1.0 - np.arange(200, dtype=np.float32) / 1000.0
        similarities = np.column_stack(
            [positive_scores, np.full(200, -1.0, dtype=np.float32)]
        )
        positives = np.zeros_like(similarities, dtype=bool)
        positives[:, 0] = True

        rows = summarize_target_pc(similarities, positives, CANONICAL_PC_TARGETS)

        self.assertEqual(
            [row["target_pair_completeness"] for row in rows],
            CANONICAL_PC_TARGETS,
        )
        self.assertEqual(
            [row["true_positives"] for row in rows],
            [190, 198, 199],
        )
        self.assertEqual(
            [row["pair_completeness"] for row in rows],
            CANONICAL_PC_TARGETS,
        )
        thresholds = [float(row["threshold"]) for row in rows]
        self.assertGreater(thresholds[0], thresholds[1])
        self.assertGreater(thresholds[1], thresholds[2])

    def test_tied_scores_keep_one_row_per_canonical_target(self):
        positive_scores = 1.0 - np.arange(200, dtype=np.float32) / 1000.0
        positive_scores[189:] = 0.5
        similarities = np.column_stack(
            [positive_scores, np.full(200, -1.0, dtype=np.float32)]
        )
        positives = np.zeros_like(similarities, dtype=bool)
        positives[:, 0] = True

        rows = summarize_target_pc(similarities, positives, CANONICAL_PC_TARGETS)

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["target_pair_completeness"] for row in rows],
            CANONICAL_PC_TARGETS,
        )
        self.assertEqual({row["threshold"] for row in rows}, {0.5})
        self.assertTrue(all(row["pair_completeness"] == 1.0 for row in rows))

    def test_multiple_targets_produce_multiple_rows_and_self_pairs_excluded(self):
        # query and candidate id sets overlap; the self match sits at the top score
        # but must be excluded via exclude_self before thresholds are chosen.
        classes = {"a": 1, "b": 1, "c": 2, "d": 2}
        image_ids = list(classes)
        records = _records(classes)
        idx = {file_id: i for i, file_id in enumerate(image_ids)}
        sims = np.full((len(image_ids), len(image_ids)), -1.0, dtype=np.float32)
        np.fill_diagonal(sims, 1.0)  # self similarity is highest but must be dropped
        for a, b, value in [("a", "b", 0.6), ("c", "d", 0.4), ("a", "d", 0.1)]:
            sims[idx[a], idx[b]] = value
            sims[idx[b], idx[a]] = value
        task = _task("cal", ["a", "b", "c", "d"], ["a", "b", "c", "d"])

        rows = calibrate_task_thresholds(task, image_ids, records, sims, [0.5, 1.0])
        self.assertEqual(len(rows), 2)
        # The self match (score 1.0) is excluded, so PC=1.0 needs the 0.4 pair.
        by_target = {row["target_pair_completeness"]: row for row in rows}
        self.assertAlmostEqual(by_target[0.5]["threshold"], 0.6, places=5)
        self.assertAlmostEqual(by_target[1.0]["threshold"], 0.4, places=5)
        self.assertAlmostEqual(by_target[0.5]["query_coverage"], 0.5)
        self.assertAlmostEqual(by_target[1.0]["query_coverage"], 1.0)
        self.assertAlmostEqual(
            evaluate_task(task, image_ids, records, sims, [1], True)[0]["metrics"][0]["query_coverage"],
            1.0,
        )

    def test_protocol_report_rejects_query_overlap(self):
        classes = {"m1": 1, "h1": 1, "m2": 2, "h2": 2}
        records = _records(classes)
        calibration = _CalibrationStub(_task("cal", ["m1"], ["h1"]))
        # Test task reuses the calibration query image -> query leakage.
        leaking = _task("test", ["m1"], ["h2"])
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(ValueError):
                _write_calibration_protocol(run_dir, calibration, [leaking], records)
            report = (run_dir / "calibration_protocol.json").read_text(encoding="utf-8")
            self.assertIn("query_ids_disjoint", report)
            self.assertIn('"passed": false', report)

    def test_protocol_report_passes_for_disjoint_populations(self):
        classes = {"m1": 1, "h1": 1, "m2": 2, "h2": 2}
        records = _records(classes)
        calibration = _CalibrationStub(_task("cal", ["m1"], ["h1"]))
        disjoint = _task("test", ["m2"], ["h2"])
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_calibration_protocol(run_dir, calibration, [disjoint], records)
            report = (run_dir / "calibration_protocol.json").read_text(encoding="utf-8")
            self.assertIn('"passed": true', report)
            self.assertIn('"target_pc": [', report)
            for target in CANONICAL_PC_TARGETS:
                self.assertIn(str(target), report)


class ThresholdCalibrationSchemaTest(unittest.TestCase):
    def _evaluation(self, **overrides) -> dict:
        calibration = {
            "method": "empirical_target_pc",
            "calibration_id": "wikidata_val",
            "target_pc": CANONICAL_PC_TARGETS,
            "task": {
                "task_id": "cal_modern_to_historic",
                "query": {"subset": "val", "role": "modern"},
                "candidates": {"subset": "val", "role": "historic"},
                "positive_policy": "same_painting",
                "exclude_self": True,
            },
        }
        calibration.update(overrides)
        return {
            "retrieval_tasks": [
                {
                    "task_id": "test_modern_to_historic",
                    "query": {"subset": "test", "role": "modern"},
                    "candidates": {"subset": "test", "role": "historic"},
                    "positive_policy": "same_painting",
                    "exclude_self": True,
                }
            ],
            "threshold_calibration": calibration,
        }

    def _reject(self, evaluation: dict, needle: str) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_threshold_calibration(evaluation, Path("exp.yml"))
        self.assertIn(needle, str(ctx.exception))

    def test_valid_calibration_block_passes(self):
        _validate_threshold_calibration(self._evaluation(), Path("exp.yml"))

    def test_rejects_test_subset_in_calibration_task(self):
        evaluation = self._evaluation()
        evaluation["threshold_calibration"]["task"]["query"]["subset"] = "test"
        self._reject(evaluation, "test subset")

    def test_rejects_all_subsets_wildcard(self):
        evaluation = self._evaluation()
        evaluation["threshold_calibration"]["task"]["candidates"]["subset"] = "ALL_SUBSETS"
        self._reject(evaluation, "ALL_SUBSETS")

    def test_rejects_target_pc_out_of_range(self):
        self._reject(self._evaluation(target_pc=[1.5]), "must be in (0, 1]")

    def test_rejects_task_id_collision_with_test_task(self):
        evaluation = self._evaluation()
        evaluation["threshold_calibration"]["task"]["task_id"] = "test_modern_to_historic"
        self._reject(evaluation, "must differ")


class _CalibrationStub:
    def __init__(self, task: dict):
        self.task = task
        self.calibration_id = "wikidata_val"
        self.method = "empirical_target_pc"
        self.target_pc = CANONICAL_PC_TARGETS


if __name__ == "__main__":
    unittest.main()
