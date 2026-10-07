import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.core.config_schema import _validate_protocol_blocks
from experiments.core.finetuning.evaluate import select_threshold_at_pair_completeness
from experiments.core.finetuning.train import (
    _checkpoint_selection_config,
    _calibration_transfer_result,
    _checkpoint_selection_label,
    _early_stop_patience,
    _is_better_for_selection,
    _mean_precision_at_k_result,
    _next_non_improving_streak,
    _select_calibration_transfer_epoch,
    _target_pc_operating_point,
    _weighted_mean_pc_at_k_and_transfer_result,
)


METRIC = "reduction_ratio_at_target_pc"
TARGET_PC = 0.99


def _metrics(*, rr: float, pc: float = TARGET_PC, threshold: float = 0.5, candidates: int = 100) -> dict:
    possible = 1000
    return {
        "calibrated_threshold": {
            "threshold": threshold,
            "target_pair_completeness": TARGET_PC,
            "pair_completeness": pc,
            "pair_quality": 0.8,
            "reduction_ratio": rr,
            "candidate_pairs": candidates,
            "possible_pairs": possible,
            "true_positives": 99,
            "false_positives": max(0, candidates - 99),
            "false_negatives": 1,
            "total_positive_pairs": 100,
        }
    }


def _selection() -> dict:
    return {
        "validation_set": "only_val",
        "metric": METRIC,
        "target_pc": TARGET_PC,
        "task_id": "all_pairs",
    }


def _validation(name: str = "only_val", *, primary: bool = True, task_specs=None):
    return SimpleNamespace(
        name=name,
        primary=primary,
        retrieval_task_specs=list(task_specs or []),
    )


class ReductionRatioCheckpointComparatorTest(unittest.TestCase):
    def test_higher_rr_wins_and_lower_rr_loses(self):
        best = _metrics(rr=0.9, candidates=100)
        self.assertTrue(
            _is_better_for_selection(
                _metrics(rr=0.92, candidates=80), best, TARGET_PC, _selection()
            )
        )
        self.assertFalse(
            _is_better_for_selection(
                _metrics(rr=0.88, candidates=120), best, TARGET_PC, _selection()
            )
        )

    def test_equal_rr_retains_earlier_checkpoint_without_other_tiebreakers(self):
        best = _metrics(rr=0.9, pc=TARGET_PC, threshold=0.4, candidates=100)
        candidate = _metrics(rr=0.9, pc=1.0, threshold=0.8, candidates=100)
        candidate["calibrated_threshold"]["pair_quality"] = 1.0
        self.assertFalse(
            _is_better_for_selection(candidate, best, TARGET_PC, _selection())
        )

    def test_first_eligible_epoch_wins(self):
        self.assertTrue(
            _is_better_for_selection(
                _metrics(rr=0.9, candidates=100), None, TARGET_PC, _selection()
            )
        )

    def test_below_target_and_malformed_operating_points_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "did not reach target"):
            _is_better_for_selection(
                _metrics(rr=0.9, pc=0.98, candidates=100),
                None,
                TARGET_PC,
                _selection(),
            )
        for key in ("threshold", "target_pair_completeness", "pair_completeness", "reduction_ratio"):
            malformed = _metrics(rr=0.9, candidates=100)
            del malformed["calibrated_threshold"][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing"):
                _target_pc_operating_point(malformed, TARGET_PC)
        malformed = _metrics(rr=0.9, candidates=100)
        malformed["calibrated_threshold"]["reduction_ratio"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            _target_pc_operating_point(malformed, TARGET_PC)

    def test_rr_must_match_candidate_counts(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            _target_pc_operating_point(_metrics(rr=0.91, candidates=100), TARGET_PC)


class CalibrationTransferCheckpointSelectionTest(unittest.TestCase):
    def test_config_and_strict_pc_eligibility(self):
        resolved = _checkpoint_selection_config(
            {
                "target_pc": 0.99,
                "checkpoint_selection": {
                    "metric": "calibration_transfer",
                    "calibration_validation_set": "syn_val",
                    "selection_validation_set": "syn_test",
                    "minimum_selection_pc": 0.98,
                    "rr_tie_margin": 0.005,
                    "epoch_tiebreak": "earliest",
                },
            },
            [_validation("syn_val"), _validation("syn_test", primary=False)],
            "syn_val",
        )
        calibration = _metrics(rr=0.9, threshold=0.4, candidates=100)
        test = {"calibrated_threshold": {"threshold": 0.4, "pair_completeness": 0.98, "reduction_ratio": 0.95}}
        result = _calibration_transfer_result(calibration, test, resolved)
        self.assertFalse(result["eligible"])
        test["calibrated_threshold"]["pair_completeness"] = 0.980001
        self.assertTrue(_calibration_transfer_result(calibration, test, resolved)["eligible"])
        self.assertIn("syn_val PC>=0.99 -> syn_test PC>0.98", _checkpoint_selection_label(resolved))

    def test_schema_accepts_transfer_protocol(self):
        _validate_protocol_blocks(
            {
                "finetuning": {
                    "data": {
                        "train": {"subset": "train", "roles": ["all"]},
                        "validation": [
                            {"name": "syn_val", "primary": True, "subset": "val", "roles": ["all"]},
                            {"name": "syn_test", "primary": False, "subset": "test", "roles": ["all"]},
                        ],
                    },
                    "evaluation": {
                        "target_pc": 0.99,
                        "checkpoint_selection": {
                            "metric": "calibration_transfer",
                            "calibration_validation_set": "syn_val",
                            "selection_validation_set": "syn_test",
                            "minimum_selection_pc": 0.98,
                            "rr_tie_margin": 0.005,
                            "epoch_tiebreak": "earliest",
                        },
                    },
                }
            },
            Path("experiment.yml"),
        )

    def test_global_rr_margin_then_earliest_epoch(self):
        base = {"eligible": True, "rr_tie_margin": 0.005}
        selected = _select_calibration_transfer_epoch(
            [
                {**base, "epoch": 1, "selection_reduction_ratio": 0.90},
                {**base, "epoch": 2, "selection_reduction_ratio": 0.904},
                {**base, "epoch": 3, "selection_reduction_ratio": 0.908},
            ]
        )
        self.assertEqual(selected["epoch"], 2)
        self.assertIsNone(_select_calibration_transfer_epoch([{**base, "eligible": False, "epoch": 1, "selection_reduction_ratio": 0.99}]))


class WeightedMeanPcTransferCheckpointSelectionTest(unittest.TestCase):
    def test_exact_weighted_score_and_config_normalization(self):
        resolved = _checkpoint_selection_config(
            {
                "target_pc": TARGET_PC,
                "top_k": [10, 20, 40, 80, 160],
                "checkpoint_selection": {
                    "metric": "weighted_mean_pc_at_k_and_transfer",
                    "calibration_validation_set": "syn_val",
                    "selection_validation_set": "syn_test",
                    "fixed_k": [10, 20, 40, 80, 160],
                    "weights": {
                        "mean_pc_at_k": 0.5,
                        "transferred_pc": 0.25,
                        "transferred_rr": 0.25,
                    },
                    "epoch_tiebreak": "earliest",
                },
            },
            [_validation("syn_val"), _validation("syn_test", primary=False)],
            "syn_val",
        )
        calibration = _metrics(rr=0.9, threshold=0.4, candidates=100)
        selection_metrics = {
            "calibrated_threshold": {
                "threshold": 0.4,
                "pair_completeness": 0.98,
                "reduction_ratio": 0.96,
            },
            "recall_at_k": [
                {"k": k, "pair_completeness": pc}
                for k, pc in zip([10, 20, 40, 80, 160], [0.8, 0.85, 0.9, 0.95, 1.0])
            ],
        }
        result = _weighted_mean_pc_at_k_and_transfer_result(
            calibration, selection_metrics, resolved
        )
        self.assertAlmostEqual(result["mean_pc_at_k"], 0.9)
        self.assertAlmostEqual(result["score"], 0.5 * 0.9 + 0.25 * 0.98 + 0.25 * 0.96)
        self.assertIn("mean PC@k=10,20,40,80,160", _checkpoint_selection_label(resolved))

    def test_schema_accepts_exact_weighted_protocol(self):
        _validate_protocol_blocks(
            {
                "finetuning": {
                    "data": {
                        "train": {"subset": "train", "roles": ["all"]},
                        "validation": [
                            {"name": "syn_val", "primary": True, "subset": "val", "roles": ["all"]},
                            {"name": "syn_test", "primary": False, "subset": "test", "roles": ["all"]},
                        ],
                    },
                    "evaluation": {
                        "target_pc": TARGET_PC,
                        "top_k": [10, 20, 40, 80, 160],
                        "checkpoint_selection": {
                            "metric": "weighted_mean_pc_at_k_and_transfer",
                            "calibration_validation_set": "syn_val",
                            "selection_validation_set": "syn_test",
                            "fixed_k": [10, 20, 40, 80, 160],
                            "weights": {
                                "mean_pc_at_k": 0.5,
                                "transferred_pc": 0.25,
                                "transferred_rr": 0.25,
                            },
                        },
                    },
                }
            },
            Path("experiment.yml"),
        )

    def test_missing_fixed_k_and_bad_weights_are_rejected(self):
        base = {
            "target_pc": TARGET_PC,
            "top_k": [10, 20],
            "checkpoint_selection": {
                "metric": "weighted_mean_pc_at_k_and_transfer",
                "calibration_validation_set": "syn_val",
                "selection_validation_set": "syn_test",
                "fixed_k": [10, 20, 40],
                "weights": {
                    "mean_pc_at_k": 0.5,
                    "transferred_pc": 0.25,
                    "transferred_rr": 0.25,
                },
            },
        }
        sets = [_validation("syn_val"), _validation("syn_test", primary=False)]
        with self.assertRaisesRegex(ValueError, "missing: 40"):
            _checkpoint_selection_config(base, sets, "syn_val")
        base["top_k"].append(40)
        base["checkpoint_selection"]["weights"]["transferred_rr"] = 0.2
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            _checkpoint_selection_config(base, sets, "syn_val")


class ReductionRatioCheckpointConfigTest(unittest.TestCase):
    def test_valid_config_is_normalized_with_inferred_set_target_and_task(self):
        resolved = _checkpoint_selection_config(
            {
                "target_pc": TARGET_PC,
                "checkpoint_selection": {"metric": METRIC},
            },
            [_validation()],
            "only_val",
        )
        self.assertEqual(
            resolved,
            {
                "validation_set": "only_val",
                "metric": METRIC,
                "target_pc": TARGET_PC,
                "task_id": "all_pairs",
            },
        )
        self.assertEqual(_checkpoint_selection_label(resolved), "only_val RR@PC>=0.99")

    def test_role_aware_config_requires_at_most_one_task_and_records_it(self):
        task = {"task_id": "modern_to_historic"}
        resolved = _checkpoint_selection_config(
            {
                "target_pc": TARGET_PC,
                "checkpoint_selection": {"metric": METRIC},
            },
            [_validation(task_specs=[task])],
            "only_val",
        )
        self.assertEqual(resolved["task_id"], "modern_to_historic")
        with self.assertRaisesRegex(ValueError, "one retrieval task"):
            _checkpoint_selection_config(
                {
                    "target_pc": TARGET_PC,
                    "checkpoint_selection": {"metric": METRIC},
                },
                [_validation(task_specs=[task, {"task_id": "other"}])],
                "only_val",
            )

    def test_multiple_sets_wrong_target_k_and_mismatching_set_are_rejected(self):
        cases = [
            (
                {"target_pc": TARGET_PC, "checkpoint_selection": {"metric": METRIC}},
                [_validation(), _validation("other", primary=False)],
                "exactly one validation set",
            ),
            (
                {"target_pc": 1.01, "checkpoint_selection": {"metric": METRIC}},
                [_validation()],
                "in \\(0, 1\\]",
            ),
            (
                {"target_pc": TARGET_PC, "checkpoint_selection": {"metric": METRIC, "k": 10}},
                [_validation()],
                "not valid",
            ),
            (
                {
                    "target_pc": TARGET_PC,
                    "checkpoint_selection": {"metric": METRIC, "validation_set": "other"},
                },
                [_validation()],
                "not configured",
            ),
        ]
        for eval_cfg, sets, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _checkpoint_selection_config(eval_cfg, sets, "only_val")

    def test_predeclared_high_recall_target_is_not_hardcoded_to_099(self):
        resolved = _checkpoint_selection_config(
            {
                "target_pc": 0.995,
                "checkpoint_selection": {"metric": METRIC},
            },
            [_validation()],
            "only_val",
        )
        self.assertEqual(resolved["target_pc"], 0.995)

    def test_existing_pc_at_k_config_is_unchanged(self):
        resolved = _checkpoint_selection_config(
            {
                "target_pc": 0.995,
                "checkpoint_selection": {"metric": "pc_at_k", "k": 25},
            },
            [_validation()],
            "only_val",
        )
        self.assertEqual(
            resolved,
            {"validation_set": "only_val", "metric": "pc_at_k", "k": 25},
        )

    def test_schema_rejects_invalid_protocol_before_training(self):
        base = {
            "finetuning": {
                "data": {
                    "train": {"subset": "train", "roles": ["all"]},
                    "validation": [
                        {
                            "name": "only_val",
                            "primary": True,
                            "subset": "val",
                            "roles": ["all"],
                        }
                    ],
                },
                "evaluation": {
                    "target_pc": TARGET_PC,
                    "checkpoint_selection": {"metric": METRIC},
                },
            }
        }
        _validate_protocol_blocks(base, Path("experiment.yml"))
        invalid = {
            **base,
            "finetuning": {
                **base["finetuning"],
                "evaluation": {
                    "target_pc": TARGET_PC,
                    "checkpoint_selection": {"metric": METRIC, "target_pc": TARGET_PC},
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "single source of truth"):
            _validate_protocol_blocks(invalid, Path("experiment.yml"))


class TargetPcThresholdSelectionTest(unittest.TestCase):
    def test_highest_feasible_threshold_keeps_target_and_actual_pc(self):
        similarities = np.asarray([[0.9], [0.8], [0.8], [0.1]], dtype=np.float32)
        positives = np.ones_like(similarities, dtype=bool)
        valid = np.ones_like(similarities, dtype=bool)

        point = select_threshold_at_pair_completeness(
            similarities, positives, valid, target_pc=0.6
        )

        self.assertAlmostEqual(point["threshold"], 0.8, places=6)
        self.assertEqual(point["target_pair_completeness"], 0.6)
        self.assertEqual(point["pair_completeness"], 0.75)
        self.assertEqual(point["candidate_pairs"], 3)
        self.assertAlmostEqual(point["reduction_ratio"], 0.25)


KS = [5, 10, 20, 40, 80]


def _precision_metrics(precision: float, reduction_ratio: float) -> dict:
    return {
        "recall_at_k": [
            {"k": k, "precision": precision, "reduction_ratio": reduction_ratio}
            for k in KS
        ]
    }


def _precision_selection() -> dict:
    return {
        "validation_set": "met_checkpoint_selection",
        "metric": "mean_precision_at_k",
        "k": KS,
        "precision_tie": 0.01,
        "epoch_tiebreak": "earliest",
    }


class MeanPrecisionCheckpointComparatorTest(unittest.TestCase):
    def test_more_than_one_point_of_precision_wins_without_looking_at_rr(self):
        best = _precision_metrics(0.20, 0.90)
        candidate = _precision_metrics(0.211, 0.10)
        self.assertTrue(
            _is_better_for_selection(candidate, best, TARGET_PC, _precision_selection())
        )

    def test_within_one_point_uses_mean_reduction_ratio(self):
        best = _precision_metrics(0.20, 0.50)
        higher_rr = _precision_metrics(0.205, 0.60)
        lower_rr = _precision_metrics(0.209, 0.40)
        self.assertTrue(
            _is_better_for_selection(higher_rr, best, TARGET_PC, _precision_selection())
        )
        self.assertFalse(
            _is_better_for_selection(lower_rr, best, TARGET_PC, _precision_selection())
        )

    def test_equal_precision_and_rr_keeps_the_earlier_epoch(self):
        best = _precision_metrics(0.20, 0.50)
        candidate = _precision_metrics(0.20, 0.50)
        self.assertFalse(
            _is_better_for_selection(candidate, best, TARGET_PC, _precision_selection())
        )

    def test_mean_is_taken_across_the_five_k_values(self):
        metrics = {
            "recall_at_k": [
                {"k": k, "precision": precision, "reduction_ratio": reduction_ratio}
                for k, precision, reduction_ratio in zip(
                    KS, [0.5, 0.4, 0.3, 0.2, 0.1], [0.9, 0.8, 0.7, 0.6, 0.5]
                )
            ]
        }
        result = _mean_precision_at_k_result(metrics, _precision_selection())
        self.assertAlmostEqual(result["mean_precision"], 0.3)
        self.assertAlmostEqual(result["mean_reduction_ratio"], 0.7)

    def test_config_requires_those_k_values_in_top_k(self):
        resolved = _checkpoint_selection_config(
            {
                "top_k": KS,
                "checkpoint_selection": {
                    "metric": "mean_precision_at_k",
                    "validation_set": "only_val",
                    "k": KS,
                    "precision_tie": 0.01,
                },
            },
            [_validation()],
            "only_val",
        )
        self.assertEqual(resolved["k"], KS)
        self.assertEqual(resolved["precision_tie"], 0.01)
        self.assertIn("mean precision@k=5,10,20,40,80", _checkpoint_selection_label(resolved))
        with self.assertRaisesRegex(ValueError, "missing"):
            _checkpoint_selection_config(
                {
                    "top_k": [10, 20, 40, 80],
                    "checkpoint_selection": {
                        "metric": "mean_precision_at_k",
                        "k": KS,
                    },
                },
                [_validation()],
                "only_val",
            )


class EarlyStopStreakTest(unittest.TestCase):
    def test_first_selected_epoch_does_not_count_as_a_miss(self):
        streak, stop = _next_non_improving_streak(True, 0, 2)
        self.assertEqual((streak, stop), (0, False))

    def test_one_miss_keeps_training(self):
        streak, stop = _next_non_improving_streak(False, 0, 2)
        self.assertEqual((streak, stop), (1, False))

    def test_second_consecutive_miss_stops_after_that_epoch(self):
        streak, stop = _next_non_improving_streak(False, 1, 2)
        self.assertEqual((streak, stop), (2, True))

    def test_a_new_best_resets_the_streak(self):
        streak, stop = _next_non_improving_streak(True, 1, 2)
        self.assertEqual((streak, stop), (0, False))

    def test_patience_can_be_disabled(self):
        self.assertIsNone(_early_stop_patience({}))
        self.assertIsNone(_early_stop_patience({"early_stop_non_improving_epochs": None}))
        streak, stop = _next_non_improving_streak(False, 4, None)
        self.assertEqual((streak, stop), (0, False))

    def test_patience_must_be_a_positive_integer(self):
        self.assertEqual(_early_stop_patience({"early_stop_non_improving_epochs": 2}), 2)
        with self.assertRaises(ValueError):
            _early_stop_patience({"early_stop_non_improving_epochs": 0})
        with self.assertRaises(ValueError):
            _early_stop_patience({"early_stop_non_improving_epochs": True})


if __name__ == "__main__":
    unittest.main()
