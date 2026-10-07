import unittest

from experiments.core.profile_summary import PROFILES, hard_synth_profile_summary


class HardSynthProfileSummaryTest(unittest.TestCase):
    def test_calibrated_macro_rows_group_by_calibration_target(self):
        tasks = []
        for index, profile in enumerate(PROFILES):
            tasks.append(
                {
                    "task_id": f"test_original_to_{profile}",
                    "num_queries": 2,
                    "num_positive_pairs": 4,
                    "metrics": [],
                    "calibrated_threshold_metrics": [
                        {
                            "calibration_target_pair_completeness": 0.95,
                            "threshold": 0.80 - index * 0.01,
                            "pair_completeness": 0.90 + index * 0.01,
                            "pair_quality": 0.20 + index * 0.01,
                            "reduction_ratio": 0.70 + index * 0.01,
                            "query_coverage": 0.80 + index * 0.01,
                        },
                        {
                            "calibration_target_pair_completeness": 0.99,
                            "threshold": 0.70 - index * 0.01,
                            "pair_completeness": 0.93 + index * 0.01,
                            "pair_quality": 0.15 + index * 0.01,
                            "reduction_ratio": 0.60 + index * 0.01,
                            "query_coverage": 0.85 + index * 0.01,
                        },
                        {
                            "calibration_target_pair_completeness": 0.995,
                            "threshold": 0.60 - index * 0.01,
                            "pair_completeness": 0.95 + index * 0.01,
                            "pair_quality": 0.10 + index * 0.01,
                            "reduction_ratio": 0.50 + index * 0.01,
                            "query_coverage": 0.90 + index * 0.01,
                        },
                    ],
                }
            )

        rows = hard_synth_profile_summary(tasks)["macro_average"]["calibrated_threshold"]

        self.assertEqual(
            [row["calibration_target_pair_completeness"] for row in rows],
            [0.95, 0.99, 0.995],
        )
        self.assertTrue(all("threshold" not in row for row in rows))
        self.assertAlmostEqual(rows[0]["pair_completeness"], 0.915)
        self.assertAlmostEqual(rows[2]["query_coverage"], 0.915)


if __name__ == "__main__":
    unittest.main()
