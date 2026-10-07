from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.core.training_seed_charts import (
    collect_training_seed_comparison,
    render_partition_training_seed_span,
    render_pc_at_k_by_training_seed,
)


def _eval(pc_at_10: float, calibrated: float) -> dict:
    return {
        "tasks": [
            {
                "task_id": "toy",
                "metrics": [
                    {"k": k, "pair_completeness": pc_at_10 if k == 10 else 0.5}
                    for k in (1, 5, 10, 20, 40, 80)
                ],
                "calibrated_threshold_metrics": [
                    {
                        "calibration_target_pair_completeness": 0.99,
                        "pair_completeness": calibrated,
                        "reduction_ratio": 0.99,
                        "pair_quality": 0.8,
                    }
                ],
            }
        ]
    }


class TrainingSeedChartsTest(unittest.TestCase):
    def test_seed_comparison_writes_figures_and_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, pc, calibrated in ((42, 0.90, 0.80), (43, 0.91, 0.81)):
                run = root / f"run-trainseed-{seed}"
                full = run / "evaluations" / "full_eval" / "models" / "m"
                part = run / "evaluations" / "parts" / "rotation-seed-7" / "models" / "m"
                full.mkdir(parents=True)
                part.mkdir(parents=True)
                (full / "eval.json").write_text(json.dumps(_eval(pc, calibrated)))
                (part / "eval.json").write_text(json.dumps(_eval(pc, calibrated)))
            comparison = collect_training_seed_comparison(
                [root / "run-trainseed-42", root / "run-trainseed-43"],
                full_evaluations={"syn": "full_eval", "wikidata_full": "full_eval"},
                partition_evaluation="parts",
            )
            output = root / "figures"
            pc_paths = render_pc_at_k_by_training_seed(comparison, output)
            span_paths = render_partition_training_seed_span(comparison, output)
            self.assertTrue((output / "training_seed_pc_at_k.pdf").is_file())
            self.assertTrue((output / "training_seed_vs_eval_split.json").is_file())
            self.assertGreaterEqual(len(pc_paths), 2)
            self.assertGreaterEqual(len(span_paths), 2)


if __name__ == "__main__":
    unittest.main()
