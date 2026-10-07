import json
import tempfile
import unittest
from pathlib import Path

from experiments.core.latex_tables import write_run_latex_tables


class LatexTablesTest(unittest.TestCase):
    def test_write_run_latex_tables_formats_metrics_and_bolds_best_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            _write_run(run_dir)
            _write_eval(
                run_dir,
                "model_b",
                "storage_b",
                top_k=[
                    {"k": 1, "reduction_ratio": 0.9, "pair_completeness": 0.7, "pair_quality": 0.2},
                    {"k": 5, "reduction_ratio": 0.5, "pair_completeness": 0.8, "pair_quality": 0.16},
                ],
                target_pc=[
                    {"target_pair_completeness": 0.95, "reduction_ratio": 0.92, "pair_completeness": 0.951, "pair_quality": 0.0119},
                    {"target_pair_completeness": 1.0, "reduction_ratio": 0.81, "pair_completeness": 1.0, "pair_quality": 0.0041},
                ],
                calibrated=[
                    {"calibration_target_pair_completeness": 0.95, "reduction_ratio": 0.92, "pair_completeness": 0.94, "pair_quality": 0.0119, "threshold": 0.2},
                    {"calibration_target_pair_completeness": 0.99, "reduction_ratio": 0.87, "pair_completeness": 0.97, "pair_quality": 0.0070, "threshold": 0.15},
                    {"calibration_target_pair_completeness": 0.995, "reduction_ratio": 0.85, "pair_completeness": 0.96, "pair_quality": 0.0060, "threshold": 0.14},
                ],
            )
            _write_eval(
                run_dir,
                "model_a",
                "storage_a",
                top_k=[
                    {"k": 1, "reduction_ratio": 0.8, "pair_completeness": 0.6, "pair_quality": 0.3},
                    {"k": 5, "reduction_ratio": 0.5, "pair_completeness": 0.9, "pair_quality": 0.18},
                ],
                target_pc=[
                    {"target_pair_completeness": 0.95, "reduction_ratio": 0.93, "pair_completeness": 0.95, "pair_quality": 0.0125},
                    {"target_pair_completeness": 1.0, "reduction_ratio": 0.82, "pair_completeness": 1.0, "pair_quality": 0.0040},
                ],
                calibrated=[
                    {"calibration_target_pair_completeness": 0.95, "reduction_ratio": 0.93, "pair_completeness": 0.93, "pair_quality": 0.0125, "threshold": 0.25},
                    {"calibration_target_pair_completeness": 0.99, "reduction_ratio": 0.88, "pair_completeness": 0.96, "pair_quality": 0.0075, "threshold": 0.18},
                    {"calibration_target_pair_completeness": 0.995, "reduction_ratio": 0.86, "pair_completeness": 0.95, "pair_quality": 0.0065, "threshold": 0.17},
                ],
            )

            paths = write_run_latex_tables(run_dir)
            fixed = paths["fixed_k"].read_text(encoding="utf-8")
            fixed_values = paths["fixed_k_values"].read_text(encoding="utf-8")
            calibrated = paths["calibrated_threshold"].read_text(encoding="utf-8")
            calibrated_values = paths["calibrated_threshold_values"].read_text(encoding="utf-8")
            combined = paths["combined"].read_text(encoding="utf-8")

            # The oracle target-PC table is no longer emitted.
            self.assertNotIn("target_pc", paths)
            self.assertEqual(combined.count("\\begin{table}[t]"), 2)
            self.assertIn("& \\multicolumn{3}{c}{$k=1$}", fixed)
            self.assertIn("& \\multicolumn{3}{c}{$PC \\geq 0.95$}", calibrated)
            self.assertIn("& \\multicolumn{3}{c}{$PC \\geq 0.99$}", calibrated)
            self.assertIn("& \\multicolumn{3}{c}{$PC \\geq 0.995$}", calibrated)
            self.assertEqual(calibrated.count("$PC \\geq 0.99$"), 1)
            self.assertNotIn("$PC = 1.00$", calibrated)
            self.assertIn("applied unchanged to the held-out test subset", calibrated)
            self.assertIn("\\input{latex_tables/fixed_k_values.tex}", fixed)
            self.assertIn("\\input{latex_tables/calibrated_threshold_values.tex}", calibrated)
            self.assertLess(fixed_values.index("Beta\\_Model"), fixed_values.index("Alpha Model"))
            self.assertIn("rev abc123", fixed_values)
            self.assertIn("& \\textbf{0.9000} & \\textbf{0.700} & 0.200", fixed_values)
            self.assertIn("& 0.8000 & 0.600 & \\textbf{0.300}", fixed_values)
            self.assertIn("& \\textbf{0.5000} & 0.800 & 0.160", fixed_values)
            self.assertIn("& \\textbf{0.5000} & \\textbf{0.900} & \\textbf{0.180}", fixed_values)
            # Calibrated table reports achieved test PC/PQ/RR grouped by calibration target.
            self.assertIn("& 0.9200 & \\textbf{0.940} & 0.0119", calibrated_values)
            self.assertIn("& \\textbf{0.9300} & 0.930 & \\textbf{0.0125}", calibrated_values)


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    run_json = {
        "run_id": "run1",
        "experiment_id": "exp",
        "status": "completed",
        "dataset": {"dataset_id": "toy_dataset"},
        "split": {"split_id": "split"},
        "model_runs": [{"model_id": "model_b", "model_revision": "abc123"}, {"model_id": "model_a"}],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    (run_dir / "resolved_config.yml").write_text(
        "models:\n"
        "  include:\n"
        "    - model_id: model_b\n"
        "      display_name: Beta_Model\n"
        "    - model_id: model_a\n"
        "      display_name: Alpha Model\n",
        encoding="utf-8",
    )


def _write_eval(
    run_dir: Path,
    model_id: str,
    storage_key: str,
    top_k: list[dict[str, float]],
    target_pc: list[dict[str, float]],
    calibrated: list[dict[str, float]] | None = None,
) -> None:
    model_dir = run_dir / "models" / storage_key
    model_dir.mkdir(parents=True)
    payload = {
        "model_id": model_id,
        "model_storage_key": storage_key,
        "tasks": [
            {
                "task_id": "modern_to_historic",
                "num_queries": 2,
                "num_candidates": 10,
                "num_positive_pairs": 4,
                "metrics": top_k,
                "target_pc_metrics": target_pc,
                "calibrated_threshold_metrics": calibrated or [],
            }
        ],
    }
    (model_dir / "eval.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
