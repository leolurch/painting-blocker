from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.core.multisplit_tables import (
    _calibrated_top_k_table,
    _calibration_table,
    _fixed_table,
    _paired_table,
    _profile_table,
    _syn_test_comparison_table,
    write_multisplit_tables,
)


def _summary(name: str, mode: str, *, k: int | None = None, target: float | None = None) -> dict:
    row = {
        "configuration_id": name,
        "analysis_model_id": name,
        "display_name": name,
        "profile": "archival",
        "selection_mode": mode,
        "k": k,
        "calibration_target_pair_completeness": target,
        "num_split_realizations": 2,
        "target_attainment_count": 2,
        "validation_selection_attainment_count": 2,
        "threshold_mean": 0.5,
        "threshold_sample_sd_split": 0.01,
        "selected_k_mean": 40.0,
        "selected_k_sample_sd_split": 0.0,
        "query_coverage_mean": 1.0,
        "candidate_pairs_mean": 100.0,
    }
    for metric, mean in (
        ("pair_completeness", 0.99),
        ("pair_quality", 0.5),
        ("reduction_ratio", 0.9),
    ):
        row[f"{metric}_mean"] = mean
        row[f"{metric}_sample_sd_split"] = 0.01
        row[f"{metric}_min"] = mean - 0.01
        row[f"{metric}_max"] = mean + 0.01
    return row


def _paired(name: str, *, k: int | None = None, target: float | None = None) -> dict:
    row = {
        "comparison_id": name,
        "k": k,
        "calibration_target_pair_completeness": target,
        "num_split_realizations": 2,
    }
    for metric in ("pair_completeness", "pair_quality", "reduction_ratio"):
        row[f"{metric}_difference_mean"] = 0.1
        row[f"{metric}_difference_sample_sd_split"] = 0.01
        row[f"{metric}_difference_min"] = 0.09
        row[f"{metric}_difference_max"] = 0.11
        row[f"{metric}_wins"] = 2
        row[f"{metric}_ties"] = 0
    return row


class MultisplitTableOperatingPointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "analysis_id": "test",
            "multisplit": {"aggregation": {"attainment_tolerance": 0.01}},
            "charts": {
                "multisplit_tables": {
                    "enabled": True,
                    "fixed_k": 40,
                    "calibration_target": 0.99,
                }
            },
            "models": [
                {
                    "analysis_model_id": model_id,
                    "include": True,
                    "presentation": {"label": model_id, "short_label": model_id},
                }
                for model_id in ("Target99", "Target95", "TopK99", "TopK95")
            ],
        }
        self.artifact = {
            "expected_split_seeds": [6, 7],
            "summary_rows": [
                _summary("K40", "top_k", k=40),
                _summary("K20", "top_k", k=20),
                _summary("Target99", "calibrated_threshold", target=0.99),
                _summary("Target95", "calibrated_threshold", target=0.95),
                _summary("TopK99", "calibrated_top_k", target=0.99),
                _summary("TopK95", "calibrated_top_k", target=0.95),
            ],
            "profile_diagnostic_summaries": [
                _summary("Target99", "calibrated_threshold", target=0.99),
                _summary("Target95", "calibrated_threshold", target=0.95),
            ],
            "paired_comparisons": [
                _paired("Fixed40", k=40),
                _paired("Fixed20", k=20),
                _paired("Calibrated99", target=0.99),
                _paired("Calibrated95", target=0.95),
            ],
        }

    def test_fixed_table_contains_only_configured_k(self):
        table = _fixed_table(self.artifact, self.config)

        self.assertIn("performance at $k=40$", table)
        self.assertIn("RR is model-independent at this budget", table)
        self.assertIn("K40", table)
        self.assertNotIn("K20", table)
        self.assertNotIn("Model & $k$", table)
        self.assertNotIn("RR mean", table)
        self.assertNotIn("K40 & 40 &", table)

    def test_fixed_table_rejects_model_dependent_reduction_ratio(self):
        self.artifact["summary_rows"][1]["k"] = 40
        self.artifact["summary_rows"][1]["reduction_ratio_mean"] = 0.8

        with self.assertRaisesRegex(ValueError, "not model-independent"):
            _fixed_table(self.artifact, self.config)

    def test_all_calibrated_tables_use_only_configured_target(self):
        calibration = _calibration_table(self.artifact, self.config)
        calibrated_k = _calibrated_top_k_table(self.artifact, self.config)
        profile = _profile_table(self.artifact, self.config)
        paired = _paired_table(self.artifact, self.config)

        self.assertIn("Target99", calibration)
        self.assertNotIn("Target95", calibration)
        self.assertIn("TopK99", calibrated_k)
        self.assertNotIn("TopK95", calibrated_k)
        self.assertIn("Target99", profile)
        self.assertNotIn("Target95", profile)
        self.assertIn("Fixed40", paired)
        self.assertNotIn("Fixed20", paired)
        self.assertIn("Calibrated99", paired)
        self.assertNotIn("Calibrated95", paired)

    def test_calibration_tables_use_short_names_and_only_included_models(self):
        self.config["models"][0]["presentation"] = {
            "label": "Long threshold model name",
            "short_label": "ShortThreshold",
        }
        self.config["models"][2]["presentation"] = {
            "label": "Long top-k model name",
            "short_label": "ShortTopK",
        }
        self.config["models"].extend([
            {
                "analysis_model_id": "ExcludedThreshold",
                "include": False,
                "presentation": {"short_label": "ExcludedThresholdShort"},
            },
            {
                "analysis_model_id": "ExcludedTopK",
                "include": False,
                "presentation": {"short_label": "ExcludedTopKShort"},
            },
        ])
        self.artifact["summary_rows"].extend([
            _summary("ExcludedThreshold", "calibrated_threshold", target=0.99),
            _summary("ExcludedTopK", "calibrated_top_k", target=0.99),
        ])

        calibration = _calibration_table(self.artifact, self.config)
        calibrated_k = _calibrated_top_k_table(self.artifact, self.config)

        self.assertIn("ShortThreshold", calibration)
        self.assertNotIn("Long threshold model name", calibration)
        self.assertNotIn("ExcludedThreshold", calibration)
        self.assertIn("ShortTopK", calibrated_k)
        self.assertNotIn("Long top-k model name", calibrated_k)
        self.assertNotIn("ExcludedTopK", calibrated_k)

    def test_calibration_tables_omit_rho_and_query_coverage_columns(self):
        self.artifact["summary_rows"][2]["pair_completeness_split_values"] = [
            {"mean_over_training_seeds": 0.985},
            {"mean_over_training_seeds": 0.970},
        ]
        calibration = _calibration_table(self.artifact, self.config)
        calibrated_k = _calibrated_top_k_table(self.artifact, self.config)

        self.assertNotIn("$\\rho$", calibration)
        self.assertNotIn("QC mean", calibration)
        self.assertNotIn("$\\rho$", calibrated_k)
        self.assertNotIn("$\\Delta$PC", calibration)
        self.assertNotIn(r"\hat{\tau}", calibration)
        self.assertIn(
            r"$\mathrm{PC}_{\mathrm{test}}$ mean $\pm\,\sigma$",
            calibration,
        )
        self.assertIn(r"$\mathrm{RR}$ mean $\pm\,\sigma$", calibration)
        self.assertIn(r"$|C|$ mean", calibration)
        self.assertNotIn("Candidate pairs", calibration)
        self.assertIn(
            r"\multicolumn{5}{c}{Calibration target $\rho=0.990$}",
            calibration,
        )
        self.assertIn(
            r"$\mathrm{PC}_{\mathrm{test}} \ge 0.980$",
            calibration,
        )
        self.assertIn("Target99 &", calibration)
        self.assertIn(r" & \textbf{1}/2 \\" , calibration)
        self.assertNotIn(r"\textbf{1/2}", calibration)
        header = next(
            line for line in calibration.splitlines() if line.strip().startswith("Model &")
        )
        self.assertTrue(
            header.endswith(
                r"$\mathrm{PC}_{\mathrm{test}} \ge 0.980$ \\"
            )
        )
        self.assertNotIn(r"\mu \pm", calibration)
        self.assertNotIn(r"$\mu$", calibration)
        self.assertNotIn(" SD", calibration)
        self.assertIn("\\begin{tabular}{lrrrr}", calibration)
        self.assertIn("\\begin{tabular}{lrrrrrrr}", calibrated_k)

    def test_calibration_caption_is_configurable(self):
        caption = r"Configured calibration caption with $\sigma$"
        self.config["charts"]["multisplit_tables"]["calibration_caption"] = caption

        table = _calibration_table(self.artifact, self.config)

        self.assertIn(f"\\caption{{{caption}}}", table)
        self.assertNotIn("Validation-calibrated performance over", table)

    def test_calibration_table_bolds_each_requested_column_winner(self):
        other = _summary("Other99", "calibrated_threshold", target=0.99)
        other.update({
            "threshold_mean": 0.6,
            "threshold_sample_sd_split": 0.02,
            "pair_completeness_mean": 0.98,
            "pair_completeness_sample_sd_split": 0.005,
            "reduction_ratio_mean": 0.8,
            "reduction_ratio_sample_sd_split": 0.005,
            "candidate_pairs_mean": 80.0,
        })
        self.artifact["summary_rows"].append(other)
        self.config["models"].append({
            "analysis_model_id": "Other99",
            "include": True,
            "presentation": {"short_label": "Other"},
        })

        table = _calibration_table(self.artifact, self.config)

        self.assertIn(
            r"Target99 & $\mathbf{0.990} \pm 0.010$",
            table,
        )
        self.assertIn(
            r"Other & $0.980 \pm \mathbf{0.005}$",
            table,
        )
        self.assertIn(r"$\mathbf{0.9000} \pm 0.0100$", table)
        self.assertIn(r"$0.8000 \pm \mathbf{0.0050}$", table)
        self.assertIn(r"$\mathbf{80.0}$", table)
        self.assertNotIn(r"$\mathbf{100.0}$", table)

    def test_syn_test_comparison_is_configurable_and_tabular_only(self):
        target = next(
            row
            for row in self.artifact["summary_rows"]
            if row["configuration_id"] == "Target99"
            and row["selection_mode"] == "calibrated_threshold"
        )
        target["pair_completeness_mean"] = 0.9946
        other = _summary("Other99", "calibrated_threshold", target=0.99)
        other["pair_completeness_mean"] = 0.995
        other["reduction_ratio_mean"] = 0.8
        self.artifact["summary_rows"].append(other)
        self.config["models"].append({
            "analysis_model_id": "Other99",
            "include": True,
            "presentation": {"short_label": "Other"},
        })
        settings = self.config["charts"]["multisplit_tables"]
        settings["syn_test_comparison"] = {
            "enabled": True,
            "models": ["Target99", "Other99"],
            "task": None,
            "target_pc": 0.99,
            "synthetic_pc_at_k": 10,
        }

        table = _syn_test_comparison_table(
            self.artifact,
            self.config,
            {"Target99": 0.7054, "Other99": 0.8},
        )

        self.assertTrue(table.startswith("\\begin{tabular}{lrrr}\n"))
        self.assertTrue(table.endswith("\\end{tabular}\n"))
        self.assertNotIn("\\begin{table}", table)
        self.assertNotIn("\\caption", table)
        self.assertIn(
            r"Model & $\mathrm{PC}_{\mathrm{test}}$ at $k=10$"
            r" & $\mathrm{PC}_{\mathrm{test}}$ at $\rho_{\mathrm{val}}=.99$"
            r" & $\mathrm{RR}_{\mathrm{test}}$ at $\rho_{\mathrm{val}}=.99$",
            table,
        )
        self.assertIn(
            r"Target99 & 0.705 & \textbf{0.995} & \textbf{0.9000}", table
        )
        self.assertIn(
            r"Other & \textbf{0.800} & \textbf{0.995} & 0.8000", table
        )
        self.assertLess(table.index("Target99 &"), table.index("Other &"))

    def test_writer_emits_syn_test_comparison_fragment(self):
        settings = self.config["charts"]["multisplit_tables"]
        settings["syn_test_comparison"] = {
            "enabled": True,
            "models": ["Target99"],
            "target_pc": 0.99,
            "synthetic_pc_at_k": 10,
            "synthetic_pc_source": "syn.json",
            "synthetic_calibrated_source": "syn_calibrated.json",
            "synthetic_model_id_aliases": {
                "Target99": "synthetic-sidecar-target"
            },
        }
        sidecar = {
            "candidate_budget": 10,
            "bars": [
                {
                    "model_id": "synthetic-sidecar-target",
                    "pair_completeness": 0.7054,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "syn.json").write_text(json.dumps(sidecar), encoding="utf-8")
            (output / "syn_calibrated.json").write_text(
                json.dumps({
                    "calibration_target_pair_completeness": 0.99,
                    "bars": [
                        {"model_id": "synthetic-sidecar-target", "pc": 0.981}
                    ],
                }),
                encoding="utf-8",
            )
            paths = write_multisplit_tables(self.artifact, self.config, output)
            comparison = output / "multisplit_syn_test_comparison.tex"
            values = output / "multisplit_syn_test_comparison_values.tex"
            metric_bundle = output / "multisplit_cross_dataset_metrics.json"

            self.assertIn(comparison, paths)
            self.assertIn(values, paths)
            self.assertIn(metric_bundle, paths)
            bundle = json.loads(metric_bundle.read_text(encoding="utf-8"))
            self.assertEqual(bundle["artifact_type"], "cross_dataset_table_metrics")
            self.assertEqual(bundle["models"][0]["syn_test_pc_at_k"], 0.7054)
            self.assertEqual(
                bundle["models"][0]["syn_test_pc_at_calibrated_threshold"],
                0.981,
            )
            self.assertEqual(bundle["models"][0]["wik_test_pc_mean"], 0.99)
            self.assertEqual(
                bundle["models"][0]["wik_test_pc_sample_sd"], 0.01
            )
            self.assertEqual(bundle["models"][0]["wik_test_pq_mean"], 0.5)
            self.assertEqual(
                bundle["models"][0]["wik_test_pq_sample_sd"], 0.01
            )
            self.assertEqual(bundle["models"][0]["wik_test_rr_mean"], 0.9)
            self.assertEqual(
                bundle["models"][0]["wik_test_rr_sample_sd"], 0.01
            )
            self.assertEqual(bundle["wik_calibration_targets"], [0.99])
            self.assertEqual(
                bundle["models"][0]["wik_calibrated_metrics_by_target"]["0.99"][
                    "wik_test_pc_mean"
                ],
                0.99,
            )
            text = comparison.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("% LaTeX dependencies"))
            self.assertIn("\\begin{tabular}", text)
            self.assertTrue(text.endswith("\\end{tabular}\n"))
            self.assertIn("\\input{multisplit_syn_test_comparison_values.tex}", text)
            self.assertIn("Target99", values.read_text(encoding="utf-8"))

    def test_analysis_id_underscores_are_not_escaped_in_labels(self):
        self.config["analysis_id"] = "hard_synth_wikidata_1_5_frozen_multisplit"

        table = _calibrated_top_k_table(self.artifact, self.config)

        self.assertIn(
            "\\label{tab:hard_synth_wikidata_1_5_frozen_multisplit-multisplit-calibrated-k}",
            table,
        )
        label_lines = [line for line in table.splitlines() if r"\label{" in line]
        self.assertTrue(all(r"\_" not in line for line in label_lines))

    def test_multiple_table_operating_points_are_rejected(self):
        self.config["charts"]["multisplit_tables"]["fixed_k"] = [20, 40]

        with self.assertRaisesRegex(ValueError, "exactly one value"):
            _fixed_table(self.artifact, self.config)


if __name__ == "__main__":
    unittest.main()
