from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.core.analysis_readme import render_analysis_readme, write_analysis_readme


class AnalysisReadmeTest(unittest.TestCase):
    def test_single_source_documents_every_chart_and_configured_values(self):
        config = {
            "analysis_id": "single",
            "charts": {
                "pc_at_k_curve": {
                    "enabled": True,
                    "task": "test_task",
                    "target_pc_lines": [0.95, 0.99],
                    "formats": ["pdf"],
                },
                "pc_at_k_curve_zoomed": {
                    "enabled": True,
                    "task": "test_task",
                    "minimum_k_exclusive": 20,
                    "pc_limits": [0.9, 1.0],
                    "output_stem": "pc_at_k_gt20_pc0p9_1p0",
                    "formats": ["pdf", "svg"],
                },
                "pc_rr_tradeoff": {"enabled": True, "task": "test_task"},
                "pc_at_k_star_bars": {
                    "enabled": True,
                    "task": "test_task",
                    "preferred_k": 80,
                },
                "calibrated_threshold_bars": {
                    "enabled": True,
                    "task": "test_task",
                    "calibration_targets": [0.95, 0.99],
                },
                "validation_rr_bars": {
                    "enabled": True,
                    "task": "test_task",
                    "calibration_targets": [0.99],
                },
                "ranked_pc_at_k_bars": {
                    "enabled": True,
                    "task": "test_task",
                    "ks": [1, 80],
                    "formats": ["svg"],
                },
                "similarity_distributions": {"enabled": True, "task": "test_task"},
                "candidate_sizes": {
                    "enabled": True,
                    "task": "test_task",
                    "calibration_targets": [0.99],
                },
                "pq_pc_curve": {"enabled": True, "task": "test_task"},
                "pq_pc_curve_zoomed": {
                    "enabled": True,
                    "task": "test_task",
                    "pc_limits": [0.995, 1.0],
                    "pq_limits": [0.0, 0.001],
                },
            },
        }

        text = render_analysis_readme(config)

        for stem in (
            "pc_at_k_<task>",
            "pc_at_k_gt20_pc0p9_1p0_<task>",
            "pc_rr_tradeoff_<task>",
            "pc<k*>_bars_<task>",
            "pc_calibrated_rho<ρ>_bars_<task>",
            "validation_rr_rho<ρ>_bars_<task>",
            "ranked_pc_at_<k>_<task>",
            "similarity_distributions_<task>",
            "candidate_sizes_rho<ρ>_<task>",
            "pq_pc_<task>",
            "pq_pc_zoomed_<task>",
        ):
            self.assertIn(stem, text)
        self.assertIn("`PC = TP / P`", text)
        self.assertIn("2,001 cutoffs", text)
        self.assertIn("Preferred budget: `80`", text)
        self.assertIn("`k > 20`", text)
        self.assertIn("`PC ∈ [0.9, 1]`", text)
        self.assertIn("PC=[0.995, 1.0]", text)
        self.assertIn("same-basename `.json` sidecar", text)

    def test_multisplit_documents_aggregation_and_enabled_charts(self):
        config = {
            "analysis_id": "multi",
            "mode": "multisplit",
            "multisplit": {"expected_split_seeds": [6, 7]},
            "model_groups": {
                "variants": {"canonical": "a", "members": ["a", "b"]}
            },
            "charts": {
                "multisplit_pc_at_k_band": {"enabled": True, "task": "test"},
                "multisplit_strip": {"enabled": True, "task": "test", "k": "all"},
                "multisplit_paired_difference": {"enabled": True, "task": "test", "k": 40},
                "multisplit_heatmap": {"enabled": True, "task": "test", "k": 40},
                "multisplit_calibration_transfer": {
                    "enabled": True,
                    "task": "test",
                    "target_pc": 0.99,
                },
                "multisplit_calibrated_top_k_transfer": {
                    "enabled": True,
                    "task": "test",
                    "target_pc": [0.95, 0.99],
                },
                "multisplit_threshold_top_k_floor_transfer": {
                    "enabled": True,
                    "task": "test",
                    "target_pc": [0.95, 0.99],
                    "minimum_top_k": [5, 10, 20, 40],
                },
                "multisplit_similarity_distributions": {
                    "enabled": True,
                    "task": "test",
                    "target_pc": 0.99,
                },
            },
        }

        text = render_analysis_readme(config)

        for stem in (
            "multisplit_pc_at_k_band",
            "multisplit_strip_pc_at_<k>",
            "multisplit_paired_pc_at_<k>",
            "multisplit_heatmap_pc_at_<k>",
            "multisplit_calibration_transfer_rho<ρ>",
            "multisplit_calibrated_top_k_transfer_rho<ρ>",
            "multisplit_threshold_top_k_floor_transfer_rho<ρ>_k<k>",
            "multisplit_similarity_distributions_<task>",
        ):
            self.assertIn(stem, text)
        self.assertIn("first averaged arithmetically across repeats within the same split", text)
        self.assertIn("not confidence intervals", text)
        self.assertIn("_canonical_models", text)
        self.assertIn("never selected from a displayed validation or test metric", text)
        self.assertIn("`ΔPC@k = PC@k(minuend) - PC@k(subtrahend)`", text)
        self.assertIn("same-basename `.json` sidecar", text)

    def test_multisplit_documents_hidden_calibration_elements(self):
        config = {
            "analysis_id": "multi",
            "mode": "multisplit",
            "multisplit": {"expected_split_seeds": [6, 7]},
            "charts": {
                "calibration_style": {
                    "include_attainment_score": False,
                    "include_specific_result_crosses": False,
                    "pc_axis_label": "",
                    "pc_axis_description": "",
                    "rr_axis_label": "",
                    "rr_axis_description": "",
                },
                "multisplit_calibrated_top_k_transfer": {
                    "enabled": True,
                    "style_from": "calibration_style",
                    "task": "test",
                    "target_pc": [0.99],
                },
            },
        }

        text = render_analysis_readme(config)

        self.assertIn("Split-level held-out PC markers are omitted", text)
        self.assertIn("Attainment counts, labels, target reference", text)
        self.assertIn("PC and RR axis labels and descriptions are omitted", text)

    def test_writer_uses_requested_filename_in_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            path = write_analysis_readme(
                {"analysis_id": "empty", "charts": {}}, output
            )

            self.assertEqual(path, output / "Readme.md")
            self.assertTrue(path.is_file())
            self.assertIn("Chart calculations", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
