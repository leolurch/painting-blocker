from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.core.multisplit_analysis import (
    MultisplitAnalysisError,
    _candidate_score,
    _compatibility_values,
    _required_source_capabilities,
    _summarize_similarity_histogram_records,
    _validate_calibrated_top_k_chart,
    _validate_calibration_chart_display_options,
    _validate_chart_models,
    _validate_threshold_top_k_floor_chart,
    _validate_model_groups,
    _validate_paired_comparisons,
    _validate_paired_difference_chart,
)
from experiments.core.multisplit_charts import (
    MODEL_GROUP_CANONICAL_SUFFIX,
    THESIS_RC_PARAMS,
    _canonical_filtered_rows,
    _chart_k_values,
    _filter_chart_models,
    _model_group_chart_modes,
    _render_calibration_transfer,
    _render_similarity_distributions,
    _save,
    _similarity_distribution_entries,
    write_multisplit_charts,
)


class MultisplitChartModelGroupsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "models": [
                {"analysis_model_id": "baseline-our-pp"},
                {"analysis_model_id": "baseline-default-pp"},
                {"analysis_model_id": "unrelated"},
            ],
            "model_groups": {
                "baseline-preprocessing": {
                    "canonical": "baseline-default-pp",
                    "members": ["baseline-our-pp", "baseline-default-pp"],
                }
            },
        }

    def test_canonical_member_is_fixed_and_ungrouped_models_remain(self):
        rows = [
            {
                "configuration_id": "baseline-our-pp",
                "pair_completeness_mean": 0.99,
            },
            {
                "configuration_id": "baseline-default-pp",
                "pair_completeness_mean": 0.50,
            },
            {"configuration_id": "unrelated", "pair_completeness_mean": 0.85},
        ]

        selected = _canonical_filtered_rows(rows, self.config)

        self.assertEqual(
            {row["configuration_id"] for row in selected},
            {"baseline-default-pp", "unrelated"},
        )

    def test_canonical_chart_can_explicitly_retain_noncanonical_models(self):
        rows = [
            {"configuration_id": "baseline-our-pp"},
            {"configuration_id": "baseline-default-pp"},
            {"configuration_id": "unrelated"},
        ]

        selected = _canonical_filtered_rows(
            rows,
            self.config,
            additional_models=["baseline-our-pp"],
        )

        self.assertEqual(
            {row["configuration_id"] for row in selected},
            {"baseline-our-pp", "baseline-default-pp", "unrelated"},
        )

    def test_grouped_charts_get_all_model_and_canonical_modes(self):
        self.assertEqual(
            _model_group_chart_modes(self.config),
            (("", False), (MODEL_GROUP_CANONICAL_SUFFIX, True)),
        )
        self.assertEqual(_model_group_chart_modes({}), (("", False),))

    def test_similarity_chart_defaults_to_all_models_and_can_select_canonical(self):
        artifact = {
            "similarity_distributions": [
                {"configuration_id": "baseline-our-pp", "task_id": "task"},
                {"configuration_id": "baseline-default-pp", "task_id": "task"},
                {"configuration_id": "unrelated", "task_id": "task"},
            ],
            "summary_rows": [
                {
                    "configuration_id": "baseline-our-pp",
                    "task_id": "task",
                    "selection_mode": "top_k",
                    "pair_completeness_mean": 0.9,
                },
                {
                    "configuration_id": "baseline-default-pp",
                    "task_id": "task",
                    "selection_mode": "top_k",
                    "pair_completeness_mean": 0.8,
                },
                {
                    "configuration_id": "unrelated",
                    "task_id": "task",
                    "selection_mode": "top_k",
                    "pair_completeness_mean": 0.85,
                },
            ],
        }
        chart = {"task": "task"}

        all_entries = _similarity_distribution_entries(artifact, self.config, chart)
        canonical_entries = _similarity_distribution_entries(
            artifact,
            self.config,
            chart,
            select_canonical_models=True,
        )

        self.assertEqual(len(all_entries), 3)
        self.assertEqual(
            {entry["configuration_id"] for entry in canonical_entries},
            {"baseline-default-pp", "unrelated"},
        )

    def test_chart_writer_appends_suffix_to_canonical_version(self):
        config = {
            **self.config,
            "models": [
                {**entry, "include": True, "presentation": {"label": entry["analysis_model_id"]}}
                for entry in self.config["models"]
            ],
            "charts": {
                "multisplit_pc_at_k_band": {"enabled": True, "formats": ["png"]}
            },
        }
        artifact = {
            "dataset_id": "dataset",
            "expected_split_seeds": [1],
            "summary_rows": [
                {
                    "configuration_id": model_id,
                    "task_id": "task",
                    "selection_mode": "top_k",
                    "k": 10,
                    "pair_completeness_mean": mean,
                    "pair_completeness_min": mean,
                    "pair_completeness_max": mean,
                }
                for model_id, mean in (
                    ("baseline-our-pp", 0.9),
                    ("baseline-default-pp", 0.8),
                    ("unrelated", 0.85),
                )
            ],
        }

        import matplotlib

        original_font_family = list(matplotlib.rcParams["font.family"])
        with tempfile.TemporaryDirectory() as directory:
            paths, warnings = write_multisplit_charts(artifact, config, Path(directory))
            self.assertEqual(list(matplotlib.rcParams["font.family"]), original_font_family)
            names = {path.name for path in paths}
            self.assertEqual(warnings, [])
            self.assertIn("multisplit_pc_at_k_band.png", names)
            self.assertIn("multisplit_pc_at_k_band.json", names)
            self.assertIn(
                f"multisplit_pc_at_k_band{MODEL_GROUP_CANONICAL_SUFFIX}.png",
                names,
            )
            self.assertIn(
                f"multisplit_pc_at_k_band{MODEL_GROUP_CANONICAL_SUFFIX}.json",
                names,
            )
            sidecar = json.loads(
                (Path(directory) / "multisplit_pc_at_k_band.json").read_text(
                    encoding="utf-8"
                )
            )
            by_model = {row["model_id"]: row for row in sidecar["series"]}
            self.assertEqual(by_model["baseline-our-pp"]["candidate_budget"], [10])
            self.assertEqual(by_model["baseline-our-pp"]["mean"], [0.9])
            canonical_sidecar = json.loads(
                (
                    Path(directory)
                    / f"multisplit_pc_at_k_band{MODEL_GROUP_CANONICAL_SUFFIX}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                {row["model_id"] for row in canonical_sidecar["series"]},
                {"baseline-default-pp", "unrelated"},
            )
            from PIL import Image

            with Image.open(Path(directory) / "multisplit_pc_at_k_band.png") as image:
                self.assertEqual(image.width, 575)


class MultisplitChartExtendedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "models": [
                {"analysis_model_id": "baseline-our-pp"},
                {"analysis_model_id": "baseline-default-pp"},
                {"analysis_model_id": "unrelated"},
            ],
            "model_groups": {
                "baseline-preprocessing": {
                    "canonical": "baseline-default-pp",
                    "members": ["baseline-our-pp", "baseline-default-pp"],
                }
            },
        }

    def _config(self) -> dict:
        return {
            "models": [
                {
                    "analysis_model_id": "trained",
                    "include": True,
                    "presentation": {"label": "Trained"},
                },
                {
                    "analysis_model_id": "frozen",
                    "include": True,
                    "presentation": {"label": "Frozen"},
                },
            ],
            "multisplit": {
                "aggregation": {"tie_tolerance": 0.001},
                "paired_comparisons": [
                    {
                        "comparison_id": "trained-minus-frozen",
                        "minuend": "trained",
                        "subtrahend": "frozen",
                    }
                ],
            },
            "charts": {
                "multisplit_paired_difference": {
                    "enabled": True,
                    "task": "test",
                    "selection_mode": "calibrated_threshold",
                    "target_pc": 0.99,
                    "metrics": ["pair_completeness", "reduction_ratio"],
                    "formats": ["png"],
                }
            },
        }

    def _artifact(self) -> dict:
        split_values = [
            {"split_seed": 6, "difference": 0.01},
            {"split_seed": 7, "difference": 0.02},
        ]
        return {
            "dataset_id": "wikidata-test",
            "expected_split_seeds": [6, 7],
            "summary_rows": [],
            "paired_comparisons": [
                {
                    "comparison_id": "trained-minus-frozen",
                    "minuend": "trained",
                    "subtrahend": "frozen",
                    "task_id": "test",
                    "selection_mode": "calibrated_threshold",
                    "calibration_target_pair_completeness": 0.99,
                    "num_split_realizations": 2,
                    "tie_tolerance": 0.001,
                    "pair_completeness_difference_mean": 0.015,
                    "pair_completeness_difference_sample_sd_split": 0.007071,
                    "pair_completeness_difference_min": 0.01,
                    "pair_completeness_difference_max": 0.02,
                    "pair_completeness_difference_split_values": split_values,
                    "pair_completeness_wins": 2,
                    "pair_completeness_ties": 0,
                    "reduction_ratio_difference_mean": 0.03,
                    "reduction_ratio_difference_sample_sd_split": 0.014142,
                    "reduction_ratio_difference_min": 0.02,
                    "reduction_ratio_difference_max": 0.04,
                    "reduction_ratio_difference_split_values": [
                        {"split_seed": 6, "difference": 0.02},
                        {"split_seed": 7, "difference": 0.04},
                    ],
                    "reduction_ratio_wins": 2,
                    "reduction_ratio_ties": 0,
                }
            ],
        }

    def test_paired_comparison_validation_is_early_and_explicit(self):
        config = self._config()
        _validate_paired_comparisons(config)
        _validate_paired_difference_chart(config)

        config["multisplit"]["paired_comparisons"].append(
            {
                "comparison_id": "trained-minus-frozen",
                "minuend": "trained",
                "subtrahend": "frozen",
            }
        )
        with self.assertRaisesRegex(
            MultisplitAnalysisError, "Duplicate paired comparison_id"
        ):
            _validate_paired_comparisons(config)

    def test_fixed_k_delta_rr_is_rejected(self):
        config = self._config()
        config["charts"]["multisplit_paired_difference"] = {
            "enabled": True,
            "task": "test",
            "selection_mode": "top_k",
            "k": 40,
            "metrics": ["reduction_ratio"],
        }
        with self.assertRaisesRegex(
            MultisplitAnalysisError, "model-independent"
        ):
            _validate_paired_difference_chart(config)

    def test_calibrated_pc_and_rr_charts_include_auditable_sidecars(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            paths, warnings = write_multisplit_charts(
                self._artifact(), config, Path(directory)
            )
            self.assertEqual(warnings, [])
            names = {path.name for path in paths}
            self.assertIn(
                "multisplit_paired_pc_threshold_rho0p99.png", names
            )
            self.assertIn(
                "multisplit_paired_rr_threshold_rho0p99.png", names
            )
            sidecar = json.loads(
                (
                    Path(directory)
                    / "multisplit_paired_rr_threshold_rho0p99.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["metric"], "reduction_ratio")
            self.assertEqual(sidecar["selection_mode"], "calibrated_threshold")
            self.assertEqual(
                sidecar["aggregation_order"],
                "mean_over_training_seeds_within_split_then_"
                "paired_difference_then_equal_split_summary",
            )
            self.assertEqual(
                [point["split_seed"] for point in sidecar["comparisons"][0]["points"]],
                [6, 7],
            )

    def test_calibrated_top_k_writer_emits_one_chart_per_target(self):
        config = {
            "models": [
                {
                    "analysis_model_id": "model",
                    "include": True,
                    "presentation": {
                        "label": "Model",
                        "color": "#4477AA",
                        "group": "frozen",
                    },
                }
            ],
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
                    "task": "test",
                    "target_pc": [0.95, 0.99],
                    "style_from": "calibration_style",
                    "formats": ["png"],
                }
            },
            "multisplit": {"aggregation": {"attainment_tolerance": 0.005}},
        }
        _validate_calibrated_top_k_chart(config)
        artifact = {
            "summary_rows": [
                {
                    "configuration_id": "model",
                    "task_id": "test",
                    "selection_mode": "calibrated_top_k",
                    "calibration_target_pair_completeness": target,
                    "selected_k_mean": 12.0,
                    "selected_k_sample_sd_split": 2.0,
                    "selected_k_min": 10.0,
                    "selected_k_max": 14.0,
                    "selection_attained_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": 1.0},
                        {"split_seed": 2, "mean_over_training_seeds": 0.0},
                    ],
                    "selected_k_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": 10.0},
                        {"split_seed": 2, "mean_over_training_seeds": 14.0},
                    ],
                    "pair_completeness_mean": target,
                    "pair_quality_mean": 0.25,
                    "reduction_ratio_mean": 0.8,
                    "target_attainment_count": 2,
                    "validation_selection_attainment_count": 1,
                    "num_split_realizations": 2,
                    "pair_completeness_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": target - 0.001},
                        {"split_seed": 2, "mean_over_training_seeds": target + 0.001},
                    ],
                    "pair_quality_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": 0.24},
                        {"split_seed": 2, "mean_over_training_seeds": 0.26},
                    ],
                    "reduction_ratio_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": 0.79},
                        {"split_seed": 2, "mean_over_training_seeds": 0.81},
                    ],
                }
                for target in (0.95, 0.99)
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            paths, warnings = write_multisplit_charts(
                artifact, config, Path(directory)
            )
            self.assertEqual(warnings, [])
            names = {path.name for path in paths}
            for target_tag in ("0p95", "0p99"):
                stem = f"multisplit_calibrated_top_k_transfer_rho{target_tag}"
                self.assertIn(f"{stem}.png", names)
                self.assertIn(f"{stem}.json", names)
            sidecar = json.loads(
                (
                    Path(directory)
                    / "multisplit_calibrated_top_k_transfer_rho0p95.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["selection_mode"], "calibrated_top_k")
            self.assertFalse(sidecar["include_attainment_score"])
            self.assertFalse(sidecar["include_specific_result_crosses"])
            self.assertEqual(sidecar["pc_axis_text"], "")
            self.assertEqual(sidecar["rr_axis_text"], "")
            self.assertEqual(sidecar["series"][0]["selected_k_mean"], 12.0)
            self.assertEqual(sidecar["series"][0]["fallback_split_count"], 1)
            self.assertEqual(sidecar["series"][0]["points"][0]["selected_k"], 10)
            self.assertFalse(
                sidecar["series"][0]["points"][0][
                    "used_least_restrictive_fallback"
                ]
            )
            self.assertTrue(
                sidecar["series"][0]["points"][1][
                    "used_least_restrictive_fallback"
                ]
            )
            self.assertEqual(sidecar["series"][0]["points"][0]["pair_quality"], 0.24)

    def test_threshold_top_k_floor_writer_emits_one_chart_per_combination(self):
        config = {
            "models": [
                {
                    "analysis_model_id": "model",
                    "include": True,
                    "presentation": {
                        "label": "Model",
                        "color": "#4477AA",
                        "group": "frozen",
                    },
                }
            ],
            "charts": {
                "multisplit_threshold_top_k_floor_transfer": {
                    "enabled": True,
                    "task": "test",
                    "target_pc": [0.95, 0.99],
                    "minimum_top_k": [5, 40],
                    "show_pc_mean_values": True,
                    "pc_mean_value_decimals": 3,
                    "formats": ["png"],
                }
            },
            "multisplit": {"aggregation": {"attainment_tolerance": 0.005}},
        }
        artifact = {
            "summary_rows": [
                {
                    "configuration_id": "model",
                    "task_id": "test",
                    "selection_mode": "calibrated_threshold_top_k_floor",
                    "calibration_target_pair_completeness": target,
                    "minimum_top_k": floor,
                    "pair_completeness_mean": target,
                    "pair_quality_mean": 0.25,
                    "reduction_ratio_mean": 0.8,
                    "target_attainment_count": 2,
                    "num_split_realizations": 2,
                    "pair_completeness_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": target - 0.001},
                        {"split_seed": 2, "mean_over_training_seeds": target + 0.001},
                    ],
                    "pair_quality_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": 0.24},
                        {"split_seed": 2, "mean_over_training_seeds": 0.26},
                    ],
                    "reduction_ratio_split_values": [
                        {"split_seed": 1, "mean_over_training_seeds": 0.79},
                        {"split_seed": 2, "mean_over_training_seeds": 0.81},
                    ],
                }
                for target in (0.95, 0.99)
                for floor in (5, 40)
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            paths, warnings = write_multisplit_charts(
                artifact, config, Path(directory)
            )
            self.assertEqual(warnings, [])
            names = {path.name for path in paths}
            for target_tag in ("0p95", "0p99"):
                for floor in (5, 40):
                    stem = (
                        "multisplit_threshold_top_k_floor_transfer_"
                        f"rho{target_tag}_k{floor}"
                    )
                    self.assertIn(f"{stem}.png", names)
                    self.assertIn(f"{stem}.json", names)
            self.assertEqual(
                len([path for path in paths if path.suffix in {".png", ".json"}]),
                8,
            )
            self.assertIn("multisplit_figure_captions.md", names)
            sidecar = json.loads(
                (
                    Path(directory)
                    / "multisplit_threshold_top_k_floor_transfer_rho0p95_k5.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["candidate_rule"], "threshold_or_top_k")
            self.assertEqual(sidecar["minimum_top_k"], 5)
            self.assertTrue(sidecar["show_pc_mean_values"])
            self.assertEqual(sidecar["pc_mean_value_decimals"], 3)
            self.assertEqual(sidecar["series"][0]["pair_quality_mean"], 0.25)
            self.assertEqual(
                sidecar["series"][0]["points"][0]["pair_quality"], 0.24
            )

    def test_floor_chart_config_and_list_compatibility_validation(self):
        config = {
            "charts": {
                "multisplit_threshold_top_k_floor_transfer": {
                    "enabled": True,
                    "task": "test",
                    "target_pc": [0.95, 0.99],
                    "minimum_top_k": [5, 10, 20, 40],
                }
            }
        }
        _validate_threshold_top_k_floor_chart(config)
        self.assertEqual(
            _compatibility_values(["old", "new"], "compatibility.test"),
            ("old", "new"),
        )
        required = _required_source_capabilities(config)
        old = {
            "run_dir": "old",
            "created_at": "2099-01-01",
            "capabilities": {"fixed_k": True, "calibrated_thresholds": True},
        }
        hybrid = {
            "run_dir": "hybrid",
            "created_at": "2020-01-01",
            "capabilities": {
                "fixed_k": True,
                "calibrated_thresholds": True,
                "calibrated_threshold_top_k_floor": True,
            },
        }
        self.assertGreater(
            _candidate_score(hybrid, "richest_then_newest", required),
            _candidate_score(old, "richest_then_newest", required),
        )
        config["charts"]["multisplit_threshold_top_k_floor_transfer"][
            "minimum_top_k"
        ] = [5, 5]
        with self.assertRaisesRegex(MultisplitAnalysisError, "duplicates"):
            _validate_threshold_top_k_floor_chart(config)

    def test_similarity_distributions_use_five_columns_and_one_line_legend(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        entries = [
            {
                "configuration_id": f"model-{index}",
                "bin_edges": [0.0, 0.5, 1.0],
                "match_density_mean": [0.5, 1.5],
                "non_match_density_mean": [1.5, 0.5],
                "match_density_min": [0.4, 1.4],
                "match_density_max": [0.6, 1.6],
                "non_match_density_min": [1.4, 0.4],
                "non_match_density_max": [1.6, 0.6],
            }
            for index in range(7)
        ]
        styles = {
            f"model-{index}": {"label": f"Model {index}"}
            for index in range(7)
        }
        thresholds = {
            f"model-{index}": {"min": 0.4, "mean": 0.5, "max": 0.6}
            for index in range(7)
        }

        with matplotlib.rc_context(THESIS_RC_PARAMS):
            fig = _render_similarity_distributions(
                plt,
                entries,
                styles,
                thresholds,
                {
                    "show_split_range": True,
                    "model_label_font_scale": 0.7,
                },
            )
        try:
            # Seven panels occupy a 2x5 grid, with the final three axes hidden.
            self.assertEqual(len(fig.axes), 10)
            self.assertEqual(sum(axis.get_visible() for axis in fig.axes), 7)
            self.assertEqual(fig.get_size_inches().tolist(), [5.75, 2.37])
            self.assertTrue(
                all(
                    axis.get_title() == f"Model {index}"
                    for index, axis in enumerate(fig.axes[:7])
                )
            )

            self.assertTrue(
                all(axis.title.get_fontsize() == 7.0 for axis in fig.axes[:7])
            )

            self.assertEqual(len(fig.legends), 1)
            legend = fig.legends[0]
            self.assertEqual(legend._ncols, 3)
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                ["non-match mean", "match mean", "mean calibrated threshold"],
            )
            self.assertTrue(
                all(text.get_fontsize() == 10 for text in legend.get_texts())
            )
            self.assertTrue(all(axis.get_legend() is None for axis in fig.axes))
            self.assertTrue(all(not axis.get_xlabel() for axis in fig.axes))
            self.assertTrue(all(not axis.get_ylabel() for axis in fig.axes))
        finally:
            plt.close(fig)

    def test_calibration_transfer_renders_vertical_pc_and_rr_axes(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [
            {
                "configuration_id": "first",
                "pair_completeness_mean": 0.985,
                "pair_completeness_split_values": [
                    {"split_seed": 1, "mean_over_training_seeds": 0.99},
                    {"split_seed": 2, "mean_over_training_seeds": 0.98},
                ],
                "reduction_ratio_mean": 0.25,
                "reduction_ratio_split_values": [
                    {"split_seed": 1, "mean_over_training_seeds": 0.20},
                    {"split_seed": 2, "mean_over_training_seeds": 0.30},
                ],
                "target_attainment_count": 1,
                "num_split_realizations": 2,
            },
            {
                "configuration_id": "second",
                "pair_completeness_mean": 0.995,
                "pair_completeness_split_values": [
                    {"split_seed": 1, "mean_over_training_seeds": 1.0},
                    {"split_seed": 2, "mean_over_training_seeds": 0.99},
                ],
                "reduction_ratio_mean": 0.60,
                "reduction_ratio_split_values": [
                    {"split_seed": 1, "mean_over_training_seeds": 0.55},
                    {"split_seed": 2, "mean_over_training_seeds": 0.65},
                ],
                "target_attainment_count": 2,
                "num_split_realizations": 2,
            },
        ]
        styles = {
            "first": {
                "label": "First",
                "color": "#0072B2",
                "marker": "o",
                "group": "finetuned",
            },
            "second": {
                "label": "Second",
                "color": "#E69F00",
                "marker": "s",
                "group": "frozen",
            },
        }

        with matplotlib.rc_context(THESIS_RC_PARAMS):
            fig = _render_calibration_transfer(
                plt,
                rows,
                styles,
                {
                    "jitter_seed": 0,
                    "dataset_label": "WIK",
                    "figure_height": 1.6632,
                    "show_explanatory_heading": False,
                    "show_axis_label_details": False,
                    "show_attainment_axis_label": True,
                    "show_attainment_region": True,
                    "attainment_region_color": "#DCEFD8",
                    "attainment_region_alpha": 0.55,
                    "model_label_layout": "alternating_guides",
                    "model_label_fontsize": 9.3,
                    "model_label_font_scale": 0.7,
                    "model_label_first_row": "bottom",
                    "model_label_row_spacing_increment_points": 0.5,
                    "pc_marker": "x",
                    "pc_marker_size": 28,
                    "pc_marker_linewidth": 1.3,
                    "pc_marker_halo_linewidth": 3.0,
                    "pc_marker_halo_color": "white",
                    "model_colors": {"first": "#112233", "second": "#445566"},
                    "rr_bar_hatches": {
                        "groups": {"finetuned": "///"},
                        "models": {"second": "xxx"},
                    },
                },
                0.99,
                0.005,
            )
        try:
            self.assertEqual(len(fig.axes), 2)
            pc_axis, rr_axis = fig.axes
            self.assertEqual(
                pc_axis.get_ylabel(),
                "$PC^{test}_{\\mathit{WIK}}$ at $\\rho_{cal}=0.99$",
            )
            self.assertEqual(
                rr_axis.get_ylabel(),
                "$RR^{test}_{\\mathit{WIK}}$",
            )
            self.assertEqual(pc_axis.get_title(), "")
            self.assertEqual(fig.get_figheight(), 1.6632)
            self.assertEqual(pc_axis.get_xticks().tolist(), [])
            self.assertIsNone(pc_axis.get_legend())
            self.assertEqual(len(pc_axis.collections), 6)
            expected_x_marker = [
                [-0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
                [0.5, -0.5],
            ]
            for collection_index in (0, 1, 3, 4):
                point_collection = pc_axis.collections[collection_index]
                self.assertEqual(
                    point_collection.get_paths()[0].vertices.tolist(),
                    expected_x_marker,
                )
                self.assertEqual(point_collection.get_sizes().tolist(), [28])
                self.assertEqual(point_collection.get_linewidths().tolist(), [1.3])
                self.assertEqual(len(point_collection.get_path_effects()), 2)
            self.assertEqual(
                [text.get_text() for text in pc_axis.texts],
                ["First", "Second"],
            )
            self.assertEqual(len(pc_axis.child_axes), 1)
            attainment_axis = pc_axis.child_axes[0]
            self.assertEqual(
                [text.get_text() for text in attainment_axis.get_xticklabels()],
                ["1/2", "2/2"],
            )
            self.assertEqual(
                attainment_axis.get_xlabel(),
                "Attainment score: $PC^{test}_{\\mathit{WIK}} \\geq 0.985$",
            )
            self.assertEqual(attainment_axis.xaxis.majorTicks[0].get_pad(), 0.5)
            self.assertEqual(attainment_axis.xaxis.labelpad, 4.5)
            reference_lines = [
                line for line in pc_axis.lines if line.get_clip_on()
            ]
            self.assertEqual(len(reference_lines), 1)
            self.assertEqual(list(reference_lines[0].get_ydata()), [0.99, 0.99])
            self.assertEqual(reference_lines[0].get_linestyle(), ":")
            guide_lines = [
                line for line in pc_axis.lines if not line.get_clip_on()
            ]
            self.assertEqual(len(guide_lines), 2)
            self.assertEqual(guide_lines[0].get_xdata().tolist(), [0, 0])
            self.assertEqual(guide_lines[1].get_xdata().tolist(), [1, 1])
            self.assertEqual(guide_lines[0].get_ydata().tolist(), [0.0, -0.13])
            self.assertEqual(guide_lines[1].get_ydata().tolist(), [0.0, -0.015])
            self.assertEqual(pc_axis.texts[0].get_position(), (0, -0.145))
            self.assertEqual(pc_axis.texts[1].get_position(), (1, -0.03))
            unshifted_second_row_y = pc_axis.get_xaxis_transform().transform(
                (0, -0.145)
            )[1]
            shifted_second_row_y = pc_axis.texts[0].xycoords.transform(
                (0, -0.145)
            )[1]
            self.assertAlmostEqual(
                shifted_second_row_y - unshifted_second_row_y,
                -0.5 * fig.dpi / 72.0,
            )
            base_bars = rr_axis.patches[:2]
            hatch_overlays = rr_axis.patches[2:4]
            attainment_regions = [
                patch
                for patch in rr_axis.patches
                if patch.get_gid() == "attainment-threshold-region"
            ]
            self.assertEqual(len(attainment_regions), 1)
            self.assertEqual(attainment_regions[0].get_y(), 0.985)
            self.assertAlmostEqual(
                attainment_regions[0].get_height(),
                pc_axis.get_ylim()[1] - 0.985,
            )
            self.assertEqual(attainment_regions[0].get_alpha(), 0.55)
            self.assertLess(attainment_regions[0].get_zorder(), base_bars[0].get_zorder())
            self.assertEqual(
                [patch.get_height() for patch in base_bars],
                [0.25, 0.60],
            )
            self.assertEqual(
                [patch.get_hatch() for patch in hatch_overlays],
                ["///", "xxx"],
            )
            self.assertTrue(
                all(patch.get_facecolor()[3] == 0.3 for patch in base_bars)
            )
            self.assertTrue(
                all(patch.get_facecolor()[3] == 0.0 for patch in hatch_overlays)
            )
            self.assertTrue(
                all(patch.get_edgecolor()[3] == 1.0 for patch in hatch_overlays)
            )
            self.assertTrue(
                all(patch.get_linewidth() == 0.45 for patch in hatch_overlays)
            )
            self.assertTrue(
                all(text.get_fontsize() == 6.51 for text in pc_axis.texts)
            )
        finally:
            plt.close(fig)

    def test_calibration_transfer_can_hide_attainment_crosses_and_axis_text(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [
            {
                "configuration_id": "model",
                "pair_completeness_mean": 0.985,
                "pair_completeness_split_values": [
                    {"split_seed": 1, "mean_over_training_seeds": 0.98},
                    {"split_seed": 2, "mean_over_training_seeds": 0.99},
                ],
                "reduction_ratio_mean": 0.6,
                "target_attainment_count": 1,
                "num_split_realizations": 2,
            }
        ]
        styles = {
            "model": {
                "label": "Model",
                "color": "#4477AA",
                "marker": "x",
                "group": "frozen",
            }
        }
        chart = {
            "include_attainment_score": False,
            "include_specific_result_crosses": False,
            "pc_axis_label": "",
            "pc_axis_description": "",
            "rr_axis_label": "",
            "rr_axis_description": "",
            "show_attainment_axis_label": True,
            "show_attainment_region": True,
            "show_explanatory_heading": False,
            "pc_ylim": [0.95, 1.0],
            "show_pc_mean_values": True,
            "pc_mean_value_decimals": 3,
            "pc_mean_value_offset_points": 2.5,
        }

        with matplotlib.rc_context(THESIS_RC_PARAMS):
            fig = _render_calibration_transfer(
                plt, rows, styles, chart, 0.99, 0.005
            )
        try:
            pc_axis, rr_axis = fig.axes
            self.assertEqual(pc_axis.get_ylabel(), "")
            self.assertEqual(rr_axis.get_ylabel(), "")
            self.assertEqual(pc_axis.child_axes, [])
            self.assertEqual(len(pc_axis.lines), 0)
            self.assertEqual(len(pc_axis.collections), 1)
            self.assertEqual(pc_axis.get_ylim(), (0.95, 1.0))
            self.assertEqual(
                [text.get_text() for text in pc_axis.texts], ["0.985"]
            )
            self.assertEqual(pc_axis.texts[0].get_position(), (0.0, -2.5))
            self.assertEqual(pc_axis.texts[0].xy, (0, 0.985))
            mean_segments = pc_axis.collections[0].get_segments()
            self.assertEqual(len(mean_segments), 1)
            self.assertEqual(mean_segments[0][:, 1].tolist(), [0.985, 0.985])
            self.assertEqual(len(rr_axis.patches), 1)
            self.assertFalse(
                any(
                    patch.get_gid() == "attainment-threshold-region"
                    for patch in rr_axis.patches
                )
            )
        finally:
            plt.close(fig)

    def test_calibration_transfer_options_are_independent(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [
            {
                "configuration_id": "model",
                "pair_completeness_mean": 0.985,
                "pair_completeness_split_values": [
                    {"split_seed": 1, "mean_over_training_seeds": 0.98},
                    {"split_seed": 2, "mean_over_training_seeds": 0.99},
                ],
                "reduction_ratio_mean": 0.6,
                "target_attainment_count": 1,
                "num_split_realizations": 2,
            }
        ]
        styles = {
            "model": {
                "label": "Model",
                "color": "#4477AA",
                "marker": "x",
                "group": "frozen",
            }
        }
        chart = {
            "include_attainment_score": False,
            "include_specific_result_crosses": True,
            "pc_axis_label": "PC",
            "pc_axis_description": "held-out completeness",
            "rr_axis_label": "RR",
            "rr_axis_description": "mean reduction ratio",
            "show_explanatory_heading": False,
        }

        with matplotlib.rc_context(THESIS_RC_PARAMS):
            fig = _render_calibration_transfer(
                plt, rows, styles, chart, 0.99, 0.005
            )
        try:
            pc_axis, rr_axis = fig.axes
            self.assertEqual(
                pc_axis.get_ylabel(), "PC\nheld-out completeness"
            )
            self.assertEqual(
                rr_axis.get_ylabel(), "RR mean reduction ratio"
            )
            self.assertEqual(pc_axis.child_axes, [])
            self.assertEqual(len(pc_axis.lines), 0)
            # One split-marker collection plus the retained mean-PC segment.
            self.assertEqual(len(pc_axis.collections), 2)
            self.assertEqual(
                len(pc_axis.collections[0].get_offsets()), 2
            )
        finally:
            plt.close(fig)

    def test_calibration_display_options_validate_inherited_types(self):
        config = {
            "charts": {
                "base": {
                    "include_attainment_score": False,
                    "include_specific_result_crosses": True,
                    "pc_axis_label": "PC",
                    "pc_axis_description": None,
                    "rr_axis_label": "",
                    "rr_axis_description": "",
                },
                "multisplit_calibrated_top_k_transfer": {
                    "enabled": True,
                    "style_from": "base",
                },
            }
        }
        _validate_calibration_chart_display_options(config)
        config["charts"]["base"]["include_attainment_score"] = "false"
        with self.assertRaisesRegex(MultisplitAnalysisError, "must be a boolean"):
            _validate_calibration_chart_display_options(config)
        config["charts"]["base"]["include_attainment_score"] = False
        config["charts"]["base"]["show_pc_mean_values"] = "true"
        with self.assertRaisesRegex(MultisplitAnalysisError, "must be a boolean"):
            _validate_calibration_chart_display_options(config)

    def test_save_can_trim_all_outer_padding(self):
        class RecordingFigure:
            def __init__(self):
                self.options = None

            def savefig(self, _path, **options):
                self.options = options

        class RecordingPyplot:
            @staticmethod
            def close(_fig):
                pass

        figure = RecordingFigure()
        with tempfile.TemporaryDirectory() as directory:
            _save(
                figure,
                RecordingPyplot(),
                Path(directory),
                "trimmed",
                {"formats": ["pdf"], "trim_outer_padding": True},
                {},
            )

        self.assertEqual(
            figure.options,
            {"format": "pdf", "bbox_inches": "tight", "pad_inches": 0},
        )

    def test_strip_k_values_can_select_every_evaluated_budget(self):
        rows = [{"k": 40}, {"k": 1}, {"k": 20}, {"k": 40}, {"k": 80}]

        self.assertEqual(_chart_k_values({"k": "all"}, rows), [1, 20, 40, 80])
        self.assertEqual(_chart_k_values({"k": [20, 40, 20, 80]}, rows), [20, 40, 80])
        self.assertEqual(_chart_k_values({"k": 40}, rows), [40])

    def test_chart_model_list_filters_rows_and_defaults_to_all_models(self):
        rows = [
            {"configuration_id": "baseline-our-pp"},
            {"configuration_id": "baseline-default-pp"},
            {"configuration_id": "unrelated"},
        ]

        self.assertEqual(_filter_chart_models(rows, {}), rows)
        self.assertEqual(
            _filter_chart_models(
                rows, {"models": ["unrelated", "baseline-our-pp"]}
            ),
            [rows[0], rows[2]],
        )

    def test_chart_model_lists_must_reference_globally_included_models(self):
        config = {
            "models": [
                {"analysis_model_id": "included", "include": True},
                {"analysis_model_id": "excluded", "include": False},
            ],
            "charts": {"figure": {"models": ["included"]}},
        }
        _validate_chart_models(config)

        config["charts"]["figure"]["models"] = ["excluded"]
        with self.assertRaisesRegex(MultisplitAnalysisError, "not globally included"):
            _validate_chart_models(config)

        config["charts"]["figure"] = {
            "canonical_additional_models": ["excluded"]
        }
        with self.assertRaisesRegex(
            MultisplitAnalysisError,
            "canonical_additional_models contains model ids that are not globally included",
        ):
            _validate_chart_models(config)

    def test_model_group_members_must_be_known_and_nonoverlapping(self):
        invalid = {
            "models": self.config["models"],
            "model_groups": {
                "first": {
                    "canonical": "baseline-our-pp",
                    "members": ["baseline-our-pp", "baseline-default-pp"],
                },
                "second": {
                    "canonical": "baseline-our-pp",
                    "members": ["baseline-our-pp", "missing"],
                },
            },
        }

        with self.assertRaisesRegex(MultisplitAnalysisError, "unknown model ids"):
            _validate_model_groups(invalid)

    def test_model_groups_require_a_predeclared_canonical_member(self):
        invalid = {
            **self.config,
            "model_groups": {
                "baseline-preprocessing": {
                    "members": ["baseline-our-pp", "baseline-default-pp"]
                }
            },
        }

        with self.assertRaisesRegex(MultisplitAnalysisError, "canonical must predeclare"):
            _validate_model_groups(invalid)

    def test_similarity_histograms_normalize_before_equal_split_averaging(self):
        records = [
            {
                "configuration_id": "model",
                "split_seed": 6,
                "bin_edges": [0.0, 0.5, 1.0],
                "match_hist": [3, 1],
                "neg_hist": [1, 3],
                "match_total": 4,
                "neg_total": 4,
            },
            {
                "configuration_id": "model",
                "split_seed": 7,
                "bin_edges": [0.0, 0.5, 1.0],
                # Ten times as many pairs: this split must still receive one vote.
                "match_hist": [10, 30],
                "neg_hist": [30, 10],
                "match_total": 40,
                "neg_total": 40,
            },
        ]

        summary = _summarize_similarity_histogram_records(records, [6, 7], "test")

        self.assertEqual(len(summary), 1)
        row = summary[0]
        np.testing.assert_allclose(row["match_density_mean"], [1.0, 1.0])
        np.testing.assert_allclose(row["non_match_density_mean"], [1.0, 1.0])
        np.testing.assert_allclose(row["match_density_min"], [0.5, 0.5])
        np.testing.assert_allclose(row["match_density_max"], [1.5, 1.5])
        widths = np.diff(row["bin_edges"])
        self.assertAlmostEqual(float(np.sum(np.asarray(row["match_density_mean"]) * widths)), 1.0)
        self.assertFalse(row["normalization"]["pair_counts_pooled"])

    def test_similarity_histograms_require_complete_split_coverage(self):
        records = [
            {
                "configuration_id": "model",
                "split_seed": 6,
                "bin_edges": [0.0, 1.0],
                "match_hist": [2],
                "neg_hist": [3],
                "match_total": 2,
                "neg_total": 3,
            }
        ]

        with self.assertRaisesRegex(MultisplitAnalysisError, "cover split seeds"):
            _summarize_similarity_histogram_records(records, [6, 7], "test")


if __name__ == "__main__":
    unittest.main()
