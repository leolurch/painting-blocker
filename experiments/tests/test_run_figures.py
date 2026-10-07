import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.core.aggregate_runs import aggregate_paths
from experiments.core.run_charts import write_run_charts
from experiments.core.run_figures import (
    FigureGenerationUnavailable,
    _import_matplotlib,
    _render_calibrated_threshold_bars,
    _render_validation_rr_bars,
    calibrated_threshold_bars_data,
    validation_rr_bars_data,
    fig1_data,
    fig2_data,
    fig3_data,
    fig4_data,
    fig5_data,
    zoomed_pc_at_k_data,
    load_model_artifacts,
    load_run_context,
    write_run_figures,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
TASK = "modern_to_historic"


def _write_run_dir(root: Path) -> Path:
    run_dir = root / "run1"
    (run_dir / "models").mkdir(parents=True)
    run_json = {
        "run_id": "run1",
        "experiment_id": "exp",
        "status": "completed",
        "dataset": {"dataset_id": "toy_dataset"},
        "split": {"split_id": "split"},
        "model_runs": [
            {"model_id": "frozen_a", "model_storage_key": "storage_a", "status": "completed"},
            {"model_id": "tuned_b", "model_storage_key": "storage_b", "status": "completed"},
            {
                "model_id": "random",
                "model_storage_key": "storage_random",
                "status": "completed",
                "kind": "analytical_baseline",
            },
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    (run_dir / "resolved_config.yml").write_text(
        "models:\n"
        "  include:\n"
        "    - model_id: frozen_a\n"
        "      display_name: Frozen A\n"
        "      color: '#111111'\n"
        "      group: frozen\n"
        "    - model_id: tuned_b\n"
        "      display_name: Tuned B\n"
        "      color: '#222222'\n"
        "      group: finetuned\n"
        "evaluation:\n"
        "  top_k: [1, 5]\n"
        "  threshold_calibration:\n"
        "    calibration_id: cal\n"
        "    target_pc: [0.95, 0.99]\n",
        encoding="utf-8",
    )
    _write_model(run_dir, "frozen_a", "storage_a", pc={1: 0.6, 5: 0.8}, with_calibration=True)
    _write_model(run_dir, "tuned_b", "storage_b", pc={1: 0.7, 5: 0.9}, with_calibration=True)
    _write_model(run_dir, "random", "storage_random", pc={1: 0.1, 5: 0.2}, with_calibration=False)
    aggregate_paths([run_dir], run_dir / "aggregate_metrics.csv", None)
    return run_dir


def _write_model(
    run_dir: Path,
    model_id: str,
    storage_key: str,
    pc: dict[int, float],
    with_calibration: bool,
) -> None:
    model_dir = run_dir / "models" / storage_key
    model_dir.mkdir(parents=True)
    metrics = [
        {
            "k": k,
            "pair_completeness": value,
            "pair_quality": value / 10,
            "reduction_ratio": 1.0 - k / 100.0,
            "query_coverage": 1.0,
        }
        for k, value in sorted(pc.items())
    ]
    calibrated = (
        [
            {
                "calibration_target_pair_completeness": rho,
                "threshold": tau,
                "pair_completeness": rho - 0.01,
                "pair_quality": 0.01,
                "calibration_reduction_ratio": 0.85,
                "reduction_ratio": 0.9,
                "query_coverage": 0.9,
            }
            for rho, tau in [(0.95, 0.55), (0.99, 0.45)]
        ]
        if with_calibration
        else []
    )
    eval_json = {
        "model_id": model_id,
        "model_storage_key": storage_key,
        "tasks": [
            {
                "task_id": TASK,
                "num_queries": 3,
                "num_candidates": 4,
                "num_positive_pairs": 3,
                "metrics": metrics,
                "target_pc_metrics": [],
                "calibrated_threshold_metrics": calibrated,
            }
        ],
    }
    (model_dir / "eval.json").write_text(json.dumps(eval_json), encoding="utf-8")
    task_curves: dict = {
        "task_id": TASK,
        "precision_recall": [
            {"pc": 1.0, "rr": 0.0},
            {"pc": 0.8, "rr": 0.95},
            {"pc": 0.5, "rr": 0.999},
        ],
        "histogram": {
            "bin_edges": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "match_hist": [0, 0, 1, 3] if with_calibration else [0, 0, 0, 0],
            "neg_hist": [4, 3, 2, 1] if with_calibration else [0, 0, 0, 0],
            "match_total": 4 if with_calibration else 0,
            "neg_total": 10 if with_calibration else 0,
        },
    }
    if with_calibration:
        task_curves["calibrated_candidate_sizes"] = [
            {
                "calibration_target_pair_completeness": 0.95,
                "threshold": 0.55,
                "candidates_per_query": [0, 1, 3],
                "query_coverage": 2 / 3,
            },
            {
                "calibration_target_pair_completeness": 0.99,
                "threshold": 0.45,
                "candidates_per_query": [1, 2, 4],
                "query_coverage": 1.0,
            },
        ]
        (model_dir / "threshold_calibration.json").write_text(
            json.dumps(
                {
                    "operating_points": [
                        {"target_pair_completeness": 0.95, "threshold": 0.55},
                        {"target_pair_completeness": 0.99, "threshold": 0.45},
                    ]
                }
            ),
            encoding="utf-8",
        )
    (model_dir / "curves.json").write_text(
        json.dumps({"model_id": model_id, "tasks": [task_curves]}), encoding="utf-8"
    )


class RunFigureDataTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = _write_run_dir(Path(self._tmp.name))
        self.context = load_run_context(self.run_dir)
        self.artifacts = {
            spec.storage_key: load_model_artifacts(spec) for spec in self.context.models
        }

    def test_context_models_styles_and_grids(self):
        specs = self.context.models
        self.assertEqual([spec.model_id for spec in specs], ["frozen_a", "tuned_b", "random"])
        self.assertEqual(specs[0].label, "Frozen A")
        self.assertEqual(specs[0].color, "#111111")
        self.assertEqual(specs[0].linestyle, "--")  # frozen -> dashed
        self.assertEqual(specs[1].linestyle, "-")
        self.assertTrue(specs[2].is_baseline)
        # No configured color for the baseline -> stable fallback palette entry.
        self.assertTrue(specs[2].color.startswith("#"))
        self.assertEqual(self.context.task_ids, [TASK])
        self.assertEqual(self.context.top_k, [1, 5])
        self.assertEqual(self.context.calibration_target_pc, [0.95, 0.99])

    def test_fig1_data_reads_pc_at_k_per_model(self):
        entries = fig1_data(self.context.models, self.artifacts, TASK)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["ks"], [1, 5])
        self.assertEqual(entries[0]["pc"], [0.6, 0.8])
        self.assertEqual(entries[2]["pc"], [0.1, 0.2])
        self.assertEqual(fig1_data(self.context.models, self.artifacts, "missing_task"), [])

    def test_zoomed_pc_at_k_data_omits_models_outside_visible_range(self):
        specs = self.context.models
        entries = [
            {
                "spec": specs[0],
                "ks": [20, 40, 80],
                "pc": [0.88, 0.89, 1.01],
            },
            {
                "spec": specs[1],
                "ks": [20, 40, 80],
                "pc": [0.70, 0.80, 0.85],
            },
            {
                "spec": specs[2],
                "ks": [20, 40, 80],
                "pc": [0.90, 0.95, 0.97],
            },
        ]

        visible = zoomed_pc_at_k_data(entries, 20, (0.9, 1.0))

        self.assertEqual(
            [entry["spec"].storage_key for entry in visible],
            ["storage_a", "storage_random"],
        )
        self.assertEqual([entry["ks"] for entry in visible], [[40, 80], [40, 80]])

    def test_fig2_data_reads_threshold_sweep(self):
        entries = fig2_data(self.context.models, self.artifacts, TASK)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["rr"], [0.0, 0.95, 0.999])
        self.assertEqual(entries[0]["pc"], [1.0, 0.8, 0.5])

    def test_fig3_k_star_selection(self):
        k_star, bars = fig3_data(self.context.models, self.artifacts, TASK)
        # 25 is not evaluated; the largest evaluated k <= 30 wins.
        self.assertEqual(k_star, 5)
        # Bars are ranked by the plotted result, highest to lowest from left to right.
        self.assertEqual([entry["pc"] for entry in bars], [0.9, 0.8, 0.2])
        self.assertEqual(
            [entry["spec"].model_id for entry in bars],
            ["tuned_b", "frozen_a", "random"],
        )

        def _with_ks(ks):
            spec = self.context.models[0]
            metrics = [{"k": k, "pair_completeness": 0.5} for k in ks]
            return {
                spec.storage_key: {
                    "eval": {"tasks": [{"task_id": TASK, "metrics": metrics}]},
                    "curves": {},
                    "calibration": {},
                }
            }

        self.assertEqual(fig3_data(self.context.models[:1], _with_ks([1, 25, 100]), TASK)[0], 25)
        self.assertEqual(fig3_data(self.context.models[:1], _with_ks([50, 100]), TASK)[0], 50)
        self.assertEqual(fig3_data(self.context.models[:1], _with_ks([]), TASK), (None, []))

    def test_calibrated_threshold_bars_read_test_pc_and_rr_and_sort_by_pc(self):
        tuned_rows = self.artifacts["storage_b"]["eval"]["tasks"][0][
            "calibrated_threshold_metrics"
        ]
        next(
            row for row in tuned_rows
            if row["calibration_target_pair_completeness"] == 0.95
        )["pair_completeness"] = 0.96

        by_rho = calibrated_threshold_bars_data(
            self.context.models, self.artifacts, TASK
        )
        self.assertEqual(sorted(by_rho), [0.95, 0.99])
        self.assertEqual(
            [entry["spec"].model_id for entry in by_rho[0.95]],
            ["tuned_b", "frozen_a"],
        )
        self.assertEqual([entry["pc"] for entry in by_rho[0.95]], [0.96, 0.94])
        self.assertTrue(all(entry["rr"] == 0.9 for entry in by_rho[0.95]))
        self.assertTrue(all(entry["threshold"] == 0.55 for entry in by_rho[0.95]))

    def test_validation_rr_bars_read_validation_rr_and_sort_by_rr(self):
        tuned_rows = self.artifacts["storage_b"]["eval"]["tasks"][0][
            "calibrated_threshold_metrics"
        ]
        next(
            row for row in tuned_rows
            if row["calibration_target_pair_completeness"] == 0.95
        )["calibration_reduction_ratio"] = 0.92
        frozen_rows = self.artifacts["storage_a"]["eval"]["tasks"][0][
            "calibrated_threshold_metrics"
        ]
        next(
            row for row in frozen_rows
            if row["calibration_target_pair_completeness"] == 0.95
        )["calibration_reduction_ratio"] = 0.88

        by_rho = validation_rr_bars_data(self.context.models, self.artifacts, TASK)
        self.assertEqual(sorted(by_rho), [0.95, 0.99])
        self.assertEqual(
            [entry["spec"].model_id for entry in by_rho[0.95]],
            ["tuned_b", "frozen_a"],
        )
        self.assertEqual([entry["rr"] for entry in by_rho[0.95]], [0.92, 0.88])

    def test_fig4_densities_and_calibrated_threshold(self):
        entries = fig4_data(self.context.models, self.artifacts, TASK)
        # The baseline has empty histograms and is excluded.
        self.assertEqual([entry["spec"].model_id for entry in entries], ["frozen_a", "tuned_b"])
        for entry in entries:
            widths = np.diff(entry["bin_edges"])
            self.assertAlmostEqual(float((entry["match_density"] * widths).sum()), 1.0)
            self.assertAlmostEqual(float((entry["neg_density"] * widths).sum()), 1.0)
            # Highest feasible calibration target (0.99) supplies the marker.
            self.assertAlmostEqual(entry["threshold"], 0.45)

    def test_fig5_sizes_from_eval_time_log(self):
        by_rho = fig5_data(self.context.models, self.artifacts, TASK)
        self.assertEqual(sorted(by_rho), [0.95, 0.99])
        dists = by_rho[0.95]
        # Baseline has no calibrated_candidate_sizes and is skipped.
        self.assertEqual([entry["spec"].model_id for entry in dists], ["frozen_a", "tuned_b"])
        # Zero counts are floored to 0.5 so log-scaled axes keep them visible.
        self.assertEqual(dists[0]["sizes"].tolist(), [0.5, 1.0, 3.0])
        self.assertAlmostEqual(dists[0]["query_coverage"], 2 / 3)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
class RunFigureRenderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = _write_run_dir(Path(self._tmp.name))

    def test_calibrated_threshold_bars_draw_calibration_target_line(self):
        context = load_run_context(self.run_dir)
        artifacts = {
            spec.storage_key: load_model_artifacts(spec) for spec in context.models
        }
        bars = calibrated_threshold_bars_data(context.models, artifacts, TASK)[0.95]
        _matplotlib, plt = _import_matplotlib()
        fig = _render_calibrated_threshold_bars(plt, 0.95, bars)
        try:
            target_lines = [
                line for line in fig.axes[0].lines
                if np.allclose(np.asarray(line.get_ydata(), dtype=float), [0.95, 0.95])
            ]
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(len(fig.axes[0].patches), len(bars))  # left PC bars
            self.assertEqual(len(fig.axes[1].patches), len(bars))  # right RR bars
            self.assertEqual(fig.axes[1].get_ylabel(), "Test Reduction Ratio (RR)")
            self.assertTrue(fig.axes[1].spines["right"].get_visible())
            self.assertEqual(len(target_lines), 1)
            self.assertIn("validation target", fig.axes[0].texts[0].get_text())
        finally:
            plt.close(fig)

    def test_validation_rr_bars_draw_one_bar_per_model(self):
        context = load_run_context(self.run_dir)
        artifacts = {
            spec.storage_key: load_model_artifacts(spec) for spec in context.models
        }
        bars = validation_rr_bars_data(context.models, artifacts, TASK)[0.95]
        _matplotlib, plt = _import_matplotlib()
        fig = _render_validation_rr_bars(plt, 0.95, bars)
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(len(fig.axes[0].patches), len(bars))
            self.assertEqual(len(fig.axes[0].texts), 0)
            self.assertIn(r"$\rho=0.95$", fig.axes[0].get_ylabel())
        finally:
            plt.close(fig)

    def test_validation_rr_bar_values_can_be_shown_vertically(self):
        context = load_run_context(self.run_dir)
        artifacts = {
            spec.storage_key: load_model_artifacts(spec) for spec in context.models
        }
        bars = validation_rr_bars_data(context.models, artifacts, TASK)[0.95]
        _matplotlib, plt = _import_matplotlib()
        fig = _render_validation_rr_bars(
            plt,
            0.95,
            bars,
            show_values=True,
            value_decimals=3,
        )
        try:
            labels = fig.axes[0].texts
            self.assertEqual(len(labels), len(bars))
            self.assertEqual(
                [label.get_text() for label in labels],
                [f"{entry['rr']:.3f}" for entry in bars],
            )
            self.assertTrue(all(label.get_rotation() == 90 for label in labels))
        finally:
            plt.close(fig)

    def test_write_run_figures_writes_all_figures(self):
        written = write_run_figures(self.run_dir)
        expected = {
            f"pc_at_k_{TASK}.pdf",
            f"pc_rr_tradeoff_{TASK}.pdf",
            f"pc5_bars_{TASK}.pdf",
            f"similarity_distributions_{TASK}.pdf",
            f"candidate_sizes_rho0p95_{TASK}.pdf",
            f"candidate_sizes_rho0p99_{TASK}.pdf",
        }
        self.assertEqual(set(written), expected)
        for path in written.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, self.run_dir / "figures")

    def test_write_run_figures_supports_other_formats(self):
        written = write_run_figures(self.run_dir, formats=("svg",))
        self.assertTrue(all(key.endswith(".svg") for key in written))

    def test_unknown_task_id_raises(self):
        with self.assertRaises(ValueError):
            write_run_figures(self.run_dir, task_id="nope")


class RunChartsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = _write_run_dir(Path(self._tmp.name))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_write_run_charts_renders_full_bundle(self):
        result = write_run_charts(self.run_dir)
        self.assertEqual(result["warnings"], [])
        self.assertFalse((self.run_dir / "charts_warning.json").is_file())
        self.assertTrue((self.run_dir / "latex_tables" / "blocking_fixed_k.tex").is_file())
        self.assertTrue((self.run_dir / "latex_tables" / "blocking_calibrated.tex").is_file())
        self.assertTrue((self.run_dir / "figures" / f"pc_at_k_{TASK}.pdf").is_file())
        self.assertTrue((self.run_dir / "figures" / f"pq_pc_{TASK}.pdf").is_file())
        self.assertTrue((self.run_dir / "bar_charts" / "ranked_pc_summary.csv").is_file())
        self.assertTrue((self.run_dir / "bar_charts" / f"ranked_pc_at_5_{TASK}.svg").is_file())

    def test_missing_matplotlib_is_recorded_without_failing(self):
        with mock.patch(
            "experiments.core.run_figures._import_matplotlib",
            side_effect=FigureGenerationUnavailable("matplotlib is not installed"),
        ):
            result = write_run_charts(self.run_dir)
        stages = {warning["stage"] for warning in result["warnings"]}
        self.assertIn("figures", stages)
        self.assertTrue((self.run_dir / "charts_warning.json").is_file())
        # Non-matplotlib stages still produce their outputs.
        self.assertTrue((self.run_dir / "latex_tables" / "blocking_fixed_k.tex").is_file())
        self.assertTrue((self.run_dir / "bar_charts" / "ranked_pc_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
