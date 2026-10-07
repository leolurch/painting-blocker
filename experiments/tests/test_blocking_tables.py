import json
import tempfile
import unittest
from pathlib import Path

from experiments.core.blocking_tables import (
    _ordered_group_row_specs,
    render_blocking_calibration_targets_table,
    render_blocking_fixed_k_table,
    render_blocking_performance_table,
    write_run_blocking_tables,
)
from experiments.core.latex_tables import write_run_latex_tables


class BlockingTablesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run1"
        _write_run(self.run_dir)

    def test_fixed_k_table_marks_best_and_second_best_per_column(self):
        _write_two_models(self.run_dir)
        paths = write_run_blocking_tables(self.run_dir)
        fixed = paths["blocking_fixed_k"].read_text(encoding="utf-8")
        values = paths["blocking_fixed_k_values"].read_text(encoding="utf-8")

        self.assertIn("\\newcommand{\\best}", fixed)  # macro comment header
        self.assertIn("\\input{latex_tables/blocking_fixed_k_values.tex}", fixed)
        self.assertIn("\\multicolumn{2}{c}{PC@$k$}", fixed)
        self.assertIn("\\multicolumn{2}{c}{PQ@$k$}", fixed)
        # PC as 0.xxx: model_b best at k=1 (0.7), model_a best at k=5 (0.9).
        self.assertIn("\\best{.700}", values)
        self.assertIn("\\secondbest{.600}", values)
        self.assertIn("\\best{.900}", values)
        self.assertIn("\\secondbest{.800}", values)
        # PQ as a fraction: model_a best in both columns.
        self.assertIn("\\best{.300}", values)
        self.assertIn("\\secondbest{.200}", values)
        self.assertIn("\\best{.180}", values)
        self.assertIn("\\secondbest{.160}", values)
        # RR is emitted once per k as a comment, reading the stored value.
        self.assertIn("% RR@1 = 0.9000", fixed)
        self.assertIn("% RR@5 = 0.5000", fixed)
        # Configured model order (run.json), not alphabetical.
        self.assertLess(values.index("Beta\\_Model"), values.index("Alpha Model"))

    def test_calibrated_table_reports_signed_transfer_gap(self):
        _write_two_models(self.run_dir)
        paths = write_run_blocking_tables(self.run_dir)
        calibrated = paths["blocking_calibrated"].read_text(encoding="utf-8")
        values = paths["blocking_calibrated_values"].read_text(encoding="utf-8")

        self.assertIn("$\\Delta PC$", calibrated)
        self.assertIn("$PC^{test}_{\\emph{WIK}}$", calibrated)
        self.assertIn("\\input{latex_tables/blocking_calibrated_values.tex}", calibrated)
        # model_a: PC 0.96 at rho 0.95 -> +0.010; model_b: PC 0.94 -> -0.010.
        self.assertIn("$+0.010$", values)
        self.assertIn("$-0.010$", values)
        # tau, RR, QC formats; missing QC renders as --.
        self.assertIn("& 0.250 &", values)
        self.assertIn("& 0.9300 &", values)
        self.assertIn("& 0.970", values)
        self.assertIn("& --", values)
        # PQ in percent.
        self.assertIn("& 1.25 &", values)

    def test_fixed_k_table_groups_orders_and_marks_models_per_group(self):
        model_ids = ["ft-z", "base-b", "ft-a", "base-a"]
        labels = {
            "ft-z": "Zeta tuned",
            "base-b": "Beta frozen",
            "ft-a": "Alpha tuned",
            "base-a": "Pinned frozen",
        }
        values = {
            "ft-z": (0.8, 0.2),
            "base-b": (0.95, 0.4),
            "ft-a": (0.9, 0.3),
            "base-a": (0.5, 0.1),
        }
        rows = [
            {
                "model_id": model_id,
                "task_id": "task",
                "selection_mode": "top_k",
                "k": 10,
                "pair_completeness": pc,
                "pair_quality": pq,
                "reduction_ratio": 0.5,
            }
            for model_id, (pc, pq) in values.items()
        ]

        table = render_blocking_fixed_k_table(
            rows,
            {"run_id": "run", "dataset": {"dataset_id": "data"}},
            model_ids,
            labels,
            "task",
            ks=[10],
            model_groups={
                "ft-z": "finetuned",
                "ft-a": "finetuned",
                "base-a": "frozen",
                "base-b": "frozen",
            },
            model_group_order=["finetuned", "frozen"],
            baseline_comparison_model="base-a",
            baseline_comparison_group="finetuned",
            baseline_comparison_position="last",
            baseline_comparison_separator="dashed",
            alphabetical_within_groups=True,
            separate_model_groups=True,
            highlight_best_per_group=True,
            comparison_metrics={
                "ft-z": {
                    "syn_test_pc_at_k": 0.7,
                    "wik_test_pc_mean": 0.8,
                    "wik_test_pc_sample_sd": 0.01,
                    "wik_test_pq_mean": 0.4,
                    "wik_test_pq_sample_sd": 0.04,
                    "wik_test_rr_mean": 0.5,
                    "wik_test_rr_sample_sd": 0.02,
                },
                "ft-a": {
                    "syn_test_pc_at_k": 0.9,
                    "wik_test_pc_mean": 0.7,
                    "wik_test_pc_sample_sd": 0.02,
                    "wik_test_pq_mean": 0.6,
                    "wik_test_pq_sample_sd": 0.03,
                    "wik_test_rr_mean": 0.6,
                    "wik_test_rr_sample_sd": 0.03,
                },
                "base-a": {
                    "syn_test_pc_at_k": None,
                    "wik_test_pc_mean": 0.9,
                    "wik_test_pc_sample_sd": 0.03,
                    "wik_test_pq_mean": 0.2,
                    "wik_test_pq_sample_sd": 0.02,
                    "wik_test_rr_mean": 0.2,
                    "wik_test_rr_sample_sd": 0.04,
                },
                "base-b": {
                    "syn_test_pc_at_k": 0.8,
                    "wik_test_pc_mean": 0.95,
                    "wik_test_pc_sample_sd": 0.04,
                    "wik_test_pq_mean": 0.5,
                    "wik_test_pq_sample_sd": 0.05,
                    "wik_test_rr_mean": 0.4,
                    "wik_test_rr_sample_sd": 0.05,
                },
            },
            performance_columns=[
                {"key": "peak_vram_gib", "label": "$\\overline{\\mathrm{VRAM}}$", "objective": "min", "decimals": 2, "leading_zero": True},
                {"key": "images_per_second", "label": "Images/s", "objective": "max", "decimals": 1},
            ],
            performance_values={
                "ft-z": {"peak_vram_gib": 1.0, "images_per_second": 10.0},
                "ft-a": {"peak_vram_gib": 2.0, "images_per_second": 20.0},
                "base-a": {"peak_vram_gib": 0.5, "images_per_second": 30.0},
                "base-b": {"peak_vram_gib": 0.4, "images_per_second": 40.0},
            },
            performance_placeholder="-",
            performance_group_heading="B200, $B=1$",
            vertical_group_rules=True,
        )

        header = table.split("\\toprule", 1)[1].split("\\midrule", 1)[0]
        self.assertLess(
            header.index("$PC^{sel}_{\\emph{SYN}}$"),
            header.index("$PC^{full}_{\\emph{WIK}}$"),
        )
        self.assertIn("$\\rho^{cal}_{\\emph{WIK}}=.99$", table)
        self.assertNotIn("$PQ^{full}_{\\emph{WIK}}$", table)
        self.assertIn("B200, $B=1$", header)
        self.assertIn("$PQ^{test}_{\\emph{WIK}}$", header)
        self.assertLess(
            header.index("$RR^{test}_{\\emph{WIK}}$"),
            header.index("$PQ^{test}_{\\emph{WIK}}$"),
        )
        self.assertLess(header.index("$PQ^{test}_{\\emph{WIK}}$"), header.index("$\\overline{\\mathrm{VRAM}}$"))
        self.assertNotIn("Peak VRAM", header)
        self.assertIn("\\begin{tabular}{l|r|r|rrr|rr}", table)
        self.assertLess(table.index("Alpha tuned"), table.index("Zeta tuned"))
        self.assertLess(table.index("Zeta tuned"), table.index("Pinned frozen"))
        self.assertLess(table.index("Pinned frozen"), table.index("Beta frozen"))
        self.assertNotIn("\\emph{(baseline", table)
        self.assertIn(
            "Alpha tuned & \\best{.900} & \\secondbest{.900} "
            "& .700 $\\pm$ .020 & \\best{.600} $\\pm$ .030 "
            "& \\best{.600} "
            "& 2.00 & 20.0",
            table,
        )
        self.assertIn(
            "Beta frozen & \\secondbest{.800} & \\best{.950} "
            "& \\best{.950} $\\pm$ \\secondbest{.040} "
            "& \\secondbest{.400} $\\pm$ \\secondbest{.050} "
            "& \\secondbest{.500} "
            "& \\best{0.40} & \\best{40.0}",
            table,
        )
        self.assertIn(
            "Zeta tuned & .700 & .800 & .800 $\\pm$ \\best{.010} "
            "& .500 $\\pm$ \\best{.020} & .400 & 1.00 & 10.0",
            table,
        )
        self.assertIn(
            "\\noalign{\\vskip-.6ex}\n        "
            "\\multicolumn{8}{@{}c@{}}{\\leaders\\hbox{"
            "\\rule[.5ex]{3pt}{.3pt}\\hspace{2pt}}\\hfill\\kern0pt} "
            "\\\\[-.8ex]\n        Pinned frozen & -- & .500 "
            "& \\secondbest{.900} $\\pm$ .030",
            table,
        )
        self.assertIn(
            "Pinned frozen & -- & .500 & \\secondbest{.900} "
            "$\\pm$ .030 & .200 $\\pm$ .040 & .200 "
            "& \\secondbest{0.50} & \\secondbest{30.0} \\\\\n"
            "        \\midrule\n        Beta frozen",
            table,
        )
        self.assertIn("\\secondbest", "\n".join(table.splitlines()[12:]))
        self.assertEqual(table.count("\\leaders\\hbox"), 1)
        self.assertNotIn("\\hdashline", table)
        self.assertEqual(table.count("\\midrule"), 2)

    def test_additional_calibration_targets_table_contains_both_targets(self):
        rows = [
            {
                "model_id": model_id,
                "task_id": "task",
                "selection_mode": "top_k",
                "k": 10,
                "pair_completeness": 0.9,
                "pair_quality": 0.1,
                "reduction_ratio": 0.5,
            }
            for model_id in ("model-a", "model-b")
        ]
        metrics = {
            "model-a": {
                "wik_calibrated_metrics_by_target": {
                    "0.95": {
                        "wik_test_pc_mean": 0.96,
                        "wik_test_pc_sample_sd": 0.01,
                        "wik_test_rr_mean": 0.5,
                        "wik_test_rr_sample_sd": 0.02,
                        "wik_test_pq_mean": 0.1,
                        "wik_test_pq_sample_sd": 0.04,
                    },
                    "0.99": {
                        "wik_test_pc_mean": 0.99,
                        "wik_test_pc_sample_sd": 0.02,
                        "wik_test_rr_mean": 0.4,
                        "wik_test_rr_sample_sd": 0.03,
                        "wik_test_pq_mean": 0.2,
                        "wik_test_pq_sample_sd": 0.05,
                    },
                }
            },
            "model-b": {
                "wik_calibrated_metrics_by_target": {
                    "0.95": {
                        "wik_test_pc_mean": 0.95,
                        "wik_test_pc_sample_sd": 0.02,
                        "wik_test_rr_mean": 0.4,
                        "wik_test_rr_sample_sd": 0.03,
                        "wik_test_pq_mean": 0.2,
                        "wik_test_pq_sample_sd": 0.03,
                    },
                    "0.99": {
                        "wik_test_pc_mean": 0.98,
                        "wik_test_pc_sample_sd": 0.03,
                        "wik_test_rr_mean": 0.3,
                        "wik_test_rr_sample_sd": 0.04,
                        "wik_test_pq_mean": 0.3,
                        "wik_test_pq_sample_sd": 0.04,
                    },
                }
            },
        }

        table = render_blocking_calibration_targets_table(
            rows,
            {"run_id": "run", "dataset": {"dataset_id": "data"}},
            ["model-a", "model-b"],
            {"model-a": "A", "model-b": "B"},
            "task",
            targets=[0.95, 0.99],
            comparison_metrics=metrics,
            model_groups={"model-a": "frozen", "model-b": "frozen"},
            model_group_order=["frozen"],
            highlight_best_per_group=True,
            vertical_group_rules=True,
        )

        self.assertIn("$\\rho^{cal}_{\\emph{WIK}}=.95$", table)
        self.assertIn("$\\rho^{cal}_{\\emph{WIK}}=.99$", table)
        self.assertIn("\\begin{table}[H]", table)
        self.assertIn("\\begin{adjustbox}{max width=\\textwidth}", table)
        self.assertNotIn("\\resizebox", table)
        self.assertIn("\\begin{tabular}{l|rrr|rrr}", table)
        self.assertIn(
            "A & \\best{.960} $\\pm$ \\best{.010} "
            "& \\best{.500} $\\pm$ \\best{.020} & .100 $\\pm$ .040 "
            "& \\best{.990} $\\pm$ \\best{.020}",
            table,
        )
        self.assertEqual(table.count("$PQ^{test}_{\\emph{WIK}}$"), 2)

    def test_additional_performance_table_contains_latency_throughput_and_vram(self):
        rows = [
            {
                "model_id": model_id,
                "task_id": "task",
                "selection_mode": "top_k",
                "k": 10,
                "pair_completeness": 0.9,
                "pair_quality": 0.1,
                "reduction_ratio": 0.5,
            }
            for model_id in ("model-a", "model-b")
        ]
        columns = [
            {"key": "p50", "label": "p50 (ms)", "objective": "min", "decimals": 1},
            {"key": "p90", "label": "p90 (ms)", "objective": "min", "decimals": 1},
            {"key": "ips", "label": "Images/s", "objective": "max", "decimals": 1},
            {
                "key": "vram",
                "label": "$\\mathrm{VRAM}_{\\max}$ (GiB)",
                "objective": "min",
                "decimals": 2,
                "leading_zero": True,
            },
        ]
        table = render_blocking_performance_table(
            rows,
            {"run_id": "run", "dataset": {"dataset_id": "data"}},
            ["model-a", "model-b"],
            {"model-a": "A", "model-b": "B"},
            "task",
            performance_columns=columns,
            performance_values={
                "model-a": {"p50": 10.0, "p90": 20.0, "ips": 50.0, "vram": 0.5},
                "model-b": {"p50": 12.0, "p90": 22.0, "ips": 40.0, "vram": 0.7},
            },
            model_groups={"model-a": "frozen", "model-b": "frozen"},
            model_group_order=["frozen"],
            highlight_best_per_group=True,
            vertical_group_rules=True,
        )

        self.assertIn("\\begin{table}[H]", table)
        self.assertIn("\\begin{adjustbox}{max width=\\textwidth}", table)
        self.assertNotIn("\\resizebox", table)
        self.assertIn("\\begin{tabular}{l|rrrr}", table)
        self.assertIn("p50 (ms) & p90 (ms) & Images/s", table)
        self.assertIn("$\\mathrm{VRAM}_{\\max}$ (GiB)", table)
        self.assertIn(
            "A & \\best{10.0} & \\best{20.0} & \\best{50.0} & \\best{0.50}",
            table,
        )

    def test_explicit_order_within_group_overrides_alphabetical_order(self):
        row_specs = [
            (("small", None), "Small"),
            (("large", None), "Large"),
            (("medium", None), "Medium"),
        ]
        ordered = _ordered_group_row_specs(
            row_specs,
            {model_id: "frozen" for model_id in ("small", "large", "medium")},
            ["frozen"],
            {"frozen": ["large", "medium", "small"]},
            None,
            None,
            True,
        )
        self.assertEqual(
            [row_key[0] for row_key, _label in ordered],
            ["large", "medium", "small"],
        )

    def test_calibrated_table_omitted_without_calibrated_rows(self):
        _write_model(self.run_dir, "model_a", "storage_a", calibrated=[])
        paths = write_run_blocking_tables(self.run_dir)
        self.assertIn("blocking_fixed_k", paths)
        self.assertNotIn("blocking_calibrated", paths)
        self.assertFalse((self.run_dir / "latex_tables" / "blocking_calibrated.tex").is_file())

    def test_existing_latex_tables_stay_byte_identical(self):
        _write_two_models(self.run_dir)
        existing = write_run_latex_tables(self.run_dir)
        before = {key: path.read_bytes() for key, path in existing.items()}

        write_run_blocking_tables(self.run_dir)

        for key, path in existing.items():
            self.assertEqual(path.read_bytes(), before[key], key)


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    run_json = {
        "run_id": "run1",
        "experiment_id": "exp",
        "status": "completed",
        "dataset": {"dataset_id": "toy_dataset"},
        "split": {"split_id": "split"},
        "model_runs": [{"model_id": "model_b"}, {"model_id": "model_a"}],
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


def _write_two_models(run_dir: Path) -> None:
    _write_model(
        run_dir,
        "model_b",
        "storage_b",
        top_k=[
            {"k": 1, "reduction_ratio": 0.9, "pair_completeness": 0.7, "pair_quality": 0.2},
            {"k": 5, "reduction_ratio": 0.5, "pair_completeness": 0.8, "pair_quality": 0.16},
        ],
        calibrated=[
            {
                "calibration_target_pair_completeness": 0.95,
                "threshold": 0.2,
                "pair_completeness": 0.94,
                "pair_quality": 0.0119,
                "reduction_ratio": 0.92,
                "query_coverage": 0.97,
            },
            {
                "calibration_target_pair_completeness": 0.99,
                "threshold": 0.15,
                "pair_completeness": 0.97,
                "pair_quality": 0.0070,
                "reduction_ratio": 0.87,
                "query_coverage": 0.99,
            },
        ],
    )
    _write_model(
        run_dir,
        "model_a",
        "storage_a",
        top_k=[
            {"k": 1, "reduction_ratio": 0.9, "pair_completeness": 0.6, "pair_quality": 0.3},
            {"k": 5, "reduction_ratio": 0.5, "pair_completeness": 0.9, "pair_quality": 0.18},
        ],
        calibrated=[
            {
                "calibration_target_pair_completeness": 0.95,
                "threshold": 0.25,
                "pair_completeness": 0.96,
                "pair_quality": 0.0125,
                "reduction_ratio": 0.93,
                # query_coverage intentionally missing -> rendered as --
            },
        ],
    )


def _write_model(
    run_dir: Path,
    model_id: str,
    storage_key: str,
    top_k: list[dict[str, float]] | None = None,
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
                "metrics": top_k
                or [
                    {"k": 1, "reduction_ratio": 0.9, "pair_completeness": 0.5, "pair_quality": 0.1}
                ],
                "target_pc_metrics": [],
                "calibrated_threshold_metrics": calibrated or [],
            }
        ],
    }
    (model_dir / "eval.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
