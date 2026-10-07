import importlib.util
import tempfile
import unittest
from pathlib import Path

from experiments.core.plot_ranked_pc_bars import (
    DEFAULT_BAR_COLOR,
    _load_model_styles,
    _write_svg_bar_chart,
    plot_ranked_bars,
)


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class RankedPcBarChartTest(unittest.TestCase):
    def test_load_model_styles_reads_experiment_labels_and_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "resolved_config.yml").write_text(
                "models:\n"
                "  include:\n"
                "    - model_id: frozen/model\n"
                "      label: Frozen backbone\n"
                "      color: '#F58518'\n"
                "    - model_id: tuned/model\n"
                "      display_name: Fine-tuned model\n",
                encoding="utf-8",
            )

            labels, colors = _load_model_styles(run_dir)

            self.assertEqual(
                labels,
                {
                    "frozen/model": "Frozen backbone",
                    "tuned/model": "Fine-tuned model",
                },
            )
            self.assertEqual(colors, {"frozen/model": "#F58518"})

    def test_svg_uses_configured_color_and_default_for_unstyled_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "chart.svg"
            rows = [
                {"model_id": "frozen/model", "pair_completeness": 0.9},
                {"model_id": "tuned/model", "pair_completeness": 0.8},
            ]

            _write_svg_bar_chart(
                rows,
                {"frozen/model": "Frozen backbone", "tuned/model": "Fine-tuned model"},
                {"frozen/model": "#F58518"},
                output,
                10,
                "test",
                None,
            )

            svg = output.read_text(encoding="utf-8")
            self.assertIn('fill="#F58518"', svg)
            self.assertIn(f'fill="{DEFAULT_BAR_COLOR}"', svg)
            self.assertIn("Frozen backbone", svg)
            self.assertIn("Fine-tuned model", svg)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_vertical_style_prints_six_decimal_pc_values_above_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = [
                {
                    "model_id": "frozen/model",
                    "task_id": "test",
                    "k": 40,
                    "pair_completeness": 0.9706933523945676,
                },
                {
                    "model_id": "tuned/model",
                    "task_id": "test",
                    "k": 40,
                    "pair_completeness": 0.9656897784131523,
                },
            ]

            outputs = plot_ranked_bars(
                rows,
                {
                    "frozen/model": "Frozen H+",
                    "tuned/model": "Finetuned H+",
                },
                output_dir,
                colors={
                    "frozen/model": "#4477AA",
                    "tuned/model": "#F0E442",
                },
                formats=("svg",),
                style={
                    "orientation": "vertical",
                    "pc_limits": [0.45, 1.025],
                    "show_values": True,
                    "value_decimals": 6,
                    "model_label_layout": "alternating_guides",
                    "model_hatches": {"tuned/model": "///"},
                },
            )

            self.assertEqual(
                outputs, [output_dir / "ranked_pc_at_40_test.svg"]
            )
            svg = outputs[0].read_text(encoding="utf-8")
            self.assertIn("0.970693", svg)
            self.assertIn("0.965690", svg)
            self.assertIn("Frozen H+", svg)
            self.assertIn("Finetuned H+", svg)
            self.assertIn("#4477aa", svg.lower())
            self.assertIn("#f0e442", svg.lower())


if __name__ == "__main__":
    unittest.main()
