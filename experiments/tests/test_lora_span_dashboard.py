"""LoRA span runs join the Met model-selection funnel once they are scored."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from experiments.core.checkpoint_naming import checkpoint_name_from_finetuning

REPO = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = REPO / "tooling/met_paint_dashboard.py"
CONFIG_DIR = REPO / "experiments/configs/experiments/finetune_dinov3_vithplus_met_paint_lora_span_v1"


def _dashboard():
    spec = importlib.util.spec_from_file_location("met_paint_dashboard_under_test", DASHBOARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _eval(mean_pc: float) -> dict:
    metrics = []
    for k in (5, 10, 20, 40, 60, 80):
        metrics.append(
            {
                "k": k,
                "pair_completeness": mean_pc,
                "reduction_ratio": 0.5,
                "precision": 0.4,
            }
        )
    return {
        "tasks": [
            {
                "metrics": metrics,
                "target_pc_metrics": [
                    {
                        "target_pair_completeness": 0.95,
                        "reduction_ratio": 0.91,
                        "pair_completeness": 0.95,
                    }
                ],
            }
        ]
    }


class LoraSpanDashboardTest(unittest.TestCase):
    def test_configs_name_the_three_spans(self):
        expected = {"all": "all", "first16": "first16", "last24": "last24"}
        found = {}
        for path in sorted(CONFIG_DIR.glob("*.yml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            span = raw["finetuning"]["model"]["lora"]["span"]
            found[path.stem] = span
            name = checkpoint_name_from_finetuning(raw)
            self.assertIn(f"__lb-{span}__", name)
            self.assertEqual(raw["finetuning"]["train"]["seed"], 42)
            self.assertNotIn("training_repeats", raw["finetuning"])
            evals = raw["finetuning"]["post_evaluations"]
            self.assertEqual(len(evals), 1)
            self.assertIn("eval_met_painting_model_selection_from_finetuning_v1", evals[0]["experiment"])
        self.assertEqual(found, expected)

    def test_scored_span_enters_the_selection_funnel(self):
        dashboard = _dashboard()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wave = root / "finetune_dinov3_vithplus_met_paint_wave1_v1"
            wave.mkdir()
            run = root / dashboard.LORA_SPAN_DIR_NAME / (
                "loramlp__p-64__k-4__r-4__la-4__lb-all__llr-3e-4__hd-1024__hlr-1e-4__hdo-0__sam-pk__ep-5"
                "__run-9001-0-trainseed-42"
            )
            eval_path = (
                run
                / "evaluations/eval_met_painting_model_selection_from_finetuning_v1/models/m/eval.json"
            )
            eval_path.parent.mkdir(parents=True)
            eval_path.write_text(json.dumps(_eval(0.99)), encoding="utf-8")
            (run / "training_summary.json").write_text(
                json.dumps({"best_epoch": 3}),
                encoding="utf-8",
            )
            selection = dashboard._model_selection(wave, [])
        spans = [model for model in selection["models"] if model.get("source") == "lora_span"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["span"], "all")
        self.assertIn("every attention block", spans[0]["label"])
        self.assertEqual(spans[0]["best_epoch"], 3)
        self.assertTrue(spans[0]["rule_ready"])
        self.assertEqual(selection["rule"]["scored"], 1)
        self.assertGreaterEqual(selection["rule"]["expected"], 3)
        self.assertFalse(selection["rule"]["complete"])


if __name__ == "__main__":
    unittest.main()
