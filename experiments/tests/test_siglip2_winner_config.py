"""The SigLIP2 winner on the charts has to be the current Met selection."""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.core.winner_config_id import winner_config_id

REPO = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = REPO / "tooling/met_paint_dashboard.py"
ID_RE = re.compile(
    r"^[0-9a-f-]{36}"
    r"__hyperparam-[0-9a-f]{12}"
    r"__training-[0-9a-f]{12}"
    r"__env-[0-9a-f]{12}"
    r"__loss-[0-9a-f]{12}"
    r"__checkpoint-[0-9a-f]{12}$"
)


def _dashboard():
    spec = importlib.util.spec_from_file_location("met_paint_dashboard_siglip2_winner", DASHBOARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _selection_eval(mean_pc: float) -> dict:
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


def _training_files(run_dir: Path, *, hidden: int, sha: str, mean_pc: float, loss_alpha: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "configuration_id": f"hidden-{hidden}",
                "training_seed": 42,
                "best_epoch": 1,
                "epochs": 5,
                "checkpoint_selection": {
                    "pc_tie": 0.005,
                    "lora_lr": 0.0003,
                    "head_lr": 0.0001,
                },
                "training_split": {"split_id": "split", "split_sha256": "aa"},
                "reproducibility": {
                    "seed": 42,
                    "dependencies": {"python": "3.12.12"},
                    "determinism": {"seed": 42},
                    "git": {"git_commit": "abc"},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "dataset": {"dataset_id": "synth"},
                    "environment": {"variables": {"SIGLIP2_MODEL_ID": "google/siglip2-giant-opt-patch16-384"}},
                    "finetuning": {
                        "image_size": 384,
                        "loss": {"name": "multi_similarity", "alpha": loss_alpha},
                        "model": {"head": {"hidden_dim": hidden}},
                        "optimizer": {"lr_lora": 0.0003, "lr_head": 0.0001},
                        "sampler": {"type": "pk"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "post_evaluations.json").write_text(
        json.dumps({"evaluations": [{"checkpoint_sha256": sha}]}),
        encoding="utf-8",
    )
    eval_path = (
        run_dir
        / "evaluations/eval_met_painting_model_selection_from_finetuning_v1/models/m/eval.json"
    )
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps(_selection_eval(mean_pc)), encoding="utf-8")


def _benchmark(root: Path, name: str, parent_name: str, pair_completeness: float) -> None:
    run_dir = root / name
    (run_dir / "models/m").mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model_runs": [{"display_name": "Finetuned"}],
                "parent_training": {"run_id": parent_name},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "models/m/eval.json").write_text(
        json.dumps({"tasks": [{"metrics": [{"k": 10, "pair_completeness": pair_completeness}]}]}),
        encoding="utf-8",
    )


class WinnerConfigIdTest(unittest.TestCase):
    def test_id_names_every_piece_and_changes_with_the_loss(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _training_files(run_dir, hidden=3072, sha="old", mean_pc=0.9, loss_alpha=2.0)
            first = winner_config_id(run_dir)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertRegex(first, ID_RE)
            loss_segment = first.split("__loss-")[1].split("__")[0]
            config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
            config["experiment"]["finetuning"]["loss"]["alpha"] = 4.0
            (run_dir / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
            second = winner_config_id(run_dir)
            assert second is not None
            self.assertNotEqual(first.split("__")[0], second.split("__")[0])
            self.assertNotEqual(loss_segment, second.split("__loss-")[1].split("__")[0])
            self.assertEqual(
                first.split("__hyperparam-")[1].split("__")[0],
                second.split("__hyperparam-")[1].split("__")[0],
            )

    def test_hyperparameter_change_keeps_the_loss_segment(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _training_files(run_dir, hidden=3072, sha="same", mean_pc=0.9, loss_alpha=2.0)
            first = winner_config_id(run_dir)
            assert first is not None
            config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
            config["experiment"]["finetuning"]["model"]["head"]["hidden_dim"] = 1024
            (run_dir / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
            second = winner_config_id(run_dir)
            assert second is not None
            self.assertNotEqual(
                first.split("__hyperparam-")[1].split("__")[0],
                second.split("__hyperparam-")[1].split("__")[0],
            )
            self.assertEqual(
                first.split("__loss-")[1].split("__")[0],
                second.split("__loss-")[1].split("__")[0],
            )


class Siglip2WinnerDashboardTest(unittest.TestCase):
    def test_charts_keep_only_the_current_config_and_warn_about_the_old_one(self):
        dashboard = _dashboard()
        old_name = (
            "siglip2loramlp__p-64__k-4__r-4__la-4__lb-last20__mh-frozen__llr-3e-4"
            "__hd-3072__hlr-1e-4__hdo-0__sam-pk__ep-5__run-100-3-trainseed-42"
        )
        new_name = (
            "siglip2loramlp__p-64__k-4__r-4__la-4__lb-last20__mh-frozen__llr-3e-4"
            "__hd-1024__hlr-1e-4__hdo-0__sam-pk__ep-5__run-200-4-trainseed-42"
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / dashboard.SIGLIP2_TRAIN_DIRS[0] / old_name
            new_dir = root / dashboard.SIGLIP2_TRAIN_DIRS[1] / new_name
            _training_files(old_dir, hidden=3072, sha="old-sha", mean_pc=0.90, loss_alpha=2.0)
            _training_files(new_dir, hidden=1024, sha="new-sha", mean_pc=0.97, loss_alpha=2.0)
            bench = root / dashboard.SIGLIP2_WIKI_FULL
            _benchmark(bench, "old-winner-eval", old_name, 0.11)
            _benchmark(bench, "new-winner-eval", new_name, 0.42)
            winner, info = dashboard._siglip2_winner(root)
            self.assertIsNotNone(winner)
            assert winner is not None
            self.assertEqual(winner.name, new_name)
            self.assertEqual(info["winner_config_id"], winner_config_id(new_dir))
            self.assertNotEqual(info["winner_config_id"], winner_config_id(old_dir))
            packed = dashboard._siglip2_chart_series(root)
            self.assertEqual(packed["curve"]["pc"], [0.42])
            self.assertEqual(packed["curve"]["winner_config_id"], info["winner_config_id"])
            self.assertEqual(packed["curve"]["label"], dashboard.SIGLIP2_WINNER_LABEL)
            stale = packed["info"]["winner_results"]["stale"]
            stale_names = [run["name"] for group in stale for run in group["runs"]]
            self.assertIn("old-winner-eval", stale_names)
            self.assertNotIn("new-winner-eval", stale_names)
            self.assertTrue(all(group["winner_config_id"] != info["winner_config_id"] for group in stale))
            self.assertIn("winnerConfigWarning", dashboard.PAGE)
            self.assertIn("renderWinnerWarning", dashboard.PAGE)

    def test_live_wave2_run_holds_met_selection_without_benchmark_scores(self):
        summary = next(
            (REPO / "experiments/runs/finetune_siglip2_giant_met_paint_wave2_v1").glob(
                "*2603007-4*/training_summary.json"
            ),
            None,
        )
        if summary is None:
            self.skipTest("wave-2 training summaries are not in this checkout")
        dashboard = _dashboard()
        winner, info = dashboard._siglip2_winner()
        self.assertIsNotNone(winner)
        assert winner is not None
        self.assertIn("2603007-4", winner.name)
        self.assertIn("hd-1024", winner.name)
        self.assertRegex(str(info["winner_config_id"]), ID_RE)
        results = info["winner_results"]
        self.assertIn("Wikidata 1.5 full set", results["missing"])
        self.assertIn("Wikidata 1.5 partitions", results["missing"])
        dirnames = [dirname for _name, dirname in dashboard.SIGLIP2_RESULT_CATALOGS]
        self.assertNotIn("eval_wikidata1_4_lostart_distractors_siglip2_winner_full_v1", dirnames)
        self.assertTrue(all(group["winner_config_id"] != info["winner_config_id"] for group in results["stale"]))
        packed = dashboard._siglip2_chart_series()
        self.assertNotIn("curve", packed)
        self.assertNotIn("multi", packed)
        lost = dashboard._wikidata_lostart()
        self.assertNotIn(
            dashboard.SIGLIP2_WINNER_LABEL,
            [item["label"] for item in lost["pc_at_k"]],
        )
        distances = [item["label"] for item in dashboard._domain_distance()]
        self.assertNotIn(dashboard.SIGLIP2_WINNER_LABEL, distances)
        competition = dashboard._siglip2_competition()
        self.assertEqual(competition["winner"], winner.name)
        self.assertEqual(competition["winner_config_id"], info["winner_config_id"])
        selected = next(model for model in competition["models"] if model["winner"])
        self.assertEqual(selected["winner_config_id"], info["winner_config_id"])
        self.assertTrue(all(model.get("winner_config_id") for model in competition["models"]))


if __name__ == "__main__":
    unittest.main()
