"""Finetuned winners scored at the K calibrated on the frozen model."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRANSFER_PATH = REPO / "tooling/compute_wikidata_frozen_k_transfer.py"
DASHBOARD_PATH = REPO / "tooling/met_paint_dashboard.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FrozenKTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transfer = _load(TRANSFER_PATH, "frozen_k_transfer_under_test")
        cls.dashboard = _load(DASHBOARD_PATH, "dashboard_frozen_k_under_test")

    def test_donated_k_is_not_replaced_by_the_winner_k(self) -> None:
        ranked = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=bool,
        )
        rows = self.transfer.margins_for_donated_k(ranked, calibrated_k=1, total_positives=2, possible_pairs=8)
        by_margin = {int(row["margin"]): row for row in rows}

        self.assertEqual(by_margin[0]["requested_k"], 1)
        self.assertEqual(by_margin[0]["applied_k"], 1)
        self.assertEqual(by_margin[0]["pair_completeness"], 0.5)
        self.assertEqual(by_margin[0]["reduction_ratio"], 0.75)
        self.assertFalse(by_margin[0]["test_target_attained"])
        self.assertEqual(by_margin[5]["requested_k"], 6)
        self.assertEqual(by_margin[5]["applied_k"], 4)
        self.assertEqual(by_margin[5]["pair_completeness"], 1.0)

    def test_pairs_require_the_same_split(self) -> None:
        donor = {"split_id": "seed-42-v2", "split_seed": 42}
        winner = {"split_id": "seed-42-v2", "split_seed": 42}
        other = {"split_id": "seed-7-v2", "split_seed": 42}
        pairs, problems = self.transfer.pair_by_split([donor], [winner, other])

        self.assertEqual(pairs, [(donor, winner)])
        self.assertTrue(any("seed-7-v2" in problem for problem in problems))

    def test_same_images_survive_a_rewritten_snapshot(self) -> None:
        roles = {
            "subsets": {
                "val": {"roles": {"modern": ["a"], "historic": ["b"]}},
                "test": {"roles": {"modern": ["c"], "historic": ["d"]}},
            }
        }
        rewritten = {
            "extra": True,
            "subsets": {
                "test": {"roles": {"historic": ["d"], "modern": ["c"]}},
                "val": {"roles": {"historic": ["b"], "modern": ["a"]}},
            },
        }
        moved = {
            "subsets": {
                "val": {"roles": {"modern": ["c"], "historic": ["b"]}},
                "test": {"roles": {"modern": ["a"], "historic": ["d"]}},
            }
        }

        self.assertTrue(self.transfer.same_partition(roles, rewritten))
        self.assertFalse(self.transfer.same_partition(roles, moved))

    def test_run_token_does_not_match_a_longer_id(self) -> None:
        self.assertTrue(self.transfer._contains_token("runs/run-2587228-12/best_checkpoint.pt", "run-2587228-12"))
        self.assertFalse(self.transfer._contains_token("runs/run-2587228-120/best_checkpoint.pt", "run-2587228-12"))

    def test_dashboard_reads_frozen_k_as_its_own_series(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "win-seed-42"
            model_dir = run_dir / "models" / "checkpoint"
            model_dir.mkdir(parents=True)
            payload = {
                "margins": [
                    {
                        "margin": 0,
                        "applied_k": 19,
                        "pair_completeness": 0.97,
                        "reduction_ratio": 0.8,
                        "selected_per_query": 19,
                    }
                ]
            }
            (model_dir / "frozen_k_margin.json").write_text(json.dumps(payload), encoding="utf-8")

            point = self.dashboard._k_margin_point(run_dir, "frozen_k_margin.json")
            own = self.dashboard._k_margin_point(run_dir)

        self.assertIsNone(own)
        assert point is not None
        packed = self.dashboard._pack_k_margin(
            [point],
            self.dashboard.HPLUS_FROZEN_K_LABEL,
            self.dashboard.HPLUS_FROZEN_K_COLOR,
            "frozen-k",
            self.dashboard.HPLUS_FROZEN_K_DETAIL,
        )
        assert packed is not None
        self.assertEqual(packed["label"], "H+ winner (frozen-k)")
        self.assertEqual(packed["by_margin"]["0"]["mean_pc"], 0.97)
        self.assertEqual(packed["by_margin"]["0"]["mean_applied_k"], 19)

    def test_k_chart_alone_receives_the_transferred_series(self) -> None:
        page = DASHBOARD_PATH.read_text(encoding="utf-8")
        chart = page.split("function drawKMarginChart", 1)[1].split("function drawUnionChart", 1)[0]
        stability = page.split("function stabilityModels", 1)[1].split("function unionEdge", 1)[0]
        margin = page.split("function drawMarginChart", 1)[1].split("function drawKMarginChart", 1)[0]

        legend = page.split("const legend = document.getElementById(\"wikiLegend\")", 1)[1].split("drawStability", 1)[0]
        self.assertIn("frozen_k_margin_multisplit", chart)
        self.assertNotIn("frozen_k_margin_multisplit", stability)
        self.assertNotIn("frozen_k_margin_multisplit", margin)
        self.assertNotIn("frozen_k_margin_multisplit", legend)


if __name__ == "__main__":
    unittest.main()
