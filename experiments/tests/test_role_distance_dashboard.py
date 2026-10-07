"""The domain-excess bars subtract the within-domain average from the cross-domain distance."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = REPO / "tooling/met_paint_dashboard.py"
SCORES_PATH = REPO / "writeups/conference-paper/figures/generated/wikidata_1_4_role_pair_similarity/scores.json"


def _dashboard():
    spec = importlib.util.spec_from_file_location("met_paint_dashboard_role_distance", DASHBOARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RoleDistanceDashboardTest(unittest.TestCase):
    def test_excess_is_cross_minus_within_average(self):
        dashboard = _dashboard()
        self.assertAlmostEqual(dashboard._domain_excess(0.2, 0.4, 0.9), 0.6)
        self.assertAlmostEqual(dashboard._domain_excess(0.5, 0.5, 0.2), 0.3)

    def test_lost_art_rows_keep_the_same_painting_pairs(self):
        dashboard = _dashboard()
        rows = dashboard._role_distance_rows("lostart")
        scores = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        for model in scores["models"]:
            summary = model["summary"]
            historic_historic = summary["historic_historic"]["mean_cosine_distance"]
            modern_modern = summary["modern_modern"]["mean_cosine_distance"]
            historic_modern = summary["historic_modern"]["mean_cosine_distance"]
            got = rows[model["label"]]
            self.assertEqual(got["n_historic_historic"], 573)
            self.assertEqual(got["n_modern_modern"], 1571)
            self.assertEqual(got["n_historic_modern"], 1603)
            self.assertAlmostEqual(
                got["domain_excess"],
                dashboard._domain_excess(historic_historic, modern_modern, historic_modern),
            )
        resnet = rows["ResNet winner"]
        self.assertEqual(resnet["n_historic_modern"], 1603)
        self.assertAlmostEqual(
            resnet["domain_excess"],
            dashboard._domain_excess(
                resnet["historic_historic"],
                resnet["modern_modern"],
                resnet["historic_modern"],
            ),
        )

    def test_histogram_loader_keeps_matching_bins(self):
        dashboard = _dashboard()
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            edges = [-0.25, 0.0, 1.0]
            payload = {
                "edges": edges,
                "models": [
                    {
                        "label": "SigLIP2",
                        "positive": [1, 2],
                        "negative": [3, 4],
                        "n_positive": 3,
                        "n_negative": 7,
                    },
                    {
                        "label": "DINOv3-H+",
                        "positive": [5, 6],
                        "negative": [7, 8],
                        "n_positive": 11,
                        "n_negative": 15,
                    },
                    {
                        "label": "dropped",
                        "positive": [1],
                        "negative": [1, 2],
                        "n_positive": 1,
                        "n_negative": 3,
                    },
                ],
            }
            (folder / "similarity_histograms_wikidata.json").write_text(json.dumps(payload), encoding="utf-8")
            dashboard._ROLE_DISTANCE_DIR = folder
            loaded = dashboard._similarity_histograms("wikidata")
        self.assertEqual([row["label"] for row in loaded["models"]], ["SigLIP2", "DINOv3-H+"])
        self.assertEqual(loaded["models"][1]["n_positive"], 11)
        self.assertEqual(loaded["edges"], edges)
