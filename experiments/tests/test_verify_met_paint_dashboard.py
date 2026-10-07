"""The chart verifier checks the catalog selected in the dashboard."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.tests.test_siglip2_winner_config import _benchmark, _dashboard, _training_files

REPO = Path(__file__).resolve().parents[2]
VERIFIER_PATH = REPO / "tooling/verify_met_paint_dashboard.py"


def _verifier():
    spec = importlib.util.spec_from_file_location("verify_met_paint_dashboard_under_test", VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DatasetVerifierTest(unittest.TestCase):
    def test_a_matching_config_from_the_other_dataset_is_still_out_of_date(self):
        verifier = _verifier()
        series = {"winner_config_id": "same-id", "dataset": "wikidata14"}
        problem = verifier._series_problem(series, "Pairs completeness", "same-id", "lostart", lambda left, right: left == right)
        self.assertIn("wikidata14", problem)
        self.assertIn("lostart", problem)
        self.assertIsNone(
            verifier._series_problem(series, "Pairs completeness", "same-id", "wikidata14", lambda left, right: left == right)
        )

    def test_selected_dataset_limits_the_chart_errors(self):
        dashboard = _dashboard()
        verifier = _verifier()
        name = (
            "siglip2loramlp__p-64__k-4__r-4__la-4__lb-last20__mh-frozen__llr-3e-4"
            "__hd-1024__hlr-1e-4__hdo-0__sam-pk__ep-5__run-200-4-trainseed-42"
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / dashboard.SIGLIP2_TRAIN_DIRS[1] / name
            _training_files(run_dir, hidden=1024, sha="new-sha", mean_pc=0.97, loss_alpha=2.0)
            _benchmark(root / dashboard.SIGLIP2_WIKI_FULL, "wiki-eval", name, 0.42)
            wiki_errors = verifier.outdated_chart_errors(root, dashboard, dataset="wikidata14")
            lost_errors = verifier.outdated_chart_errors(root, dashboard, dataset="lostart")
        self.assertTrue(wiki_errors)
        self.assertTrue(lost_errors)
        self.assertTrue(all(error.startswith("error: wikidata14:") for error in wiki_errors))
        self.assertTrue(all(error.startswith("error: lostart:") for error in lost_errors))
        self.assertFalse(any("wikidata14:wikiPc:" in error for error in wiki_errors))
        self.assertFalse(any("lostart:wikiPc:" in error for error in lost_errors))
        self.assertIn("payload.chart_warnings", dashboard.PAGE)

    def test_dashboard_warnings_stay_on_the_selected_dataset(self):
        dashboard = _dashboard()
        wiki = dashboard.chart_warnings("wikidata14")
        lost = dashboard.chart_warnings("lostart")
        self.assertFalse(any(item.startswith("error: lostart:") for item in wiki))
        self.assertFalse(any(item.startswith("error: wikidata14:") for item in lost))


if __name__ == "__main__":
    unittest.main()
