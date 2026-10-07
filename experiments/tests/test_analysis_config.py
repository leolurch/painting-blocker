import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.core.analysis_charts import write_analysis_charts
from experiments.core.analysis_config import (
    config_family_sha256,
    initialize_analysis_config,
    load_analysis_config,
    portable_config_family_sha256,
    update_analysis_config,
    write_analysis_config,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
TASK = "test_modern_to_historic"


def _resolved(models, *, experiment_id="compatible_eval"):
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "description": "same protocol, incrementally extended models",
        "dataset": {"dataset_id": "toy", "path": "/dataset"},
        "split": {"file": "/split.json"},
        "models": {"include": models},
        "embedding": {"cache_policy": "reuse_if_config_hash_matches", "batch_size": 4},
        "evaluation": {
            "similarity": "cosine",
            "top_k": [1, 5],
            "retrieval_tasks": [{"task_id": TASK}],
            "threshold_calibration": {"target_pc": [0.95]},
        },
        "run": {"output_dir": "/runs/compatible_eval"},
    }


def _model(storage_key, *, preprocessing, label):
    return {
        "model_id": "shared/checkpoint",
        "model_storage_key": storage_key,
        "revision": "abc",
        "adapter_name": "fake_adapter",
        "adapter_kwargs": {},
        "embedding": {"normalize": True, "pooling": "default"},
        "preprocessing": preprocessing,
        "label": label,
        "color": "#123456",
    }


def _write_run(
    root: Path,
    name: str,
    models,
    *,
    changed_protocol=False,
    experiment_id="compatible_eval",
):
    run_dir = root / experiment_id / name
    (run_dir / "models").mkdir(parents=True)
    resolved = _resolved(models, experiment_id=experiment_id)
    if changed_protocol:
        resolved["evaluation"]["similarity"] = "dot"
    (run_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    model_runs = []
    for model in models:
        key = model["model_storage_key"]
        model_dir = run_dir / "models" / key
        model_dir.mkdir()
        eval_data = {
            "schema_version": 1,
            "model_id": model["model_id"],
            "model_storage_key": key,
            "tasks": [
                {
                    "task_id": TASK,
                    "query_ids_hash": "queries",
                    "candidate_ids_hash": "candidates",
                    "positive_pairs_hash": "positives",
                    "num_queries": 2,
                    "num_candidates": 5,
                    "num_positive_pairs": 2,
                    "metrics": [
                        {"k": 1, "pair_completeness": 0.5, "pair_quality": 0.5, "reduction_ratio": 0.8},
                        {"k": 5, "pair_completeness": 1.0, "pair_quality": 0.2, "reduction_ratio": 0.0},
                    ],
                    "target_pc_metrics": [],
                    "calibrated_threshold_metrics": [
                        {
                            "calibration_target_pair_completeness": 0.95,
                            "calibration_pair_completeness": 0.96,
                            "calibration_reduction_ratio": 0.85,
                            "threshold": 0.4,
                            "pair_completeness": 0.75,
                            "pair_quality": 0.1,
                            "reduction_ratio": 0.9,
                            "query_coverage": 1.0,
                        }
                    ],
                }
            ],
        }
        (model_dir / "eval.json").write_text(json.dumps(eval_data), encoding="utf-8")
        curves = {
            "model_id": model["model_id"],
            "tasks": [
                {
                    "task_id": TASK,
                    "precision_recall": [
                        {"pc": 1.0, "rr": 0.0, "pair_completeness": 1.0, "pair_quality": 0.1},
                        {"pc": 0.5, "rr": 0.9, "pair_completeness": 0.5, "pair_quality": 0.8},
                    ],
                    "histogram": {
                        "bin_edges": [-1.0, 0.0, 1.0],
                        "match_hist": [0, 2],
                        "neg_hist": [2, 1],
                    },
                }
            ],
        }
        (model_dir / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
        model_runs.append(
            {
                "model_id": model["model_id"],
                "model_storage_key": key,
                "model_revision": "abc",
                "status": "completed",
                "path": f"models/{key}/eval.json",
            }
        )
    run_json = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": name,
        "created_at": f"2026-01-0{name[-1]}T00:00:00+00:00",
        "status": "completed",
        "dataset": {"dataset_id": "toy", "source_db_sha256": "db"},
        "split": {"split_id": "split", "split_sha256": "split-sha"},
        "code": {"git_commit": "commit"},
        "model_runs": model_runs,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


class AnalysisConfigTest(unittest.TestCase):
    def test_portable_family_hash_ignores_historical_repo_root_layout(self):
        current = {
            **_resolved([]),
            "repo_root": "/cluster/project",
            "split": {"file": "/cluster/project/experiments/configs/splits/test.json"},
        }
        historical = {
            **_resolved([]),
            "repo_root": "/cluster/project/experiments/configs",
            "split": {"file": "/cluster/project/experiments/configs/splits/test.json"},
        }
        self.assertEqual(
            portable_config_family_sha256(current),
            portable_config_family_sha256(historical),
        )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        self.a = _model(
            "shared_checkpoint_letterbox_aaaaaaaaaa",
            preprocessing={"resize": {"mode": "letterbox", "size": 512}},
            label="Current preprocessing",
        )
        self.b = _model(
            "shared_checkpoint_default_bbbbbbbbbb",
            preprocessing={"resize": {"mode": "model_default"}},
            label="Model-default preprocessing",
        )

    def test_update_adds_models_and_preserves_user_fields(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        config = load_analysis_config(config_path)
        self.assertEqual(len(config["models"]), 1)
        entry = config["models"][0]
        entry["include"] = False
        entry["presentation"]["label"] = "My retained label"
        config["charts"]["pc_at_k_curve"]["enabled"] = False
        write_analysis_config(config_path, config)

        _write_run(self.runs, "run2", [self.a, self.b])
        # Same experiment_id but a changed non-model protocol must not be added.
        _write_run(self.runs, "run3", [self.a, self.b], changed_protocol=True)
        _path, updated, summary = update_analysis_config(config_path)

        self.assertEqual(summary["compatible_runs"], 2)
        self.assertEqual(summary["discovered_models"], 2)
        by_key = {entry["analysis_model_id"]: entry for entry in updated["models"]}
        retained = by_key[self.a["model_storage_key"]]
        added = by_key[self.b["model_storage_key"]]
        self.assertFalse(retained["include"])
        self.assertEqual(retained["presentation"]["label"], "My retained label")
        self.assertFalse(updated["charts"]["pc_at_k_curve"]["enabled"])
        self.assertTrue(added["include"])
        self.assertEqual(added["discovered"]["preprocessing"]["resize"]["mode"], "model_default")
        # prefer_existing keeps the original source for existing identities.
        self.assertTrue(retained["source"]["selected_run"].endswith("run1"))
        self.assertTrue(added["source"]["selected_run"].endswith("run2"))

    def test_update_accepts_experiment_id_aliases_but_keeps_protocol_strict(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        config = load_analysis_config(config_path)
        config["compatibility"]["experiment_id"] = [
            "compatible_eval",
            "compatible_eval_frozen",
        ]
        write_analysis_config(config_path, config)

        alias_run = _write_run(
            self.runs,
            "run2",
            [self.b],
            experiment_id="compatible_eval_frozen",
        )
        _write_run(
            self.runs,
            "run3",
            [self.a, self.b],
            experiment_id="compatible_eval_frozen",
            changed_protocol=True,
        )
        _write_run(
            self.runs,
            "run4",
            [self.a, self.b],
            experiment_id="unlisted_eval",
        )

        _path, updated, summary = update_analysis_config(config_path)

        self.assertEqual(summary["compatible_runs"], 2)
        self.assertEqual(summary["discovered_models"], 2)
        by_key = {entry["analysis_model_id"]: entry for entry in updated["models"]}
        self.assertEqual(
            (
                config_path.parent
                / by_key[self.b["model_storage_key"]]["source"]["selected_model_dir"]
            ).resolve(),
            (alias_run / "models" / self.b["model_storage_key"]).resolve(),
        )

    def test_update_accepts_explicit_family_hashes_for_non_protocol_metadata(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)

        alias_run = _write_run(
            self.runs,
            "run2",
            [self.b],
            experiment_id="compatible_eval_static",
        )
        resolved_path = alias_run / "resolved_config.yml"
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved["description"] = "Static model-list evaluation"
        resolved["run"]["output_dir"] = "/runs/compatible_eval_static"
        resolved_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )

        config = load_analysis_config(config_path)
        config["compatibility"]["experiment_id"] = [
            "compatible_eval",
            "compatible_eval_static",
        ]
        config["compatibility"]["config_family_sha256"] = [
            config["compatibility"]["config_family_sha256"],
            config_family_sha256(resolved),
        ]
        write_analysis_config(config_path, config)

        _path, updated, summary = update_analysis_config(config_path)

        self.assertEqual(summary["compatible_runs"], 2)
        self.assertEqual(summary["discovered_models"], 2)
        self.assertTrue(
            next(
                entry for entry in updated["models"]
                if entry["analysis_model_id"] == self.b["model_storage_key"]
            )["availability"]["available"]
        )

    def test_load_rejects_invalid_experiment_id_aliases(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        config = load_analysis_config(config_path)
        config["compatibility"]["experiment_id"] = []
        write_analysis_config(config_path, config)

        with self.assertRaisesRegex(ValueError, "compatibility.experiment_id"):
            load_analysis_config(config_path)

    def test_checkpoint_name_is_mined_when_template_display_name_has_no_label(self):
        checkpoint = _model(
            "named_checkpoint_cccccccccc",
            preprocessing={"resize": {"mode": "letterbox", "size": 512}},
            label=None,
        )
        checkpoint["model_id"] = (
            "local/finetuned/example/"
            "loramlp__p-32__k-4__r-4__la-8__llr-3e-5__hd-1024__hlr-3e-4__hdo-0p1__sam-pk__ep-15"
            "__run-2369999-2/best"
        )
        checkpoint["display_name"] = "Finetuned checkpoint (static template)"
        run1 = _write_run(self.runs, "run1", [checkpoint])
        config_path = self.root / "analysis.yml"

        _path, config, _summary = initialize_analysis_config(run1, self.runs, config_path)

        entry = config["models"][0]
        expected = (
            "loramlp__p-32__k-4__r-4__la-8__llr-3e-5__hd-1024__hlr-3e-4__hdo-0p1__sam-pk__ep-15"
        )
        self.assertEqual(entry["presentation"]["label"], expected)
        self.assertEqual(entry["generated"]["label"], expected)
        self.assertEqual(entry["discovered"]["checkpoint_naming"]["sam"], "pk")
        self.assertEqual(entry["discovered"]["checkpoint_naming"]["run"], "2369999-2")

    def test_explicit_label_precedes_inherited_template_display_name(self):
        checkpoint = _model(
            "explicit_checkpoint_dddddddddd",
            preprocessing={"resize": {"mode": "letterbox", "size": 512}},
            label="LoRA MLP P32/K4",
        )
        checkpoint["display_name"] = "Finetuned checkpoint (static template)"
        run1 = _write_run(self.runs, "run1", [checkpoint])
        config_path = self.root / "analysis.yml"

        _path, config, _summary = initialize_analysis_config(run1, self.runs, config_path)

        self.assertEqual(config["models"][0]["presentation"]["label"], "LoRA MLP P32/K4")

    def test_render_skips_models_without_sources_and_marks_output_incomplete(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        config = load_analysis_config(config_path)
        missing_key = "selected_model_without_source"
        config["models"].append(
            {
                "analysis_model_id": missing_key,
                "model_id": "local/missing/model",
                "include": True,
                "presentation": {"label": "Missing model", "order": 0},
            }
        )
        for chart in config["charts"].values():
            chart["enabled"] = False
        config["output"]["directory"] = "analysis-output"
        write_analysis_config(config_path, config)

        result = write_analysis_charts(config_path)

        output = self.root / "analysis-output"
        self.assertEqual(result["models"], 1)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(result["warnings"][0]["model"], missing_key)
        self.assertEqual(result["warnings"][0]["policy"], "skip_model")
        readme = output / "Readme.md"
        self.assertTrue(readme.is_file())
        self.assertEqual(result["paths"]["readme"], str(readme))
        incomplete = output / "INCOMPLETE.txt"
        self.assertTrue(incomplete.is_file())
        self.assertIn(missing_key, incomplete.read_text(encoding="utf-8"))
        manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["models"]), 1)
        self.assertEqual(manifest["warnings"], result["warnings"])

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_render_writes_numeric_json_sidecar_for_every_graph(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        config = load_analysis_config(config_path)
        for chart in config["charts"].values():
            chart["enabled"] = False
        config["charts"]["pc_at_k_curve"].update(
            {"enabled": True, "formats": ["pdf", "png"]}
        )
        config["charts"]["pc_at_k_curve_zoomed"] = {
            "enabled": True,
            "task": TASK,
            "minimum_k_exclusive": 1,
            "pc_limits": [0.4, 1.0],
            "require_source_model_dirs_for_all": True,
            "source_model_dirs": {
                self.a["model_storage_key"]: str(
                    run1 / "models" / self.a["model_storage_key"]
                ),
            },
            "formats": ["pdf", "svg"],
        }
        config["charts"]["pq_pc_curve"].update(
            {"enabled": True, "formats": ["pdf"]}
        )
        config["charts"]["ranked_pc_at_k_bars"].update(
            {"enabled": True, "formats": ["svg"], "ks": [1]}
        )
        config["output"]["directory"] = "analysis-output"
        write_analysis_config(config_path, config)

        write_analysis_charts(config_path)

        output = self.root / "analysis-output"
        graphs = [
            path for path in output.rglob("*")
            if path.suffix in {".pdf", ".png", ".svg"}
        ]
        self.assertGreaterEqual(len(graphs), 4)
        for graph in graphs:
            with self.subTest(graph=graph.name):
                sidecar = graph.with_suffix(".json")
                self.assertTrue(sidecar.is_file())
                self.assertIsInstance(json.loads(sidecar.read_text(encoding="utf-8")), dict)
        pc_data = json.loads(
            (output / "figures" / f"pc_at_k_{TASK}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(pc_data["series"][0]["ks"], [1, 5])
        self.assertEqual(pc_data["series"][0]["pc"], [0.5, 1.0])
        zoom_data = json.loads(
            (
                output
                / "bar_charts"
                / f"pc_at_k_gt1_zoomed_{TASK}.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(zoom_data["candidate_budget_filter"], {"minimum_exclusive": 1})
        self.assertEqual(zoom_data["pair_completeness_limits"], [0.4, 1.0])
        self.assertEqual(zoom_data["series"][0]["ks"], [5])
        self.assertEqual(zoom_data["legend_model_ids"], [self.a["model_storage_key"]])
        self.assertEqual(len(zoom_data["source_eval_artifacts"]), 1)
        self.assertEqual(
            zoom_data["source_eval_artifacts"][0]["analysis_model_id"],
            self.a["model_storage_key"],
        )
        manifest = json.loads(
            (output / "source_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["chart_sources"][0]["chart"], "pc_at_k_curve_zoomed"
        )

    def test_render_blocking_tables_does_not_require_reference_run(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        config = load_analysis_config(config_path)
        config["discovery"].pop("reference_run")
        for chart in config["charts"].values():
            chart["enabled"] = False
        config["charts"]["blocking_tables"]["enabled"] = True
        config["charts"]["blocking_tables"]["ks"] = [1]
        config["output"]["directory"] = "analysis-output"
        write_analysis_config(config_path, config)

        result = write_analysis_charts(config_path)

        output = self.root / "analysis-output"
        table = output / "latex_tables" / "blocking_fixed_k.tex"
        values = output / "latex_tables" / "blocking_fixed_k_values.tex"
        self.assertEqual(result["models"], 1)
        self.assertTrue(table.is_file())
        contents = table.read_text(encoding="utf-8")
        self.assertIn("on toy", contents)
        self.assertIn("tab:compatible-eval-analysis", contents)
        self.assertEqual(contents.count("$k=1$"), 2)
        self.assertNotIn("$k=5$", contents)
        value_text = values.read_text(encoding="utf-8")
        self.assertIn("& \\best{.500} & \\best{.500}", value_text)
        self.assertNotIn("1.000", value_text)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_render_uses_selected_cross_run_model_and_chart_config(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        _write_run(self.runs, "run2", [self.a, self.b])
        _path, config, _summary = update_analysis_config(config_path)
        for entry in config["models"]:
            entry["include"] = entry["analysis_model_id"] == self.b["model_storage_key"]
        for chart in config["charts"].values():
            chart["enabled"] = False
        config["charts"]["pc_at_k_curve"]["enabled"] = True
        config["charts"]["calibrated_threshold_bars"]["enabled"] = True
        config["charts"]["validation_rr_bars"]["enabled"] = True
        config["charts"]["ranked_pc_at_k_bars"]["enabled"] = True
        config["charts"]["pq_pc_curve_zoomed"] = {
            "enabled": True,
            "task": TASK,
            "formats": ["pdf"],
            "pc_limits": [0.8, 1.0],
            "pq_limits": [0.0, 0.2],
        }
        config["output"]["directory"] = "analysis-output"
        write_analysis_config(config_path, config)

        result = write_analysis_charts(config_path)

        output = self.root / "analysis-output"
        self.assertEqual(result["models"], 1)
        self.assertTrue((output / "figures" / f"pc_at_k_{TASK}.pdf").is_file())
        calibrated_bars = output / "figures" / f"pc_calibrated_rho0p95_bars_{TASK}.pdf"
        self.assertTrue(calibrated_bars.is_file())
        validation_rr_bars = output / "figures" / f"validation_rr_rho0p95_bars_{TASK}.pdf"
        self.assertTrue(validation_rr_bars.is_file())
        self.assertTrue((output / "figures" / f"pq_pc_zoomed_{TASK}.pdf").is_file())
        self.assertTrue((output / "bar_charts" / f"ranked_pc_at_1_{TASK}.svg").is_file())
        self.assertIn(
            f"calibrated_threshold_bars:pc_calibrated_rho0p95_bars_{TASK}.pdf",
            result["paths"],
        )
        self.assertIn(
            f"validation_rr_bars:validation_rr_rho0p95_bars_{TASK}.pdf",
            result["paths"],
        )
        self.assertIn(
            f"pq_pc_curve_zoomed:pq_pc_zoomed_{TASK}.pdf",
            result["paths"],
        )
        summary = (output / "bar_charts" / "ranked_pc_summary.csv").read_text(encoding="utf-8")
        self.assertIn(self.b["model_storage_key"], summary)
        self.assertNotIn(self.a["model_storage_key"], summary)
        manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["models"][0]["analysis_model_id"], self.b["model_storage_key"])

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_new_protocols_repeat_threshold_bar_filenames_without_margins(self):
        run1 = _write_run(self.runs, "run1", [self.a])
        model_dir = run1 / "models" / self.a["model_storage_key"]
        eval_path = model_dir / "eval.json"
        document = json.loads(eval_path.read_text(encoding="utf-8"))
        document["tasks"][0]["calibrated_top_k_metrics"] = [
            {
                "calibration_target_pair_completeness": 0.95,
                "calibration_reduction_ratio": 0.7,
                "k": 4,
                "pair_completeness": 0.8,
                "reduction_ratio": 0.6,
            }
        ]
        eval_path.write_text(json.dumps(document), encoding="utf-8")
        (model_dir / "calibrated_union.json").write_text(
            json.dumps(
                {
                    "target_pc": 0.95,
                    "k_margins": [0, 5],
                    "threshold_margins": [0.0, 0.01],
                    "pair_completeness": [[0.55, 0.99], [0.98, 0.99]],
                    "reduction_ratio": [[0.4, 0.1], [0.2, 0.1]],
                    "applied_k": [3, 8],
                    "applied_threshold": [0.5, 0.49],
                }
            ),
            encoding="utf-8",
        )
        config_path = self.root / "analysis.yml"
        initialize_analysis_config(run1, self.runs, config_path)
        config = load_analysis_config(config_path)
        for chart in config["charts"].values():
            chart["enabled"] = False
        config["charts"]["calibrated_threshold_bars"]["enabled"] = True
        config["charts"]["validation_rr_bars"]["enabled"] = True
        config["charts"]["calibration_protocol_bars"]["enabled"] = True
        config["output"]["directory"] = "analysis-output"
        write_analysis_config(config_path, config)

        write_analysis_charts(config_path)

        output = self.root / "analysis-output" / "figures"
        threshold_names = {
            f"pc_calibrated_rho0p95_bars_{TASK}.pdf",
            f"pc_calibrated_rho0p95_bars_{TASK}.json",
            f"validation_rr_rho0p95_bars_{TASK}.pdf",
            f"validation_rr_rho0p95_bars_{TASK}.json",
        }
        self.assertTrue(threshold_names <= {path.name for path in output.iterdir() if path.is_file()})
        calibrated_k = output / "calibration_protocols" / "calibrated_k"
        union = output / "calibration_protocols" / "union"
        self.assertEqual(threshold_names, {path.name for path in calibrated_k.iterdir()})
        self.assertEqual(
            {
                f"pc_calibrated_rho0p95_bars_{TASK}.pdf",
                f"pc_calibrated_rho0p95_bars_{TASK}.json",
            },
            {path.name for path in union.iterdir()},
        )
        sidecar = json.loads(
            (union / f"pc_calibrated_rho0p95_bars_{TASK}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sidecar["bars"][0]["pc"], 0.55)
        self.assertFalse(sidecar["safety_margins"])
        protocol_names = {
            path.name
            for path in (output / "calibration_protocols").rglob("*")
            if path.is_file()
        }
        self.assertFalse(any("margin" in name for name in protocol_names))


if __name__ == "__main__":
    unittest.main()
