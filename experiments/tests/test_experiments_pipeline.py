import csv
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.core.aggregate_resources import aggregate_resource_paths
from experiments.core.aggregate_runs import aggregate_paths
from experiments.core.config_schema import DatasetConfig
from experiments.core.eval_pipeline import cosine_cache, evaluate_task
from experiments.core.random_baseline import evaluate_random_task
from experiments.core.runner import ModelEvaluationError, run_experiment
from experiments.core.split_schema import (
    SplitImage,
    ids_for_multirole_selector,
    materialize_baseline_split,
    materialize_class_disjoint_role_split,
    resolve_retrieval_tasks,
    validate_split_shape,
    write_split,
)

# Class labels for the toy dataset built by ``make_dataset`` below.
_TOY_CLASS = {"h1": 1, "m1": 1, "h2": 2, "m2": 2}


def _toy_records(ids: list[str]) -> dict[str, SplitImage]:
    return {file_id: SplitImage(file_id, _TOY_CLASS[file_id]) for file_id in ids}


SCHEMA = """
CREATE TABLE classes(class_id INTEGER PRIMARY KEY, qid TEXT NOT NULL UNIQUE, label TEXT);
CREATE TABLE image_files(file_id TEXT PRIMARY KEY, file_ext TEXT NOT NULL, local_rel_path TEXT, download_status TEXT NOT NULL, source_role TEXT);
CREATE TABLE image_file_classes(image_file_class_id INTEGER PRIMARY KEY AUTOINCREMENT, file_id TEXT NOT NULL, class_id INTEGER NOT NULL, primary_label TEXT);
CREATE TABLE image_file_class_tags(image_file_class_id INTEGER NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(image_file_class_id, tag));
CREATE TABLE splits(split_name TEXT NOT NULL, file_id TEXT NOT NULL, PRIMARY KEY(split_name, file_id));
"""


class ExperimentPipelineTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> DatasetConfig:
        db_path = root / "dataset.db"
        image_root = root / "images"
        image_root.mkdir()
        conn = sqlite3.connect(db_path)
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO classes(class_id, qid, label) VALUES (?, ?, ?)", [(1, "Q1", "one"), (2, "Q2", "two")])
        rows = [
            ("h1", ".jpg", "images/h1.jpg", "downloaded", "historic"),
            ("m1", ".jpg", "images/m1.jpg", "downloaded", "modern"),
            ("h2", ".jpg", "images/h2.jpg", "downloaded", "historic"),
            ("m2", ".jpg", "images/m2.jpg", "downloaded", "modern"),
        ]
        conn.executemany("INSERT INTO image_files(file_id, file_ext, local_rel_path, download_status, source_role) VALUES (?, ?, ?, ?, ?)", rows)
        rels = [("h1", 1, "same_painting"), ("m1", 1, "same_painting"), ("h2", 2, "same_painting"), ("m2", 2, "same_painting")]
        conn.executemany("INSERT INTO image_file_classes(file_id, class_id, primary_label) VALUES (?, ?, ?)", rels)
        conn.executemany("INSERT INTO splits(split_name, file_id) VALUES (?, ?)", [("same_painting", r[0]) for r in rows])
        conn.commit()
        conn.close()
        return DatasetConfig(Path("dataset.yml"), "toy", db_path, image_root, {"expected_num_classes": 2, "expected_num_images": 4}, {})

    def make_role_dataset(self, root: Path, num_classes: int) -> DatasetConfig:
        db_path = root / "dataset.db"
        image_root = root / "images"
        image_root.mkdir()
        conn = sqlite3.connect(db_path)
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO classes(class_id, qid, label) VALUES (?, ?, ?)",
            [(i, f"Q{i}", f"c{i}") for i in range(1, num_classes + 1)],
        )
        rows: list[tuple[str, str, str, str, str]] = []
        rels: list[tuple[str, int, str]] = []
        for i in range(1, num_classes + 1):
            # One modern original + two historic ablations per class.
            for file_id, role in ((f"m{i}", "modern"), (f"h{i}a", "historic"), (f"h{i}b", "historic")):
                rows.append((file_id, ".jpg", f"images/{file_id}.jpg", "downloaded", role))
                rels.append((file_id, i, "same_painting"))
        conn.executemany(
            "INSERT INTO image_files(file_id, file_ext, local_rel_path, download_status, source_role) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.executemany("INSERT INTO image_file_classes(file_id, class_id, primary_label) VALUES (?, ?, ?)", rels)
        conn.executemany("INSERT INTO splits(split_name, file_id) VALUES (?, ?)", [("same_painting", r[0]) for r in rows])
        conn.commit()
        conn.close()
        return DatasetConfig(
            Path("dataset.yml"),
            "toy",
            db_path,
            image_root,
            {"expected_num_classes": num_classes, "expected_num_images": len(rows)},
            {},
        )

    def test_class_disjoint_role_split_directional_val_test_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_role_dataset(Path(tmp), num_classes=10)
            split = materialize_class_disjoint_role_split(dataset, "class_disjoint_role_v1")
            validate_split_shape(split)

            tasks = resolve_retrieval_tasks(split, [
                {
                    "task_id": "val_modern_to_historic",
                    "query": {"subset": "val", "role": "modern"},
                    "candidates": {"subset": "val", "role": "historic"},
                    "positive_policy": "same_painting",
                    "exclude_self": False,
                },
                {
                    "task_id": "test_modern_to_historic",
                    "query": {"subset": "test", "role": "modern"},
                    "candidates": {"subset": "test", "role": "historic"},
                    "positive_policy": "same_painting",
                    "exclude_self": False,
                },
            ])
            self.assertEqual({task["task_id"] for task in tasks}, {"val_modern_to_historic", "test_modern_to_historic"})
            for task in tasks:
                self.assertTrue(all(qid.startswith("m") for qid in task["query_ids"]))
                self.assertTrue(all(cid.startswith("h") for cid in task["candidate_ids"]))

            all_subset_task = resolve_retrieval_tasks(split, [
                {
                    "task_id": "all_modern_to_historic",
                    "query": {"subset": "ALL_SUBSETS", "role": "modern"},
                    "candidates": {"subset": "ALL_SUBSETS", "role": "historic"},
                    "positive_policy": "same_painting",
                    "exclude_self": False,
                }
            ])[0]
            self.assertEqual(len(all_subset_task["query_ids"]), 10)
            self.assertEqual(len(all_subset_task["candidate_ids"]), 20)
            self.assertTrue(all(qid.startswith("m") for qid in all_subset_task["query_ids"]))
            self.assertTrue(all(cid.startswith("h") for cid in all_subset_task["candidate_ids"]))

            subsets = split["subsets"]
            self.assertEqual(set(subsets), {"train", "val", "test"})
            subset_classes = []
            records = split["images"]
            for name in ("train", "val", "test"):
                ids = ids_for_multirole_selector(split, name, ["ALL_ROLES"])
                subset_classes.append({records[file_id]["class_id"] for file_id in ids})
            self.assertEqual(set.union(*subset_classes), set(range(1, 11)))
            for a, b in ((0, 1), (0, 2), (1, 2)):
                self.assertEqual(subset_classes[a] & subset_classes[b], set())

            train_ids = ids_for_multirole_selector(split, "train", ["ALL_ROLES"])
            self.assertTrue(any(fid.startswith("m") for fid in train_ids))
            self.assertTrue(any(fid.startswith("h") for fid in train_ids))

    def test_class_disjoint_role_split_accepts_custom_train_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_role_dataset(Path(tmp), num_classes=10)
            split = materialize_class_disjoint_role_split(
                dataset,
                "class_disjoint_role_v1",
                train_share=0.4,
            )

            records = split["images"]
            class_counts = {
                name: len({
                    records[file_id]["class_id"]
                    for file_id in ids_for_multirole_selector(split, name, ["ALL_ROLES"])
                })
                for name in split["subsets"]
            }
            self.assertEqual(class_counts, {"train": 4, "val": 3, "test": 3})
            self.assertEqual(
                split["split_strategy"]["ratios"],
                {"train": 0.4, "val": 0.3, "test": 0.3},
            )

    def test_zero_train_share_creates_calibration_test_only_role_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_role_dataset(Path(tmp), num_classes=10)
            split = materialize_class_disjoint_role_split(
                dataset,
                "class_disjoint_role_v1",
                train_share=0.0,
            )
            validate_split_shape(split)

            self.assertEqual(set(split["subsets"]), {"val", "test"})
            for name in ("val", "test"):
                ids = ids_for_multirole_selector(split, name, ["ALL_ROLES"])
                self.assertEqual(len(ids), 15)
                self.assertTrue(any(file_id.startswith("m") for file_id in ids))
                self.assertTrue(any(file_id.startswith("h") for file_id in ids))

    def test_zero_train_and_validation_shares_create_test_only_role_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_role_dataset(Path(tmp), num_classes=10)
            split = materialize_class_disjoint_role_split(
                dataset,
                "class_disjoint_role_v1",
                train_share=0.0,
                validation_share=0.0,
            )
            validate_split_shape(split)

            self.assertEqual(set(split["subsets"]), {"test"})
            test_ids = ids_for_multirole_selector(split, "test", ["ALL_ROLES"])
            self.assertEqual(len(test_ids), 30)
            self.assertEqual(
                split["split_strategy"]["ratios"],
                {"train": 0.0, "val": 0.0, "test": 1.0},
            )
            self.assertIn("_train_0_val_0_test_1_", split["split_id"])

    def test_class_disjoint_role_split_rejects_invalid_shares(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_role_dataset(Path(tmp), num_classes=10)
            for train_share in (-0.1, 1.0):
                with self.subTest(train_share=train_share):
                    with self.assertRaisesRegex(ValueError, "train_share"):
                        materialize_class_disjoint_role_split(
                            dataset,
                            "class_disjoint_role_v1",
                            train_share=train_share,
                        )
            for validation_share in (-0.1, 1.0):
                with self.subTest(validation_share=validation_share):
                    with self.assertRaisesRegex(ValueError, "validation_share"):
                        materialize_class_disjoint_role_split(
                            dataset,
                            "class_disjoint_role_v1",
                            train_share=0.0,
                            validation_share=validation_share,
                        )
            with self.assertRaisesRegex(ValueError, r"train_share \+ validation_share"):
                materialize_class_disjoint_role_split(
                    dataset,
                    "class_disjoint_role_v1",
                    train_share=0.6,
                    validation_share=0.4,
                )

    def test_materialize_baseline_split_uses_ids_not_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_dataset(Path(tmp))
            split = materialize_baseline_split(dataset, "baseline_full")
            validate_split_shape(split)
            self.assertEqual(split["dataset"]["dataset_id"], "toy")
            roles = split["subsets"]["eval"]["roles"]
            self.assertEqual(roles["modern"], ["m1.jpg", "m2.jpg"])
            self.assertEqual(roles["historic"], ["h1.jpg", "h2.jpg"])
            self.assertEqual(split["coverage_filter"]["excluded_num_classes"], 0)
            # IDs are bare filenames; the DB-relative "images/" prefix never leaks in.
            self.assertNotIn("images/h1.jpg", json.dumps(split))
            self.assertEqual(split["images"], {"h1.jpg": {"class_id": 1}, "m1.jpg": {"class_id": 1}, "h2.jpg": {"class_id": 2}, "m2.jpg": {"class_id": 2}})

    def test_evaluate_task_with_explicit_ids(self):
        image_ids = ["h1", "h2", "m1", "m2"]
        records = _toy_records(image_ids)
        embeddings = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        sims = cosine_cache(embeddings, "cpu", "float32", "float32")
        task = {"task_id": "all", "query_ids": image_ids, "candidate_ids": image_ids, "exclude_self": True, "positive_policy": "same_painting"}
        result, curves = evaluate_task(task, image_ids, records, sims, [1], True, [1.0])
        row = result["metrics"][0]
        target_row = result["target_pc_metrics"][0]
        self.assertEqual(result["num_positive_pairs"], 4)
        self.assertEqual(result["num_possible_pairs"], 12)
        self.assertEqual(result["num_excluded_pairs"], 4)
        self.assertEqual(row["true_positives"], 4)
        self.assertAlmostEqual(row["precision"], 1.0)
        self.assertAlmostEqual(row["query_coverage"], 1.0)
        self.assertEqual(len(result["metrics"]), 1)
        self.assertAlmostEqual(target_row["pair_completeness"], 1.0)
        self.assertAlmostEqual(target_row["pair_quality"], 1.0)
        self.assertIn("histogram", curves)

    def test_random_baseline_uses_split_label_counts_without_embeddings(self):
        image_ids = ["h1", "h2", "m1", "m2"]
        records = _toy_records(image_ids)
        task = {
            "task_id": "all",
            "query_ids": image_ids,
            "candidate_ids": image_ids,
            "exclude_self": True,
            "positive_policy": "same_painting",
        }

        targets = [0.95, 0.99, 0.995]
        result, curves = evaluate_random_task(task, records, [1], True, targets)
        row = result["metrics"][0]
        target_rows = result["target_pc_metrics"]

        self.assertEqual(result["num_positive_pairs"], 4)
        self.assertEqual(result["num_possible_pairs"], 12)
        self.assertEqual(result["num_excluded_pairs"], 4)
        self.assertEqual(row["candidate_pairs"], 4)
        self.assertAlmostEqual(row["true_positives"], 4 / 3)
        self.assertAlmostEqual(row["precision"], 1 / 3)
        self.assertAlmostEqual(row["recall"], 1 / 3)
        self.assertAlmostEqual(row["pair_quality"], 1 / 3)
        self.assertAlmostEqual(row["pair_completeness"], 1 / 3)
        self.assertAlmostEqual(row["reduction_ratio"], 2 / 3)
        self.assertAlmostEqual(row["query_coverage"], 1.0)
        self.assertEqual(
            [target_row["target_pair_completeness"] for target_row in target_rows],
            targets,
        )
        self.assertEqual(len(target_rows), 3)
        self.assertTrue(all(target_row["pair_quality"] == 1 / 3 for target_row in target_rows))
        self.assertTrue(all(target_row["threshold"] is None for target_row in target_rows))
        self.assertEqual(curves["precision_recall"], [])

    def test_possible_pairs_exposes_self_exclusion_after_dropping_empty_queries(self):
        image_ids = ["h1", "h2", "m1"]
        records = _toy_records(image_ids)
        embeddings = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        sims = cosine_cache(embeddings, "cpu", "float32", "float32")
        task = {
            "task_id": "self_overlap",
            "query_ids": ["h1", "h2"],
            "candidate_ids": ["h1", "h2", "m1"],
            "exclude_self": True,
            "positive_policy": "same_painting",
        }

        result, _ = evaluate_task(task, image_ids, records, sims, [1], True, [1.0])
        random_result, _ = evaluate_random_task(task, records, [1], True, [1.0])

        for task_result in (result, random_result):
            self.assertEqual(task_result["num_original_queries"], 2)
            self.assertEqual(task_result["num_queries"], 1)
            self.assertEqual(task_result["num_queries_dropped_without_positives"], 1)
            self.assertEqual(task_result["num_candidates"], 3)
            self.assertEqual(task_result["num_possible_pairs"], 2)
            self.assertEqual(task_result["num_excluded_pairs"], 1)
            self.assertEqual(task_result["num_positive_pairs"], 1)

    def test_standalone_calibration_run_reports_all_canonical_pc_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.make_role_dataset(root, num_classes=10)
            split_json = root / "split.json"
            model_yml = root / "models.yml"
            experiment_yml = root / "experiment.yml"
            write_split(
                split_json,
                materialize_class_disjoint_role_split(
                    dataset,
                    "class_disjoint_role_v1",
                    train_share=0.0,
                ),
            )
            model_yml.write_text(
                "schema_version: 1\n"
                "adapter_name: siglip2_adapter\n"
                "models:\n"
                "  - model_id: fake/calibrated\n"
                "    display_name: Calibrated fake\n",
                encoding="utf-8",
            )
            experiment_yml.write_text(
                "schema_version: 1\n"
                "experiment_id: standalone_calibration_targets\n"
                "dataset:\n"
                f"  path: {root}\n"
                "split:\n"
                f"  file: {split_json}\n"
                "models:\n"
                "  include:\n"
                f"    - config: {model_yml}\n"
                "      model_id: fake/calibrated\n"
                "evaluation:\n"
                "  similarity: cosine\n"
                "  top_k: [1]\n"
                "  target_pc: [0.95, 0.99, 0.995]\n"
                "  threshold_calibration:\n"
                "    method: empirical_target_pc\n"
                "    calibration_id: toy_val\n"
                "    target_pc: [0.95, 0.99, 0.995]\n"
                "    task:\n"
                "      task_id: calibration_val_modern_to_historic\n"
                "      query: {subset: val, role: modern}\n"
                "      candidates: {subset: val, role: historic}\n"
                "      positive_policy: same_painting\n"
                "      exclude_self: false\n"
                "  retrieval_tasks:\n"
                "    - task_id: test_modern_to_historic\n"
                "      query: {subset: test, role: modern}\n"
                "      candidates: {subset: test, role: historic}\n"
                "      positive_policy: same_painting\n"
                "      exclude_self: false\n"
                "run:\n"
                f"  output_dir: {root / 'runs'}\n"
                "  fail_on_empty_positives: true\n",
                encoding="utf-8",
            )

            def fake_prepare_embeddings(_dataset, model, image_ids, records, model_dir, *_args):
                embeddings = np.zeros((len(image_ids), 10), dtype=np.float32)
                for row, file_id in enumerate(image_ids):
                    embeddings[row, int(records[file_id].class_id) - 1] = 1.0
                np.save(model_dir / "embeddings.npy", embeddings)
                return embeddings, {
                    "normalized": True,
                    "embedding_sha256": f"sha-{model.model_id}",
                    "resources": {
                        "embedding": {"wall_time_seconds": 0.0},
                        "model": {},
                    },
                }

            with mock.patch(
                "experiments.core.runner.prepare_embeddings",
                side_effect=fake_prepare_embeddings,
            ):
                run_dir = run_experiment(experiment_yml, run_id="canonical-targets")

            model_dir = next((run_dir / "models").iterdir())
            eval_data = json.loads((model_dir / "eval.json").read_text(encoding="utf-8"))
            calibration = json.loads(
                (model_dir / "threshold_calibration.json").read_text(encoding="utf-8")
            )
            protocol = json.loads(
                (run_dir / "calibration_protocol.json").read_text(encoding="utf-8")
            )
            aggregate = json.loads(
                (run_dir / "aggregate_metrics.json").read_text(encoding="utf-8")
            )["rows"]
            with (run_dir / "aggregate_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                aggregate_csv = list(csv.DictReader(handle))
            table = (run_dir / "latex_tables" / "calibrated_threshold.tex").read_text(
                encoding="utf-8"
            )
            targets = [0.95, 0.99, 0.995]

            self.assertEqual(protocol["target_pc"], targets)
            self.assertTrue(protocol["passed"])
            self.assertEqual(eval_data["evaluation"]["target_pc"], targets)
            self.assertEqual(
                eval_data["evaluation"]["threshold_calibration"]["target_pc"],
                targets,
            )
            self.assertEqual(
                [row["target_pair_completeness"] for row in calibration["operating_points"]],
                targets,
            )
            task = eval_data["tasks"][0]
            self.assertEqual(
                [row["target_pair_completeness"] for row in task["target_pc_metrics"]],
                targets,
            )
            self.assertEqual(
                [
                    row["calibration_target_pair_completeness"]
                    for row in task["calibrated_threshold_metrics"]
                ],
                targets,
            )
            calibrated_rows = [
                row for row in aggregate if row["selection_mode"] == "calibrated_threshold"
            ]
            self.assertEqual(
                [row["calibration_target_pair_completeness"] for row in calibrated_rows],
                targets,
            )
            calibrated_csv = [
                row
                for row in aggregate_csv
                if row["selection_mode"] == "calibrated_threshold"
            ]
            self.assertEqual(
                [float(row["calibration_target_pair_completeness"]) for row in calibrated_csv],
                targets,
            )
            self.assertIn("$PC \\geq 0.95$", table)
            self.assertIn("$PC \\geq 0.99$", table)
            self.assertIn("$PC \\geq 0.995$", table)

            # Figure-5 data is logged at eval time into curves.json (never
            # reconstructed post-hoc from the similarity cache).
            curves = json.loads((model_dir / "curves.json").read_text(encoding="utf-8"))
            sizes_rows = curves["tasks"][0]["calibrated_candidate_sizes"]
            self.assertEqual(
                [row["calibration_target_pair_completeness"] for row in sizes_rows],
                targets,
            )
            num_queries = eval_data["tasks"][0]["num_queries"]
            for row in sizes_rows:
                self.assertEqual(len(row["candidates_per_query"]), num_queries)
                self.assertGreaterEqual(row["query_coverage"], 0.0)
            # Per-query arrays must never leak into the flat aggregate CSV.
            self.assertTrue(all("candidates_per_query" not in row for row in aggregate_csv[:1]))

            # The default chart bundle runs with every validation run.
            self.assertTrue((run_dir / "latex_tables" / "blocking_fixed_k.tex").is_file())
            self.assertTrue((run_dir / "latex_tables" / "blocking_calibrated.tex").is_file())
            task_tag = "test_modern_to_historic"
            if importlib.util.find_spec("matplotlib"):
                self.assertFalse((run_dir / "charts_warning.json").is_file())
                figures = run_dir / "figures"
                self.assertTrue((figures / f"pc_at_k_{task_tag}.pdf").is_file())
                self.assertTrue((figures / f"pc_rr_tradeoff_{task_tag}.pdf").is_file())
                self.assertTrue((figures / f"similarity_distributions_{task_tag}.pdf").is_file())
                self.assertTrue((figures / f"candidate_sizes_rho0p995_{task_tag}.pdf").is_file())
                self.assertTrue((figures / f"pq_pc_{task_tag}.pdf").is_file())
                self.assertTrue((run_dir / "bar_charts" / f"ranked_pc_at_1_{task_tag}.svg").is_file())
            else:
                self.assertTrue((run_dir / "charts_warning.json").is_file())

    def test_run_experiment_can_compare_frozen_and_checkpoint_models_in_one_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.make_dataset(root)
            dataset_yml = root / "dataset.yml"
            split_json = root / "split.json"
            frozen_yml = root / "frozen.yml"
            finetuned_yml = root / "finetuned.yml"
            experiment_yml = root / "experiment.yml"
            dataset_yml.write_text(
                "schema_version: 1\n"
                "dataset_id: toy\n"
                "paths:\n"
                f"  dataset_db: {dataset.dataset_db}\n"
                f"  image_root: {dataset.image_root}\n"
                "identity:\n"
                "  expected_num_classes: 2\n"
                "  expected_num_images: 4\n",
                encoding="utf-8",
            )
            write_split(split_json, materialize_baseline_split(dataset, "baseline_full"))
            frozen_yml.write_text(
                "schema_version: 1\n"
                "adapter_name: siglip2_adapter\n"
                "models:\n"
                "  - model_id: fake/frozen\n"
                "    display_name: Frozen fake\n",
                encoding="utf-8",
            )
            finetuned_yml.write_text(
                "schema_version: 1\n"
                "adapter_name: resnet50_projection_adapter\n"
                "models:\n"
                "  - model_id: local/finetuned\n"
                "    display_name: Finetuned fake\n"
                "    adapter_kwargs:\n"
                "      checkpoint_path_env: SMARTMATCH_TEST_CHECKPOINT\n",
                encoding="utf-8",
            )
            experiment_yml.write_text(
                "schema_version: 1\n"
                "experiment_id: compare_frozen_checkpoint\n"
                "dataset:\n"
                f"  path: {root}\n"
                "split:\n"
                f"  file: {split_json}\n"
                "models:\n"
                "  include:\n"
                f"    - config: {frozen_yml}\n"
                "      model_id: fake/frozen\n"
                f"    - config: {finetuned_yml}\n"
                "      model_id: local/finetuned\n"
                "evaluation:\n"
                "  similarity: cosine\n"
                "  top_k: [1]\n"
                "  target_pc: [1.0]\n"
                "  retrieval_tasks:\n"
                "    - task_id: modern_to_historic\n"
                "      query: {subset: eval, role: modern}\n"
                "      candidates: {subset: eval, role: historic}\n"
                "      positive_policy: same_painting\n"
                "      exclude_self: false\n"
                "run:\n"
                f"  output_dir: {root / 'runs'}\n"
                "  fail_on_empty_positives: true\n",
                encoding="utf-8",
            )

            def fake_prepare_embeddings(_dataset, model, image_ids, _records, model_dir, *_args):
                vectors = {
                    "h1.jpg": [1.0, 0.0],
                    "m1.jpg": [1.0, 0.0],
                    "h2.jpg": [0.0, 1.0],
                    "m2.jpg": [0.0, 1.0],
                }
                embeddings = np.asarray([vectors[file_id] for file_id in image_ids], dtype=np.float32)
                np.save(model_dir / "embeddings.npy", embeddings)
                return embeddings, {
                    "normalized": True,
                    "embedding_sha256": f"sha-{model.model_id}",
                    "resources": {"embedding": {"wall_time_seconds": 0.0}, "model": {}},
                }

            with mock.patch(
                "experiments.core.runner.prepare_embeddings",
                side_effect=fake_prepare_embeddings,
            ):
                run_dir = run_experiment(experiment_yml, run_id="compare")

            rows = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))["rows"]
            model_ids = {row["model_id"] for row in rows}
            tables = (run_dir / "latex_tables" / "tables.tex").read_text(encoding="utf-8")
            table_values = (run_dir / "latex_tables" / "tables_values.tex").read_text(encoding="utf-8")
            self.assertEqual(model_ids, {"fake/frozen", "local/finetuned"})
            self.assertIn("\\input{latex_tables/tables_values.tex}", tables)
            self.assertIn("Frozen fake", table_values)
            self.assertIn("Finetuned fake", table_values)
            self.assertIn("PQ", tables)
            self.assertIn("PC", tables)
            self.assertIn("RR", tables)

    def test_run_experiment_fails_run_and_retains_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.make_dataset(root)
            dataset_yml = root / "dataset.yml"
            split_json = root / "split.json"
            model_yml = root / "models.yml"
            experiment_yml = root / "experiment.yml"
            dataset_yml.write_text(
                "schema_version: 1\n"
                "dataset_id: toy\n"
                "paths:\n"
                f"  dataset_db: {dataset.dataset_db}\n"
                f"  image_root: {dataset.image_root}\n"
                "identity:\n"
                "  expected_num_classes: 2\n"
                "  expected_num_images: 4\n",
                encoding="utf-8",
            )
            write_split(split_json, materialize_baseline_split(dataset, "baseline_full"))
            model_yml.write_text(
                "schema_version: 1\n"
                "adapter_name: siglip2_adapter\n"
                "models:\n"
                "  - model_id: fake/good\n"
                "    display_name: Good fake\n"
                "  - model_id: fake/failing\n"
                "    display_name: Failing fake\n",
                encoding="utf-8",
            )
            experiment_yml.write_text(
                "schema_version: 1\n"
                "experiment_id: skip_failed_model\n"
                "dataset:\n"
                f"  path: {root}\n"
                "split:\n"
                f"  file: {split_json}\n"
                "models:\n"
                "  include:\n"
                f"    - config: {model_yml}\n"
                "      model_id: fake/good\n"
                f"    - config: {model_yml}\n"
                "      model_id: fake/failing\n"
                "evaluation:\n"
                "  similarity: cosine\n"
                "  top_k: [1]\n"
                "  target_pc: [1.0]\n"
                "  retrieval_tasks:\n"
                "    - task_id: modern_to_historic\n"
                "      query: {subset: eval, role: modern}\n"
                "      candidates: {subset: eval, role: historic}\n"
                "      positive_policy: same_painting\n"
                "      exclude_self: false\n"
                "run:\n"
                f"  output_dir: {root / 'runs'}\n"
                "  fail_on_empty_positives: true\n",
                encoding="utf-8",
            )

            def fake_prepare_embeddings(_dataset, model, image_ids, _records, model_dir, *_args):
                if model.model_id == "fake/failing":
                    raise RuntimeError("synthetic model crash")
                vectors = {
                    "h1.jpg": [1.0, 0.0],
                    "m1.jpg": [1.0, 0.0],
                    "h2.jpg": [0.0, 1.0],
                    "m2.jpg": [0.0, 1.0],
                }
                embeddings = np.asarray([vectors[file_id] for file_id in image_ids], dtype=np.float32)
                np.save(model_dir / "embeddings.npy", embeddings)
                return embeddings, {
                    "normalized": True,
                    "embedding_sha256": f"sha-{model.model_id}",
                    "resources": {"embedding": {"wall_time_seconds": 0.0}, "model": {}},
                }

            with mock.patch(
                "experiments.core.runner.prepare_embeddings", side_effect=fake_prepare_embeddings
            ):
                with self.assertRaises(ModelEvaluationError) as raised:
                    run_experiment(experiment_yml, run_id="failed")

            run_dir = raised.exception.run_dir
            failure_text = (run_dir / "MODEL_FAILURES.txt").read_text(encoding="utf-8")
            rows = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))["rows"]
            run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

            self.assertIn("fake/failing", failure_text)
            self.assertIn("synthetic model crash", failure_text)
            self.assertTrue((run_dir / "aggregate_metrics.csv").is_file())
            self.assertTrue((run_dir / "latex_tables" / "tables.tex").is_file())
            self.assertEqual({row["model_id"] for row in rows}, {"fake/good"})
            self.assertEqual(run_data["status"], "failed")
            statuses = {entry["model_id"]: entry["status"] for entry in run_data["model_runs"]}
            self.assertEqual(statuses["fake/good"], "completed")
            self.assertEqual(statuses["fake/failing"], "failed")
            self.assertEqual(run_data["artifacts"]["model_failures"], "MODEL_FAILURES.txt")

    def test_run_experiment_can_write_random_baseline_without_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.make_dataset(root)
            dataset_yml = root / "dataset.yml"
            split_json = root / "split.json"
            model_yml = root / "model.yml"
            experiment_yml = root / "experiment.yml"
            dataset_yml.write_text(
                "schema_version: 1\n"
                "dataset_id: toy\n"
                "paths:\n"
                f"  dataset_db: {dataset.dataset_db}\n"
                f"  image_root: {dataset.image_root}\n"
                "identity:\n"
                "  expected_num_classes: 2\n"
                "  expected_num_images: 4\n",
                encoding="utf-8",
            )
            write_split(split_json, materialize_baseline_split(dataset, "baseline_full"))
            model_yml.write_text(
                "schema_version: 1\n"
                "adapter_name: siglip2_adapter\n"
                "models:\n"
                "  - model_id: fake/model\n",
                encoding="utf-8",
            )
            experiment_yml.write_text(
                "schema_version: 1\n"
                "experiment_id: random_only\n"
                "dataset:\n"
                f"  path: {root}\n"
                "split:\n"
                f"  file: {split_json}\n"
                "models:\n"
                "  include:\n"
                f"    - config: {model_yml}\n"
                "      model_id: fake/model\n"
                "evaluation:\n"
                "  similarity: cosine\n"
                "  random_baseline: true\n"
                "  top_k: [1]\n"
                "  target_pc: [1.0]\n"
                "  retrieval_tasks:\n"
                "    - task_id: modern_to_historic\n"
                "      query: {subset: eval, role: modern}\n"
                "      candidates: {subset: eval, role: historic}\n"
                "      positive_policy: same_painting\n"
                "      exclude_self: false\n"
                "run:\n"
                f"  output_dir: {root / 'runs'}\n"
                "  fail_on_empty_positives: true\n",
                encoding="utf-8",
            )

            run_dir = run_experiment(
                experiment_yml,
                model_filter="random_selection_expected",
                run_id="random-only",
            )
            model_dir = run_dir / "models" / "random_selection_expected"
            eval_data = json.loads((model_dir / "eval.json").read_text(encoding="utf-8"))
            run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

            self.assertTrue(eval_data["embedding"]["skipped"])
            self.assertEqual(eval_data["baseline"]["type"], "random_selection_expected")
            self.assertFalse((model_dir / "embeddings.npy").exists())
            self.assertFalse((model_dir / "resource_metrics.json").exists())
            self.assertEqual(run_data["model_runs"][0]["kind"], "analytical_baseline")

    def test_aggregate_eval_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            model_dir = run_dir / "models" / "model_a"
            model_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps({"run_id": "run1", "experiment_id": "exp", "status": "completed", "dataset": {"dataset_id": "toy"}, "split": {"split_id": "split"}}))
            (model_dir / "eval.json").write_text(
                json.dumps(
                    {
                        "model_id": "model_a",
                        "model_revision": "abc123",
                        "tasks": [
                            {
                                "task_id": "all",
                                "num_queries": 2,
                                "num_candidates": 2,
                                "num_possible_pairs": 3,
                                "num_excluded_pairs": 1,
                                "num_positive_pairs": 2,
                                "metrics": [{"k": 1, "precision": 1.0}],
                                "target_pc_metrics": [
                                    {"target_pair_completeness": 1.0, "pair_completeness": 1.0}
                                ],
                                "calibrated_threshold_metrics": [
                                    {
                                        "calibration_target_pair_completeness": target,
                                        "threshold": threshold,
                                        "pair_completeness": achieved,
                                    }
                                    for target, threshold, achieved in (
                                        (0.95, 0.7, 0.94),
                                        (0.99, 0.6, 0.97),
                                        (0.995, 0.5, 0.96),
                                    )
                                ],
                            }
                        ],
                    }
                )
            )
            rows = aggregate_paths([run_dir])
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["experiment_id"], "exp")
            self.assertEqual(rows[0]["model_id"], "model_a")
            self.assertEqual(rows[0]["model_revision"], "abc123")
            self.assertEqual(rows[0]["num_possible_pairs"], 3)
            self.assertEqual(rows[0]["num_excluded_pairs"], 1)
            self.assertEqual(
                {row["selection_mode"] for row in rows},
                {"top_k", "target_pc", "calibrated_threshold"},
            )
            calibrated = [
                row for row in rows if row["selection_mode"] == "calibrated_threshold"
            ]
            self.assertEqual(
                [row["calibration_target_pair_completeness"] for row in calibrated],
                [0.95, 0.99, 0.995],
            )
            self.assertEqual(
                [row["pair_completeness"] for row in calibrated],
                [0.94, 0.97, 0.96],
            )

    def test_aggregate_filters_incomplete_runs_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_eval_run(root, "complete", status="completed", model_id="model_a")
            self._write_eval_run(root, "running", status="running", model_id="model_b")
            json_output = root / "aggregate.json"

            rows = aggregate_paths([root], json_output=json_output)
            payload = json.loads(json_output.read_text(encoding="utf-8"))

            self.assertEqual({row["run_id"] for row in rows}, {"complete"})
            self.assertEqual(payload["provenance"]["discovered_eval_files"], 2)
            self.assertEqual(payload["provenance"]["skipped_runs"][0]["reason"], "run_status_not_completed")
            self.assertEqual(
                {row["run_id"] for row in aggregate_paths([root], include_incomplete=True)},
                {"complete", "running"},
            )

    def test_aggregate_duplicate_task_keys_fail_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_eval_run(root, "run1", status="completed", model_id="model_a")
            self._write_eval_run(root, "run2", status="completed", model_id="model_a")

            with self.assertRaisesRegex(ValueError, "Duplicate aggregate task keys"):
                aggregate_paths([root])

            rows = aggregate_paths([root], allow_duplicates=True)
            self.assertEqual(len(rows), 2)

    def test_aggregate_manifest_filters_approved_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_eval_run(root, "run1", status="completed", model_id="model_a")
            self._write_eval_run(root, "run2", status="completed", model_id="model_b")
            manifest = root / "approved.yml"
            manifest.write_text("schema_version: 1\napproved_runs:\n  - run_dir: run1\n", encoding="utf-8")
            json_output = root / "aggregate.json"

            rows = aggregate_paths([root], json_output=json_output, approved_run_manifest=manifest)
            payload = json.loads(json_output.read_text(encoding="utf-8"))

            self.assertEqual({row["run_id"] for row in rows}, {"run1"})
            self.assertEqual(payload["provenance"]["approved_run_manifest"]["path"], str(manifest.resolve()))
            self.assertEqual(payload["provenance"]["skipped_runs"][0]["reason"], "not_in_approved_manifest")

    def _write_eval_run(
        self,
        root: Path,
        run_id: str,
        *,
        status: str,
        model_id: str,
        split_id: str = "split",
        task_id: str = "all",
    ) -> Path:
        run_dir = root / run_id
        model_dir = run_dir / "models" / model_id
        model_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "experiment_id": "exp",
                    "status": status,
                    "dataset": {"dataset_id": "toy"},
                    "split": {"split_id": split_id},
                    "model_runs": [{"model_id": model_id, "model_storage_key": model_id, "model_revision": "abc123"}],
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "eval.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "model_storage_key": model_id,
                    "model_revision": "abc123",
                    "tasks": [
                        {
                            "task_id": task_id,
                            "num_queries": 2,
                            "num_candidates": 2,
                            "num_possible_pairs": 3,
                            "num_excluded_pairs": 1,
                            "num_positive_pairs": 2,
                            "metrics": [{"k": 1, "precision": 1.0}],
                            "target_pc_metrics": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_aggregate_resource_rows_filters_cached_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            resource_dir = run_dir / "resource_metrics"
            resource_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps({"run_id": "run1", "experiment_id": "exp", "dataset": {"dataset_id": "toy"}, "split": {"split_id": "split"}}))
            base = {
                "experiment_id": "exp",
                "dataset_id": "toy",
                "model_id": "model_a",
                "model_storage_key": "model_a_key",
                "summary": {"embedding_dimensionality": 3, "parameter_count": 123, "inference_engine": "fake"},
                "model": {},
                "system": {"cpu_cores": 8, "gpu": [{"name": "Dedicated GPU"}], "memory_total_bytes": 1024},
                "embedding": {
                    "num_images": 10,
                    "wall_time_seconds": 2.0,
                    "time_per_100_images_seconds": 20.0,
                    "gpu_memory": {"nvidia_smi_peak_used_bytes": 100},
                },
                "retrieval": {
                    "num_queries": 5,
                    "wall_time_seconds": 1.0,
                    "similarity_time_seconds": 0.7,
                    "evaluation_time_seconds": 0.3,
                    "latency_per_query_seconds": 0.2,
                    "gpu_memory": {"nvidia_smi_peak_used_bytes": 200},
                },
            }
            uncached = dict(base)
            uncached["embedding"] = {**base["embedding"], "cache_reused_for_run": False}
            cached = {**base, "model_id": "model_b", "model_storage_key": "model_b_key"}
            cached["embedding"] = {**base["embedding"], "cache_reused_for_run": True}
            (resource_dir / "model_a.json").write_text(json.dumps(uncached))
            (resource_dir / "model_b.json").write_text(json.dumps(cached))

            rows = aggregate_resource_paths([run_dir])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model_id"], "model_a")
            self.assertEqual(rows[0]["embedding_cache_reused_for_run"], False)
            self.assertEqual(rows[0]["embedding_images_per_second"], 5.0)
            self.assertEqual(rows[0]["gpu_memory_peak_used_bytes"], 200)
            self.assertEqual(len(aggregate_resource_paths([run_dir], include_cached=True)), 2)


if __name__ == "__main__":
    unittest.main()
