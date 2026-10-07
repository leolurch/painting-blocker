import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from experiments.core.cross_dataset_threshold_transfer import (
    EXPERIMENT_TYPE,
    load_cross_dataset_experiment,
    validate_cross_dataset_experiment,
)
from experiments.core.finetuning.post_evaluation import checkpoint_model_config
from experiments.core.runner import run_experiment


def _split(
    dataset_id: str,
    image_root: Path,
    *,
    split_id: str,
    classes: dict[str, int],
) -> dict:
    return {
        "schema_version": 2,
        "split_id": split_id,
        "dataset": {
            "dataset_id": dataset_id,
            "image_root": str(image_root),
            "source_db_sha256": f"sha-{dataset_id}",
            "identity": {
                "num_classes": len(set(classes.values())),
                "num_images": len(classes),
            },
        },
        "images": {
            file_id: {"class_id": class_id}
            for file_id, class_id in classes.items()
        },
        "subsets": {
            "test": {
                "roles": {
                    "query": ["q1", "q2"],
                    "candidate": ["c1", "c2"],
                }
            }
        },
        "split_strategy": {"type": "test_only", "seed": 42},
    }


class CrossDatasetThresholdTransferTest(unittest.TestCase):
    def _write_experiment(self, root: Path, *, evaluation_target_pc=None) -> Path:
        calibration_root = root / "calibration_dataset"
        evaluation_root = root / "evaluation_dataset"
        (calibration_root / "images").mkdir(parents=True)
        (evaluation_root / "images").mkdir(parents=True)
        calibration_split = root / "calibration_split.json"
        evaluation_split = root / "evaluation_split.json"
        # Bare IDs and numeric labels intentionally overlap. Dataset identity is
        # the namespace boundary; the runner must never combine these records.
        calibration_split.write_text(
            json.dumps(
                _split(
                    "toy-calibration",
                    calibration_root / "images",
                    split_id="toy-calibration-test",
                    classes={"q1": 101, "q2": 102, "c1": 101, "c2": 102},
                )
            ),
            encoding="utf-8",
        )
        evaluation_split.write_text(
            json.dumps(
                _split(
                    "toy-evaluation",
                    evaluation_root / "images",
                    split_id="toy-evaluation-test-only",
                    classes={"q1": 101, "q2": 102, "c1": 101, "c2": 102},
                )
            ),
            encoding="utf-8",
        )
        model_yml = root / "model.yml"
        model_yml.write_text(
            "schema_version: 1\n"
            "adapter_name: siglip2_adapter\n"
            "models:\n"
            "  - model_id: fake/cross-dataset\n"
            "    display_name: Cross-dataset fake\n",
            encoding="utf-8",
        )
        evaluation = {
            "similarity": "cosine",
            "top_k": [1],
            "cache_dtype": "float32",
            "device": "cpu",
            "gpu_dtype": "float32",
            "fp32_matmul_precision": "ieee",
            "retrieval_tasks": [
                {
                    "task_id": "evaluation_query_to_candidate",
                    "query": {"subset": "test", "role": "query"},
                    "candidates": {"subset": "test", "role": "candidate"},
                    "positive_policy": "same_painting",
                    "exclude_self": True,
                }
            ],
        }
        if evaluation_target_pc is not None:
            evaluation["target_pc"] = evaluation_target_pc
        experiment = {
            "schema_version": 1,
            "experiment_type": EXPERIMENT_TYPE,
            "experiment_id": "toy_cross_dataset_transfer",
            "dataset": {"path": str(evaluation_root)},
            "split": {"file": str(evaluation_split)},
            "models": {
                "include": [
                    {"config": str(model_yml), "model_id": "fake/cross-dataset"}
                ]
            },
            "calibration": {
                "dataset": {"path": str(calibration_root)},
                "split": {"file": str(calibration_split)},
                "method": "empirical_target_pc",
                "calibration_id": "toy_calibration_test",
                "target_pc": [1.0],
                "task": {
                    "task_id": "calibration_query_to_candidate",
                    "query": {"subset": "test", "role": "query"},
                    "candidates": {"subset": "test", "role": "candidate"},
                    "positive_policy": "same_painting",
                    "exclude_self": True,
                },
            },
            "embedding": {
                "cache_policy": "reuse_if_config_hash_matches",
                "cache_dir": str(root / "cache"),
                "batch_size": 4,
            },
            "evaluation": evaluation,
            "run": {
                "output_dir": str(root / "runs"),
                "save_similarity_cache": False,
                "fail_on_empty_positives": True,
            },
        }
        experiment_yml = root / "experiment.yml"
        experiment_yml.write_text(
            yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8"
        )
        return experiment_yml

    def test_dynamic_checkpoint_injection_uses_the_separate_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._write_experiment(root)
            raw = yaml.safe_load(experiment.read_text(encoding="utf-8"))
            raw["models"] = {"from_finetuning_checkpoint": True}
            experiment.write_text(
                yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
            )
            checkpoint = root / "best_checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            model = checkpoint_model_config(
                checkpoint, model_id="local/cross-dataset/best"
            )
            context = load_cross_dataset_experiment(
                experiment, injected_models=[model]
            )
            self.assertEqual(
                [entry.model_id for entry in context.experiment.models],
                ["local/cross-dataset/best"],
            )
            self.assertEqual(
                context.experiment.resolved["calibration"]["split"]["file"],
                str((root / "calibration_split.json").resolve()),
            )

    def test_separate_type_rejects_evaluation_oracle_threshold_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = self._write_experiment(
                Path(tmp), evaluation_target_pc=[1.0]
            )
            with self.assertRaisesRegex(ValueError, "oracle thresholds"):
                load_cross_dataset_experiment(experiment)

    def test_calibrates_on_source_and_applies_threshold_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._write_experiment(root)
            calls: list[tuple[str, Path, list[str]]] = []

            def fake_prepare_embeddings(
                dataset,
                model,
                image_ids,
                _records,
                model_dir,
                *_args,
            ):
                calls.append((dataset.dataset_id, model_dir, list(image_ids)))
                if dataset.dataset_id == "toy-calibration":
                    vectors = {
                        "q1": [1.0, 0.0],
                        "q2": [0.0, 1.0],
                        "c1": [0.8, 0.6],
                        "c2": [0.6, 0.8],
                    }
                else:
                    vectors = {
                        "q1": [1.0, 0.0],
                        "q2": [0.0, 1.0],
                        "c1": [0.9, float(np.sqrt(1.0 - 0.9**2))],
                        "c2": [float(np.sqrt(1.0 - 0.7**2)), 0.7],
                    }
                embeddings = np.asarray(
                    [vectors[file_id] for file_id in image_ids], dtype=np.float32
                )
                model_dir.mkdir(parents=True, exist_ok=True)
                np.save(model_dir / "embeddings.npy", embeddings)
                (model_dir / "embeddings_metadata.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                return embeddings, {
                    "normalized": True,
                    "embedding_sha256": f"embedding-{dataset.dataset_id}",
                    "cache_reused_for_run": False,
                    "resources": {
                        "embedding": {
                            "num_images": len(image_ids),
                            "wall_time_seconds": 0.0,
                        },
                        "model": {},
                    },
                }

            with mock.patch(
                "experiments.core.cross_dataset_threshold_transfer.prepare_embeddings",
                side_effect=fake_prepare_embeddings,
            ):
                run_dir = run_experiment(experiment, run_id="cross-transfer")

            self.assertEqual(
                [dataset_id for dataset_id, _, _ in calls],
                ["toy-calibration", "toy-evaluation"],
            )
            self.assertEqual(calls[0][1].name, "calibration")
            self.assertEqual(calls[1][1].parent.name, "models")

            model_dir = next((run_dir / "models").iterdir())
            calibration = json.loads(
                (model_dir / "threshold_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
            evaluation = json.loads(
                (model_dir / "eval.json").read_text(encoding="utf-8")
            )
            threshold = calibration["operating_points"][0]["threshold"]
            transferred = evaluation["tasks"][0]["calibrated_threshold_metrics"][0]
            self.assertAlmostEqual(threshold, 0.8, places=5)
            self.assertAlmostEqual(transferred["threshold"], threshold, places=7)
            self.assertAlmostEqual(transferred["calibration_pair_completeness"], 1.0)
            self.assertAlmostEqual(transferred["pair_completeness"], 0.5)
            self.assertAlmostEqual(transferred["calibration_error"], -0.5)
            self.assertEqual(evaluation["tasks"][0]["target_pc_metrics"], [])

            run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["experiment_type"], EXPERIMENT_TYPE)
            self.assertEqual(
                run_meta["calibration_source"]["dataset_id"], "toy-calibration"
            )
            self.assertEqual(
                run_meta["dataset"]["dataset_id"], "toy-evaluation"
            )
            self.assertTrue((run_dir / "calibration_split.snapshot.json").is_file())
            protocol = json.loads(
                (run_dir / "calibration_protocol.json").read_text(encoding="utf-8")
            )
            self.assertTrue(protocol["passed"])
            self.assertTrue(
                next(
                    row
                    for row in protocol["checks"]
                    if row["check"] == "distinct_dataset_namespaces"
                )["ok"]
            )

            aggregate = json.loads(
                (run_dir / "aggregate_metrics.json").read_text(encoding="utf-8")
            )["rows"]
            calibrated = next(
                row
                for row in aggregate
                if row["selection_mode"] == "calibrated_threshold"
            )
            self.assertEqual(calibrated["calibration_dataset_id"], "toy-calibration")
            self.assertEqual(
                calibrated["calibration_split_id"], "toy-calibration-test"
            )
            with (run_dir / "aggregate_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertIn("calibration_dataset_id", csv_rows[0])


if __name__ == "__main__":
    unittest.main()
