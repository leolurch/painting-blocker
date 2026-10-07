from __future__ import annotations

import argparse
import json
import tempfile
import unittest

import numpy as np
from pathlib import Path
from unittest.mock import patch

from experiments import cli
from experiments.core.aggregate_runs import aggregate_paths
from experiments.core.config_schema import (
    DatasetConfig,
    ExperimentConfig,
    load_experiment_config,
    resolve_post_evaluations,
)
from experiments.core.finetuning.post_evaluation import (
    checkpoint_model_config,
    run_post_evaluations,
)
from experiments.core.runner import run_experiment
from experiments.core.split_rotation_evaluation import (
    SplitRotationSpec,
    StaticSplitRealization,
    run_split_rotation_evaluation,
)
from experiments.core.split_schema import write_split


class PostFinetuningEvaluationTest(unittest.TestCase):
    def _write_eval_template(self, root: Path) -> Path:
        path = root / "eval.yml"
        path.write_text(
            """schema_version: 1
experiment_id: linked_eval_v1
dataset:
  path: ./dataset
split:
  file: split.json
models:
  from_finetuning_checkpoint: true
evaluation:
  similarity: cosine
  top_k: [1]
  retrieval_tasks:
    - task_id: test_task
      query: {subset: test, role: modern}
      candidates: {subset: test, role: historic}
      positive_policy: same_painting
      exclude_self: true
run:
  output_dir: ./unused
""",
            encoding="utf-8",
        )
        return path

    def test_train_cli_runs_linked_evaluations_after_training(self) -> None:
        args = argparse.Namespace(
            experiment=Path("train.yml"),
            run_id="finetune_1",
            epochs=None,
            device="cuda",
            seed=123,
        )
        training_exp = object()
        train_result = {
            "status": "ok",
            "run_dir": "/tmp/finetune_1",
            "best_checkpoint": "/tmp/finetune_1/best_checkpoint.pt",
        }
        eval_results = [{"status": "completed", "experiment_id": "eval_v1"}]
        with patch.object(cli, "load_experiment_config", return_value=training_exp), patch.object(
            cli, "preflight_post_evaluations", return_value=[(object(), object())]
        ), patch(
            "experiments.core.finetuning.train.train_from_experiment",
            return_value=dict(train_result),
        ) as train_mock, patch.object(
            cli, "run_post_evaluations", return_value=eval_results
        ) as post_mock, patch(
            "experiments.core.resource_metrics.release_torch_cuda_memory"
        ) as release_mock:
            result = cli._train(args)
        train_mock.assert_called_once()
        self.assertEqual(train_mock.call_args.kwargs["seed_override"], 123)
        release_mock.assert_called_once_with(reset_compiler=True)
        post_mock.assert_called_once()
        post_args, post_kwargs = post_mock.call_args
        self.assertEqual(post_args, (training_exp, train_result["run_dir"]))
        self.assertEqual(post_kwargs["training_result"]["best_checkpoint"], train_result["best_checkpoint"])
        self.assertEqual(result["post_evaluations"], eval_results)

    def test_dynamic_eval_requires_checkpoint_and_resolves_injected_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = self._write_eval_template(root)
            checkpoint = root / "best_checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint-content")
            model = checkpoint_model_config(checkpoint, model_id="local/test/run/best")

            with self.assertRaisesRegex(ValueError, "requires a finetuning checkpoint"):
                load_experiment_config(eval_path)
            template = load_experiment_config(eval_path, allow_dynamic_models=True)
            self.assertEqual(template.models, [])
            resolved = load_experiment_config(eval_path, injected_models=[model])
            self.assertEqual([entry.model_id for entry in resolved.models], ["local/test/run/best"])
            self.assertEqual(
                resolved.models[0].adapter.adapter_name,
                "finetuned_checkpoint_adapter",
            )
            self.assertEqual(
                resolved.resolved["models"]["include"][0]["adapter_kwargs"]["checkpoint_path"],
                str(checkpoint.resolve()),
            )

    def test_post_evaluation_specs_resolve_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = self._write_eval_template(root)
            dataset = DatasetConfig(root, "toy", root / "dataset.db", root, {}, {})
            exp = ExperimentConfig(
                path=root / "train.yml",
                repo_root=root,
                experiment_id="train_v1",
                dataset=dataset,
                split_file=root / "train-split.json",
                models=[],
                raw={
                    "finetuning": {
                        "post_evaluations": [
                            {"experiment": eval_path.name, "checkpoint": "last", "label": "Grid model"}
                        ]
                    }
                },
                resolved={},
            )
            specs = resolve_post_evaluations(exp)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].experiment, eval_path.resolve())
            self.assertEqual(specs[0].checkpoint, "last")
            self.assertEqual(specs[0].label, "Grid model")

    def test_normal_runner_accepts_injected_checkpoint_and_keeps_calculation_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            (dataset_root / "images").mkdir(parents=True)
            split_path = root / "split.json"
            write_split(
                split_path,
                {
                    "schema_version": 2,
                    "split_id": "dynamic_split",
                    "dataset": {
                        "dataset_id": "dynamic-toy",
                        "image_root": str(dataset_root / "images"),
                        "identity": {"num_classes": 2, "num_images": 4},
                        "source_db_sha256": "toy-sha",
                    },
                    "subsets": {
                        "test": {
                            "roles": {
                                "modern": ["m1.png", "m2.png"],
                                "historic": ["h1.png", "h2.png"],
                            }
                        }
                    },
                    "images": {
                        "m1.png": {"class_id": 1},
                        "h1.png": {"class_id": 1},
                        "m2.png": {"class_id": 2},
                        "h2.png": {"class_id": 2},
                    },
                },
            )
            eval_path = root / "runner-eval.yml"
            eval_path.write_text(
                f"""schema_version: 1
experiment_id: dynamic_runner_eval_v1
dataset:
  path: {dataset_root}
split:
  file: {split_path}
models:
  from_finetuning_checkpoint: true
evaluation:
  top_k: [1]
  retrieval_tasks:
    - task_id: test_modern_to_historic
      query: {{subset: test, role: modern}}
      candidates: {{subset: test, role: historic}}
      positive_policy: same_painting
      exclude_self: true
run:
  save_similarity_cache: false
""",
                encoding="utf-8",
            )
            checkpoint = root / "best_checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            model = checkpoint_model_config(
                checkpoint,
                model_id="local/finetuned/grid/run/best",
                display_name="Grid run",
            )
            # The runner embeds sorted [h1, h2, m1, m2]. Matching classes get
            # identical vectors, exercising the real retrieval metric path.
            embeddings = np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            )
            embedding_meta = {
                "embedding_sha256": "embedding-sha",
                "normalized": True,
                "cache_reused_for_run": False,
                "resources": {},
            }
            parent = {
                "experiment_id": "grid_v1",
                "run_id": "finetune_1",
                "checkpoint_kind": "best",
                "checkpoint_sha256": "checkpoint-sha",
            }
            with patch(
                "experiments.core.runner.prepare_embeddings",
                return_value=(embeddings, embedding_meta),
            ), patch("experiments.core.run_charts.write_run_charts", return_value={}):
                produced = run_experiment(
                    eval_path,
                    run_id="linked",
                    injected_models=[model],
                    output_root_override=root / "outputs",
                    parent_training=parent,
                )
            run_meta = json.loads((produced / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["status"], "completed")
            self.assertEqual(run_meta["parent_training"], parent)
            rows = aggregate_paths([produced])
            self.assertEqual(rows[0]["pair_completeness"], 1.0)
            self.assertEqual(rows[0]["finetuning_experiment_id"], "grid_v1")
            self.assertEqual(rows[0]["display_name"], "Grid run")

    def test_nested_eval_aggregation_flattens_training_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_run = root / "finetune_1" / "evaluations" / "eval_v1"
            model_dir = eval_run / "models" / "model-key"
            model_dir.mkdir(parents=True)
            (eval_run / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "eval_v1",
                        "experiment_id": "eval_protocol_v1",
                        "status": "completed",
                        "dataset": {"dataset_id": "toy"},
                        "split": {"split_id": "split-v1", "split_seed": 67, "split_sha256": "split-sha"},
                        "parent_training": {
                            "experiment_id": "grid_variant_v1",
                            "configuration_id": "final_lora_linear",
                            "training_seed": 43,
                            "run_id": "finetune_1",
                            "run_dir": str(root / "finetune_1"),
                            "checkpoint_kind": "best",
                            "checkpoint_path": "/tmp/best_checkpoint.pt",
                            "checkpoint_sha256": "abc123",
                        },
                        "model_runs": [
                            {
                                "model_id": "local/finetuned/grid_variant_v1/finetune_1/best",
                                "model_storage_key": "model-key",
                                "display_name": "Grid variant 1",
                                "status": "completed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "eval.json").write_text(
                json.dumps(
                    {
                        "model_id": "local/finetuned/grid_variant_v1/finetune_1/best",
                        "model_storage_key": "model-key",
                        "tasks": [
                            {
                                "task_id": "test_task",
                                "metrics": [{"k": 1, "pair_completeness": 0.75}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows = aggregate_paths([root])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["finetuning_experiment_id"], "grid_variant_v1")
            self.assertEqual(rows[0]["configuration_id"], "final_lora_linear")
            self.assertEqual(rows[0]["training_seed"], 43)
            self.assertEqual(rows[0]["split_seed"], 67)
            self.assertEqual(rows[0]["finetuning_run_id"], "finetune_1")
            self.assertEqual(rows[0]["checkpoint_sha256"], "abc123")
            self.assertEqual(rows[0]["display_name"], "Grid variant 1")

    def test_rotation_runner_propagates_dynamic_model_and_training_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = DatasetConfig(root, "toy", root / "dataset.db", root, {}, {})
            eval_exp = ExperimentConfig(
                path=root / "eval.yml",
                repo_root=root,
                experiment_id="rotation_v1",
                dataset=dataset,
                split_file=root / "default.json",
                models=[],
                raw={},
                resolved={},
            )
            realizations = tuple(
                StaticSplitRealization(
                    seed=seed,
                    file=root / f"split-{seed}.json",
                    split_id=f"split-{seed}",
                    split_sha256=f"sha-{seed}",
                    source_db_sha256="db-sha",
                )
                for seed in (6, 7)
            )
            spec = SplitRotationSpec(eval_exp, realizations, root / "unused", True)
            checkpoint = root / "best_checkpoint.pt"
            checkpoint.write_bytes(b"weights")
            model = checkpoint_model_config(checkpoint, model_id="local/test/best")
            parent = {"checkpoint_sha256": "checkpoint-sha", "checkpoint_kind": "best"}

            with patch(
                "experiments.core.split_rotation_evaluation.load_split_rotation_spec",
                return_value=spec,
            ), patch(
                "experiments.core.split_rotation_evaluation.run_experiment",
                side_effect=lambda _experiment, **kwargs: Path(kwargs["output_root_override"])
                / kwargs["run_id"],
            ) as run_mock:
                result = run_split_rotation_evaluation(
                    eval_exp.path,
                    all_splits=True,
                    run_id="rotation",
                    injected_models=[model],
                    output_root_override=root / "outputs",
                    parent_training=parent,
                )

            self.assertEqual(result["status"], "completed")
            self.assertEqual([row["seed"] for row in result["completed"]], [6, 7])
            self.assertEqual(run_mock.call_count, 2)
            for call in run_mock.call_args_list:
                self.assertEqual(call.kwargs["injected_models"], [model])
                self.assertEqual(call.kwargs["parent_training"], parent)
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["parent_training"], parent)

    def test_post_evaluation_runs_all_split_rotations_with_injected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "rotation-eval.yml"
            eval_path.write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "train-1"
            run_dir.mkdir(parents=True)
            (run_dir / "best_checkpoint.pt").write_bytes(b"weights")
            dataset = DatasetConfig(root, "toy", root / "dataset.db", root, {}, {})
            training_exp = ExperimentConfig(
                path=root / "train.yml",
                repo_root=root,
                experiment_id="training_v1",
                dataset=dataset,
                split_file=root / "train-split.json",
                models=[],
                raw={
                    "finetuning": {
                        "post_evaluations": [
                            {"experiment": eval_path.name, "checkpoint": "best"}
                        ]
                    }
                },
                resolved={},
            )
            eval_exp = ExperimentConfig(
                path=eval_path,
                repo_root=root,
                experiment_id="wikidata_rotation_v1",
                dataset=dataset,
                split_file=root / "eval-split.json",
                models=[],
                raw={
                    "models": {"from_finetuning_checkpoint": True},
                    "split_rotation_evaluation": {"schema_version": 1},
                },
                resolved={},
            )
            rotation_manifest = (
                run_dir
                / "evaluations"
                / "wikidata_rotation_v1"
                / "rotation.rotation_manifest.json"
            )
            with patch(
                "experiments.core.finetuning.post_evaluation.preflight_post_evaluations",
                return_value=[(resolve_post_evaluations(training_exp)[0], eval_exp)],
            ), patch(
                "experiments.core.split_rotation_evaluation.run_split_rotation_evaluation",
                return_value={"manifest": str(rotation_manifest), "status": "completed"},
            ) as rotation_mock:
                results = run_post_evaluations(training_exp, run_dir)

            self.assertEqual(results[0]["rotation_manifest"], str(rotation_manifest))
            self.assertEqual(results[0]["status"], "completed")
            kwargs = rotation_mock.call_args.kwargs
            self.assertTrue(kwargs["all_splits"])
            self.assertEqual(kwargs["run_id"], "rotation")
            self.assertEqual(len(kwargs["injected_models"]), 1)
            self.assertEqual(kwargs["parent_training"]["checkpoint_kind"], "best")

    def test_post_evaluation_is_nested_and_records_training_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = self._write_eval_template(root)
            run_dir = root / "train-runs" / "finetune_123-4"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best_checkpoint.pt"
            checkpoint.write_bytes(b"trained-weights")
            (run_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "configuration_id": "stable_lora_mlp_v1",
                        "training_seed": 43,
                        "training_repeats": {
                            "seeds": [42, 43, 44],
                            "require_complete": True,
                        },
                        "reproducibility": {"seed": 43},
                    }
                ),
                encoding="utf-8",
            )
            dataset = DatasetConfig(root, "toy", root / "dataset.db", root, {}, {})
            training_exp = ExperimentConfig(
                path=root / "train.yml",
                repo_root=root,
                experiment_id="grid_variant_v1",
                dataset=dataset,
                split_file=root / "train-split.json",
                models=[],
                raw={"finetuning": {"post_evaluations": [eval_path.name]}},
                resolved={},
            )
            eval_exp = load_experiment_config(eval_path, allow_dynamic_models=True)

            def fake_run(experiment, **kwargs):
                produced = Path(kwargs["output_root_override"]) / kwargs["run_id"]
                produced.mkdir(parents=True)
                (produced / "run.json").write_text(
                    json.dumps({"status": "completed"}), encoding="utf-8"
                )
                self.assertEqual(kwargs["parent_training"]["run_id"], run_dir.name)
                self.assertEqual(kwargs["parent_training"]["experiment_id"], "grid_variant_v1")
                self.assertEqual(
                    kwargs["parent_training"]["configuration_id"],
                    "stable_lora_mlp_v1",
                )
                self.assertEqual(kwargs["parent_training"]["training_seed"], 43)
                self.assertEqual(
                    kwargs["parent_training"]["training_repeats"]["seeds"],
                    [42, 43, 44],
                )
                self.assertEqual(len(kwargs["injected_models"]), 1)
                self.assertIn(run_dir.name, kwargs["injected_models"][0].model_id)
                return produced

            with patch(
                "experiments.core.finetuning.post_evaluation.preflight_post_evaluations",
                return_value=[(resolve_post_evaluations(training_exp)[0], eval_exp)],
            ), patch(
                "experiments.core.finetuning.post_evaluation.run_experiment",
                side_effect=fake_run,
            ):
                results = run_post_evaluations(training_exp, run_dir)

            self.assertEqual(results[0]["status"], "completed")
            expected = run_dir / "evaluations" / "linked_eval_v1"
            self.assertEqual(Path(results[0]["run_dir"]), expected)
            manifest = json.loads((run_dir / "post_evaluations.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["evaluations"][0]["checkpoint_kind"], "best")


if __name__ == "__main__":
    unittest.main()
