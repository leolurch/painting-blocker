from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
import numpy as np

from experiments.core.analysis_config import load_analysis_config
from experiments.core.config_schema import load_experiment_config
from experiments.core.multisplit_analysis import (
    _analysis_model_id_for_source,
    _validate_training_selection_protocol,
    multisplit_family_sha256,
    portable_multisplit_family_sha256,
)
from experiments.core.runner import _profile_diagnostics
from experiments.core.split_schema import SplitImage
from experiments.core.split_rotation_evaluation import load_split_rotation_spec


def _split(seed: int, val_classes: set[int]) -> dict:
    images = {
        f"{role}{class_id}.jpg": {"class_id": class_id, "profile": "archival"}
        for class_id in range(4)
        for role in ("m", "h")
    }
    test_classes = set(range(4)) - val_classes
    return {
        "schema_version": 2,
        "split_id": f"wikidata-test-seed-{seed}",
        "dataset": {
            "dataset_id": "wikidata-test",
            "image_root": "/unused/images",
            "source_db_sha256": "db-sha",
            "identity": {"num_classes": 4, "num_images": 8},
        },
        "split_strategy": {
            "type": "class_disjoint_role",
            "seed": seed,
            "ratios": {"train": 0.0, "val": 0.5, "test": 0.5},
        },
        "subsets": {
            name: {
                "roles": {
                    "modern": [f"m{class_id}.jpg" for class_id in sorted(classes)],
                    "historic": [f"h{class_id}.jpg" for class_id in sorted(classes)],
                }
            }
            for name, classes in (("val", val_classes), ("test", test_classes))
        },
        "images": images,
    }


class MultisplitModeTest(unittest.TestCase):
    def test_portable_multisplit_family_hash_ignores_checkout_location(self):
        first = {"experiment_id": "eval", "repo_root": "/cluster/old", "models": {}}
        second = {"experiment_id": "eval", "repo_root": "/cluster/new", "models": {}}
        self.assertNotEqual(multisplit_family_sha256(first), multisplit_family_sha256(second))
        self.assertEqual(
            portable_multisplit_family_sha256(first),
            portable_multisplit_family_sha256(second),
        )

    def _fixture(self, root: Path) -> Path:
        for seed, classes in ((6, {0, 1}), (7, {0, 2})):
            (root / f"split-{seed}.json").write_text(
                json.dumps(_split(seed, classes)), encoding="utf-8"
            )
        (root / "model.yml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "adapter_name": "resnet50_hf_adapter",
                    "models": [{"model_id": "test/model"}],
                }
            ),
            encoding="utf-8",
        )
        experiment = {
            "schema_version": 1,
            "experiment_id": "multisplit-test",
            "dataset": {"path": str(root / "dataset")},
            "split": {"file": "split-6.json"},
            "split_rotation_evaluation": {
                "schema_version": 1,
                "protocol": {"frozen_before_evaluation": True},
                "expected_split_seeds": [6, 7],
                "static_splits": [
                    {"seed": 6, "file": "split-6.json"},
                    {"seed": 7, "file": "split-7.json"},
                ],
                "output_dir": str(root / "runs"),
            },
            "models": {
                "include": [
                    {
                        "config": "model.yml",
                        "model_id": "test/model",
                        "group": "frozen",
                    }
                ]
            },
            "evaluation": {
                "top_k": [1, 2],
                "threshold_calibration": {
                    "method": "empirical_target_pc",
                    "calibration_id": "val",
                    "target_pc": [0.99],
                    "task": {
                        "task_id": "cal",
                        "query": {"subset": "val", "role": "modern"},
                        "candidates": {"subset": "val", "role": "historic"},
                        "positive_policy": "same_painting",
                        "exclude_self": True,
                    },
                },
                "retrieval_tasks": [
                    {
                        "task_id": "test",
                        "query": {"subset": "test", "role": "modern"},
                        "candidates": {"subset": "test", "role": "historic"},
                        "positive_policy": "same_painting",
                        "exclude_self": True,
                    }
                ],
            },
        }
        path = root / "experiment.yml"
        path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
        return path

    def test_rotation_spec_validates_explicit_static_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._fixture(root)
            spec = load_split_rotation_spec(experiment)
            self.assertEqual([row.seed for row in spec.realizations], [6, 7])
            self.assertTrue(spec.frozen_before_evaluation)

    def test_rotation_accepts_static_finetuned_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._fixture(root)
            model_path = root / "model.yml"
            model_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "finetuned_checkpoint_adapter",
                        "models": [
                            {
                                "model_id": "local/template",
                                "adapter_kwargs": {
                                    "checkpoint_path": "runs/train/best_checkpoint.pt"
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            document = yaml.safe_load(experiment.read_text(encoding="utf-8"))
            document["models"]["include"][0].update(
                {
                    "model_id": "local/template",
                    "variant_id": "local/finetuned/train/best",
                    "group": "finetuned",
                }
            )
            experiment.write_text(yaml.safe_dump(document), encoding="utf-8")

            spec = load_split_rotation_spec(experiment)

            self.assertEqual(spec.experiment.models[0].model_id, "local/finetuned/train/best")

    def test_rotation_rejects_nonstatic_finetuned_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._fixture(root)
            document = yaml.safe_load(experiment.read_text(encoding="utf-8"))
            document["models"]["include"][0]["group"] = "finetuned"
            experiment.write_text(yaml.safe_dump(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not fixed frozen models"):
                load_split_rotation_spec(experiment)

    def test_optional_split_override_does_not_change_ordinary_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._fixture(root)
            ordinary = load_experiment_config(experiment)
            overridden = load_experiment_config(
                experiment, split_file_override=root / "split-7.json"
            )
            self.assertEqual(ordinary.split_file.name, "split-6.json")
            self.assertEqual(overridden.split_file.name, "split-7.json")

    def test_new_model_is_discovered_when_config_already_has_models(self):
        configured = [
            {
                "analysis_model_id": "known-analysis-id",
                "model_id": "known/model",
                "model_storage_key": "known-storage-key",
            }
        ]

        analysis_id = _analysis_model_id_for_source(
            configured,
            {"new/model", "new-storage-key"},
            {"model_id": "new/model", "model_storage_key": "new-storage-key"},
            Path("new/eval.json"),
        )

        self.assertEqual(analysis_id, "new-storage-key")

    def test_existing_model_alias_keeps_persisted_analysis_id(self):
        configured = [
            {
                "analysis_model_id": "persisted-analysis-id",
                "model_id": "known/model",
                "model_storage_key": "known-storage-key",
            }
        ]

        analysis_id = _analysis_model_id_for_source(
            configured,
            {"known/model", "known-storage-key"},
            {"model_id": "known/model", "model_storage_key": "known-storage-key"},
            Path("known/eval.json"),
        )

        self.assertEqual(analysis_id, "persisted-analysis-id")

    def test_frozen_protocol_accepts_explicit_static_finetuned_checkpoints(self):
        config = {
            "multisplit": {
                "protocol": {
                    "frozen_before_evaluation": True,
                    "allow_static_finetuned_checkpoints": True,
                }
            },
            "models": [
                {
                    "analysis_model_id": "finetuned-model",
                    "include": True,
                    "presentation": {"group": "finetuned"},
                }
            ],
        }
        rows = [
            {
                "configuration_id": "finetuned-model",
                # Post-finetuning rotation evaluations preserve this lineage;
                # the checkpoint state is nevertheless frozen before WIK evaluation.
                "training_seed": 42,
                "model_state_kind": "static_finetuned_checkpoint",
            }
        ]

        _validate_training_selection_protocol(rows, config)

    def test_frozen_protocol_rejects_unverified_finetuned_state(self):
        config = {
            "multisplit": {
                "protocol": {
                    "frozen_before_evaluation": True,
                    "allow_static_finetuned_checkpoints": True,
                }
            },
            "models": [
                {
                    "analysis_model_id": "finetuned-model",
                    "include": True,
                    "presentation": {"group": "finetuned"},
                }
            ],
        }
        rows = [
            {
                "configuration_id": "finetuned-model",
                "training_seed": None,
                "model_state_kind": "unsupported",
            }
        ]

        with self.assertRaisesRegex(ValueError, "static finetuned checkpoints"):
            _validate_training_selection_protocol(rows, config)

    def test_multisplit_family_hash_neutralizes_only_split_path(self):
        base = {
            "split": {"file": "/split-6.json"},
            "models": {"include": [{"model_id": "a"}]},
            "evaluation": {"top_k": [1, 5]},
            "embedding": {"batch_size": 8},
        }
        other_split = {**base, "split": {"file": "/split-7.json"}}
        changed_protocol = {**base, "evaluation": {"top_k": [1, 10]}}
        self.assertEqual(
            multisplit_family_sha256(base), multisplit_family_sha256(other_split)
        )
        self.assertNotEqual(
            multisplit_family_sha256(base), multisplit_family_sha256(changed_protocol)
        )

    def test_multisplit_family_hash_canonicalizes_hf_token_aliases(self):
        seed_6 = {
            "environment": {
                "secret_presence": {
                    "HF_TOKEN": True,
                    "HUGGINGFACE_HUB_TOKEN": False,
                    "HUGGING_FACE_HUB_TOKEN": False,
                }
            }
        }
        later_seed = {
            "environment": {
                "secret_presence": {
                    "HF_TOKEN": True,
                    "HUGGINGFACE_HUB_TOKEN": True,
                    "HUGGING_FACE_HUB_TOKEN": True,
                }
            }
        }
        alias_only = {
            "environment": {
                "secret_presence": {
                    "HF_TOKEN": False,
                    "HUGGINGFACE_HUB_TOKEN": True,
                    "HUGGING_FACE_HUB_TOKEN": False,
                }
            }
        }
        no_credentials = {
            "environment": {
                "secret_presence": {
                    "HF_TOKEN": False,
                    "HUGGINGFACE_HUB_TOKEN": False,
                    "HUGGING_FACE_HUB_TOKEN": False,
                }
            }
        }

        expected = multisplit_family_sha256(seed_6)
        self.assertEqual(expected, multisplit_family_sha256(later_seed))
        self.assertEqual(expected, multisplit_family_sha256(alias_only))
        self.assertNotEqual(expected, multisplit_family_sha256(no_credentials))

    def test_profile_diagnostics_apply_global_operating_point_without_retuning(self):
        image_ids = ["q1", "q2", "c1", "c2"]
        records = {
            "q1": SplitImage("q1", 1),
            "q2": SplitImage("q2", 2),
            "c1": SplitImage("c1", 1),
            "c2": SplitImage("c2", 2),
        }
        similarities = np.full((4, 4), -1.0, dtype=np.float32)
        similarities[0, 2] = 0.9
        similarities[1, 3] = 0.8
        task = {
            "task_id": "test",
            "query_ids": ["q1", "q2"],
            "candidate_ids": ["c1", "c2"],
            "exclude_self": True,
            "positive_policy": "same_painting",
        }
        split = {
            "images": {
                "q1": {"class_id": 1},
                "q2": {"class_id": 2},
                "c1": {"class_id": 1, "profile": "archival"},
                "c2": {"class_id": 2, "profile": "print"},
            }
        }
        global_calibration = [
            {
                "target_pair_completeness": 0.99,
                "threshold": 0.75,
                "pair_completeness": 1.0,
                "pair_quality": 1.0,
                "reduction_ratio": 0.5,
            }
        ]
        rows = _profile_diagnostics(
            task,
            image_ids,
            records,
            similarities,
            [1],
            global_calibration,
            "global-pooled",
            [],
            None,
            split,
            {"candidate_metadata_field": "profile"},
        )
        self.assertEqual([row["profile"] for row in rows], ["archival", "print"])
        self.assertTrue(all(row["selection_role"] == "diagnostic_only" for row in rows))
        self.assertTrue(
            all(
                metric["threshold"] == 0.75
                for row in rows
                for metric in row["calibrated_threshold_metrics"]
            )
        )

    def test_old_analysis_config_without_mode_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.yml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "analysis_id": "old",
                        "discovery": {},
                        "models": [],
                        "charts": {},
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_analysis_config(path)
            self.assertNotIn("mode", loaded)


if __name__ == "__main__":
    unittest.main()
