import csv
import io
import json
import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from experiments.core.adapter_registry import known_adapter_names
from experiments.cli import build_parser
from experiments.core.finetuning.console_logging import ConsoleTrainingLogger, format_duration
from experiments.core.finetuning.evaluate import apply_threshold, evaluate_embeddings
from experiments.core.validate.embedding_adapters.checkpoint_utils import CheckpointProjectionAdapter
from experiments.core.finetuning.losses import MultiSimilarityLoss, SupConLoss, build_loss
from experiments.core.finetuning.mining import mine_hard_negatives_from_embeddings
from experiments.core.finetuning.samplers import (
    HardNegativePKBatchSampler,
    OriginBalancedPKBatchSampler,
    PKBatchSampler,
    RoleStratifiedPKBatchSampler,
)
from experiments.core.finetuning.models.dinov3 import (
    layer_span_settings,
    qv_lora_candidate_modules,
    resolve_qv_lora_targets,
)
from experiments.core.finetuning.models.pooling import GeMPooling2d, TokenGeMPooling
from experiments.core.finetuning.train import (
    _grad_scaler,
    _gradient_cache_backward,
    _gradient_cache_config,
)
from experiments.core.config_schema import (
    _validate_gradient_cache,
    _validate_training_repeats,
    training_repeat_seeds,
)
from experiments.core.split_schema import write_split


class FinetuningSkeletonTest(unittest.TestCase):
    def test_training_repeat_seed_contract(self):
        finetuning = {
            "configuration_id": "vitb-lora-mlp-v1",
            "training_repeats": {
                "seeds": [42, 43, 44],
                "require_complete": True,
            },
            "train": {"seed": 42},
        }
        _validate_training_repeats(finetuning, Path("experiment.yml"))
        self.assertEqual(training_repeat_seeds(finetuning), (42, 43, 44))

        with self.assertRaisesRegex(ValueError, "configuration_id"):
            _validate_training_repeats(
                {
                    "training_repeats": {"seeds": [42, 43]},
                    "train": {"seed": 42},
                },
                Path("experiment.yml"),
            )
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            training_repeat_seeds(
                {
                    "training_repeats": {"seeds": [42, 42]},
                }
            )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _validate_training_repeats(
                {
                    "configuration_id": "vitb-lora-mlp-v1",
                    "training_repeats": {"seeds": [42, 43]},
                    "train": {"seed": 44},
                },
                Path("experiment.yml"),
            )

    def test_training_repeat_rejects_undeclared_cli_seed_before_gpu_work(self):
        from experiments.core.finetuning import train as train_module

        experiment = SimpleNamespace(
            raw={
                "finetuning": {
                    "configuration_id": "vitb-lora-mlp-v1",
                    "training_repeats": {"seeds": [42, 43]},
                    "model": {"type": "fake"},
                    "train": {"seed": 42},
                }
            },
            split_file=Path("split.json"),
        )
        with mock.patch.object(
            train_module, "load_experiment_config", return_value=experiment
        ), mock.patch.object(
            train_module, "load_split", return_value={"images": {"x": {"class_id": 1}}}
        ), mock.patch.object(
            train_module, "validate_split_manifest"
        ), self.assertRaisesRegex(
            ValueError, "is not declared"
        ):
            train_module.train_from_experiment(
                Path("experiment.yml"), seed_override=44
            )

    def test_console_training_logger_formats_heartbeat(self):
        stream = io.StringIO()
        logger = ConsoleTrainingLogger(enabled=True, heartbeat_seconds=120, device="cpu", stream=stream)
        logger.configure(phase="train", epoch=1, epochs=2, batch=3, batches_per_epoch=5, global_step=3, total_steps=10)
        logger.heartbeat()
        output = stream.getvalue()
        self.assertIn("heartbeat", output)
        self.assertIn("epoch=1/2", output)
        self.assertIn("batch=3/5", output)
        self.assertIn("eta_total=", output)
        self.assertEqual(format_duration(3661), "01:01:01")

    def test_gradient_cache_matches_graph_preserving_chunked_forward_with_dropout(self):
        class TinyEmbeddingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(5, 4)
                self.dropout = torch.nn.Dropout(0.35)

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.normalize(
                    self.dropout(self.projection(values)), p=2, dim=1
                )

        torch.manual_seed(11)
        reference_model = TinyEmbeddingModel()
        cached_model = TinyEmbeddingModel()
        cached_model.load_state_dict(reference_model.state_dict())
        values = torch.randn(8, 5)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        reference_loss_fn = MultiSimilarityLoss()
        cached_loss_fn = MultiSimilarityLoss()

        torch.manual_seed(97)
        reference_embeddings = torch.cat(
            [reference_model(values[start : start + 2]) for start in range(0, 8, 2)],
            dim=0,
        )
        reference_loss = reference_loss_fn(reference_embeddings, labels)
        reference_loss.backward()
        reference_rng_state = torch.get_rng_state().clone()

        torch.manual_seed(97)
        cached_loss, _observed = _gradient_cache_backward(
            cached_model,
            cached_loss_fn,
            values,
            labels,
            chunk_size=2,
            device="cpu",
            precision="float32",
            scaler=_grad_scaler("cpu", "float32"),
            verify_replay=True,
        )
        cached_rng_state = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(reference_rng_state, cached_rng_state))
        self.assertTrue(torch.allclose(reference_loss, cached_loss, atol=1e-7, rtol=1e-6))
        for reference_parameter, cached_parameter in zip(
            reference_model.parameters(), cached_model.parameters()
        ):
            self.assertIsNotNone(reference_parameter.grad)
            self.assertIsNotNone(cached_parameter.grad)
            self.assertTrue(
                torch.allclose(
                    reference_parameter.grad,
                    cached_parameter.grad,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )

    def test_gradient_cache_config_and_schema_enforce_physical_chunk(self):
        sampler = SimpleNamespace(classes_per_batch=64, images_per_class=4)
        resolved = _gradient_cache_config(
            {"gradient_cache": {"enabled": True, "chunk_size": 128}}, sampler
        )
        self.assertEqual(resolved["effective_batch_size"], 256)
        self.assertEqual(resolved["chunk_size"], 128)
        self.assertEqual(resolved["chunks_per_step"], 2)
        overridden = _gradient_cache_config(
            {"gradient_cache": {"enabled": True, "chunk_size": 32}},
            sampler,
            chunk_size_override=128,
        )
        self.assertEqual(overridden["chunk_size"], 128)
        self.assertEqual(overridden["effective_batch_size"], 256)
        self.assertEqual(overridden["chunks_per_step"], 2)
        _validate_gradient_cache(
            {
                "sampler": {"classes_per_batch": 64, "images_per_class": 4},
                "train": {"gradient_cache": {"enabled": True, "chunk_size": 128}},
            },
            Path("experiment.yml"),
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _validate_gradient_cache(
                {
                    "sampler": {"classes_per_batch": 32, "images_per_class": 4},
                    "train": {"gradient_cache": {"enabled": True, "chunk_size": 256}},
                },
                Path("experiment.yml"),
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _validate_gradient_cache(
                {
                    "sampler": {"classes_per_batch": 32.5, "images_per_class": 4},
                    "train": {"gradient_cache": {"enabled": True, "chunk_size": 128}},
                },
                Path("experiment.yml"),
            )

    def test_multi_similarity_loss_skips_singleton_anchors(self):
        embeddings = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.0, 0.9],
                    [0.7, 0.7],
                ]
            ),
            p=2,
            dim=1,
        )
        labels = torch.tensor([0, 0, 1, 1, 2])
        loss = MultiSimilarityLoss()(embeddings, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertEqual(MultiSimilarityLoss()(embeddings[:1], labels[:1]).item(), 0.0)
        self.assertIsInstance(build_loss({"name": "multi_similarity", "miner_epsilon": 0.2}), MultiSimilarityLoss)
        cross = build_loss({"name": "multi_similarity", "cross_origin_only": True})
        self.assertTrue(cross.cross_origin_only)

    def test_cross_origin_loss_ignores_same_origin_pairs(self):
        embeddings = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            p=2,
            dim=1,
        )
        labels = torch.tensor([0, 0, 1, 1])
        # 0 = historic, 1 = modern. The only same-painting pairs are same-origin.
        origins = torch.tensor([0, 0, 1, 1])
        loss_fn = MultiSimilarityLoss()
        loss_fn.cross_origin_only = True
        loss = loss_fn(embeddings, labels, origins)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(loss_fn.last_stats.valid_anchors, 0)
        self.assertEqual(loss_fn.last_stats.positive_pairs, 0)
        # Same paintings, opposite origins: those pairs count.
        mixed = torch.tensor([0, 1, 0, 1])
        counted = loss_fn(embeddings, labels, mixed)
        self.assertTrue(torch.isfinite(counted))
        self.assertGreater(loss_fn.last_stats.positive_pairs, 0)
        self.assertGreater(loss_fn.last_stats.negative_pairs, 0)
        self.assertIsInstance(build_loss({"name": "supcon"}), SupConLoss)

    def test_dinov3_lora_qv_target_resolution_uses_actual_module_names(self):
        fake = torch.nn.Module()
        fake.encoder = torch.nn.Module()
        fake.encoder.layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attention": torch.nn.ModuleDict(
                            {
                                "query": torch.nn.Linear(4, 4),
                                "value": torch.nn.Linear(4, 4),
                                "key": torch.nn.Linear(4, 4),
                            }
                        )
                    }
                )
            ]
        )
        self.assertEqual(resolve_qv_lora_targets(fake), ["query", "value"])
        self.assertEqual(resolve_qv_lora_targets(fake, ["attention.query", "attention.value"]), ["attention.query", "attention.value"])
        candidates = qv_lora_candidate_modules(fake)
        self.assertIn("encoder.layer.0.attention.query", candidates)
        with self.assertRaisesRegex(ValueError, "did not match"):
            resolve_qv_lora_targets(fake, ["q_proj", "v_proj"])

    def test_dinov3_lora_can_target_only_final_layers(self):
        fake = torch.nn.Module()
        fake.layer = torch.nn.ModuleList()
        for _ in range(6):
            block = torch.nn.Module()
            block.attention = torch.nn.ModuleDict(
                {
                    "q_proj": torch.nn.Linear(4, 4),
                    "v_proj": torch.nn.Linear(4, 4),
                    "k_proj": torch.nn.Linear(4, 4),
                }
            )
            fake.layer.append(block)
        targets = resolve_qv_lora_targets(
            fake,
            ["q_proj", "v_proj"],
            adapt_last_n_layers=2,
        )
        self.assertEqual(
            targets,
            [
                "layer.4.attention.q_proj",
                "layer.4.attention.v_proj",
                "layer.5.attention.q_proj",
                "layer.5.attention.v_proj",
            ],
        )
        with self.assertRaisesRegex(ValueError, "exceeds discovered layer count"):
            resolve_qv_lora_targets(fake, ["q_proj", "v_proj"], adapt_last_n_layers=7)
        first = resolve_qv_lora_targets(
            fake,
            ["q_proj", "v_proj"],
            adapt_first_n_layers=2,
        )
        self.assertEqual(
            first,
            [
                "layer.0.attention.q_proj",
                "layer.0.attention.v_proj",
                "layer.1.attention.q_proj",
                "layer.1.attention.v_proj",
            ],
        )
        with self.assertRaisesRegex(ValueError, "only one"):
            resolve_qv_lora_targets(
                fake,
                ["q_proj", "v_proj"],
                adapt_first_n_layers=2,
                adapt_last_n_layers=2,
            )

    def test_lora_span_selects_all_first_or_last_blocks(self):
        self.assertEqual(layer_span_settings({"span": "all"}), ("all", None, None))
        self.assertEqual(layer_span_settings({"span": "first16"}), ("first16", 16, None))
        self.assertEqual(layer_span_settings({"span": "last24"}), ("last24", None, 24))
        self.assertEqual(layer_span_settings({"adapt_last_n_layers": 16}), (None, None, 16))
        with self.assertRaisesRegex(ValueError, "every attention block"):
            layer_span_settings({"span": "all", "adapt_last_n_layers": 16})

    def test_pk_sampler_repeats_small_classes(self):
        sampler = PKBatchSampler([0, 0, 1], classes_per_batch=2, images_per_class=4, batches_per_epoch=2, seed=7)
        batches = list(sampler)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch) == 8 for batch in batches))
        self.assertTrue(all(0 <= idx <= 2 for batch in batches for idx in batch))
        self.assertEqual(
            batches,
            [[2, 0, 1, 0, 2, 0, 2, 2], [2, 2, 0, 1, 0, 2, 0, 2]],
        )

    def test_origin_balanced_sampler_keeps_the_requested_mix(self):
        labels = []
        roles = []
        for class_id in (0, 1, 2):
            labels.extend([class_id] * 8)
            roles.extend(
                ["modern_original", "modern_generated", "modern_generated", "modern_generated"]
                + ["historic_archival", "historic_print", "historic_cropped_record", "historic_framed_photo"]
            )
        sampler = OriginBalancedPKBatchSampler(
            labels,
            roles,
            classes_per_batch=2,
            per_origin={"historic": 2, "modern": 2},
            batches_per_epoch=1,
            seed=3,
        )
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 8)
        self.assertEqual(sampler.images_per_class, 4)
        by_class: dict[int, list[str]] = {}
        for index in batch:
            by_class.setdefault(labels[index], []).append(roles[index])
        self.assertEqual(len(by_class), 2)
        for class_roles in by_class.values():
            historic = [role for role in class_roles if role.startswith("historic_")]
            modern = [role for role in class_roles if role.startswith("modern_")]
            self.assertEqual(len(historic), 2)
            self.assertEqual(len(modern), 2)
            self.assertEqual(len(set(class_roles)), 4)

    def test_selected_split_roles_are_preserved_for_sampling(self):
        from experiments.core.finetuning.data import _selected_role_memberships

        split = {
            "subsets": {
                "train": {
                    "roles": {
                        "modern_original": ["a0.png", "b0.png"],
                        "historic_print": ["a1.png", "b1.png"],
                    }
                }
            }
        }
        memberships = _selected_role_memberships(
            split,
            "train",
            ["modern_original", "historic_print"],
        )
        self.assertEqual(memberships["a0.png"], {"modern_original"})
        self.assertEqual(memberships["a1.png"], {"historic_print"})

    @staticmethod
    def _role_sampler_inputs():
        labels = []
        roles = []
        for class_id in ("a", "b", "c"):
            labels.extend([class_id] * 10)
            roles.extend(
                [
                    "modern_original",
                    "historic_archival",
                    "historic_archival",
                    "historic_print",
                    "historic_print",
                    "historic_cropped_record",
                    "historic_cropped_record",
                    "historic_framed_photo",
                    "historic_framed_photo",
                    "modern_generated",
                ]
            )
        return labels, roles

    def test_role_stratified_pk_sampler_k2_enforces_original_to_historic_pair(self):
        labels, roles = self._role_sampler_inputs()
        sampler = RoleStratifiedPKBatchSampler(
            labels,
            roles,
            classes_per_batch=3,
            images_per_class=2,
            anchor_role="modern_original",
            positive_roles=[
                "historic_archival",
                "historic_print",
                "historic_cropped_record",
                "historic_framed_photo",
            ],
            batches_per_epoch=5,
            seed=17,
        )
        batches = list(sampler)
        self.assertEqual(sampler.ignored_role_counts, {"modern_generated": 3})
        self.assertEqual(len(batches), 5)
        for batch in batches:
            self.assertEqual(len(batch), 6)
            by_class = {}
            for index in batch:
                by_class.setdefault(labels[index], []).append(roles[index])
            self.assertEqual(set(by_class), {"a", "b", "c"})
            for sampled_roles in by_class.values():
                self.assertEqual(sampled_roles.count("modern_original"), 1)
                self.assertEqual(len(sampled_roles), 2)
                self.assertEqual(sum(role.startswith("historic_") for role in sampled_roles), 1)

    def test_role_stratified_pk_sampler_k4_uses_distinct_historic_profiles(self):
        labels, roles = self._role_sampler_inputs()
        kwargs = dict(
            classes_per_batch=3,
            images_per_class=4,
            anchor_role="modern_original",
            positive_roles=[
                "historic_archival",
                "historic_print",
                "historic_cropped_record",
                "historic_framed_photo",
            ],
            batches_per_epoch=3,
            seed=23,
        )
        first = RoleStratifiedPKBatchSampler(labels, roles, **kwargs)
        second = RoleStratifiedPKBatchSampler(labels, roles, **kwargs)
        first_batches = list(first)
        self.assertEqual(first_batches, list(second))
        for batch in first_batches:
            self.assertEqual(len(batch), 12)
            by_class = {}
            for index in batch:
                by_class.setdefault(labels[index], []).append(roles[index])
            for sampled_roles in by_class.values():
                self.assertEqual(sampled_roles.count("modern_original"), 1)
                historic_roles = [role for role in sampled_roles if role.startswith("historic_")]
                self.assertEqual(len(historic_roles), 3)
                self.assertEqual(len(set(historic_roles)), 3)

    def test_role_stratified_sampler_dispatch_uses_dataset_roles(self):
        from experiments.core.finetuning.train import _build_sampler

        labels, roles = self._role_sampler_inputs()
        dataset = SimpleNamespace(
            class_ids=labels,
            labels=list(range(len(labels))),
            roles=roles,
            items=[],
        )
        sampler = _build_sampler(
            dataset,
            {
                "type": "role_stratified_pk",
                "classes_per_batch": 3,
                "images_per_class": 4,
                "anchor_role": "modern_original",
                "positive_roles": [
                    "historic_archival",
                    "historic_print",
                    "historic_cropped_record",
                    "historic_framed_photo",
                ],
            },
            {"seed": 19, "batches_per_epoch": 1},
            SimpleNamespace(),
        )
        self.assertIsInstance(sampler, RoleStratifiedPKBatchSampler)
        self.assertEqual(len(next(iter(sampler))), 12)

    def test_role_stratified_pk_sampler_rejects_invalid_role_coverage(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            RoleStratifiedPKBatchSampler(
                ["a", "a"],
                ["modern_original"],
                1,
                2,
                anchor_role="modern_original",
                positive_roles=["historic"],
            )
        with self.assertRaisesRegex(ValueError, "Every class"):
            RoleStratifiedPKBatchSampler(
                ["a", "a", "b", "b"],
                ["modern_original", "historic", "modern_original", "other"],
                2,
                2,
                anchor_role="modern_original",
                positive_roles=["historic"],
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            RoleStratifiedPKBatchSampler(
                ["a", "a"],
                ["modern_original", "historic"],
                1,
                3,
                anchor_role="modern_original",
                positive_roles=["historic"],
            )

    def test_hard_negative_sampler_forces_mined_images(self):
        import pandas as pd

        labels = ["a", "a", "b", "b"]
        image_ids = ["a0", "a1", "b0", "b1"]
        hard_pairs = [{0, 3}, {1, 2}]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hard_negatives.parquet"
            pd.DataFrame(
                [
                    {"anchor_image_id": "a0", "negative_image_id": "b1", "anchor_class_id": "a", "negative_class_id": "b", "score": 0.9},
                    {"anchor_image_id": "b0", "negative_image_id": "a1", "anchor_class_id": "b", "negative_class_id": "a", "score": 0.8},
                ]
            ).to_parquet(path, index=False)
            sampler = HardNegativePKBatchSampler(
                labels,
                classes_per_batch=2,
                images_per_class=1,
                hard_negative_parquet=path,
                hard_negatives_per_anchor_class=1,
                image_ids=image_ids,
                batches_per_epoch=3,
                seed=11,
            )
            batches = list(sampler)
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        self.assertTrue(all(set(batch) in hard_pairs for batch in batches))

    def test_hard_negative_sampler_falls_back_to_pk_without_mined_negatives(self):
        sampler = HardNegativePKBatchSampler([0, 0, 1], classes_per_batch=2, images_per_class=4, batches_per_epoch=2, seed=7)
        batches = list(sampler)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch) == 8 for batch in batches))
        self.assertTrue(all(0 <= idx <= 2 for batch in batches for idx in batch))

    def test_gem_pooling_shapes(self):
        gem = GeMPooling2d(p=3.0)
        self.assertEqual(tuple(gem(torch.ones(2, 3, 4, 5)).shape), (2, 3))
        token_gem = TokenGeMPooling(p=3.0)
        self.assertEqual(tuple(token_gem(torch.ones(2, 7, 5)).shape), (2, 5))

    def test_calibrated_eval_and_mining(self):
        image_ids = ["a1", "a2", "b1", "b2"]
        class_ids = ["a", "a", "b", "b"]
        embeddings = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
        result = evaluate_embeddings(image_ids, class_ids, embeddings, target_pc=1.0, top_k=[1])
        metrics = result["calibrated_threshold"]
        self.assertAlmostEqual(metrics["pair_completeness"], 1.0)
        self.assertGreater(metrics["pair_quality"], 0.5)
        self.assertAlmostEqual(metrics["query_coverage"], 1.0)
        mined = mine_hard_negatives_from_embeddings(image_ids, class_ids, embeddings, top_m=1)
        self.assertEqual(len(mined), 4)
        self.assertTrue((mined["anchor_class_id"] != mined["negative_class_id"]).all())

    def test_fixed_threshold_query_coverage_counts_queries_not_pairs(self):
        similarities = np.asarray(
            [[1.0, 0.9, 0.2], [0.9, 1.0, 0.3], [0.2, 0.3, 1.0]],
            dtype=np.float32,
        )
        positives = np.zeros_like(similarities, dtype=bool)
        valid = ~np.eye(3, dtype=bool)
        metrics = apply_threshold(similarities, positives, valid, threshold=0.8)
        self.assertEqual(metrics["candidate_pairs"], 2)
        self.assertAlmostEqual(metrics["query_coverage"], 2.0 / 3.0)

    def test_checkpoint_path_env_resolution_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "SMARTMATCH_MISSING_CHECKPOINT"):
            CheckpointProjectionAdapter._resolve_checkpoint_path(
                "local/model",
                None,
                "SMARTMATCH_MISSING_CHECKPOINT",
            )
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            with mock.patch.dict(os.environ, {"SMARTMATCH_TEST_CHECKPOINT": str(checkpoint)}):
                resolved = CheckpointProjectionAdapter._resolve_checkpoint_path(
                    "local/model",
                    None,
                    "SMARTMATCH_TEST_CHECKPOINT",
                )
        self.assertEqual(resolved, checkpoint.resolve())

    def test_cli_and_adapter_names_include_finetuning_commands(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--experiment", "exp.yml", "--epochs", "1"])
        self.assertEqual(args.command, "train")
        args = parser.parse_args(["mine-hard-negatives", "--experiment", "exp.yml", "--checkpoint", "ckpt.pt", "--subset", "train"])
        self.assertEqual(args.command, "mine-hard-negatives")
        self.assertEqual(args.subset, "train")
        self.assertIn("dinov3_projection_adapter", known_adapter_names())
        self.assertIn("resnet50_projection_adapter", known_adapter_names())
        self.assertIn("clip_gem_projection_adapter", known_adapter_names())

    def test_finetuning_seed_reproducibility_smoke_with_tiny_model(self):
        from experiments.core.finetuning.train import train_from_experiment

        class TinyModel(torch.nn.Module):
            model_type = "tiny_projection"
            backbone_name = "tiny/fake"
            backbone_revision = "tiny-rev"
            pooling_type = "flatten"
            projection_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3 * 4 * 4, self.projection_dim)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                flat = pixel_values.flatten(1)
                return torch.nn.functional.normalize(self.linear(flat), p=2, dim=1)

            def checkpoint_metadata(self) -> dict[str, object]:
                return {
                    "model_type": self.model_type,
                    "backbone_name": self.backbone_name,
                    "backbone_revision": self.backbone_revision,
                    "pooling_type": self.pooling_type,
                    "projection_dim": self.projection_dim,
                }

        def tensor_transform(image: Image.Image) -> torch.Tensor:
            arr = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._write_tiny_finetune_experiment(root)
            with mock.patch(
                "experiments.core.finetuning.train.build_trainable_model",
                side_effect=lambda _cfg: TinyModel(),
            ), mock.patch(
                "experiments.core.finetuning.train.build_image_transform",
                side_effect=lambda _cfg, train: tensor_transform,
            ):
                first = train_from_experiment(experiment, run_id="first")
                second = train_from_experiment(experiment, run_id="second")

            first_dir = Path(first["run_dir"])
            second_dir = Path(second["run_dir"])
            first_history = json.loads((first_dir / "training_history.json").read_text(encoding="utf-8"))
            second_history = json.loads((second_dir / "training_history.json").read_text(encoding="utf-8"))
            validation_csv = (first_dir / "validation_metrics.csv").read_text(encoding="utf-8")
            validation_svg = (first_dir / "training_metrics.svg").read_text(encoding="utf-8")
            from experiments.core.finetuning.checkpointing import load_checkpoint

            first_checkpoint = load_checkpoint(first_dir / "last_checkpoint.pt", map_location="cpu")
            second_checkpoint = load_checkpoint(second_dir / "last_checkpoint.pt", map_location="cpu")

        self.assertEqual(first_history, second_history)
        self.assertIn("validation_set", validation_csv)
        self.assertIn("primary_val", validation_csv)
        self.assertIn("pc_at_1", validation_csv)
        self.assertIn("pq_at_1", validation_csv)
        self.assertIn("qc_at_1", validation_csv)
        self.assertIn("is_best_checkpoint", validation_csv)
        self.assertIn("primary_val PC@1", validation_svg)
        self.assertIn("last e1", validation_svg)
        for name, value in first_checkpoint["model_state_dict"].items():
            self.assertTrue(torch.equal(value, second_checkpoint["model_state_dict"][name]), name)
        repro = first_checkpoint["reproducibility"]
        self.assertEqual(repro["seed"], 123)
        self.assertTrue(repro["determinism"]["torch_deterministic_algorithms"])
        self.assertEqual(repro["model"]["revision"], "tiny-config-rev")
        self.assertIn("rng_state_at_checkpoint", repro)
        self.assertIsNotNone(first_checkpoint["validation_threshold"])

    def test_role_aware_finetuning_validation_uses_directional_tasks(self):
        from experiments.core.finetuning.train import train_from_experiment

        class TinyModel(torch.nn.Module):
            model_type = "tiny_projection"
            backbone_name = "tiny/fake"
            backbone_revision = "tiny-rev"
            pooling_type = "flatten"
            projection_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3 * 4 * 4, self.projection_dim)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.normalize(self.linear(pixel_values.flatten(1)), p=2, dim=1)

            def checkpoint_metadata(self) -> dict[str, object]:
                return {
                    "model_type": self.model_type,
                    "backbone_name": self.backbone_name,
                    "backbone_revision": self.backbone_revision,
                    "pooling_type": self.pooling_type,
                    "projection_dim": self.projection_dim,
                }

        def tensor_transform(image: Image.Image) -> torch.Tensor:
            arr = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._write_tiny_finetune_experiment(root, role_aware=True)
            with mock.patch(
                "experiments.core.finetuning.train.build_trainable_model",
                side_effect=lambda _cfg: TinyModel(),
            ), mock.patch(
                "experiments.core.finetuning.train.build_image_transform",
                side_effect=lambda _cfg, train: tensor_transform,
            ):
                result = train_from_experiment(experiment, run_id="role-aware")
            history = json.loads((Path(result["run_dir"]) / "training_history.json").read_text(encoding="utf-8"))

        validation = history["history"][0]["validation"]["sets"]["primary_val"]
        task = validation["retrieval_tasks"][0]
        self.assertEqual(validation["primary_task_id"], "val_modern_to_historic")
        self.assertEqual(task["task_id"], "val_modern_to_historic")
        self.assertEqual(task["num_queries"], 2)
        self.assertEqual(task["num_candidates"], 2)
        self.assertEqual(task["num_positive_pairs"], 2)
        calibrated = validation["calibrated_threshold"]
        self.assertIsNotNone(calibrated.get("threshold"))
        self.assertEqual(calibrated["target_pair_completeness"], 1.0)
        self.assertEqual(calibrated["pair_completeness"], 1.0)
        self.assertEqual(calibrated["possible_pairs"], task["num_possible_pairs"])
        self.assertEqual(calibrated["total_positive_pairs"], task["num_positive_pairs"])

    def test_secondary_validation_set_is_reported_without_driving_checkpoint_selection(self):
        from experiments.core.finetuning.train import train_from_experiment

        class TinyModel(torch.nn.Module):
            model_type = "tiny_projection"
            backbone_name = "tiny/fake"
            backbone_revision = "tiny-rev"
            pooling_type = "flatten"
            projection_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3 * 4 * 4, self.projection_dim)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.normalize(self.linear(pixel_values.flatten(1)), p=2, dim=1)

            def checkpoint_metadata(self) -> dict[str, object]:
                return {"model_type": self.model_type, "backbone_name": self.backbone_name, "projection_dim": self.projection_dim}

        def tensor_transform(image: Image.Image) -> torch.Tensor:
            arr = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._write_tiny_finetune_experiment(root, secondary_validation=True)
            with mock.patch("experiments.core.finetuning.train.build_trainable_model", side_effect=lambda _cfg: TinyModel()), mock.patch(
                "experiments.core.finetuning.train.build_image_transform", side_effect=lambda _cfg, train: tensor_transform
            ):
                result = train_from_experiment(experiment, run_id="secondary")
            run_dir = Path(result["run_dir"])
            history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
            validation_csv = (run_dir / "validation_metrics.csv").read_text(encoding="utf-8")
            validation_svg = (run_dir / "training_metrics.svg").read_text(encoding="utf-8")

        validation = history["history"][0]["validation"]
        self.assertEqual(validation["primary"], "primary_val")
        self.assertIn("primary_val", validation["sets"])
        self.assertIn("secondary_val", validation["sets"])
        self.assertEqual(validation["sets"]["secondary_val"]["primary_task_id"], "secondary_modern_to_historic")
        self.assertIn("secondary_val", validation_csv)
        self.assertIn("secondary_val PC@1", validation_svg)
        self.assertIn("stroke-dasharray", validation_svg)

    def test_gradient_cache_runs_through_training_loop_and_persists_metadata(self):
        from experiments.core.finetuning.train import train_from_experiment

        class TinyDropoutModel(torch.nn.Module):
            model_type = "tiny_projection"
            backbone_name = "tiny/fake"
            projection_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3 * 4 * 4, self.projection_dim)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                values = self.dropout(self.linear(pixel_values.flatten(1)))
                return torch.nn.functional.normalize(values, p=2, dim=1)

            def checkpoint_metadata(self) -> dict[str, object]:
                return {
                    "model_type": self.model_type,
                    "backbone_name": self.backbone_name,
                    "projection_dim": self.projection_dim,
                }

        def tensor_transform(image: Image.Image) -> torch.Tensor:
            array = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1).contiguous()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._write_tiny_finetune_experiment(root, gradient_cache=True)
            with mock.patch(
                "experiments.core.config_schema.find_repo_root", return_value=Path.cwd()
            ), mock.patch(
                "experiments.core.finetuning.train.build_trainable_model",
                side_effect=lambda _cfg: TinyDropoutModel(),
            ), mock.patch(
                "experiments.core.finetuning.train.build_image_transform",
                side_effect=lambda _cfg, train: tensor_transform,
            ), mock.patch(
                "experiments.core.finetuning.train.evaluate_references", return_value={}
            ):
                result = train_from_experiment(experiment, run_id="gradcache")
            summary = json.loads(
                (Path(result["run_dir"]) / "training_summary.json").read_text(encoding="utf-8")
            )
            history = json.loads(
                (Path(result["run_dir"]) / "training_history.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["global_steps"], 1)
        self.assertTrue(summary["gradient_cache"]["enabled"])
        self.assertEqual(summary["gradient_cache"]["effective_batch_size"], 4)
        self.assertEqual(summary["gradient_cache"]["chunk_size"], 2)
        self.assertIsNotNone(history["history"][0]["train_mining"])

    def test_reduction_ratio_checkpoint_selection_keeps_earlier_epoch_on_tie(self):
        from experiments.core.finetuning.checkpointing import load_checkpoint
        from experiments.core.finetuning.train import train_from_experiment

        class TinyModel(torch.nn.Module):
            model_type = "tiny_projection"
            backbone_name = "tiny/fake"
            projection_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3 * 4 * 4, self.projection_dim)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.normalize(self.linear(pixel_values.flatten(1)), p=2, dim=1)

            def checkpoint_metadata(self) -> dict[str, object]:
                return {
                    "model_type": self.model_type,
                    "backbone_name": self.backbone_name,
                    "projection_dim": self.projection_dim,
                }

        def tensor_transform(image: Image.Image) -> torch.Tensor:
            arr = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        def validation_metrics(
            *, threshold: float, pc: float, rr: float, candidates: int, tp: int
        ) -> dict[str, object]:
            return {
                "num_images": 4,
                "image_ids": ["c3_0.png", "c3_1.png", "c4_0.png", "c4_1.png"],
                "target_pc": 0.99,
                "calibrated_threshold": {
                    "threshold": threshold,
                    "target_pair_completeness": 0.99,
                    "pair_completeness": pc,
                    "pair_quality": tp / candidates,
                    "reduction_ratio": rr,
                    "query_coverage": 1.0,
                    "candidate_pairs": candidates,
                    "possible_pairs": 1000,
                    "true_positives": tp,
                    "false_positives": candidates - tp,
                    "false_negatives": 100 - tp,
                    "total_positive_pairs": 100,
                },
                "recall_at_k": [
                    {
                        "k": 1,
                        "pair_completeness": pc,
                        "pair_quality": tp / candidates,
                        "query_coverage": 1.0,
                        "candidate_pairs": candidates,
                    }
                ],
            }

        sequence = [
            validation_metrics(threshold=0.5, pc=0.99, rr=0.88, candidates=120, tp=99),
            validation_metrics(threshold=0.6, pc=0.99, rr=0.90, candidates=100, tp=99),
            # Better PC/PQ and a different threshold must not break the RR tie.
            validation_metrics(threshold=0.7, pc=1.0, rr=0.90, candidates=100, tp=100),
        ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._write_tiny_finetune_experiment(
                root, rr_checkpoint_selection=True, epochs=3
            )
            with mock.patch(
                "experiments.core.finetuning.train.build_trainable_model",
                side_effect=lambda _cfg: TinyModel(),
            ), mock.patch(
                "experiments.core.finetuning.train.build_image_transform",
                side_effect=lambda _cfg, train: tensor_transform,
            ), mock.patch(
                "experiments.core.finetuning.train._evaluate_model",
                side_effect=sequence,
            ):
                result = train_from_experiment(experiment, run_id="rr-selection")

            run_dir = Path(result["run_dir"])
            best = load_checkpoint(run_dir / "best_checkpoint.pt", map_location="cpu")
            last = load_checkpoint(run_dir / "last_checkpoint.pt", map_location="cpu")
            summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
            history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
            csv_rows = list(csv.DictReader((run_dir / "validation_metrics.csv").read_text().splitlines()))
            svg = (run_dir / "training_metrics.svg").read_text(encoding="utf-8")

        self.assertEqual(best["epoch"], 2)
        self.assertEqual(last["epoch"], 3)
        self.assertEqual(best["validation_threshold"], 0.6)
        self.assertEqual(best["validation_metrics"]["calibrated_threshold"]["target_pair_completeness"], 0.99)
        self.assertEqual(best["validation_metrics"]["calibrated_threshold"]["pair_completeness"], 0.99)
        self.assertEqual(best["training_config"]["checkpoint_selection"]["metric"], "reduction_ratio_at_target_pc")
        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["best_validation_threshold"], 0.6)
        self.assertEqual(summary["best_validation_reduction_ratio"], 0.9)
        self.assertEqual(summary["best_validation_candidate_pairs"], 100)
        self.assertEqual(summary["best_validation_possible_pairs"], 1000)
        self.assertEqual(history["history"][1]["checkpoint_selection_result"]["is_new_best"], True)
        self.assertEqual(history["history"][2]["checkpoint_selection_result"]["is_new_best"], False)
        self.assertEqual([row["epoch"] for row in csv_rows if row["is_best_checkpoint"] == "true"], ["2"])
        self.assertEqual([row["epoch"] for row in csv_rows if row["is_last_checkpoint"] == "true"], ["3"])
        self.assertTrue(all(row["target_pair_completeness"] == "0.99" for row in csv_rows))
        self.assertIn("RR@PC&gt;=0.99", svg)

    def _write_tiny_finetune_experiment(
        self,
        root: Path,
        role_aware: bool = False,
        secondary_validation: bool = False,
        *,
        rr_checkpoint_selection: bool = False,
        gradient_cache: bool = False,
        epochs: int = 1,
    ) -> Path:
        image_root = root / "images"
        image_root.mkdir()
        db_path = root / "dataset.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE classes(class_id INTEGER PRIMARY KEY, qid TEXT NOT NULL UNIQUE, label TEXT);
            CREATE TABLE image_files(file_id TEXT PRIMARY KEY, file_ext TEXT NOT NULL, local_rel_path TEXT, download_status TEXT NOT NULL, source_role TEXT);
            CREATE TABLE image_file_classes(image_file_class_id INTEGER PRIMARY KEY AUTOINCREMENT, file_id TEXT NOT NULL, class_id INTEGER NOT NULL, primary_label TEXT);
            CREATE TABLE image_file_class_tags(image_file_class_id INTEGER NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(image_file_class_id, tag));
            CREATE TABLE splits(split_name TEXT NOT NULL, file_id TEXT NOT NULL, PRIMARY KEY(split_name, file_id));
            """
        )
        class_rows = [(idx, f"Q{idx}", f"class {idx}") for idx in range(1, 5)]
        conn.executemany("INSERT INTO classes(class_id, qid, label) VALUES (?, ?, ?)", class_rows)
        file_rows = []
        rel_rows = []
        split_rows = []
        for class_id in range(1, 5):
            for image_index in range(2):
                file_id = f"c{class_id}_{image_index}"
                color = (class_id * 40, image_index * 80, 255 - class_id * 30)
                Image.new("RGB", (8, 8), color).save(image_root / f"{file_id}.png")
                file_rows.append((file_id, ".png", f"images/{file_id}.png", "downloaded", "synthetic"))
                rel_rows.append((file_id, class_id, "same_painting"))
                split_rows.append(("same_painting", file_id))
        conn.executemany(
            "INSERT INTO image_files(file_id, file_ext, local_rel_path, download_status, source_role) VALUES (?, ?, ?, ?, ?)",
            file_rows,
        )
        conn.executemany("INSERT INTO image_file_classes(file_id, class_id, primary_label) VALUES (?, ?, ?)", rel_rows)
        conn.executemany("INSERT INTO splits(split_name, file_id) VALUES (?, ?)", split_rows)
        conn.commit()
        conn.close()

        dataset_yml = root / "dataset.yml"
        model_yml = root / "models.yml"
        split_json = root / "split.json"
        experiment_yml = root / "experiment.yml"
        dataset_yml.write_text(
            "schema_version: 1\n"
            "dataset_id: tiny\n"
            "paths:\n"
            f"  dataset_db: {db_path}\n"
            f"  image_root: {image_root}\n"
            "identity:\n"
            "  expected_num_classes: 4\n"
            "  expected_num_images: 8\n",
            encoding="utf-8",
        )
        model_yml.write_text(
            "schema_version: 1\n"
            "adapter_name: siglip2_adapter\n"
            "models:\n"
            "  - model_id: fake/tiny\n"
            "    revision: tiny-config-rev\n",
            encoding="utf-8",
        )
        train_ids = ["c1_0.png", "c1_1.png", "c2_0.png", "c2_1.png"]
        val_ids = ["c3_0.png", "c3_1.png", "c4_0.png", "c4_1.png"]
        val_roles = {"all": val_ids}
        if role_aware:
            val_roles.update({"modern": ["c3_0.png", "c4_0.png"], "historic": ["c3_1.png", "c4_1.png"]})
        write_split(
            split_json,
            {
                "schema_version": 2,
                "split_id": "tiny_split",
                "dataset": {
                    "dataset_id": "tiny",
                    "image_root": str(image_root),
                    "identity": {"num_classes": 4, "num_images": 8},
                    "source_db_sha256": "test",
                },
                "subsets": {
                    "train": {"roles": {"all": train_ids}},
                    "val": {"roles": val_roles},
                },
                "images": {
                    "c1_0.png": {"class_id": 1}, "c1_1.png": {"class_id": 1},
                    "c2_0.png": {"class_id": 2}, "c2_1.png": {"class_id": 2},
                    "c3_0.png": {"class_id": 3}, "c3_1.png": {"class_id": 3},
                    "c4_0.png": {"class_id": 4}, "c4_1.png": {"class_id": 4},
                },
            },
        )
        secondary_dataset_yaml = ""
        if secondary_validation:
            secondary_root = root / "secondary"
            secondary_image_root = secondary_root / "images"
            secondary_image_root.mkdir(parents=True)
            secondary_split = secondary_root / "split.json"
            secondary_ids = []
            secondary_images = {}
            modern_ids = []
            historic_ids = []
            for class_id in (5, 6):
                for role_index, role in enumerate(("modern", "historic")):
                    file_id = f"s{class_id}_{role}.png"
                    Image.new("RGB", (8, 8), (class_id * 25, role_index * 120, 80)).save(secondary_image_root / file_id)
                    secondary_ids.append(file_id)
                    secondary_images[file_id] = {"class_id": class_id}
                    (modern_ids if role == "modern" else historic_ids).append(file_id)
            write_split(
                secondary_split,
                {
                    "schema_version": 2,
                    "split_id": "secondary_split",
                    "dataset": {
                        "dataset_id": "secondary_tiny",
                        "image_root": str(secondary_image_root),
                        "identity": {"num_classes": 2, "num_images": 4},
                        "source_db_sha256": "test",
                    },
                    "subsets": {"val": {"roles": {"modern": modern_ids, "historic": historic_ids}}},
                    "images": secondary_images,
                },
            )
            secondary_dataset_yaml = (
                "      - name: secondary_val\n"
                "        primary: false\n"
                "        dataset:\n"
                f"          path: {secondary_root}\n"
                "        split:\n"
                f"          file: {secondary_split}\n"
                "        subset: val\n"
                "        roles: [ALL_ROLES]\n"
                "        retrieval_tasks:\n"
                "        - task_id: secondary_modern_to_historic\n"
                "          query: {subset: val, role: modern}\n"
                "          candidates: {subset: val, role: historic}\n"
                "          positive_policy: same_painting\n"
                "          exclude_self: true\n"
            )
        retrieval_tasks_yaml = ""
        if role_aware:
            retrieval_tasks_yaml = (
                "    retrieval_tasks:\n"
                "    - task_id: val_modern_to_historic\n"
                "      query: {subset: val, role: modern}\n"
                "      candidates: {subset: val, role: historic}\n"
                "      positive_policy: same_painting\n"
                "      exclude_self: false\n"
            )
        checkpoint_selection_yaml = ""
        target_pc = 1.0
        if rr_checkpoint_selection:
            target_pc = 0.99
            checkpoint_selection_yaml = (
                "    checkpoint_selection:\n"
                "      metric: reduction_ratio_at_target_pc\n"
            )
        gradient_cache_yaml = (
            "    gradient_cache:\n"
            "      enabled: true\n"
            "      chunk_size: 2\n"
            "      verify_replay: true\n"
            if gradient_cache
            else ""
        )
        experiment_yml.write_text(
            "schema_version: 1\n"
            "experiment_id: tiny_finetune\n"
            "dataset:\n"
            f"  path: {root}\n"
            "split:\n"
            f"  file: {split_json}\n"
            "models:\n"
            "  include:\n"
            f"    - config: {model_yml}\n"
            "      model_id: fake/tiny\n"
            "finetuning:\n"
            "  model:\n"
            "    type: resnet50_gem_projection\n"
            "    backbone: tiny/fake\n"
            "    revision: tiny-config-rev\n"
            "    projection_dim: 4\n"
            "  sampler:\n"
            "    classes_per_batch: 2\n"
            "    images_per_class: 2\n"
            "  optimizer:\n"
            "    name: adamw\n"
            "    lr: 0.001\n"
            "    weight_decay: 0.0\n"
            "  train:\n"
            f"    epochs: {epochs}\n"
            "    precision: float32\n"
            "    num_workers: 0\n"
            "    seed: 123\n"
            "    deterministic: true\n"
            f"{gradient_cache_yaml}"
            "  evaluation:\n"
            f"    target_pc: {target_pc}\n"
            "    top_k: [1]\n"
            "    batch_size: 4\n"
            f"{checkpoint_selection_yaml}"
            f"{retrieval_tasks_yaml}"
            "  data:\n"
            "    train: {subset: train, roles: [ALL_ROLES]}\n"
            "    validation:\n"
            "      - name: primary_val\n"
            "        primary: true\n"
            "        subset: val\n"
            "        roles: [ALL_ROLES]\n"
            f"{secondary_dataset_yaml}"
            "run:\n"
            f"  output_dir: {root / 'runs'}\n",
            encoding="utf-8",
        )
        return experiment_yml


if __name__ == "__main__":
    unittest.main()
