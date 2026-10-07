"""Domain-specific training switches. The plain multi-similarity loss stays the default."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.core.checkpoint_naming import checkpoint_name_from_finetuning
from experiments.core.finetuning.cross_era import neighbor_lists, probe_plan
from experiments.core.finetuning.data import (
    MetricImageItem,
    MetricLearningImageDataset,
    append_hard_historic_views,
    append_online_historic_views,
    origin_code,
)
from experiments.core.finetuning.emulsion import historic_emulsion
from experiments.core.finetuning.losses import (
    CompositeMetricLoss,
    HistoricCatalogMemory,
    MultiSimilarityLoss,
    build_loss,
    koleo_historic,
    origin_prototype_pull,
    threshold_consistent_margin,
)
from experiments.core.finetuning.samplers import CrossEraGraphPKBatchSampler, RoleQuotaPKBatchSampler
from tooling.cross_era_linear_map import apply_map, blocking_metrics, fit_cca, fit_pair_whitening


def _unit(*rows: list[float]) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=1)


class DomainTrainingTests(unittest.TestCase):
    def test_plain_multi_similarity_stays_the_default(self) -> None:
        loss = build_loss({"name": "multi_similarity", "cross_origin_only": True})
        self.assertIsInstance(loss, MultiSimilarityLoss)
        self.assertNotIsInstance(loss, CompositeMetricLoss)
        self.assertTrue(loss.cross_origin_only)

    def test_threshold_margin_ignores_pairs_already_past_the_margins(self) -> None:
        embeddings = _unit([1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99])
        labels = torch.tensor([0, 0, 1, 1])
        origins = torch.tensor([0, 1, 0, 1])
        easy = threshold_consistent_margin(
            embeddings, labels, origins, m_pos=0.2, m_neg=0.2, weight_pos=1, weight_neg=1, cross_origin_only=True
        )
        self.assertEqual(float(easy), 0.0)
        hard = threshold_consistent_margin(
            embeddings, labels, origins, m_pos=0.99, m_neg=0.0, weight_pos=1, weight_neg=1, cross_origin_only=True
        )
        self.assertGreater(float(hard), 0.0)

    def test_koleo_is_smaller_when_historic_paintings_are_spread_out(self) -> None:
        labels = torch.tensor([0, 1, 0, 1])
        origins = torch.tensor([0, 0, 1, 1])
        packed = _unit([1, 0], [0.99, 0.1], [0, 1], [0, 1])
        spread = _unit([1, 0], [0, 1], [0, 1], [1, 0])
        packed_loss = float(koleo_historic(packed, labels, origins, weight=1))
        spread_loss = float(koleo_historic(spread, labels, origins, weight=1))
        self.assertGreater(packed_loss, spread_loss)

    def test_catalog_memory_adds_a_historic_negative_and_skips_the_same_painting(self) -> None:
        loss = build_loss({"name": "multi_similarity", "memory": {"enabled": True, "warmup_steps": 0}})
        self.assertIsInstance(loss, CompositeMetricLoss)
        assert isinstance(loss, CompositeMetricLoss)
        memory = loss.memory
        assert isinstance(memory, HistoricCatalogMemory)
        remembered = _unit([0, 1], [1, 0])
        memory.observe(remembered, torch.tensor([1, 0]), torch.tensor([0, 0]), torch.tensor([10, 11]))
        embeddings = _unit([1, 0], [0.9, 0.1])
        labels = torch.tensor([0, 0])
        origins = torch.tensor([1, 0])
        scores = memory.negative_scores(embeddings, labels, origins, torch.tensor([0, 1]), cross_origin_only=True)
        assert scores is not None
        self.assertTrue(torch.isfinite(scores[0, 0]))
        self.assertFalse(torch.isfinite(scores[0, 1]))
        self.assertFalse(torch.isfinite(scores[1]).any())
        value = loss(embeddings, labels, origins, indices=torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(value))

    def test_mixup_grows_the_batch_and_keeps_a_gradient(self) -> None:
        loss = build_loss(
            {
                "name": "multi_similarity",
                "embedding_mixup": {"cross_era_positives": True, "synthetic_hard_negatives": True},
            }
        )
        embeddings = _unit([1, 0], [0, 1], [0.2, 0.8], [0.8, 0.2]).detach().requires_grad_(True)
        labels = torch.tensor([0, 0, 1, 1])
        origins = torch.tensor([1, 0, 1, 0])
        value = loss(embeddings, labels, origins)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        assert embeddings.grad is not None
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)

    def test_graph_batch_keeps_the_nearest_confusers(self) -> None:
        labels: list[int] = []
        roles: list[str] = []
        for class_id in range(4):
            labels.extend([class_id, class_id, class_id, class_id])
            roles.extend(["modern_original", "modern_generated", "historic_archival", "historic_print"])
        sampler = CrossEraGraphPKBatchSampler(
            labels, roles, classes_per_batch=3, images_per_class=4, hard_fraction=1.0, neighbors_pool=2, batches_per_epoch=1, seed=1
        )
        sampler.set_neighbors({0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [0, 1]})
        batch = next(iter(sampler))
        selected = sampler.last_selected_classes[0]
        self.assertEqual(selected[1:], sampler.neighbors[selected[0]][:2])
        self.assertEqual(len(batch), 12)
        for class_id in selected:
            indexes = [index for index in batch if labels[index] == class_id]
            drawn = [roles[index] for index in indexes]
            self.assertEqual(sum(role.startswith("historic_") for role in drawn), 2)
            self.assertEqual(sum(role.startswith("modern_") for role in drawn), 2)

    def test_neighbor_guard_drops_a_confuser_closer_than_the_true_historic_view(self) -> None:
        modern = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        historic = np.array([[0.8, 0.6], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        modern /= np.linalg.norm(modern, axis=1, keepdims=True)
        historic /= np.linalg.norm(historic, axis=1, keepdims=True)
        neighbors, guarded = neighbor_lists(modern, historic, [0, 1, 2], pool=2)
        self.assertGreater(guarded, 0)
        self.assertNotIn(1, neighbors[0])

    def test_probe_plan_prefers_the_original_and_the_archival_view(self) -> None:
        labels = [0, 0, 0, 0]
        roles = ["modern_generated", "modern_original", "historic_print", "historic_archival"]
        self.assertEqual(probe_plan(labels, roles), [(0, 1, 3)])

    def test_role_quota_keeps_one_online_view_and_one_static_historic_view(self) -> None:
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        roles = [
            "modern_original",
            "modern_generated",
            "historic_archival",
            "historic_print",
            "historic_online",
            "modern_original",
            "modern_generated",
            "historic_print",
            "historic_framed_photo",
            "historic_online",
        ]
        sampler = RoleQuotaPKBatchSampler(
            labels,
            roles,
            classes_per_batch=2,
            quotas={"modern_original": 1, "modern_generated": 1, "historic": 1, "historic_online": 1},
            batches_per_epoch=1,
            seed=2,
        )
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 8)
        for class_id in (0, 1):
            drawn = [roles[index] for index in batch if labels[index] == class_id]
            self.assertEqual(drawn.count("historic_online"), 1)
            self.assertEqual(sum(role in {"historic_archival", "historic_print", "historic_framed_photo"} for role in drawn), 1)

    def test_emulsion_film_models_drop_color_and_blue_sensitive_drops_red(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = 255
        rng = np.random.default_rng(0)
        blue = historic_emulsion(image, rng, kind=0)
        self.assertTrue(np.array_equal(blue[..., 0], blue[..., 1]))
        luma = historic_emulsion(image, np.random.default_rng(0), kind=2)
        self.assertTrue(np.array_equal(luma[..., 0], luma[..., 1]))
        self.assertLess(float(blue.mean()), float(luma.mean()))
        again = historic_emulsion(image, np.random.default_rng(1))
        repeat = historic_emulsion(image, np.random.default_rng(1))
        self.assertTrue(np.array_equal(again, repeat))

    def test_online_and_hard_views_join_the_training_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "original.jpg"
            Image.new("RGB", (4, 4), (10, 20, 30)).save(image_path)
            hard_path = root / "hard.jpg"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(hard_path)
            items = [
                MetricImageItem("original.jpg", image_path, "7", "modern_original"),
                MetricImageItem("generated.jpg", image_path, "7", "modern_generated"),
            ]
            online = append_online_historic_views(items)
            self.assertEqual(online[-1].role, "historic_online")
            self.assertEqual(origin_code("historic_online"), 0)
            manifest = root / "manifest.json"
            manifest.write_text(
                '{"image_root": "%s", "views": [{"class_id": "7", "file_id": "hard.jpg", "role": "historic_archival"}]}'
                % root,
                encoding="utf-8",
            )
            extended = append_hard_historic_views(items, manifest)
            self.assertEqual(extended[-1].class_id, "7")
            self.assertTrue(extended[-1].path.is_file())
            dataset = MetricLearningImageDataset(online)
            sample = dataset[len(online) - 1]
            self.assertEqual(sample["role"], "historic_online")
            self.assertEqual(tuple(sample["pixel_values"].shape), (3, 4, 4))

    def test_checkpoint_name_accepts_the_new_samplers(self) -> None:
        config = {
            "model": {
                "type": "dinov3_lora_qv_projection",
                "lora": {"rank": 4, "alpha": 4},
                "head": {"type": "mlp_projection", "hidden_dim": 1024, "dropout": 0.0},
            },
            "optimizer": {"lr_lora": 3e-4, "lr_head": 1e-4},
            "train": {"epochs": 5},
            "sampler": {"type": "cross_era_graph_pk", "classes_per_batch": 64, "images_per_class": 4},
        }
        self.assertIn("sam-gpk", checkpoint_name_from_finetuning(config) or "")
        config["sampler"] = {
            "type": "role_quota_pk",
            "classes_per_batch": 64,
            "quotas": {"modern_original": 1, "historic": 1, "historic_online": 2},
        }
        self.assertIn("k-4", checkpoint_name_from_finetuning(config) or "")

    def test_whitening_shrinks_a_pure_era_direction(self) -> None:
        count = 32
        shared = np.random.default_rng(0).normal(size=(count, 2)).astype(np.float32)
        modern = shared + np.array([0.0, 2.0], dtype=np.float32)
        historic = shared + np.array([0.0, -2.0], dtype=np.float32)
        mapping = fit_pair_whitening(modern, historic)
        gap = modern.mean(axis=0) - historic.mean(axis=0)
        mapped_gap = gap @ mapping
        self.assertLess(abs(float(mapped_gap[1])), abs(float(gap[1])))

    def test_blocking_metrics_count_pairs(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 1, 0, 1, 2])
        metrics = blocking_metrics(embeddings, query_index=[0, 1], target_index=[2, 3, 4], labels=labels, ks=(1,))
        self.assertEqual(metrics["pc_at_1"], 1.0)
        self.assertGreater(metrics["reduction_ratio_at_0_99"], 0.0)

    def test_cca_lines_up_the_two_sides(self) -> None:
        rng = np.random.default_rng(1)
        shared = rng.normal(size=(40, 3)).astype(np.float32)
        modern = shared + rng.normal(scale=0.01, size=shared.shape).astype(np.float32)
        historic = shared @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        left, right = fit_cca(modern, historic)
        aligned_modern = apply_map(modern, left)
        aligned_historic = apply_map(historic, right)
        correlation = float(np.mean(np.sum(aligned_modern * aligned_historic, axis=1)))
        self.assertGreater(correlation, 0.5)

    def test_unit_origin_weights_match_the_unweighted_loss(self) -> None:
        embeddings = _unit([1, 0], [0.2, 0.8], [0, 1], [0.7, 0.3])
        labels = torch.tensor([0, 0, 1, 1])
        origins = torch.tensor([1, 0, 1, 0])
        plain = MultiSimilarityLoss()
        weighted = build_loss({"name": "multi_similarity", "origin_weights": {"same": 1, "cross": 1}})
        self.assertIsInstance(weighted, MultiSimilarityLoss)
        self.assertFalse(weighted.needs_origins)
        self.assertTrue(
            torch.allclose(plain(embeddings, labels), weighted(embeddings, labels, origins))
        )

    def test_zero_same_origin_weight_matches_the_hard_cross_origin_mask(self) -> None:
        embeddings = _unit([1, 0], [0.9, 0.2], [0.2, 0.9], [0, 1], [0.6, 0.4], [0.4, 0.6])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        origins = torch.tensor([1, 1, 0, 0, 0, 1])
        hard = build_loss({"name": "multi_similarity", "cross_origin_only": True})
        soft = build_loss({"name": "multi_similarity", "origin_weights": {"same": 0, "cross": 1}})
        assert isinstance(hard, MultiSimilarityLoss)
        assert isinstance(soft, MultiSimilarityLoss)
        self.assertTrue(soft.needs_origins)
        self.assertTrue(torch.allclose(hard(embeddings, labels, origins), soft(embeddings, labels, origins)))
        self.assertEqual(hard.last_stats.positive_pairs, soft.last_stats.positive_pairs)
        with self.assertRaisesRegex(ValueError, "origin_weights"):
            build_loss(
                {
                    "name": "multi_similarity",
                    "cross_origin_only": True,
                    "origin_weights": {"same": 0.5, "cross": 1},
                }
            )

    def test_half_same_origin_weight_keeps_same_origin_pairs_and_changes_the_loss(self) -> None:
        # The same-origin positive is orthogonal, so multi-similarity mines it.
        embeddings = _unit([1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9])
        labels = torch.tensor([0, 0, 0, 1])
        origins = torch.tensor([1, 1, 0, 0])
        half = build_loss({"name": "multi_similarity", "origin_weights": {"same": 0.5, "cross": 1}})
        dropped = build_loss({"name": "multi_similarity", "origin_weights": {"same": 0, "cross": 1}})
        half_value = half(embeddings, labels, origins)
        dropped_value = dropped(embeddings, labels, origins)
        self.assertTrue(torch.isfinite(half_value))
        self.assertGreater(half.last_stats.positive_pairs, dropped.last_stats.positive_pairs)
        self.assertFalse(torch.allclose(half_value, dropped_value))

    def test_modern_anchors_skip_historic_images(self) -> None:
        embeddings = _unit([1, 0], [0.2, 0.9], [0, 1], [0.8, 0.2])
        labels = torch.tensor([0, 0, 1, 1])
        origins = torch.tensor([0, 1, 0, 1])
        loss = build_loss({"name": "multi_similarity", "anchor_origin": "modern"})
        assert isinstance(loss, MultiSimilarityLoss)
        value = loss(embeddings, labels, origins)
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(loss.last_stats.valid_anchors, 2)

    def test_origin_prototypes_are_zero_when_the_two_means_match(self) -> None:
        embeddings = _unit([1, 0], [1, 0], [0, 1], [0, 1])
        labels = torch.tensor([0, 0, 1, 1])
        origins = torch.tensor([1, 0, 1, 0])
        matched = origin_prototype_pull(embeddings, labels, origins, weight=1)
        self.assertLess(float(matched), 1e-5)
        shifted = embeddings.clone()
        shifted[1] = _unit([0, 1])[0]
        apart = origin_prototype_pull(shifted, labels, origins, weight=1)
        self.assertGreater(float(apart), float(matched))
        built = build_loss(
            {"name": "multi_similarity", "regularizers": [{"name": "origin_prototypes", "weight": 0.5}]}
        )
        self.assertIsInstance(built, CompositeMetricLoss)
        self.assertEqual(built.prototype_weight, 0.5)
