import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.core.embedding_pipeline import (
    _validate_model_descriptor_policy,
    postprocess_descriptors,
)
from experiments.core.eval_pipeline import cosine_cache
from experiments.core.finetuning.evaluate import evaluate_embeddings
from experiments.core.retrieval_metrics import cosine_similarity_matrix
from experiments.core.runner import _evaluation_options


class EvaluationPrecisionPolicyTest(unittest.TestCase):
    def test_runner_defaults_to_mandatory_fp32(self) -> None:
        options = _evaluation_options(SimpleNamespace(raw={"evaluation": {}}))
        self.assertEqual(options.gpu_dtype, "float32")
        self.assertEqual(options.cache_dtype, "float32")
        self.assertEqual(options.fp32_matmul_precision, "ieee")

    def test_runner_rejects_non_fp32_compute_or_cache_dtype(self) -> None:
        for key in ("gpu_dtype", "cache_dtype"):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "FP32 evaluation is mandatory"
            ):
                _evaluation_options(
                    SimpleNamespace(raw={"evaluation": {key: "float16"}})
                )


class DescriptorPostprocessingTest(unittest.TestCase):
    def test_model_descriptor_policy_rejects_non_fp32_or_unnormalized_configs(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed to 'float32'"):
            _validate_model_descriptor_policy(
                SimpleNamespace(
                    model_id="example/model",
                    raw={"embedding": {"dtype": "float16", "normalize": True}},
                )
            )
        with self.assertRaisesRegex(ValueError, "requires normalized descriptors"):
            _validate_model_descriptor_policy(
                SimpleNamespace(
                    model_id="example/model",
                    raw={"embedding": {"dtype": "float32", "normalize": False}},
                )
            )

    def test_fp32_input_is_copied_and_normalized_without_mutation(self) -> None:
        source = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        original = source.copy()

        result = postprocess_descriptors(source, normalize=True, dtype=np.dtype("float32"))

        self.assertIsNot(result, source)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_allclose(
            result,
            np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_invalid_descriptors_fail_loudly(self) -> None:
        invalid_cases = (
            (np.asarray([1.0, 2.0], dtype=np.float32), "shape"),
            (np.asarray([[np.nan, 1.0]], dtype=np.float32), "NaN or infinity"),
            (np.asarray([[np.inf, 1.0]], dtype=np.float32), "NaN or infinity"),
            (np.asarray([[0.0, 0.0]], dtype=np.float32), "zero or near-zero"),
        )
        for values, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                postprocess_descriptors(values, normalize=True, dtype=np.dtype("float32"))
        with self.assertRaisesRegex(TypeError, "fixed to float32"):
            postprocess_descriptors(
                np.asarray([[1.0, 2.0]], dtype=np.float32),
                dtype=np.dtype("float16"),
            )
        with self.assertRaisesRegex(TypeError, "must be floating"):
            postprocess_descriptors(np.asarray([[1, 2]], dtype=np.int64))

    def test_large_finite_descriptors_are_normalized_without_overflow(self) -> None:
        result = postprocess_descriptors(
            np.asarray([[1e20, 1e20], [1e30, -1e30]], dtype=np.float32)
        )

        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=1),
            np.ones(2, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_normalized_cpu_similarity_is_direct_fp32_dot_product(self) -> None:
        raw = np.asarray([[3.0, 4.0], [4.0, -3.0]], dtype=np.float64)
        descriptors = postprocess_descriptors(raw, normalize=True, dtype=np.dtype("float32"))
        reference = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        reference = reference @ reference.T

        with mock.patch("numpy.linalg.norm", side_effect=AssertionError("double normalization")):
            similarities, metadata = cosine_similarity_matrix(
                descriptors,
                descriptors,
                device_preference="cpu",
                gpu_dtype="float32",
                inputs_normalized=True,
                fp32_matmul_precision="ieee",
                return_metadata=True,
            )

        np.testing.assert_allclose(similarities, reference, rtol=1e-6, atol=1e-7)
        self.assertEqual(similarities.dtype, np.float32)
        self.assertEqual(metadata["similarity_input_dtype"], "float32")
        self.assertEqual(metadata["similarity_output_dtype"], "float32")
        self.assertEqual(metadata["fp32_matmul_precision"], "ieee")
        self.assertFalse(metadata["tf32_enabled"])
        self.assertTrue(metadata["inputs_normalized"])

    def test_similarity_rejects_non_fp32_compute_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "FP32 similarity is mandatory"):
            cosine_similarity_matrix(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                device_preference="cpu",
                gpu_dtype="float16",
                inputs_normalized=True,
            )

    def test_similarity_cache_rejects_non_fp32_storage(self) -> None:
        descriptors = postprocess_descriptors(
            np.asarray([[1.0, 0.0]], dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "FP32 similarity caching is mandatory"):
            cosine_cache(
                descriptors,
                "cpu",
                "float32",
                "float16",
                inputs_normalized=True,
            )

    def test_finetuning_checkpoint_metrics_use_normalized_direct_dot(self) -> None:
        descriptors = postprocess_descriptors(
            np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
                dtype=np.float32,
            )
        )
        with mock.patch(
            "experiments.core.retrieval_metrics.postprocess_descriptors",
            side_effect=AssertionError("unexpected second normalization"),
        ):
            result = evaluate_embeddings(
                ["a1", "a2", "b1", "b2"],
                ["a", "a", "b", "b"],
                descriptors,
                target_pc=1.0,
                top_k=[1],
                inputs_normalized=True,
            )

        self.assertEqual(result["calibrated_threshold"]["pair_completeness"], 1.0)

    def test_normalized_similarity_rejects_non_unit_or_non_fp32_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "not unit length"):
            cosine_similarity_matrix(
                np.asarray([[2.0, 0.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                device_preference="cpu",
                inputs_normalized=True,
            )
        with self.assertRaisesRegex(TypeError, "must be float32"):
            cosine_similarity_matrix(
                np.asarray([[1.0, 0.0]], dtype=np.float16),
                np.asarray([[1.0, 0.0]], dtype=np.float16),
                device_preference="cpu",
                inputs_normalized=True,
            )


if __name__ == "__main__":
    unittest.main()
