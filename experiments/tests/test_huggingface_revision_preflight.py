import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.core.adapter_registry import AdapterConfig
from experiments.cli import main
from experiments.core.config_schema import ModelConfig
from experiments.core.preflight import (
    HuggingFaceRevisionError,
    validate_huggingface_revisions,
)
from experiments.core.runner import ModelEvaluationError


class HuggingFaceRevisionPreflightTest(unittest.TestCase):
    def _model(
        self,
        revision: str | None,
        *,
        model_id: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        model_name_or_path: str | None = None,
    ) -> ModelConfig:
        kwargs = {}
        if model_name_or_path is not None:
            kwargs["model_name_or_path"] = model_name_or_path
        adapter = AdapterConfig(
            adapter_name="dino_adapter",
            model_id=model_id,
            kwargs=kwargs,
            revision=revision,
        )
        return ModelConfig(
            path=Path("dinov3.yml"),
            model_id=adapter.model_id,
            adapter=adapter,
            raw={
                "model_id": adapter.model_id,
                "adapter_kwargs": kwargs,
                "revision": revision,
            },
            revision=revision,
        )

    def test_verifies_pinned_revision_without_downloading_weights(self) -> None:
        revision = "5931719e67bbdb9737e363e781fb0c67687896bc"
        api = mock.Mock()
        api.model_info.return_value = SimpleNamespace(sha=revision)

        with mock.patch("huggingface_hub.HfApi", return_value=api):
            results = validate_huggingface_revisions([self._model(revision)], token="token")

        api.model_info.assert_called_once_with(
            repo_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
            revision=revision,
            token="token",
        )
        self.assertEqual(results[0]["status"], "verified")
        self.assertEqual(results[0]["hub_repo_id"], self._model(revision).model_id)
        self.assertEqual(results[0]["resolved_revision"], revision)

    def test_validates_logical_alias_against_model_name_or_path(self) -> None:
        revision = "5931719e67bbdb9737e363e781fb0c67687896bc"
        hub_repo_id = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        model = self._model(
            revision,
            model_id="local/dinov3-vith16plus-pretrain-lvd1689m-model-default",
            model_name_or_path=hub_repo_id,
        )
        api = mock.Mock()
        api.model_info.return_value = SimpleNamespace(sha=revision)

        with mock.patch("huggingface_hub.HfApi", return_value=api):
            results = validate_huggingface_revisions([model], token="token")

        api.model_info.assert_called_once_with(
            repo_id=hub_repo_id,
            revision=revision,
            token="token",
        )
        self.assertEqual(results[0]["model_id"], model.model_id)
        self.assertEqual(results[0]["hub_repo_id"], hub_repo_id)
        self.assertEqual(results[0]["status"], "verified")

    def test_variant_identity_verifies_adapter_source_model(self) -> None:
        revision = "5931719e67bbdb9737e363e781fb0c67687896bc"
        source_model_id = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        base = self._model(revision, model_id=source_model_id)
        variant = ModelConfig(
            path=base.path,
            model_id=f"model-default/{source_model_id}",
            adapter=base.adapter,
            raw={**base.raw, "source_model_id": source_model_id},
            revision=revision,
        )
        api = mock.Mock()
        api.model_info.return_value = SimpleNamespace(sha=revision)

        with mock.patch("huggingface_hub.HfApi", return_value=api):
            results = validate_huggingface_revisions([variant], token="token")

        api.model_info.assert_called_once_with(
            repo_id=source_model_id,
            revision=revision,
            token="token",
        )
        self.assertEqual(results[0]["model_id"], variant.model_id)
        self.assertEqual(results[0]["hub_repo_id"], source_model_id)

    def test_corrupted_but_well_formed_sha_fails_preflight(self) -> None:
        corrupted = "5931719e67bbdb9737e363e781fb0c65127896bc"
        api = mock.Mock()
        api.model_info.side_effect = RuntimeError("Revision Not Found")

        with mock.patch("huggingface_hub.HfApi", return_value=api):
            with self.assertRaises(HuggingFaceRevisionError) as raised:
                validate_huggingface_revisions([self._model(corrupted)])

        failure = raised.exception.results[0]
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["configured_revision"], corrupted)
        self.assertIn("Revision Not Found", failure["error"])

    def test_rejects_non_commit_revision(self) -> None:
        api = mock.Mock()
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            with self.assertRaises(HuggingFaceRevisionError) as raised:
                validate_huggingface_revisions([self._model("main")])

        api.model_info.assert_not_called()
        self.assertIn("40-character", raised.exception.results[0]["error"])

    def test_models_without_revision_do_not_make_hub_request(self) -> None:
        with mock.patch("huggingface_hub.HfApi") as api_class:
            results = validate_huggingface_revisions([self._model(None)])

        self.assertEqual(results, [])
        api_class.assert_not_called()

    def test_cli_reports_preflight_failure_as_nonzero(self) -> None:
        error = HuggingFaceRevisionError(
            [
                {
                    "model_id": "fake/model",
                    "configured_revision": "0" * 40,
                    "status": "error",
                    "error": "Revision Not Found",
                }
            ]
        )
        with mock.patch("experiments.cli._validate", side_effect=error), mock.patch(
            "sys.stderr"
        ) as stderr, mock.patch("sys.stdout"):
            code = main(["validate", "--experiment", "experiment.yml"])

        self.assertEqual(code, 1)
        self.assertTrue(stderr.write.called)

    def test_cli_reports_model_evaluation_failure_as_nonzero(self) -> None:
        failure = {
            "model_id": "fake/model",
            "status": "failed",
            "error": "model load failed",
        }
        error = ModelEvaluationError(Path("runs/failed"), [failure])
        with mock.patch("experiments.cli._run", side_effect=error), mock.patch(
            "sys.stderr"
        ) as stderr, mock.patch("sys.stdout"):
            code = main(["run", "--experiment", "experiment.yml"])

        self.assertEqual(code, 1)
        self.assertTrue(stderr.write.called)


if __name__ == "__main__":
    unittest.main()
