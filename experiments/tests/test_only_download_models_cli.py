import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from experiments.core.adapter_registry import AdapterConfig
from experiments.cli import build_parser, main
from experiments.core.config_schema import ModelConfig
from experiments.core.model_downloader import (
    ModelDownloadError,
    _download_and_verify_model,
    _verify_snapshot,
    download_experiment_models,
)


class OnlyDownloadModelsCliTest(unittest.TestCase):
    def test_cli_only_download_models_dispatches_downloader(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--experiment",
                "experiment.yml",
                "--only-download-models",
            ]
        )

        with mock.patch(
            "experiments.cli.download_experiment_models",
            return_value=[{"model_id": "fake/one", "status": "verified"}],
        ) as downloader:
            result = args.handler(args)

        downloader.assert_called_once_with(Path("experiment.yml"), model_filter=None)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["num_downloaded_models"], 1)
        self.assertEqual(result["downloaded_models"][0]["model_id"], "fake/one")

    def test_download_experiment_models_uses_all_models_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_yml = self._write_experiment(root)

            with mock.patch(
                "experiments.core.model_downloader._download_and_verify_model",
                side_effect=lambda model, _token: {"model_id": model.model_id, "status": "verified"},
            ) as download:
                downloaded = download_experiment_models(experiment_yml)

            self.assertEqual([model["model_id"] for model in downloaded], ["fake/one", "fake/two"])
            self.assertEqual([call.args[0].model_id for call in download.call_args_list], ["fake/one", "fake/two"])
            self.assertFalse((root / "runs").exists())

    def test_checkpoint_backed_models_are_skipped_by_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_yml = self._write_experiment(root, adapter_name="resnet50_projection_adapter")
            model_yml = root / "models.yml"
            model_yml.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "adapter_name": "resnet50_projection_adapter",
                        "models": [
                            {
                                "model_id": "local/finetuned",
                                "adapter_kwargs": {"checkpoint_path_env": "SMARTMATCH_TEST_CHECKPOINT"},
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            experiment = yaml.safe_load(experiment_yml.read_text(encoding="utf-8"))
            experiment["models"]["include"] = [{"config": str(model_yml), "model_id": "local/finetuned"}]
            experiment_yml.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")

            downloaded = download_experiment_models(experiment_yml)

        self.assertEqual(downloaded[0]["status"], "skipped_local_checkpoint")

    def test_variant_download_uses_adapter_source_model_id(self) -> None:
        source_model_id = "google/siglip2-giant-opt-patch16-384"
        model = ModelConfig(
            path=Path("siglip2.yml"),
            model_id=f"model-default/{source_model_id}",
            adapter=AdapterConfig("siglip2_adapter", source_model_id, revision="a" * 40),
            raw={"source_model_id": source_model_id},
            revision="a" * 40,
        )
        snapshot = Path("/cache/snapshots") / ("b" * 40)
        with mock.patch(
            "experiments.core.model_downloader._download_snapshot", return_value=snapshot
        ) as download, mock.patch(
            "experiments.core.model_downloader._local_snapshot", return_value=snapshot
        ) as local, mock.patch(
            "experiments.core.model_downloader._remote_model_files", return_value={"model.safetensors": 3}
        ) as remote, mock.patch(
            "experiments.core.model_downloader._verify_snapshot", return_value={"num_files": 1}
        ):
            result = _download_and_verify_model(model, token="token")

        download.assert_called_once_with(source_model_id, "token", "a" * 40)
        local.assert_called_once_with(source_model_id, "token", "b" * 40)
        remote.assert_called_once_with(source_model_id, "token", "b" * 40)
        self.assertEqual(result["model_id"], model.model_id)
        self.assertEqual(result["status"], "verified")

    def test_download_verification_failure_exits_nonzero_and_prints_error(self) -> None:
        error = ModelDownloadError(
            [
                {
                    "model_id": "fake/one",
                    "model_storage_key": "fake_one",
                    "status": "error",
                    "error": "Missing downloaded files: ['model.safetensors']",
                }
            ]
        )
        with mock.patch(
            "experiments.cli.download_experiment_models",
            side_effect=error,
        ), mock.patch("sys.stderr") as stderr, mock.patch("sys.stdout"):
            code = main(["run", "--experiment", "experiment.yml", "--only-download-models"])

        self.assertEqual(code, 1)
        self.assertTrue(stderr.write.called)

    def test_verify_snapshot_requires_all_remote_files_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"abc")

            summary = _verify_snapshot(
                snapshot,
                {"config.json": 2, "model.safetensors": 3},
            )
            self.assertEqual(summary["num_files"], 2)
            self.assertEqual(summary["num_weight_files"], 1)

            with self.assertRaisesRegex(RuntimeError, "Missing downloaded files"):
                _verify_snapshot(snapshot, {"missing.safetensors": 1})

    def _write_experiment(self, root: Path, adapter_name: str = "siglip2_adapter") -> Path:
        dataset_yml = root / "dataset.yml"
        model_yml = root / "models.yml"
        experiment_yml = root / "experiment.yml"

        dataset_yml.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "dataset_id": "toy",
                    "paths": {"dataset_db": str(root / "missing.db")},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        model_yml.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "adapter_name": adapter_name,
                    "models": [
                        {"model_id": "fake/one", "embedding": {"pooling": "default"}},
                        {"model_id": "fake/two", "embedding": {"pooling": "default"}},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        experiment_yml.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "experiment_id": "warmup",
                    "dataset": {"path": str(root), "dataset_id": "toy"},
                    "split": {"file": str(root / "missing_split.json")},
                    "models": {
                        "include": [
                            {"config": str(model_yml), "model_id": "fake/one"},
                            {"config": str(model_yml), "model_id": "fake/two"},
                        ]
                    },
                    "embedding": {"cache_dir": str(root / "cache")},
                    "run": {"output_dir": str(root / "runs")},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return experiment_yml


if __name__ == "__main__":
    unittest.main()
