from __future__ import annotations

from pathlib import Path

import pytest

from deduplication import file_io
from deduplication.config import Config, ModelConfig, RetrievalConfig


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "deduplication-project"
    root.mkdir()
    monkeypatch.setattr(file_io, "PROJECT_DIR", root)
    monkeypatch.setattr(file_io, "WORK_DIR", root / "work")
    return root


@pytest.fixture
def config(project: Path) -> Config:
    model = ModelConfig(
        model_id="fixture/dino",
        revision="abc123",
        pooling="default",
        resize_mode="letterbox",
        resize_size=32,
        normalize=True,
        dtype="float16",
        batch_size=2,
        preprocess_workers=2,
        output_dim=3,
    )
    retrieval = RetrievalConfig(
        min_retrieval_window=2, cosine_threshold=0.5, comparison=">=", query_batch_size=2
    )
    raw = {
        "model": {"id": model.model_id, "revision": model.revision},
        "retrieval": {"min_retrieval_window": 2, "cosine_threshold": 0.5, "comparison": ">="},
        "calibration": {"run_id": "fixture", "artifact_path": "fixture", "threshold_source": "fixture"},
    }
    # image_matching/matching/classifier.pkl in https://github.com/HPI-Information-Systems/smARTmatch
    classifier = file_io.REPO_ROOT / "experiments/datasets/deduplication/deduplication/classifier.pkl"
    return Config(
        path=project / "config.toml",
        raw=raw,
        images_dir=project / "work/images",
        extensions=(".jpg", ".png"),
        model=model,
        retrieval=retrieval,
        superpoint_max_num_keypoints=128,
        classifier_path=classifier,
        classifier_sha256="f9b313a44d86b7352583cd365d2af9181ec018708768aeaeaa4c762ebe7ff24d",
        keeper_source_priority=("alpha", "beta"),
        calibration=raw["calibration"],
    )
