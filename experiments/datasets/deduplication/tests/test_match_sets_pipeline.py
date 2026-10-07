from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from deduplication import embedding, file_io, manifest, verification
from deduplication.pipeline import match_sets


class PipelineEmbedder:
    output_dim = 3
    calls = 0

    def __init__(self, _model):
        type(self).calls += 1

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        rows = [[1.0, 0.0, 0.0] if "red" in path.name else [0.0, 1.0, 0.0] for path in paths]
        return np.asarray(rows, dtype=np.float32)

    def close(self):
        pass


def test_match_sets_runs_all_stages_and_resumes(project, config, tmp_path, monkeypatch):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    Image.new("RGB", (8, 8), "red").save(a / "red.jpg")
    Image.new("RGB", (8, 8), "blue").save(a / "blue.jpg")
    Image.new("RGB", (8, 8), "red").save(b / "red-copy.jpg")
    Image.new("RGB", (8, 8), "green").save(b / "green.jpg")

    real_embed = embedding.embed
    monkeypatch.setattr(
        embedding,
        "embed",
        lambda cfg, **_kwargs: real_embed(cfg, embedder_factory=PipelineEmbedder),
    )
    monkeypatch.setattr(
        verification,
        "precompute_superpoint_features",
        lambda _config, **_kwargs: {
            "complete": True,
            "feature_dir": str(project / "work/verification/features"),
            "unique_content_count": 3,
        },
    )

    def fake_verify(_config, **_kwargs):
        candidates = file_io.read_csv_rows(project / "work/candidates/candidates.csv")
        rows = [
            {
                "pair_id": row["pair_id"],
                "status": "ok",
                "lightglue_prediction": "1",
                "lightglue_confidence": "0.9",
                "match_count": "10",
                "classifier_sha256": config.classifier_sha256,
                "superpoint_fingerprint": "fixture",
                "error": "",
            }
            for row in candidates
        ]
        path = project / "work/verification/verification.csv"
        file_io.atomic_write_csv(path, manifest.VERIFICATION_FIELDS, rows)
        summary = {
            "candidate_count": len(rows),
            "candidates_sha256": file_io.sha256_file(project / "work/candidates/candidates.csv"),
            "verification_sha256": file_io.sha256_file(path),
        }
        file_io.atomic_write_json(project / "work/verification/summary.json", summary)
        return summary

    monkeypatch.setattr(verification, "verify", fake_verify)
    PipelineEmbedder.calls = 0
    first = match_sets(config, a, b)
    assert first["mode"] == "bipartite"
    assert first["review_summary"]["review_pair_count"] > 0
    assert PipelineEmbedder.calls == 1
    state = file_io.read_json(project / "work/pipeline_state.json")
    assert all(stage["status"] == "completed" for stage in state["stages"].values())

    second = match_sets(config, a, b)
    assert second["retrieval_summary"]["candidates_sha256"] == first["retrieval_summary"]["candidates_sha256"]
    assert PipelineEmbedder.calls == 1
