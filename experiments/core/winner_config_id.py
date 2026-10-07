"""One identity string for a selected training configuration.

A stored benchmark may be shown as that winner only when it references the
same string. The string starts with a UUID and then names the pieces that
have to match:

    {uuid}__hyperparam-{hash}__training-{hash}__env-{hash}__loss-{hash}__checkpoint-{hash}

The UUID is UUID5 of those pieces, so the same training files always produce
the same id. A change to any piece changes the UUID and that piece's hash.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from experiments.core.artifacts import sha256_json

WINNER_CONFIG_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "painting-blocker/winner-config-id/v1",
)
SEGMENTS = ("hyperparam", "training", "env", "loss", "checkpoint")


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _finetuning(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment")
    if isinstance(experiment, dict) and isinstance(experiment.get("finetuning"), dict):
        return experiment["finetuning"]
    finetuning = config.get("finetuning")
    return finetuning if isinstance(finetuning, dict) else {}


def _checkpoint_sha(posted: dict[str, Any] | None) -> str:
    if not isinstance(posted, dict):
        return ""
    for item in posted.get("evaluations") or []:
        if isinstance(item, dict) and item.get("checkpoint_sha256"):
            return str(item["checkpoint_sha256"])
    return ""


def winner_config_parts(run_dir: Path) -> dict[str, Any] | None:
    """Return the five pieces, or None when the run has no training record."""
    config = _load(run_dir / "training_config.json") or {}
    summary = _load(run_dir / "training_summary.json") or {}
    posted = _load(run_dir / "post_evaluations.json")
    if not config and not summary and not posted:
        return None
    experiment = config.get("experiment") if isinstance(config.get("experiment"), dict) else {}
    environment = experiment.get("environment") if isinstance(experiment.get("environment"), dict) else {}
    dataset = experiment.get("dataset") if isinstance(experiment.get("dataset"), dict) else {}
    finetuning = _finetuning(config)
    evaluation = finetuning.get("evaluation") if isinstance(finetuning.get("evaluation"), dict) else {}
    reproducibility = summary.get("reproducibility") if isinstance(summary.get("reproducibility"), dict) else {}
    split = summary.get("training_split") if isinstance(summary.get("training_split"), dict) else {}
    git = reproducibility.get("git") if isinstance(reproducibility.get("git"), dict) else {}
    return {
        "hyperparam": {
            "model": finetuning.get("model"),
            "optimizer": finetuning.get("optimizer"),
            "sampler": finetuning.get("sampler"),
            "image_size": finetuning.get("image_size"),
            "preprocessing": finetuning.get("preprocessing"),
            "normalization": finetuning.get("normalization"),
        },
        "training": {
            "configuration_id": summary.get("configuration_id") or finetuning.get("configuration_id"),
            "training_seed": summary.get("training_seed", reproducibility.get("seed")),
            "epochs": summary.get("epochs"),
            "best_epoch": summary.get("best_epoch"),
            "checkpoint_selection": summary.get("checkpoint_selection") or evaluation.get("checkpoint_selection"),
            "split_id": split.get("split_id"),
            "split_sha256": split.get("split_sha256"),
            "split_strategy": split.get("split_strategy"),
            "dataset_id": dataset.get("dataset_id"),
            "git_commit": git.get("git_commit"),
            "train": finetuning.get("train"),
            "gradient_cache": summary.get("gradient_cache"),
        },
        "env": {
            "dependencies": reproducibility.get("dependencies"),
            "determinism": reproducibility.get("determinism"),
            "variables": environment.get("variables"),
        },
        "loss": finetuning.get("loss") if "loss" in finetuning else summary.get("loss"),
        "checkpoint": {
            "checkpoint_sha256": _checkpoint_sha(posted),
            "best_epoch": summary.get("best_epoch"),
        },
    }


def winner_config_id(run_dir: Path) -> str | None:
    """Return the global id, or None when the directory has no training record."""
    parts = winner_config_parts(run_dir)
    if parts is None:
        return None
    digests = {name: sha256_json(parts[name]) for name in SEGMENTS}
    identity = uuid.uuid5(WINNER_CONFIG_NAMESPACE, sha256_json(digests))
    segments = "__".join(f"{name}-{digests[name][:12]}" for name in SEGMENTS)
    return f"{identity}__{segments}"


def same_winner_config(left: str | None, right: str | None) -> bool:
    return bool(left) and left == right
