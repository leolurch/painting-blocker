"""Download and verify Hugging Face model snapshots for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapter_registry import resolve_hf_token
from .config_schema import ExperimentConfig, ModelConfig, load_experiment_config
from .random_baseline import parse_random_baseline_config

_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".msgpack",
    ".onnx",
)

_CHECKPOINT_ADAPTERS = frozenset(
    {
        "clip_gem_projection_adapter",
        "clip_projection_adapter",
        "dinov3_gem_projection_adapter",
        "dinov3_projection_adapter",
        "resnet50_projection_adapter",
    }
)
_SUCCESS_STATUSES = frozenset({"verified", "skipped_local_checkpoint"})


class ModelDownloadError(RuntimeError):
    """Raised when one or more configured model snapshots failed verification."""

    def __init__(self, results: list[dict[str, Any]]):
        self.results = results
        failures = [result for result in results if result.get("status") not in _SUCCESS_STATUSES]
        detail = "; ".join(
            f"{failure.get('model_id')}: {failure.get('error', 'unknown error')}"
            for failure in failures
        )
        super().__init__(f"Model download verification failed for {len(failures)} model(s): {detail}")


def download_experiment_models(
    experiment_path: Path | str,
    model_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Download and verify all adapter-backed model snapshots for an experiment."""
    exp = load_experiment_config(experiment_path)
    models = [model for model in exp.models if model_filter in (None, model.model_id)]
    if not models:
        if _is_random_baseline_filter(exp, model_filter):
            return []
        raise ValueError(f"Model {model_filter!r} is not included in {exp.path}")

    token = resolve_hf_token((dict(exp.raw.get("embedding") or {})).get("hf_token"))
    results = [_download_and_verify_model(model, token) for model in models]
    if any(result.get("status") not in _SUCCESS_STATUSES for result in results):
        raise ModelDownloadError(results)
    return results


def _is_random_baseline_filter(exp: ExperimentConfig, model_filter: str | None) -> bool:
    random_baseline = parse_random_baseline_config(
        (dict(exp.raw.get("evaluation") or {})).get("random_baseline")
    )
    return random_baseline is not None and model_filter == random_baseline.model_id


def _download_and_verify_model(model: ModelConfig, token: str | None) -> dict[str, Any]:
    result = {
        "model_id": model.model_id,
        "model_storage_key": model.storage_key,
        "model_revision": model.revision,
        "adapter_name": model.adapter.adapter_name,
    }
    if model.adapter.adapter_name in _CHECKPOINT_ADAPTERS:
        return {
            **result,
            "status": "skipped_local_checkpoint",
            "reason": "checkpoint-backed adapters load local checkpoint_path/checkpoint_path_env values at run time",
        }
    try:
        source_model_id = str(
            model.adapter.kwargs.get("model_name_or_path") or model.adapter.model_id
        )
        snapshot_path = _download_snapshot(source_model_id, token, model.revision)
        revision = snapshot_path.name
        verified_path = _local_snapshot(source_model_id, token, revision)
        expected_files = _remote_model_files(source_model_id, token, revision)
        verification = _verify_snapshot(verified_path, expected_files)
        return {
            **result,
            "status": "verified",
            "configured_revision": model.revision,
            "resolved_revision": revision,
            "revision": revision,
            "cache_path": str(verified_path),
            "verification": verification,
        }
    except Exception as exc:
        return {**result, "status": "error", "error": str(exc)}


def _download_snapshot(repo_id: str, token: str | None, revision: str | None) -> Path:
    _validate_repo_id(repo_id)
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": repo_id, "repo_type": "model", "token": token}
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs))


def _local_snapshot(repo_id: str, token: str | None, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            token=token,
            local_files_only=True,
        )
    )


def _remote_model_files(repo_id: str, token: str | None, revision: str) -> dict[str, int | None]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(
        repo_id=repo_id,
        revision=revision,
        token=token,
        files_metadata=True,
    )
    files = {
        str(sibling.rfilename): getattr(sibling, "size", None)
        for sibling in info.siblings or []
        if getattr(sibling, "rfilename", None) and not str(sibling.rfilename).endswith("/")
    }
    if not files:
        raise RuntimeError(f"No files found in remote model snapshot {repo_id}@{revision}")
    return files


def _verify_snapshot(snapshot_path: Path, expected_files: dict[str, int | None]) -> dict[str, Any]:
    if not snapshot_path.is_dir():
        raise RuntimeError(f"Snapshot path is not a directory: {snapshot_path}")

    missing, size_mismatches, lfs_pointers = [], [], []
    weight_files = []
    total_bytes = 0
    for filename, expected_size in sorted(expected_files.items()):
        relative = _safe_relative_path(filename)
        path = snapshot_path / relative
        if not path.exists() or not path.is_file():
            missing.append(filename)
            continue
        actual_size = path.stat().st_size
        total_bytes += actual_size
        if expected_size is not None and actual_size != int(expected_size):
            size_mismatches.append(
                {"file": filename, "expected_bytes": int(expected_size), "actual_bytes": actual_size}
            )
        if _is_weight_file(filename):
            weight_files.append(filename)
            if _looks_like_lfs_pointer(path):
                lfs_pointers.append(filename)

    incomplete_files = [
        str(path.relative_to(snapshot_path))
        for path in snapshot_path.rglob("*.incomplete")
    ]
    if missing:
        raise RuntimeError(f"Missing downloaded files: {missing[:10]}")
    if size_mismatches:
        raise RuntimeError(f"Downloaded file size mismatches: {size_mismatches[:10]}")
    if incomplete_files:
        raise RuntimeError(f"Incomplete download marker files remain: {incomplete_files[:10]}")
    if lfs_pointers:
        raise RuntimeError(f"Weight files are unresolved Git LFS pointers: {lfs_pointers[:10]}")
    if not weight_files:
        raise RuntimeError("No model weight files were found in the downloaded snapshot")

    return {
        "num_files": len(expected_files),
        "num_weight_files": len(weight_files),
        "total_bytes": total_bytes,
    }


def _validate_repo_id(repo_id: str) -> None:
    if "/" not in repo_id or repo_id.strip() != repo_id:
        raise ValueError(f"Expected a Hugging Face model repo id like 'org/name', got {repo_id!r}")


def _safe_relative_path(filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe filename in remote model snapshot: {filename!r}")
    return relative


def _is_weight_file(filename: str) -> bool:
    return filename.lower().endswith(_WEIGHT_SUFFIXES)


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False
