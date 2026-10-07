"""Fail-fast checks performed before model evaluation starts."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .adapter_registry import resolve_hf_token
from .config_schema import ExperimentConfig, ModelConfig

_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


class HuggingFaceRevisionError(RuntimeError):
    """Raised when a configured Hugging Face revision cannot be verified."""

    def __init__(self, results: list[dict[str, Any]]):
        self.results = results
        failures = [result for result in results if result.get("status") == "error"]
        detail = "; ".join(
            f"{failure['model_id']}@{failure.get('configured_revision') or '<missing>'}: "
            f"{failure.get('error', 'unknown error')}"
            for failure in failures
        )
        super().__init__(
            f"Hugging Face revision preflight failed for {len(failures)} model(s): {detail}"
        )


def validate_huggingface_revisions(
    models: Iterable[ModelConfig],
    *,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Verify each configured, pinned model revision with the Hub metadata API.

    Models without a revision are ignored. A model may use a logical model_id
    while adapter_kwargs.model_name_or_path names the actual Hub repository; in
    that case the revision is verified against the latter. This check intentionally
    performs no model or weight download.
    """
    pinned = [model for model in models if model.revision]
    if not pinned:
        return []

    from huggingface_hub import HfApi

    api = HfApi()
    results: list[dict[str, Any]] = []
    for model in pinned:
        revision = str(model.revision)
        hub_repo_id = str(
            model.adapter.kwargs.get("model_name_or_path") or model.adapter.model_id
        )
        result = {
            "model_id": model.model_id,
            "hub_repo_id": hub_repo_id,
            "model_storage_key": model.storage_key,
            "configured_revision": revision,
        }
        try:
            _validate_model_revision(hub_repo_id, revision)
            info = api.model_info(
                repo_id=hub_repo_id,
                revision=revision,
                token=token,
            )
            resolved_revision = str(getattr(info, "sha", "") or "")
            if not resolved_revision:
                raise RuntimeError("Hub response did not include a resolved commit SHA")
            if resolved_revision.lower() != revision.lower():
                raise RuntimeError(
                    f"revision resolved to {resolved_revision}, expected the pinned commit {revision}"
                )
            results.append(
                {
                    **result,
                    "status": "verified",
                    "resolved_revision": resolved_revision,
                }
            )
        except Exception as exc:
            results.append(
                {
                    **result,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if any(result["status"] == "error" for result in results):
        raise HuggingFaceRevisionError(results)
    return results


def preflight_huggingface_revisions(
    exp: ExperimentConfig,
    models: Iterable[ModelConfig] | None = None,
) -> list[dict[str, Any]]:
    """Resolve authentication and verify revisions for an experiment."""
    embedding = dict(exp.raw.get("embedding") or {})
    token = resolve_hf_token(embedding.get("hf_token"))
    return validate_huggingface_revisions(exp.models if models is None else models, token=token)


def _validate_model_revision(model_id: str, revision: str) -> None:
    if "/" not in model_id or model_id.strip() != model_id:
        raise ValueError(
            f"expected a Hugging Face model repo id like 'org/name', got {model_id!r}"
        )
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise ValueError(
            "revision must be a full 40-character hexadecimal commit SHA, "
            f"got {revision!r}"
        )
