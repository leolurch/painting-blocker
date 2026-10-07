"""Persistent multisplit analysis discovery and rendering.

This is deliberately separate from the historical single-source analysis mode.
It selects one immutable evaluation source per model/training-repeat/split cell,
then delegates descriptive split-level aggregation to the shared sensitivity
implementation.  Evaluation metrics are only read, never recomputed.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .aggregate_runs import discover_eval_files, rows_from_eval
from .analysis_readme import write_analysis_readme
from .artifacts import PACKAGE_DIR, atomic_write_json, sha256_file, sha256_json, utc_now_iso
from .calibration_protocol_bars import (
    CalibrationProtocolBarsError,
    parse_protocol_chart,
    prune_incomplete_union_rows,
    union_rows_for_batch,
)
from .split_sensitivity import (
    SplitSensitivityError,
    _validate_static_splits,
    _write_csv,
    aggregate_split_sensitivity_rows,
)


class MultisplitAnalysisError(ValueError):
    """Raised when a multisplit source matrix is incomplete or incompatible."""


_HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _relative(config_path: Path, value: Path) -> str:
    return os.path.relpath(value.resolve(), config_path.parent.resolve())


def _canonicalize_hf_secret_presence(payload: dict[str, Any]) -> None:
    """Treat synonymous Hugging Face credential variables identically."""
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return
    secret_presence = environment.get("secret_presence")
    if not isinstance(secret_presence, dict):
        return
    if not any(name in secret_presence for name in _HF_TOKEN_ENV_NAMES):
        return

    credential_available = any(
        bool(secret_presence.get(name)) for name in _HF_TOKEN_ENV_NAMES
    )
    # Keep the historical payload shape and represent all equivalent aliases
    # through the canonical HF_TOKEN slot. This preserves existing family
    # hashes created when only HF_TOKEN was initially present.
    secret_presence["HF_TOKEN"] = credential_available
    secret_presence["HUGGINGFACE_HUB_TOKEN"] = False
    secret_presence["HUGGING_FACE_HUB_TOKEN"] = False


def multisplit_family_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    """Strict run-family identity with non-scientific differences neutralized."""
    payload = copy.deepcopy(resolved)
    payload.pop("models", None)
    split = payload.get("split")
    if isinstance(split, dict):
        split["file"] = "<multisplit-static-realization>"
    _canonicalize_hf_secret_presence(payload)
    return payload


def multisplit_family_sha256(resolved: dict[str, Any]) -> str:
    return sha256_json(multisplit_family_payload(resolved))


def portable_multisplit_family_sha256(resolved: dict[str, Any]) -> str:
    payload = multisplit_family_payload(resolved)
    payload.pop("repo_root", None)
    return sha256_json(payload)


def _compatibility_values(value: Any, qualified_name: str) -> tuple[str, ...]:
    """Normalize a scalar or list-valued compatibility identity."""
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise MultisplitAnalysisError(
            f"{qualified_name} must be a string or a non-empty list of strings"
        )
    result: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise MultisplitAnalysisError(
                f"{qualified_name} must contain non-empty strings"
            )
        if item not in result:
            result.append(item)
    return tuple(result)


def multisplit_compatibility_family_hashes(value: Any) -> tuple[str, ...]:
    return _compatibility_values(
        value, "compatibility.multisplit_config_family_sha256"
    )


def _validate_model_groups(config: dict[str, Any]) -> None:
    groups = config.get("model_groups") or {}
    if not isinstance(groups, dict):
        raise MultisplitAnalysisError("model_groups must be a mapping")
    known_models = {
        str(entry["analysis_model_id"])
        for entry in config.get("models") or []
        if isinstance(entry, dict) and entry.get("analysis_model_id") is not None
    }
    assigned: dict[str, str] = {}
    for group_id, settings in groups.items():
        if not isinstance(settings, dict):
            raise MultisplitAnalysisError(f"model_groups.{group_id} must be a mapping")
        members = settings.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise MultisplitAnalysisError(
                f"model_groups.{group_id}.members must contain at least two model ids"
            )
        normalized = [str(value) for value in members]
        if len(normalized) != len(set(normalized)):
            raise MultisplitAnalysisError(f"model_groups.{group_id}.members contains duplicates")
        unknown = [value for value in normalized if value not in known_models]
        if unknown:
            raise MultisplitAnalysisError(
                f"model_groups.{group_id}.members contains unknown model ids: "
                + ", ".join(unknown)
            )
        canonical = settings.get("canonical")
        if canonical is None:
            raise MultisplitAnalysisError(
                f"model_groups.{group_id}.canonical must predeclare one member"
            )
        if str(canonical) not in normalized:
            raise MultisplitAnalysisError(
                f"model_groups.{group_id}.canonical must be listed in members"
            )
        overlapping = [value for value in normalized if value in assigned]
        if overlapping:
            other = assigned[overlapping[0]]
            raise MultisplitAnalysisError(
                f"Model {overlapping[0]!r} belongs to both model groups {other!r} and {group_id!r}"
            )
        assigned.update({value: str(group_id) for value in normalized})


def _validate_chart_models(config: dict[str, Any]) -> None:
    included_models = {
        str(entry["analysis_model_id"])
        for entry in config.get("models") or []
        if isinstance(entry, dict)
        and entry.get("analysis_model_id") is not None
        and bool(entry.get("include", False))
    }
    for chart_name, settings in (config.get("charts") or {}).items():
        if not isinstance(settings, dict):
            continue
        for setting_name in ("models", "canonical_additional_models"):
            if setting_name not in settings:
                continue
            models = settings[setting_name]
            qualified_name = f"charts.{chart_name}.{setting_name}"
            if not isinstance(models, list) or not models:
                raise MultisplitAnalysisError(
                    f"{qualified_name} must be a non-empty list of analysis model ids"
                )
            if any(not isinstance(value, str) or not value for value in models):
                raise MultisplitAnalysisError(
                    f"{qualified_name} must contain non-empty strings"
                )
            if len(models) != len(set(models)):
                raise MultisplitAnalysisError(
                    f"{qualified_name} contains duplicates"
                )
            unavailable = [value for value in models if value not in included_models]
            if unavailable:
                raise MultisplitAnalysisError(
                    f"{qualified_name} contains model ids that are not globally included: "
                    + ", ".join(unavailable)
                )


_CALIBRATION_CHART_NAMES = (
    "multisplit_calibration_transfer",
    "multisplit_calibrated_top_k_transfer",
    "multisplit_threshold_top_k_floor_transfer",
)


def _effective_calibration_chart(
    config: dict[str, Any], chart_name: str
) -> dict[str, Any] | None:
    chart = (config.get("charts") or {}).get(chart_name)
    if not isinstance(chart, dict):
        return None
    style_from = chart.get("style_from")
    if style_from is None:
        return chart
    inherited = (config.get("charts") or {}).get(str(style_from))
    if not isinstance(inherited, dict):
        raise MultisplitAnalysisError(
            f"charts.{chart_name}.style_from must name another chart mapping"
        )
    return {**inherited, **chart}


def _validate_calibration_chart_display_options(config: dict[str, Any]) -> None:
    boolean_options = (
        "include_attainment_score",
        "include_specific_result_crosses",
        "show_pc_mean_values",
    )
    text_options = (
        "pc_axis_label",
        "pc_axis_description",
        "rr_axis_label",
        "rr_axis_description",
    )
    for chart_name in _CALIBRATION_CHART_NAMES:
        chart = _effective_calibration_chart(config, chart_name)
        if chart is None:
            continue
        for option in boolean_options:
            if option in chart and not isinstance(chart[option], bool):
                raise MultisplitAnalysisError(
                    f"charts.{chart_name}.{option} must be a boolean"
                )
        for option in text_options:
            value = chart.get(option)
            if option in chart and value is not None and not isinstance(value, str):
                raise MultisplitAnalysisError(
                    f"charts.{chart_name}.{option} must be a string or null"
                )


def _validate_threshold_top_k_floor_chart(config: dict[str, Any]) -> None:
    chart = (config.get("charts") or {}).get(
        "multisplit_threshold_top_k_floor_transfer"
    )
    if not isinstance(chart, dict) or not bool(chart.get("enabled", False)):
        return
    if not str(chart.get("task") or ""):
        raise MultisplitAnalysisError(
            "charts.multisplit_threshold_top_k_floor_transfer.task is required"
        )
    targets = chart.get("target_pc")
    if not isinstance(targets, list) or not targets:
        raise MultisplitAnalysisError(
            "charts.multisplit_threshold_top_k_floor_transfer.target_pc must be "
            "a non-empty list"
        )
    normalized_targets = [float(value) for value in targets]
    if any(not 0.0 < value <= 1.0 for value in normalized_targets):
        raise MultisplitAnalysisError(
            "charts.multisplit_threshold_top_k_floor_transfer.target_pc values "
            "must lie in (0, 1]"
        )
    if len(normalized_targets) != len(set(normalized_targets)):
        raise MultisplitAnalysisError(
            "charts.multisplit_threshold_top_k_floor_transfer.target_pc contains duplicates"
        )
    floors = chart.get("minimum_top_k")
    if (
        not isinstance(floors, list)
        or not floors
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in floors)
    ):
        raise MultisplitAnalysisError(
            "charts.multisplit_threshold_top_k_floor_transfer.minimum_top_k must "
            "be a non-empty list of positive integers"
        )
    if len(floors) != len(set(floors)):
        raise MultisplitAnalysisError(
            "charts.multisplit_threshold_top_k_floor_transfer.minimum_top_k contains duplicates"
        )


def _validate_calibration_protocol_bars(config: dict[str, Any]) -> None:
    try:
        parse_protocol_chart(config)
    except CalibrationProtocolBarsError as exc:
        raise MultisplitAnalysisError(str(exc)) from exc


def _validate_calibrated_top_k_chart(config: dict[str, Any]) -> None:
    chart = (config.get("charts") or {}).get(
        "multisplit_calibrated_top_k_transfer"
    )
    if not isinstance(chart, dict) or not bool(chart.get("enabled", False)):
        return
    if not str(chart.get("task") or ""):
        raise MultisplitAnalysisError(
            "charts.multisplit_calibrated_top_k_transfer.task is required"
        )
    targets = chart.get("target_pc")
    if not isinstance(targets, list) or not targets:
        raise MultisplitAnalysisError(
            "charts.multisplit_calibrated_top_k_transfer.target_pc must be a "
            "non-empty list"
        )
    normalized = [float(value) for value in targets]
    if any(not 0.0 < value <= 1.0 for value in normalized):
        raise MultisplitAnalysisError(
            "charts.multisplit_calibrated_top_k_transfer.target_pc values must "
            "lie in (0, 1]"
        )
    if len(normalized) != len(set(normalized)):
        raise MultisplitAnalysisError(
            "charts.multisplit_calibrated_top_k_transfer.target_pc contains duplicates"
        )


def _validate_paired_comparisons(config: dict[str, Any]) -> None:
    block = config["multisplit"]
    comparisons = block.get("paired_comparisons") or []
    if not isinstance(comparisons, list):
        raise MultisplitAnalysisError(
            "multisplit.paired_comparisons must be a list"
        )
    included_models = {
        str(entry["analysis_model_id"])
        for entry in config.get("models") or []
        if isinstance(entry, dict)
        and entry.get("analysis_model_id") is not None
        and bool(entry.get("include", False))
    }
    seen: set[str] = set()
    for entry in comparisons:
        if not isinstance(entry, dict):
            raise MultisplitAnalysisError(
                "multisplit.paired_comparisons entries must be mappings"
            )
        comparison_id = str(entry.get("comparison_id") or "")
        minuend = str(entry.get("minuend") or "")
        subtrahend = str(entry.get("subtrahend") or "")
        if not comparison_id or not minuend or not subtrahend:
            raise MultisplitAnalysisError(
                "Each paired comparison requires comparison_id, minuend, and "
                "subtrahend"
            )
        if comparison_id in seen:
            raise MultisplitAnalysisError(
                f"Duplicate paired comparison_id: {comparison_id!r}"
            )
        seen.add(comparison_id)
        if minuend == subtrahend:
            raise MultisplitAnalysisError(
                f"Paired comparison {comparison_id!r} must compare distinct models"
            )
        missing = [
            model_id
            for model_id in (minuend, subtrahend)
            if model_id not in included_models
        ]
        if missing:
            raise MultisplitAnalysisError(
                f"Paired comparison {comparison_id!r} references models that are "
                f"not included: {', '.join(missing)}"
            )

    aggregation = block.get("aggregation") or {}
    if not isinstance(aggregation, dict):
        raise MultisplitAnalysisError("multisplit.aggregation must be a mapping")
    tie_tolerance = float(aggregation.get("tie_tolerance", 1e-12))
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise MultisplitAnalysisError(
            "multisplit.aggregation.tie_tolerance must be a finite non-negative number"
        )


def _validate_paired_difference_chart(config: dict[str, Any]) -> None:
    chart = (config.get("charts") or {}).get("multisplit_paired_difference")
    if not isinstance(chart, dict) or not bool(chart.get("enabled", False)):
        return
    if not str(chart.get("task") or ""):
        raise MultisplitAnalysisError(
            "charts.multisplit_paired_difference.task is required"
        )
    metrics = chart.get("metrics")
    if metrics is None:
        metrics = [chart.get("metric") or "pair_completeness"]
    if not isinstance(metrics, list) or not metrics:
        raise MultisplitAnalysisError(
            "charts.multisplit_paired_difference.metrics must be a non-empty list"
        )
    allowed_metrics = {
        "pair_completeness",
        "pair_quality",
        "reduction_ratio",
    }
    normalized_metrics = [str(value) for value in metrics]
    if (
        len(normalized_metrics) != len(set(normalized_metrics))
        or any(metric not in allowed_metrics for metric in normalized_metrics)
    ):
        raise MultisplitAnalysisError(
            "charts.multisplit_paired_difference.metrics must contain unique "
            "pair_completeness, pair_quality, or reduction_ratio values"
        )
    selection_mode = str(chart.get("selection_mode") or "top_k")
    allowed_modes = {
        "top_k",
        "calibrated_threshold",
        "calibrated_top_k",
        "calibrated_threshold_top_k_floor",
    }
    if selection_mode not in allowed_modes:
        raise MultisplitAnalysisError(
            "charts.multisplit_paired_difference.selection_mode is unsupported"
        )
    if selection_mode == "top_k":
        if int(chart.get("k", 0)) <= 0:
            raise MultisplitAnalysisError(
                "charts.multisplit_paired_difference.k must be positive"
            )
        if "reduction_ratio" in normalized_metrics:
            raise MultisplitAnalysisError(
                "Fixed-k paired reduction ratio is model-independent; use a "
                "variable-budget calibrated operating point"
            )
    else:
        target_pc = float(chart.get("target_pc", 0.0))
        if not 0.0 < target_pc <= 1.0:
            raise MultisplitAnalysisError(
                "charts.multisplit_paired_difference.target_pc must lie in (0, 1]"
            )
        if (
            selection_mode == "calibrated_threshold_top_k_floor"
            and int(chart.get("minimum_top_k", 0)) <= 0
        ):
            raise MultisplitAnalysisError(
                "charts.multisplit_paired_difference.minimum_top_k must be positive"
            )


def validate_multisplit_config(config_path: Path, config: dict[str, Any]) -> None:
    if config.get("mode") != "multisplit":
        raise MultisplitAnalysisError("Multisplit analysis requires mode: multisplit")
    for key, expected in (("discovery", dict), ("compatibility", dict), ("multisplit", dict), ("models", list), ("charts", dict)):
        if not isinstance(config.get(key), expected):
            raise MultisplitAnalysisError(f"Multisplit analysis requires a {key} {expected.__name__}")
    block = config["multisplit"]
    protocol = block.get("protocol") or {}
    if not isinstance(protocol, dict):
        raise MultisplitAnalysisError("multisplit.protocol must be a mapping")
    if protocol.get("frozen_before_evaluation") is not True and not block.get("allow_trained_models", False):
        raise MultisplitAnalysisError(
            "multisplit.protocol.frozen_before_evaluation: true is required unless trained models are explicitly enabled"
        )
    static = block.get("static_splits")
    if not isinstance(static, list) or not static:
        raise MultisplitAnalysisError("multisplit.static_splits must be a non-empty list")
    seeds = [int(entry["seed"]) for entry in static if isinstance(entry, dict) and "seed" in entry]
    if len(seeds) != len(static) or len(seeds) != len(set(seeds)):
        raise MultisplitAnalysisError("Every static split needs a unique integer seed")
    expected = block.get("expected_split_seeds")
    if not isinstance(expected, list) or [int(value) for value in expected] != seeds:
        raise MultisplitAnalysisError(
            "multisplit.expected_split_seeds must exactly match static_splits order"
        )
    compatibility = config.get("compatibility") or {}
    _compatibility_values(
        compatibility.get("experiment_id"), "compatibility.experiment_id"
    )
    _compatibility_values(
        compatibility.get("evaluation_protocol_sha256"),
        "compatibility.evaluation_protocol_sha256",
    )
    multisplit_compatibility_family_hashes(
        compatibility.get("multisplit_config_family_sha256")
    )
    dataset_id = str(compatibility.get("dataset_id") or "")
    if not dataset_id:
        raise MultisplitAnalysisError("compatibility.dataset_id is required")
    sensitivity_config = {
        "dataset_id": dataset_id,
        "expected_split_seeds": seeds,
        "static_splits": static,
    }
    _validate_static_splits(config_path, sensitivity_config)
    _validate_model_groups(config)
    _validate_chart_models(config)
    _validate_paired_comparisons(config)
    _validate_paired_difference_chart(config)
    _validate_calibration_chart_display_options(config)
    _validate_threshold_top_k_floor_chart(config)
    _validate_calibrated_top_k_chart(config)
    _validate_calibration_protocol_bars(config)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MultisplitAnalysisError(f"Expected YAML mapping: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MultisplitAnalysisError(f"Expected JSON mapping: {path}")
    return data


def _static_records(config_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    block = config["multisplit"]
    return _validate_static_splits(
        config_path,
        {
            "dataset_id": config["compatibility"]["dataset_id"],
            "expected_split_seeds": [int(entry["seed"]) for entry in block["static_splits"]],
            "static_splits": block["static_splits"],
        },
    )


def _model_state_kind(resolved_model: dict[str, Any]) -> str:
    group = str(resolved_model.get("group") or "")
    if group == "frozen":
        return "frozen_encoder"
    adapter_kwargs = resolved_model.get("adapter_kwargs") or {}
    checkpoint_path = adapter_kwargs.get("checkpoint_path") if isinstance(adapter_kwargs, dict) else None
    if (
        group == "finetuned"
        and resolved_model.get("adapter_name") == "finetuned_checkpoint_adapter"
        and isinstance(checkpoint_path, str)
        and bool(checkpoint_path.strip())
        and checkpoint_path != "REQUIRED_OVERRIDE_BY_EXPERIMENT"
    ):
        return "static_finetuned_checkpoint"
    return "unsupported"


def _model_aliases(entry: dict[str, Any]) -> set[str]:
    values = {
        entry.get("analysis_model_id"),
        entry.get("model_storage_key"),
        entry.get("model_id"),
        *(entry.get("source_ids") or []),
    }
    return {str(value) for value in values if value not in (None, "")}


def _candidate_identity(rows: list[dict[str, Any]], data: dict[str, Any]) -> tuple[str | None, int | None]:
    first = rows[0] if rows else {}
    configuration = first.get("configuration_id") or data.get("model_storage_key") or data.get("model_id")
    training_seed = first.get("training_seed")
    return (str(configuration) if configuration else None, int(training_seed) if training_seed not in (None, "") else None)


def _cell_key(split_seed: int, training_seed: int | None) -> str:
    return str(split_seed) if training_seed is None else f"{split_seed}/training-{training_seed}"


def _analysis_model_id_for_source(
    configured_models: list[dict[str, Any]],
    source_identities: set[str],
    data: dict[str, Any],
    eval_path: Path,
) -> str:
    matches = [entry for entry in configured_models if _model_aliases(entry) & source_identities]
    if len(matches) > 1:
        raise MultisplitAnalysisError(
            f"Evaluation {eval_path} maps to multiple configured analysis models"
        )
    if matches:
        return str(matches[0]["analysis_model_id"])
    return str(data.get("model_storage_key") or data.get("model_id"))


def _required_source_capabilities(config: dict[str, Any]) -> tuple[str, ...]:
    charts = config.get("charts") or {}
    required: list[str] = []
    floor_chart = charts.get("multisplit_threshold_top_k_floor_transfer")
    if isinstance(floor_chart, dict) and bool(floor_chart.get("enabled", False)):
        required.append("calibrated_threshold_top_k_floor")
    top_k_chart = charts.get("multisplit_calibrated_top_k_transfer")
    if isinstance(top_k_chart, dict) and bool(top_k_chart.get("enabled", False)):
        required.append("calibrated_top_k")
    return tuple(required)


def _candidate_score(
    candidate: dict[str, Any],
    policy: str,
    required_capabilities: tuple[str, ...] = (),
) -> tuple[Any, ...]:
    capabilities = candidate.get("capabilities") or {}
    required = all(bool(capabilities.get(value)) for value in required_capabilities)
    if policy == "newest_complete":
        return (
            required,
            str(candidate.get("created_at") or ""),
            str(candidate["run_dir"]),
        )
    if policy in {"prefer_existing", "richest_then_newest"}:
        richness = sum(bool(value) for value in capabilities.values())
        return (
            required,
            richness,
            str(candidate.get("created_at") or ""),
            str(candidate["run_dir"]),
        )
    raise MultisplitAnalysisError(f"Unknown source_policy: {policy!r}")


def _discover_candidates(
    config_path: Path,
    config: dict[str, Any],
    static: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    discovery = config["discovery"]
    root = _resolve(config_path, discovery["runs_root"])
    compatibility = config["compatibility"]
    static_by_identity = {
        (int(row["split_seed"]), str(row["split_id"]), str(row["split_sha256"])): row
        for row in static
    }
    expected_families = multisplit_compatibility_family_hashes(
        compatibility.get("multisplit_config_family_sha256")
    )
    expected_protocols = set(
        _compatibility_values(
            compatibility.get("evaluation_protocol_sha256"),
            "compatibility.evaluation_protocol_sha256",
        )
    )
    expected_experiments = set(
        _compatibility_values(
            compatibility.get("experiment_id"),
            "compatibility.experiment_id",
        )
    )
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    discovered_models: dict[str, dict[str, Any]] = {}
    accepted_runs: dict[str, dict[str, Any]] = {}
    configured_models = [entry for entry in config.get("models") or [] if isinstance(entry, dict)]

    for eval_path in discover_eval_files([root]):
        run_dir = eval_path.parents[2]
        run_json = run_dir / "run.json"
        resolved_path = run_dir / "resolved_config.yml"
        if not run_json.is_file() or not resolved_path.is_file():
            continue
        run = _load_json(run_json)
        if discovery.get("completed_runs_only", True) and run.get("status") != "completed":
            continue
        if str(run.get("experiment_id") or "") not in expected_experiments:
            continue
        dataset = run.get("dataset") or {}
        if str(dataset.get("dataset_id")) != str(compatibility["dataset_id"]):
            continue
        expected_db = compatibility.get("dataset_source_db_sha256")
        if expected_db and dataset.get("source_db_sha256") != expected_db:
            continue
        protocol_hash = str((run.get("config") or {}).get("evaluation_protocol_sha256") or "")
        if protocol_hash not in expected_protocols:
            continue
        resolved = _load_yaml(resolved_path)
        family_hash = multisplit_family_sha256(resolved)
        portable_family_hash = portable_multisplit_family_sha256(resolved)
        if not ({family_hash, portable_family_hash} & set(expected_families)):
            continue
        split = run.get("split") or {}
        seed = split.get("split_seed")
        identity = (
            int(seed) if seed is not None else -1,
            str(split.get("split_id") or ""),
            str(split.get("split_sha256") or ""),
        )
        if identity not in static_by_identity:
            continue
        data = _load_json(eval_path)
        resolved_model = next(
            (
                dict(value)
                for value in ((resolved.get("models") or {}).get("include") or [])
                if isinstance(value, dict)
                and str(value.get("model_storage_key") or "") == str(data.get("model_storage_key") or "")
            ),
            {},
        )
        rows = rows_from_eval(eval_path)
        if not rows:
            continue
        configuration_id, training_seed = _candidate_identity(rows, data)
        source_identities = {
            str(value) for value in (
                configuration_id,
                data.get("model_storage_key"),
                data.get("model_id"),
                rows[0].get("finetuning_experiment_id"),
            ) if value not in (None, "")
        }
        analysis_id = _analysis_model_id_for_source(
            configured_models, source_identities, data, eval_path
        )
        model_dir = eval_path.parent
        curves_path = model_dir / "curves.json"
        curves = _load_json(curves_path) if curves_path.is_file() else {}
        candidate = {
            "analysis_model_id": analysis_id,
            "split_seed": identity[0],
            "training_seed": training_seed,
            "cell_key": _cell_key(identity[0], training_seed),
            "run_dir": _relative(config_path, run_dir),
            "model_dir": _relative(config_path, model_dir),
            "eval_path": _relative(config_path, eval_path),
            "run_id": run.get("run_id", run_dir.name),
            "created_at": run.get("created_at"),
            "split_id": identity[1],
            "split_sha256": identity[2],
            "evaluation_protocol_sha256": protocol_hash,
            "multisplit_config_family_sha256": family_hash,
            "model_id": data.get("model_id"),
            "model_storage_key": data.get("model_storage_key"),
            "checkpoint_sha256": rows[0].get("checkpoint_sha256"),
            "model_state_kind": _model_state_kind(resolved_model),
            "capabilities": {
                "fixed_k": any(task.get("metrics") for task in data.get("tasks") or []),
                "calibrated_thresholds": any(
                    task.get("calibrated_threshold_metrics") for task in data.get("tasks") or []
                ),
                "calibrated_threshold_top_k_floor": any(
                    task.get("calibrated_threshold_top_k_floor_metrics")
                    for task in data.get("tasks") or []
                ),
                "calibrated_top_k": any(
                    task.get("calibrated_top_k_metrics")
                    for task in data.get("tasks") or []
                ),
                "similarity_histogram": any(
                    (task.get("histogram") or {}).get("match_hist")
                    and (task.get("histogram") or {}).get("neg_hist")
                    for task in curves.get("tasks") or []
                ),
            },
        }
        candidates[analysis_id].append(candidate)
        discovered_models.setdefault(
            analysis_id,
            {
                "analysis_model_id": analysis_id,
                "model_id": data.get("model_id") or analysis_id,
                "model_storage_key": data.get("model_storage_key"),
                "label": (
                    resolved_model.get("display_name")
                    or resolved_model.get("label")
                    or resolved_model.get("name")
                    or data.get("model_id")
                    or analysis_id
                ),
                "color": resolved_model.get("color"),
                "group": resolved_model.get("group"),
            },
        )
        accepted_runs.setdefault(
            str(run_dir.resolve()),
            {
                "run_dir": _relative(config_path, run_dir),
                "run_id": run.get("run_id", run_dir.name),
                "split_seed": identity[0],
                "split_id": identity[1],
                "split_sha256": identity[2],
            },
        )
    return candidates, discovered_models, sorted(accepted_runs.values(), key=lambda row: (row["split_seed"], row["run_dir"]))


def update_multisplit_analysis_config(
    config_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_multisplit_config(config_path, config)
    static = _static_records(config_path, config)
    candidates, discovered, runs = _discover_candidates(config_path, config, static)
    existing_list = [entry for entry in config.get("models") or [] if isinstance(entry, dict)]
    existing = {str(entry["analysis_model_id"]): entry for entry in existing_list}
    policy = str(config["discovery"].get("source_policy") or "prefer_existing")
    include_new = bool(config["discovery"].get("new_models_include", True))
    expected_seeds = [int(row["split_seed"]) for row in static]
    required_capabilities = _required_source_capabilities(config)
    merged: list[dict[str, Any]] = []
    ids = list(existing)
    ids.extend(value for value in discovered if value not in existing)
    for order, analysis_id in enumerate(ids):
        entry = copy.deepcopy(existing.get(analysis_id) or {})
        meta = discovered.get(analysis_id) or {}
        entry.setdefault("analysis_model_id", analysis_id)
        entry.setdefault("model_id", meta.get("model_id") or analysis_id)
        if meta.get("model_storage_key"):
            entry.setdefault("model_storage_key", meta["model_storage_key"])
        entry.setdefault("include", include_new)
        presentation = entry.setdefault("presentation", {})
        presentation.setdefault("label", meta.get("label") or entry["model_id"])
        if meta.get("color") is not None:
            presentation.setdefault("color", meta["color"])
        if meta.get("group") is not None:
            presentation.setdefault("group", meta["group"])
        presentation.setdefault("order", order)
        source = entry.setdefault("source", {})
        source.setdefault("mode", "auto")
        previous = source.get("selected_by_split") or {}
        selected: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        expected_training = entry.get("training_seeds")
        normalized_training_seeds = [
            int(value) for value in expected_training or []
        ]
        for candidate in candidates.get(analysis_id, []):
            normalized_candidate = candidate
            # Static checkpoint reevaluations do not carry the originating
            # training seed in eval.json. For a configured model with exactly
            # one predeclared repeat, inherit that repeat identity so the rich
            # reevaluation competes with (rather than duplicates) its legacy
            # source in the same split/repeat cell.
            if (
                candidate.get("training_seed") is None
                and len(normalized_training_seeds) == 1
                and required_capabilities
                and all(
                    bool((candidate.get("capabilities") or {}).get(capability))
                    for capability in required_capabilities
                )
            ):
                normalized_candidate = copy.deepcopy(candidate)
                normalized_candidate["training_seed"] = normalized_training_seeds[0]
                normalized_candidate["cell_key"] = _cell_key(
                    int(candidate["split_seed"]), normalized_training_seeds[0]
                )
            grouped[normalized_candidate["cell_key"]].append(
                normalized_candidate
            )
        for cell, cell_candidates in grouped.items():
            previous_path = (previous.get(cell) or {}).get("run_dir")
            chosen = None
            if source["mode"] == "pinned" or (policy == "prefer_existing" and previous_path):
                chosen = next((row for row in cell_candidates if row["run_dir"] == previous_path), None)
            if chosen is None:
                chosen = max(
                    cell_candidates,
                    key=lambda row: _candidate_score(
                        row, policy, required_capabilities
                    ),
                )
            selected[cell] = chosen
        source["selected_by_split"] = selected
        observed = {int(row["split_seed"]) for row in selected.values()}
        missing_cells: list[str] = []
        if expected_training:
            for split_seed in expected_seeds:
                for training_seed in [int(value) for value in expected_training]:
                    cell = _cell_key(split_seed, training_seed)
                    if cell not in selected:
                        missing_cells.append(cell)
        else:
            missing_cells = [str(seed) for seed in expected_seeds if str(seed) not in selected]
            seeded = [row for row in selected.values() if row.get("training_seed") is not None]
            if seeded:
                raise MultisplitAnalysisError(
                    f"Model {analysis_id!r} has training-seeded sources but no explicit training_seeds"
                )
        entry["availability"] = {
            "available": not missing_cells,
            "expected_split_seeds": expected_seeds,
            "available_split_seeds": sorted(observed),
            "missing_cells": missing_cells,
            "source_count": len(candidates.get(analysis_id, [])),
        }
        merged.append(entry)
    merged.sort(key=lambda row: (int((row.get("presentation") or {}).get("order", 10**9)), str(row["analysis_model_id"])))
    config["models"] = merged
    config["inventory"] = {
        "updated_at": utc_now_iso(),
        "compatible_runs": runs,
        "num_compatible_runs": len(runs),
        "num_discovered_models": len(discovered),
        "static_splits": static,
    }
    return config, {
        "compatible_runs": len(runs),
        "discovered_models": len(discovered),
        "incomplete_models": [row["analysis_model_id"] for row in merged if not row["availability"]["available"]],
    }


def initialize_multisplit_analysis_config(
    experiment_path: Path | str,
    reference_run: Path | str,
    runs_root: Path | str,
    output: Path | str,
    *,
    analysis_id: str | None = None,
    new_models_include: bool = True,
    source_policy: str = "prefer_existing",
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create a multisplit analysis config from one rotation experiment/run."""
    from .config_schema import load_experiment_config

    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Analysis config already exists: {output_path}")
    experiment_path = Path(experiment_path).expanduser().resolve()
    experiment = load_experiment_config(experiment_path)
    document = _load_yaml(experiment_path)
    rotation = document.get("split_rotation_evaluation")
    if not isinstance(rotation, dict):
        raise MultisplitAnalysisError(
            f"Experiment has no split_rotation_evaluation block: {experiment_path}"
        )
    reference = Path(reference_run).expanduser().resolve()
    run_json = reference / "run.json"
    resolved_path = reference / "resolved_config.yml"
    if not run_json.is_file() or not resolved_path.is_file():
        raise FileNotFoundError(f"Reference run is incomplete: {reference}")
    run = _load_json(run_json)
    if run.get("status") != "completed":
        raise MultisplitAnalysisError("Reference run must be completed")
    resolved = _load_yaml(resolved_path)
    static_splits = [
        {
            "seed": int(entry["seed"]),
            "file": _relative(output_path, _resolve(experiment_path, entry["file"])),
        }
        for entry in rotation.get("static_splits") or []
    ]
    identifier = analysis_id or output_path.stem
    evaluation = experiment.raw.get("evaluation") or {}
    tasks = evaluation.get("retrieval_tasks") or []
    task_id = str(tasks[0].get("task_id")) if tasks else None
    targets = [
        float(value)
        for value in ((evaluation.get("threshold_calibration") or {}).get("target_pc") or [])
    ]
    top_k = [int(value) for value in evaluation.get("top_k") or []]
    floor_evaluation = evaluation.get("threshold_top_k_floor") or {}
    floor_values = [int(value) for value in floor_evaluation.get("minimum_k") or []]
    config: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": identifier,
        "mode": "multisplit",
        "discovery": {
            "runs_root": _relative(output_path, Path(runs_root).expanduser().resolve()),
            "reference_run": _relative(output_path, reference),
            "completed_runs_only": True,
            "new_models_include": bool(new_models_include),
            "source_policy": source_policy,
        },
        "compatibility": {
            "experiment_id": run.get("experiment_id"),
            "dataset_id": (run.get("dataset") or {}).get("dataset_id") or experiment.dataset.dataset_id,
            "dataset_source_db_sha256": (run.get("dataset") or {}).get("source_db_sha256"),
            "evaluation_protocol_sha256": (run.get("config") or {}).get("evaluation_protocol_sha256"),
            "multisplit_config_family_sha256": [
                multisplit_family_sha256(resolved),
                portable_multisplit_family_sha256(resolved),
            ],
        },
        "multisplit": {
            "expected_split_seeds": [int(entry["seed"]) for entry in static_splits],
            "protocol": {
                "frozen_before_evaluation": bool(
                    ((rotation.get("protocol") or {}).get("frozen_before_evaluation"))
                ),
                "allow_static_finetuned_checkpoints": True,
                "require_complete_matrix": True,
            },
            "static_splits": static_splits,
            "aggregation": {
                "unit": "split",
                "split_weighting": "equal",
                "sample_sd_ddof": 1,
                "range": "observed_min_max",
                "attainment_tolerance": 1e-12,
                "tie_tolerance": 0.0,
            },
            "paired_comparisons": [],
        },
        "output": {
            "directory": _relative(output_path, PACKAGE_DIR / "analyses" / identifier),
            "clean_generated_directories": True,
        },
        "model_groups": {},
        "models": [],
        "charts": {
            "multisplit_pc_at_k_band": {"enabled": True, "task": task_id, "metric": "pair_completeness", "formats": ["pdf"]},
            "multisplit_strip": {"enabled": True, "task": task_id, "metric": "pair_completeness", "k": 40, "jitter_seed": 0, "formats": ["pdf"]},
            "multisplit_paired_difference": {"enabled": False, "task": task_id, "metric": "pair_completeness", "k": 40, "jitter_seed": 0, "formats": ["pdf"]},
            "multisplit_heatmap": {"enabled": True, "task": task_id, "metric": "pair_completeness", "k": 40, "formats": ["pdf"]},
            "multisplit_calibration_transfer": {
                "enabled": bool(targets),
                "task": task_id,
                "target_pc": max(targets) if targets else None,
                "jitter_seed": 0,
                "rr_bar_alpha": 0.30,
                "include_attainment_score": True,
                "include_specific_result_crosses": True,
                "pc_axis_label": None,
                "pc_axis_description": None,
                "rr_axis_label": None,
                "rr_axis_description": None,
                "formats": ["pdf"],
            },
            "calibration_protocol_bars": {
                "enabled": bool(targets),
                "safety_margins": False,
                "protocols": ["calibrated_k", "union"],
                "style_from": "multisplit_calibration_transfer",
                "formats": ["pdf"],
            },
            "multisplit_threshold_top_k_floor_transfer": {
                "enabled": bool(targets and floor_values),
                "task": task_id,
                "target_pc": targets,
                "minimum_top_k": floor_values,
                "style_from": "multisplit_calibration_transfer",
                "jitter_seed": 0,
                "formats": ["pdf"],
            },
            "multisplit_similarity_distributions": {
                "enabled": True,
                "task": task_id,
                "primary_metric": "pair_completeness",
                "target_pc": max(targets) if targets else None,
                "show_split_range": True,
                "formats": ["pdf"],
            },
            "multisplit_tables": {
                "enabled": True,
                "fixed_k": 40 if 40 in top_k else (max(top_k) if top_k else None),
                "calibration_target": max(targets) if targets else None,
                "calibration_caption": None,
                "syn_test_comparison": {
                    "enabled": False,
                    "models": [],
                    "task": task_id,
                    "target_pc": max(targets) if targets else None,
                    "synthetic_pc_at_k": 10,
                    "synthetic_pc_source": None,
                },
            },
        },
        "inventory": {},
    }
    validate_multisplit_config(output_path, config)
    config, summary = update_multisplit_analysis_config(output_path, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .artifacts import atomic_write_text

    atomic_write_text(output_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=120))
    return output_path, config, summary


def _selected_rows(config_path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    require_complete = bool(((config["multisplit"].get("protocol") or {}).get("require_complete_matrix", True)))
    for entry in config.get("models") or []:
        if not isinstance(entry, dict) or not bool(entry.get("include", False)):
            continue
        if require_complete and not (entry.get("availability") or {}).get("available", False):
            raise MultisplitAnalysisError(
                f"Selected model {entry.get('analysis_model_id')!r} does not cover the complete split/repeat matrix"
            )
        analysis_id = str(entry["analysis_model_id"])
        selected = ((entry.get("source") or {}).get("selected_by_split") or {})
        for cell, source in selected.items():
            eval_path = _resolve(config_path, source["eval_path"])
            eval_rows = rows_from_eval(eval_path)
            if not eval_rows:
                raise MultisplitAnalysisError(f"Selected evaluation has no metric rows: {eval_path}")
            batch: list[dict[str, Any]] = []
            for row in eval_rows:
                normalized = dict(row)
                normalized["configuration_id"] = analysis_id
                normalized["display_name"] = (entry.get("presentation") or {}).get("label") or analysis_id
                normalized["model_state_kind"] = source.get("model_state_kind")
                if (
                    normalized.get("training_seed") is None
                    and source.get("training_seed") is not None
                ):
                    normalized["training_seed"] = int(source["training_seed"])
                batch.append(normalized)
            rows.extend(batch)
            rows.extend(union_rows_for_batch(batch, eval_path))
            sources.append(
                {
                    "analysis_model_id": analysis_id,
                    "cell": cell,
                    **source,
                    "eval_sha256": sha256_file(eval_path),
                    "curves_sha256": (
                        sha256_file(eval_path.parent / "curves.json")
                        if (eval_path.parent / "curves.json").is_file()
                        else None
                    ),
                }
            )
    rows = prune_incomplete_union_rows(rows)
    if not rows:
        raise MultisplitAnalysisError("Multisplit analysis selects no evaluation rows")
    missing_hashes = [
        row for row in rows
        if any(not row.get(field) for field in ("query_ids_hash", "candidate_ids_hash", "positive_pairs_hash"))
    ]
    if missing_hashes:
        raise MultisplitAnalysisError(
            "Every multisplit task row requires query/candidate/positive-pair population hashes; "
            "rerun legacy evaluations with the current evaluator"
        )
    return rows, sources


def _histogram_density(
    counts: list[int] | list[float],
    edges: list[float],
    *,
    context: str,
) -> np.ndarray:
    """Normalize one histogram to unit area without pooling pair counts."""
    count_values = np.asarray(counts, dtype=np.float64)
    edge_values = np.asarray(edges, dtype=np.float64)
    if edge_values.ndim != 1 or count_values.ndim != 1:
        raise MultisplitAnalysisError(f"Similarity histogram must be one-dimensional: {context}")
    if edge_values.size != count_values.size + 1 or count_values.size == 0:
        raise MultisplitAnalysisError(f"Similarity histogram shape mismatch: {context}")
    if not np.all(np.isfinite(edge_values)) or not np.all(np.isfinite(count_values)):
        raise MultisplitAnalysisError(f"Similarity histogram contains non-finite values: {context}")
    if np.any(count_values < 0):
        raise MultisplitAnalysisError(f"Similarity histogram contains negative counts: {context}")
    widths = np.diff(edge_values)
    if np.any(widths <= 0):
        raise MultisplitAnalysisError(f"Similarity histogram edges are not strictly increasing: {context}")
    total = float(count_values.sum())
    if total <= 0:
        raise MultisplitAnalysisError(f"Similarity histogram is empty: {context}")
    return count_values / (total * widths)


def _summarize_similarity_histogram_records(
    records: list[dict[str, Any]],
    expected_split_seeds: list[int],
    task_id: str,
) -> list[dict[str, Any]]:
    """Average normalized histograms over repeats, then equally over splits.

    Pair counts are normalized within every source evaluation. Training repeats
    are averaged within a split before the split-level densities receive equal
    weight in the final mean and observed bin-wise range.
    """
    expected = [int(value) for value in expected_split_seeds]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    canonical_edges: np.ndarray | None = None
    for record in records:
        model_id = str(record["configuration_id"])
        split_seed = int(record["split_seed"])
        context = f"model={model_id}, split_seed={split_seed}, task={task_id}"
        edges = np.asarray(record.get("bin_edges") or [], dtype=np.float64)
        match_counts = list(record.get("match_hist") or [])
        non_match_counts = list(record.get("neg_hist") or [])
        match_density = _histogram_density(match_counts, edges.tolist(), context=f"match, {context}")
        non_match_density = _histogram_density(
            non_match_counts, edges.tolist(), context=f"non-match, {context}"
        )
        if canonical_edges is None:
            canonical_edges = edges
        elif canonical_edges.shape != edges.shape or not np.array_equal(canonical_edges, edges):
            raise MultisplitAnalysisError(
                f"Similarity histogram bin edges differ across selected sources: {context}"
            )
        stored_match_total = record.get("match_total")
        stored_non_match_total = record.get("neg_total")
        match_total = int(sum(match_counts) if stored_match_total is None else stored_match_total)
        non_match_total = int(
            sum(non_match_counts) if stored_non_match_total is None else stored_non_match_total
        )
        if match_total != int(sum(match_counts)) or non_match_total != int(sum(non_match_counts)):
            raise MultisplitAnalysisError(
                f"Similarity histogram stored totals do not match bin counts: {context}"
            )
        grouped[(model_id, split_seed)].append(
            {
                "match_density": match_density,
                "non_match_density": non_match_density,
                "match_total": match_total,
                "non_match_total": non_match_total,
            }
        )
    if canonical_edges is None:
        raise MultisplitAnalysisError(
            f"No similarity histograms were found for task {task_id!r}"
        )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model_id, split_seed), repeats in grouped.items():
        by_model[model_id].append(
            {
                "split_seed": split_seed,
                "num_training_repeats": len(repeats),
                "match_density": np.mean(
                    np.stack([row["match_density"] for row in repeats]), axis=0
                ),
                "non_match_density": np.mean(
                    np.stack([row["non_match_density"] for row in repeats]), axis=0
                ),
                "match_totals": [int(row["match_total"]) for row in repeats],
                "non_match_totals": [int(row["non_match_total"]) for row in repeats],
            }
        )

    summaries: list[dict[str, Any]] = []
    for model_id, split_rows in sorted(by_model.items()):
        split_rows.sort(key=lambda row: int(row["split_seed"]))
        observed = [int(row["split_seed"]) for row in split_rows]
        if observed != sorted(expected):
            raise MultisplitAnalysisError(
                f"Similarity histograms for model {model_id!r} cover split seeds "
                f"{observed}, expected {sorted(expected)}"
            )
        match = np.stack([row["match_density"] for row in split_rows])
        non_match = np.stack([row["non_match_density"] for row in split_rows])
        summaries.append(
            {
                "configuration_id": model_id,
                "task_id": task_id,
                "num_split_realizations": len(split_rows),
                "num_source_evaluations": sum(
                    int(row["num_training_repeats"]) for row in split_rows
                ),
                "bin_edges": canonical_edges.tolist(),
                "normalization": {
                    "within_source": "separate_unit_area_match_and_non_match_densities",
                    "training_repeat_aggregation": "equal_mean_within_split",
                    "split_aggregation": "equal_mean",
                    "pair_counts_pooled": False,
                    "range": "observed_binwise_min_max_across_split_level_means",
                },
                "match_density_mean": np.mean(match, axis=0).tolist(),
                "match_density_min": np.min(match, axis=0).tolist(),
                "match_density_max": np.max(match, axis=0).tolist(),
                "non_match_density_mean": np.mean(non_match, axis=0).tolist(),
                "non_match_density_min": np.min(non_match, axis=0).tolist(),
                "non_match_density_max": np.max(non_match, axis=0).tolist(),
                "split_values": [
                    {
                        "split_seed": int(row["split_seed"]),
                        "num_training_repeats": int(row["num_training_repeats"]),
                        "match_density": row["match_density"].tolist(),
                        "non_match_density": row["non_match_density"].tolist(),
                        "match_totals": row["match_totals"],
                        "non_match_totals": row["non_match_totals"],
                    }
                    for row in split_rows
                ],
            }
        )
    return summaries


def _selected_similarity_distributions(
    config_path: Path,
    sources: list[dict[str, Any]],
    expected_split_seeds: list[int],
    task_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        model_dir = _resolve(config_path, source["model_dir"])
        curves_path = model_dir / "curves.json"
        if not curves_path.is_file():
            raise MultisplitAnalysisError(
                f"Similarity-distribution chart requires curves.json: {curves_path}"
            )
        curves = _load_json(curves_path)
        tasks = [
            task for task in curves.get("tasks") or []
            if str(task.get("task_id")) == task_id
        ]
        if len(tasks) != 1:
            raise MultisplitAnalysisError(
                f"Expected exactly one curves task {task_id!r}: {curves_path}"
            )
        histogram = tasks[0].get("histogram") or {}
        records.append(
            {
                "configuration_id": source["analysis_model_id"],
                "split_seed": source["split_seed"],
                "training_seed": source.get("training_seed"),
                "bin_edges": histogram.get("bin_edges") or [],
                "match_hist": histogram.get("match_hist") or [],
                "neg_hist": histogram.get("neg_hist") or [],
                "match_total": histogram.get("match_total"),
                "neg_total": histogram.get("neg_total"),
            }
        )
    return _summarize_similarity_histogram_records(records, expected_split_seeds, task_id)


def _profile_diagnostic_rows(
    config_path: Path, sources: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    for source in sources:
        eval_path = _resolve(config_path, source["eval_path"])
        data = _load_json(eval_path)
        for task in data.get("tasks") or []:
            for profile in task.get("profile_diagnostics") or []:
                base = {
                    "analysis_model_id": source["analysis_model_id"],
                    "split_seed": int(source["split_seed"]),
                    "training_seed": source.get("training_seed"),
                    "split_id": source.get("split_id"),
                    "split_sha256": source.get("split_sha256"),
                    "task_id": task.get("task_id"),
                    "profile": profile.get("profile"),
                    "selection_role": "diagnostic_only",
                    "num_queries": profile.get("num_queries"),
                    "num_candidates": profile.get("num_candidates"),
                    "num_positive_pairs": profile.get("num_positive_pairs"),
                }
                for metric in profile.get("metrics") or []:
                    raw_rows.append({**base, "selection_mode": "top_k", **metric})
                for metric in profile.get("calibrated_threshold_metrics") or []:
                    raw_rows.append({**base, "selection_mode": "calibrated_threshold", "k": None, **metric})
                for metric in profile.get("calibrated_top_k_metrics") or []:
                    raw_rows.append(
                        {
                            **base,
                            "selection_mode": "calibrated_top_k",
                            **metric,
                            "selected_k": metric.get("k"),
                            "k": None,
                        }
                    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        operation = (
            row["analysis_model_id"], row["task_id"], row["profile"],
            row["selection_mode"], row.get("k"),
            row.get("calibration_target_pair_completeness"), row["split_seed"],
        )
        grouped[operation].append(row)
    split_rows: list[dict[str, Any]] = []
    metrics = ("pair_completeness", "pair_quality", "reduction_ratio", "query_coverage", "candidate_pairs")
    for key, repeats in grouped.items():
        model, task, profile, mode, k, target, split_seed = key
        record = {
            "analysis_model_id": model,
            "task_id": task,
            "profile": profile,
            "selection_mode": mode,
            "k": k,
            "calibration_target_pair_completeness": target,
            "split_seed": split_seed,
            "num_training_repeats": len(repeats),
            "selection_role": "diagnostic_only",
        }
        for metric in metrics:
            values = [float(row[metric]) for row in repeats if row.get(metric) is not None]
            if values:
                record[metric] = float(statistics.fmean(values))
        split_rows.append(record)
    summary_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        key = (
            row["analysis_model_id"], row["task_id"], row["profile"],
            row["selection_mode"], row.get("k"),
            row.get("calibration_target_pair_completeness"),
        )
        summary_groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, split_values in summary_groups.items():
        model, task, profile, mode, k, target = key
        record = {
            "analysis_model_id": model,
            "task_id": task,
            "profile": profile,
            "selection_mode": mode,
            "k": k,
            "calibration_target_pair_completeness": target,
            "num_split_realizations": len(split_values),
            "selection_role": "diagnostic_only",
        }
        for metric in metrics:
            values = [float(row[metric]) for row in split_values if row.get(metric) is not None]
            if not values:
                continue
            record[f"{metric}_mean"] = float(statistics.fmean(values))
            record[f"{metric}_sample_sd_split"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
            record[f"{metric}_min"] = min(values)
            record[f"{metric}_max"] = max(values)
            record[f"{metric}_split_values"] = [
                {"split_seed": row["split_seed"], "value": row.get(metric)}
                for row in sorted(split_values, key=lambda value: int(value["split_seed"]))
            ]
        summaries.append(record)
    return raw_rows, summaries


def _validate_training_selection_protocol(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    protocol = config["multisplit"].get("protocol") or {}
    frozen = protocol.get("frozen_before_evaluation") is True
    seeded_rows = [row for row in rows if row.get("training_seed") is not None]
    if frozen:
        nonstatic_seeded = [
            row
            for row in seeded_rows
            if row.get("model_state_kind") != "static_finetuned_checkpoint"
        ]
        if nonstatic_seeded:
            raise MultisplitAnalysisError(
                "frozen_before_evaluation analysis contains training-seeded model states "
                "that are not fixed finetuned checkpoints"
            )
        allow_static_finetuned = protocol.get("allow_static_finetuned_checkpoints") is True
        selected_models = [
            entry for entry in config.get("models") or []
            if isinstance(entry, dict) and entry.get("include")
        ]
        invalid_models: list[str] = []
        for entry in selected_models:
            analysis_id = str(entry.get("analysis_model_id"))
            group = str((entry.get("presentation") or {}).get("group") or "")
            if group == "frozen":
                continue
            model_rows = [row for row in rows if row.get("configuration_id") == analysis_id]
            is_static_finetuned = (
                allow_static_finetuned
                and group == "finetuned"
                and bool(model_rows)
                and all(
                    row.get("model_state_kind") == "static_finetuned_checkpoint"
                    for row in model_rows
                )
            )
            if not is_static_finetuned:
                invalid_models.append(analysis_id)
        if invalid_models:
            raise MultisplitAnalysisError(
                "frozen_before_evaluation requires frozen encoders or explicitly allowed "
                "static finetuned checkpoints: " + ", ".join(invalid_models)
            )
        return
    for row in seeded_rows:
        if row.get("training_split_sha256") != row.get("split_sha256"):
            raise MultisplitAnalysisError(
                f"Training/evaluation split mismatch for {row.get('configuration_id')}, "
                f"training_seed={row.get('training_seed')}, split_seed={row.get('split_seed')}"
            )
        training_protocol = row.get("training_protocol") or {}
        selector = training_protocol.get("train_selector") or {}
        if str(selector.get("subset") or "") != "train":
            raise MultisplitAnalysisError("Every trained repeat must train only on subset: train")
        validations = training_protocol.get("validation_sets") or []
        if not validations or any(str(value.get("subset") or "") == "test" for value in validations):
            raise MultisplitAnalysisError(
                "Checkpoint selection validation sets must be configured and must not use test"
            )
        selection = row.get("checkpoint_selection") or {}
        if selection.get("metric") != "reduction_ratio_at_target_pc":
            raise MultisplitAnalysisError(
                "Multisplit trained models must select checkpoints by validation "
                "reduction_ratio_at_target_pc"
            )
        target = selection.get("target_pc")
        if target is None or not (0.0 < float(target) <= 1.0):
            raise MultisplitAnalysisError("Checkpoint selection requires a predeclared target_pc")


def render_multisplit_analysis(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    from .multisplit_charts import write_multisplit_charts
    from .multisplit_tables import write_multisplit_tables

    validate_multisplit_config(config_path, config)
    static = _static_records(config_path, config)
    rows, sources = _selected_rows(config_path, config)
    _validate_training_selection_protocol(rows, config)
    model_entries = [entry for entry in config["models"] if isinstance(entry, dict) and entry.get("include")]
    expected_seeds = [int(row["split_seed"]) for row in static]
    comparisons = (config["multisplit"].get("paired_comparisons") or [])
    aggregate_config = {
        "analysis_id": config["analysis_id"],
        "dataset_id": config["compatibility"]["dataset_id"],
        "expected_split_seeds": expected_seeds,
        "protocol": config["multisplit"].get("protocol") or {},
        "aggregation": config["multisplit"].get("aggregation") or {},
        "configurations": [
            {
                "configuration_id": str(entry["analysis_model_id"]),
                "source_ids": list(_model_aliases(entry)),
                "label": (entry.get("presentation") or {}).get("label") or entry["analysis_model_id"],
                "color": (entry.get("presentation") or {}).get("color"),
                "group": (entry.get("presentation") or {}).get("group"),
                **({"training_seeds": entry["training_seeds"]} if entry.get("training_seeds") else {}),
            }
            for entry in model_entries
        ],
        "comparisons": comparisons,
        "require_protocol_hashes": True,
        "require_static_split_hashes": True,
        "require_common_operating_points": True,
    }
    try:
        artifact = aggregate_split_sensitivity_rows(rows, aggregate_config)
    except SplitSensitivityError as exc:
        raise MultisplitAnalysisError(str(exc)) from exc
    static_by_seed = {int(row["split_seed"]): row for row in static}
    for population in artifact["split_populations"]:
        expected = static_by_seed[int(population["split_seed"])]
        if population.get("split_id") != expected["split_id"] or population.get("split_sha256") != expected["split_sha256"]:
            raise MultisplitAnalysisError(
                f"Selected run identity does not match static split seed {population['split_seed']}"
            )
    artifact["static_splits"] = static
    profile_rows, profile_summaries = _profile_diagnostic_rows(config_path, sources)
    artifact["profile_diagnostic_rows"] = profile_rows
    artifact["profile_diagnostic_summaries"] = profile_summaries
    similarity_chart = (config.get("charts") or {}).get("multisplit_similarity_distributions")
    if isinstance(similarity_chart, dict) and bool(similarity_chart.get("enabled", False)):
        similarity_task = str(similarity_chart.get("task") or "")
        if not similarity_task:
            raise MultisplitAnalysisError(
                "charts.multisplit_similarity_distributions.task is required"
            )
        similarity_model_ids = similarity_chart.get("models")
        similarity_sources = sources
        if similarity_model_ids is not None:
            selected_ids = {str(value) for value in similarity_model_ids}
            similarity_sources = [
                source for source in sources
                if str(source.get("analysis_model_id")) in selected_ids
            ]
        artifact["similarity_distributions"] = _selected_similarity_distributions(
            config_path, similarity_sources, expected_seeds, similarity_task
        )
    else:
        artifact["similarity_distributions"] = []
    artifact["scientific_interpretation"] = {
        "experimental_unit": "equal-weighted split-level mean after averaging training repeats",
        "sample_sd_ddof": 1,
        "ranges_are_confidence_intervals": False,
        "split_results_independent": False,
        "pair_counts_pooled": False,
        "similarity_distribution_normalization": (
            "unit-area match/non-match densities per source; equal mean over training "
            "repeats within split; equal mean over splits"
        ),
        "inference_claimed": False,
        "statement": "Results are descriptive across the evaluated splits; observed ranges are not confidence intervals.",
    }
    output_cfg = config.get("output") or {}
    output = _resolve(config_path, output_cfg.get("directory") or f"../analyses/{config['analysis_id']}")
    output.mkdir(parents=True, exist_ok=True)
    if bool(output_cfg.get("clean_generated_directories", True)):
        for generated in (output / "figures", output / "latex_tables"):
            if generated.is_dir():
                shutil.rmtree(generated)
    readme_path = write_analysis_readme(config, output)
    raw_csv = _write_csv(output / "raw_metrics.csv", artifact["raw_rows"])
    split_csv = _write_csv(output / "split_level_metrics.csv", artifact["split_level_rows"])
    summary_csv = _write_csv(output / "multisplit_summary.csv", artifact["summary_rows"])
    paired_csv = _write_csv(output / "paired_comparisons.csv", artifact["paired_comparisons"])
    denom_csv = _write_csv(output / "split_metric_denominators.csv", artifact["split_denominators"])
    profile_raw_csv = _write_csv(output / "profile_diagnostics_raw.csv", profile_rows)
    profile_summary_csv = _write_csv(output / "profile_diagnostics_summary.csv", profile_summaries)
    summary_json = output / "multisplit_summary.json"
    atomic_write_json(summary_json, artifact)
    table_paths = write_multisplit_tables(artifact, config, output / "latex_tables")
    chart_paths, chart_warnings = write_multisplit_charts(artifact, config, output / "figures")
    source_manifest = {
        "schema_version": 1,
        "artifact_type": "multisplit_source_manifest",
        "created_at": utc_now_iso(),
        "analysis_id": config["analysis_id"],
        "analysis_config": str(config_path),
        "analysis_config_sha256": sha256_file(config_path),
        "static_splits": static,
        "sources": sources,
    }
    manifest_path = output / "source_manifest.json"
    atomic_write_json(manifest_path, source_manifest)
    result = {
        "output_dir": str(output),
        "models": len(model_entries),
        "split_realizations": len(expected_seeds),
        "warnings": [*artifact.get("warnings", []), *chart_warnings],
        "outputs": {
            "readme": str(readme_path),
            "raw_csv": str(raw_csv),
            "split_level_csv": str(split_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "paired_csv": str(paired_csv),
            "denominators_csv": str(denom_csv),
            "profile_diagnostics_raw_csv": str(profile_raw_csv),
            "profile_diagnostics_summary_csv": str(profile_summary_csv),
            "tables": [str(path) for path in table_paths],
            "charts": [str(path) for path in chart_paths],
            "source_manifest": str(manifest_path),
        },
    }
    atomic_write_json(output / "analysis_result.json", result)
    return result
