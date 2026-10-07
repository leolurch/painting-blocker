"""Persistent, refreshable cross-run chart analysis configurations.

An analysis config separates immutable evaluation artifacts from mutable model
selection and presentation choices.  Discovery groups completed evaluation runs
by a canonical resolved-config fingerprint with the model list removed, then
merges newly found model artifacts without resetting user-owned fields such as
``include`` or ``presentation``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_write_text, sha256_json, utc_now_iso
from .checkpoint_naming import parse_checkpoint_name

ANALYSIS_SCHEMA_VERSION = 1


class AnalysisConfigError(ValueError):
    """Raised for malformed or incompatible analysis configuration."""


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AnalysisConfigError(f"Expected a JSON mapping: {path}")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AnalysisConfigError(f"Expected a YAML mapping: {path}")
    return data


def load_analysis_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Analysis config not found: {config_path}")
    data = _load_yaml(config_path)
    if data.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise AnalysisConfigError(
            f"Unsupported analysis schema_version in {config_path}: "
            f"{data.get('schema_version')!r}"
        )
    if not isinstance(data.get("discovery"), dict):
        raise AnalysisConfigError("Analysis config requires a discovery mapping")
    compatibility = data.get("compatibility")
    if compatibility is not None and not isinstance(compatibility, dict):
        raise AnalysisConfigError("Analysis config compatibility must be a mapping")
    if isinstance(compatibility, dict):
        compatibility_experiment_ids(compatibility)
        compatibility_family_hashes(compatibility)
    if not isinstance(data.get("models"), list):
        raise AnalysisConfigError("Analysis config requires a models list")
    if not isinstance(data.get("charts"), dict):
        raise AnalysisConfigError("Analysis config requires a charts mapping")
    mode = str(data.get("mode") or "single_source")
    if mode not in {"single_source", "multisplit"}:
        raise AnalysisConfigError(f"Unknown analysis mode: {mode!r}")
    if mode == "multisplit":
        from .multisplit_analysis import validate_multisplit_config

        validate_multisplit_config(config_path, data)
    model_ids = [
        str(entry.get("analysis_model_id"))
        for entry in data["models"]
        if isinstance(entry, dict) and entry.get("analysis_model_id")
    ]
    if len(model_ids) != len(set(model_ids)):
        raise AnalysisConfigError("Analysis config contains duplicate analysis_model_id entries")
    return data


def write_analysis_config(path: Path | str, config: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=120)
    atomic_write_text(target, text)
    return target


def _resolve_from_config(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _relative_to_config(config_path: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), config_path.parent.resolve())


def config_family_payload(resolved_config: dict[str, Any]) -> dict[str, Any]:
    """Return the strict run-family identity: resolved config minus models."""
    payload = copy.deepcopy(resolved_config)
    payload.pop("models", None)
    return payload


def config_family_sha256(resolved_config: dict[str, Any]) -> str:
    return sha256_json(config_family_payload(resolved_config))


def _replace_path_prefix(value: Any, prefix: str, replacement: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, prefix, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_path_prefix(item, prefix, replacement) for item in value]
    if isinstance(value, str) and (value == prefix or value.startswith(prefix + os.sep)):
        return replacement + value[len(prefix):]
    return value


def portable_config_family_payload(resolved_config: dict[str, Any]) -> dict[str, Any]:
    """Family identity independent of the checkout location.

    Historical cluster artifacts recorded ``repo_root`` as the configs directory,
    while current artifacts record the repository root. Both layouts resolve the
    same scientific protocol and should join one analysis family.
    """
    payload = config_family_payload(resolved_config)
    repo_root = str(payload.pop("repo_root", "") or "")
    if not repo_root:
        return payload
    root = Path(repo_root)
    config_root = root if root.name == "configs" and root.parent.name == "experiments" else root / "experiments" / "configs"
    return _replace_path_prefix(payload, str(config_root), "<config_root>")


def portable_config_family_sha256(resolved_config: dict[str, Any]) -> str:
    return sha256_json(portable_config_family_payload(resolved_config))


def compatibility_experiment_ids(compatibility: dict[str, Any]) -> tuple[str, ...]:
    """Return accepted experiment IDs from the scalar-or-list config field."""
    value = compatibility.get("experiment_id")
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise AnalysisConfigError(
            "compatibility.experiment_id must be a string or a non-empty list of strings"
        )
    result: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise AnalysisConfigError(
                "compatibility.experiment_id must be a string or a non-empty list of strings"
            )
        if item not in result:
            result.append(item)
    return tuple(result)


def primary_experiment_id(compatibility: dict[str, Any]) -> str | None:
    """Return the canonical experiment ID used for presentation metadata."""
    values = compatibility_experiment_ids(compatibility)
    return values[0] if values else None


def compatibility_family_hashes(compatibility: dict[str, Any]) -> tuple[str, ...]:
    """Return explicitly accepted strict family hashes from a scalar-or-list field."""
    value = compatibility.get("config_family_sha256")
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise AnalysisConfigError(
            "compatibility.config_family_sha256 must be a string or a non-empty list of strings"
        )
    result: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise AnalysisConfigError(
                "compatibility.config_family_sha256 must be a string or a non-empty list of strings"
            )
        if item not in result:
            result.append(item)
    return tuple(result)


def _family_matches(
    resolved: dict[str, Any],
    family_hashes: tuple[str, ...],
    experiment_ids: tuple[str, ...],
) -> bool:
    """Match a family while allowing only the configured experiment-ID aliases."""
    resolved_experiment = resolved.get("experiment_id")
    if experiment_ids and resolved_experiment not in experiment_ids:
        return False
    if (
        config_family_sha256(resolved) in family_hashes
        or portable_config_family_sha256(resolved) in family_hashes
    ):
        return True
    if not experiment_ids:
        return False
    # The stored family hash was created from one member of the alias list.
    # Substitute every accepted ID so all other protocol fields remain under
    # the same strict family-hash check.
    for experiment_id in experiment_ids:
        if experiment_id == resolved_experiment:
            continue
        normalized = copy.deepcopy(resolved)
        normalized["experiment_id"] = experiment_id
        if (
            config_family_sha256(normalized) in family_hashes
            or portable_config_family_sha256(normalized) in family_hashes
        ):
            return True
    return False


def _run_candidates(runs_root: Path) -> list[Path]:
    # run.json is the completion marker. This also finds post-evaluation runs
    # nested below finetuning runs while naturally ignoring training-only dirs.
    return sorted(path.parent for path in runs_root.glob("**/run.json"))


def _artifact_capabilities(model_dir: Path) -> dict[str, bool]:
    eval_path = model_dir / "eval.json"
    curves_path = model_dir / "curves.json"
    calibration_path = model_dir / "threshold_calibration.json"
    capabilities = {
        "fixed_k": False,
        "threshold_curve": False,
        "similarity_histogram": False,
        "calibrated_thresholds": False,
        "calibrated_candidate_sizes": False,
    }
    if eval_path.is_file():
        data = _load_json(eval_path)
        tasks = data.get("tasks") or []
        capabilities["fixed_k"] = any(task.get("metrics") for task in tasks)
        capabilities["calibrated_thresholds"] = any(
            task.get("calibrated_threshold_metrics") for task in tasks
        ) or calibration_path.is_file()
    if curves_path.is_file():
        data = _load_json(curves_path)
        tasks = data.get("tasks") or []
        capabilities["threshold_curve"] = any(task.get("precision_recall") for task in tasks)
        capabilities["similarity_histogram"] = any(
            (task.get("histogram") or {}).get("match_hist") for task in tasks
        )
        capabilities["calibrated_candidate_sizes"] = any(
            task.get("calibrated_candidate_sizes") for task in tasks
        )
    return capabilities


def _resolved_models(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "resolved_config.yml"
    if not path.is_file():
        return {}
    resolved = _load_yaml(path)
    entries = ((resolved.get("models") or {}).get("include") or [])
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        storage_key = str(entry.get("model_storage_key") or "")
        if storage_key:
            result[storage_key] = dict(entry)
    return result


def _source_record(
    config_path: Path,
    run_dir: Path,
    run_meta: dict[str, Any],
    model_run: dict[str, Any],
) -> dict[str, Any] | None:
    storage_key = str(model_run.get("model_storage_key") or "")
    if not storage_key:
        return None
    model_dir = run_dir / "models" / storage_key
    eval_path = model_dir / "eval.json"
    if not eval_path.is_file():
        return None
    source = {
        "run_dir": _relative_to_config(config_path, run_dir),
        "model_dir": _relative_to_config(config_path, model_dir),
        "created_at": run_meta.get("created_at"),
        "run_id": run_meta.get("run_id", run_dir.name),
        "git_commit": (run_meta.get("code") or {}).get("git_commit"),
        "capabilities": _artifact_capabilities(model_dir),
    }
    return source


def _discover(
    config_path: Path,
    runs_root: Path,
    family_hashes: tuple[str, ...],
    compatibility: dict[str, Any],
    *,
    completed_runs_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    sources: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    identity_payloads: dict[str, str] = {}
    experiment_ids = compatibility_experiment_ids(compatibility)
    for run_dir in _run_candidates(runs_root):
        try:
            run_meta = _load_json(run_dir / "run.json")
            resolved_path = run_dir / "resolved_config.yml"
            if not resolved_path.is_file():
                continue
            resolved = _load_yaml(resolved_path)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            continue
        if completed_runs_only and run_meta.get("status") != "completed":
            continue
        run_identity = {
            "experiment_id": run_meta.get("experiment_id"),
            "dataset_id": (run_meta.get("dataset") or {}).get("dataset_id"),
            "dataset_source_db_sha256": (run_meta.get("dataset") or {}).get("source_db_sha256"),
            "split_id": (run_meta.get("split") or {}).get("split_id"),
            "split_sha256": (run_meta.get("split") or {}).get("split_sha256"),
        }
        if experiment_ids and run_identity["experiment_id"] not in experiment_ids:
            continue
        if any(
            key != "experiment_id"
            and compatibility.get(key) is not None
            and run_identity.get(key) != compatibility.get(key)
            for key in run_identity
        ):
            continue
        if not _family_matches(resolved, family_hashes, experiment_ids):
            continue
        resolved_by_storage = _resolved_models(run_dir)
        accepted = 0
        for model_run in run_meta.get("model_runs") or []:
            if not isinstance(model_run, dict) or model_run.get("status") != "completed":
                continue
            storage_key = str(model_run.get("model_storage_key") or "")
            source = _source_record(config_path, run_dir, run_meta, model_run)
            if not storage_key or source is None:
                continue
            resolved_model = resolved_by_storage.get(storage_key, {})
            # Verify that a storage identity has consistent discovered semantics.
            identity_view = {
                key: value
                for key, value in resolved_model.items()
                if key not in {
                    "config_path", "display_name", "label", "name", "color", "group",
                    "model_storage_key",
                }
            }
            if identity_view:
                identity_json = json.dumps(identity_view, sort_keys=True, separators=(",", ":"))
                previous = identity_payloads.get(storage_key)
                if previous is not None and previous != identity_json:
                    raise AnalysisConfigError(
                        f"Conflicting resolved model semantics for storage key {storage_key!r}"
                    )
                identity_payloads[storage_key] = identity_json
            sources.setdefault(storage_key, []).append(source)
            model_id = str(
                model_run.get("model_id") or resolved_model.get("model_id") or storage_key
            )
            checkpoint_naming = parse_checkpoint_name(model_id)
            if checkpoint_naming is None:
                adapter_kwargs = resolved_model.get("adapter_kwargs") or {}
                if isinstance(adapter_kwargs, dict) and adapter_kwargs.get("checkpoint_path"):
                    checkpoint_naming = parse_checkpoint_name(
                        str(adapter_kwargs["checkpoint_path"])
                    )
            explicit_label = resolved_model.get("label") or resolved_model.get("name")
            discovered_metadata = {
                "analysis_model_id": storage_key,
                "model_id": model_id,
                "model_storage_key": storage_key,
                "kind": model_run.get("kind"),
                "discovered": {
                    "revision": model_run.get("model_revision", resolved_model.get("revision")),
                    "adapter_name": resolved_model.get("adapter_name"),
                    "embedding": resolved_model.get("embedding"),
                    "preprocessing": resolved_model.get("preprocessing"),
                    "source_model_id": resolved_model.get("source_model_id"),
                    "checkpoint_naming": checkpoint_naming,
                },
                "defaults": {
                    "label": (
                        explicit_label
                        or (checkpoint_naming or {}).get("title")
                        or resolved_model.get("display_name")
                        or model_run.get("display_name")
                        or model_run.get("model_id")
                        or storage_key
                    ),
                    "color": resolved_model.get("color"),
                    "group": resolved_model.get("group"),
                },
            }
            if storage_key not in metadata or (
                resolved_model and not metadata[storage_key]["discovered"].get("adapter_name")
            ):
                metadata[storage_key] = discovered_metadata
            accepted += 1
        runs.append(
            {
                "run_dir": _relative_to_config(config_path, run_dir),
                "run_id": run_meta.get("run_id", run_dir.name),
                "created_at": run_meta.get("created_at"),
                "config_sha256": (run_meta.get("config") or {}).get("config_sha256"),
                "git_commit": (run_meta.get("code") or {}).get("git_commit"),
                "models": accepted,
            }
        )
    for candidates in sources.values():
        candidates.sort(key=lambda item: (str(item.get("created_at") or ""), item["run_dir"]))
    runs.sort(key=lambda item: (str(item.get("created_at") or ""), item["run_dir"]))
    return runs, sources, metadata


def _source_score(source: dict[str, Any]) -> tuple[int, str, str]:
    richness = sum(bool(value) for value in (source.get("capabilities") or {}).values())
    return richness, str(source.get("created_at") or ""), str(source.get("run_dir") or "")


def _choose_source(candidates: list[dict[str, Any]], policy: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    if policy == "newest_complete":
        return max(
            candidates,
            key=lambda source: (
                str(source.get("created_at") or ""), str(source.get("run_dir") or "")
            ),
        )
    if policy in {"prefer_existing", "richest_then_newest"}:
        return max(candidates, key=_source_score)
    raise AnalysisConfigError(f"Unknown source_policy: {policy!r}")


def _merge_model(
    existing: dict[str, Any] | None,
    discovered: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    new_models_include: bool,
    source_policy: str,
    order: int,
) -> dict[str, Any]:
    model = copy.deepcopy(existing) if existing is not None else {}
    model.update(
        {
            "analysis_model_id": discovered["analysis_model_id"],
            "model_id": discovered["model_id"],
            "model_storage_key": discovered["model_storage_key"],
            "discovered": discovered["discovered"],
        }
    )
    if discovered.get("kind") is not None:
        model["kind"] = discovered["kind"]
    model.setdefault("include", bool(new_models_include))
    presentation = model.setdefault("presentation", {})
    defaults = discovered["defaults"]
    generated = model.setdefault("generated", {})
    generated_label = str(defaults.get("label") or discovered["model_id"])
    previous_generated_label = generated.get("label")
    current_label = presentation.get("label")
    placeholder_labels = {
        "Finetuned checkpoint (static template)",
        "Finetuned checkpoint",
    }
    if (
        current_label is None
        or current_label == previous_generated_label
        or current_label in placeholder_labels
    ):
        presentation["label"] = generated_label
    generated["label"] = generated_label
    if defaults.get("color") is not None:
        presentation.setdefault("color", defaults["color"])
    if defaults.get("group") is not None:
        presentation.setdefault("group", defaults["group"])
    presentation.setdefault("order", order)

    source = model.setdefault("source", {})
    source.setdefault("mode", "auto")
    if source.get("mode") not in {"auto", "pinned"}:
        raise AnalysisConfigError(
            f"Unknown source.mode for {discovered['analysis_model_id']}: {source.get('mode')!r}"
        )
    available_runs = {candidate["run_dir"] for candidate in candidates}
    selected = source.get("selected_run")
    if source.get("mode") == "pinned":
        pass
    elif source_policy == "prefer_existing" and selected in available_runs:
        source["selected_run"] = selected
    else:
        chosen_source = _choose_source(candidates, source_policy)
        selected = chosen_source["run_dir"] if chosen_source else selected
        source["selected_run"] = selected

    chosen = next((candidate for candidate in candidates if candidate["run_dir"] == source.get("selected_run")), None)
    source["selected_model_dir"] = chosen.get("model_dir") if chosen else source.get("selected_model_dir")
    model["availability"] = {
        "available": chosen is not None,
        "source_count": len(candidates),
        "source_runs": [candidate["run_dir"] for candidate in candidates],
        "selected_capabilities": (chosen or {}).get("capabilities", {}),
    }
    return model


def _default_charts(task_id: str | None, top_k: list[int], targets: list[float]) -> dict[str, Any]:
    common = {"enabled": True, "task": task_id, "formats": ["pdf"]}
    return {
        "pc_at_k_curve": {**common, "target_pc_lines": targets},
        "pc_rr_tradeoff": {**common, "target_pc_lines": targets},
        "pc_at_k_star_bars": {**common, "preferred_k": 25},
        "calibrated_threshold_bars": {
            **common,
            "calibration_targets": targets,
            "missing_data": "warn",
        },
        "validation_rr_bars": {
            **common,
            "calibration_targets": targets,
            "missing_data": "warn",
            "show_values": False,
            "value_decimals": 4,
        },
        "ranked_pc_at_k_bars": {
            "enabled": True,
            "task": task_id,
            "ks": top_k,
            "formats": ["svg"],
        },
        "similarity_distributions": dict(common),
        "candidate_sizes": {
            **common,
            "calibration_targets": targets,
            "missing_data": "warn",
        },
        "calibration_protocol_bars": {
            **common,
            "calibration_targets": targets,
            "missing_data": "warn",
            "safety_margins": False,
            "protocols": ["calibrated_k", "union"],
            "style_from": "calibrated_threshold_bars",
        },
        "pq_pc_curve": dict(common),
        "blocking_tables": {"enabled": True, "task": task_id, "ks": top_k},
    }


def initialize_analysis_config(
    reference_run: Path | str,
    runs_root: Path | str,
    output: Path | str,
    *,
    analysis_id: str | None = None,
    new_models_include: bool = True,
    source_policy: str = "prefer_existing",
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Analysis config already exists; use analysis-update instead: {output_path}"
        )
    reference = Path(reference_run).expanduser().resolve()
    root = Path(runs_root).expanduser().resolve()
    if not (reference / "run.json").is_file() or not (reference / "resolved_config.yml").is_file():
        raise FileNotFoundError(f"Reference evaluation run is incomplete: {reference}")
    run_meta = _load_json(reference / "run.json")
    if run_meta.get("status") != "completed":
        raise AnalysisConfigError(
            f"Reference run must be completed: {reference} status={run_meta.get('status')!r}"
        )
    resolved = _load_yaml(reference / "resolved_config.yml")
    family_hash = config_family_sha256(resolved)
    evaluation = dict(resolved.get("evaluation") or {})
    tasks = evaluation.get("retrieval_tasks") or []
    task_id = str(tasks[0].get("task_id")) if tasks and tasks[0].get("task_id") else None
    calibration = dict(evaluation.get("threshold_calibration") or {})
    config: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": analysis_id or output_path.stem,
        "discovery": {
            "runs_root": _relative_to_config(output_path, root),
            "reference_run": _relative_to_config(output_path, reference),
            "completed_runs_only": True,
            "new_models_include": bool(new_models_include),
            "source_policy": source_policy,
        },
        "compatibility": {
            "experiment_id": run_meta.get("experiment_id"),
            "config_family_sha256": family_hash,
            "dataset_id": (run_meta.get("dataset") or {}).get("dataset_id"),
            "dataset_source_db_sha256": (run_meta.get("dataset") or {}).get("source_db_sha256"),
            "split_id": (run_meta.get("split") or {}).get("split_id"),
            "split_sha256": (run_meta.get("split") or {}).get("split_sha256"),
        },
        "output": {
            "directory": f"../analyses/{analysis_id or output_path.stem}",
            "clean_generated_directories": True,
        },
        "models": [],
        "charts": _default_charts(
            task_id,
            [int(value) for value in evaluation.get("top_k") or []],
            [float(value) for value in calibration.get("target_pc") or []],
        ),
        "inventory": {},
    }
    config, summary = _refresh(output_path, config)
    if not summary["compatible_runs"] or not summary["discovered_models"]:
        raise AnalysisConfigError(
            "No completed compatible model artifacts were found below the configured runs root"
        )
    write_analysis_config(output_path, config)
    return output_path, config, summary


def _refresh(config_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    discovery = config["discovery"]
    runs_root = _resolve_from_config(config_path, discovery["runs_root"])
    compatibility = dict(config.get("compatibility") or {})
    compatibility_experiment_ids(compatibility)
    family_hashes = compatibility_family_hashes(compatibility)
    if not family_hashes:
        raise AnalysisConfigError("compatibility.config_family_sha256 is required")
    runs, sources, metadata = _discover(
        config_path,
        runs_root,
        family_hashes,
        compatibility,
        completed_runs_only=bool(discovery.get("completed_runs_only", True)),
    )
    existing_list = [entry for entry in config.get("models") or [] if isinstance(entry, dict)]
    existing = {str(entry.get("analysis_model_id")): entry for entry in existing_list if entry.get("analysis_model_id")}
    merged: list[dict[str, Any]] = []
    added: list[str] = []
    discovery_order = {key: index for index, key in enumerate(metadata)}
    ordered_keys = sorted(
        metadata,
        key=lambda key: (
            0 if key in existing else 1,
            int((existing.get(key, {}).get("presentation") or {}).get("order", 10**9)),
            discovery_order[key],
        ),
    )
    for index, storage_key in enumerate(ordered_keys):
        if storage_key not in existing:
            added.append(storage_key)
        merged.append(
            _merge_model(
                existing.get(storage_key),
                metadata[storage_key],
                sources.get(storage_key, []),
                new_models_include=bool(discovery.get("new_models_include", True)),
                source_policy=str(discovery.get("source_policy") or "prefer_existing"),
                order=index,
            )
        )
    # Preserve no-longer-discovered entries and user choices; mark unavailable.
    missing: list[str] = []
    for entry in existing_list:
        key = str(entry.get("analysis_model_id") or "")
        if key and key not in metadata:
            stale = copy.deepcopy(entry)
            stale["availability"] = {
                "available": False,
                "source_count": 0,
                "source_runs": [],
                "selected_capabilities": {},
            }
            merged.append(stale)
            missing.append(key)
    merged.sort(key=lambda entry: (
        int((entry.get("presentation") or {}).get("order", 10**9)),
        str(entry.get("analysis_model_id")),
    ))
    now = utc_now_iso()
    config["models"] = merged
    config["inventory"] = {
        "updated_at": now,
        "compatible_runs": runs,
        "num_compatible_runs": len(runs),
        "num_discovered_models": len(metadata),
    }
    summary = {
        "compatible_runs": len(runs),
        "discovered_models": len(metadata),
        "added_models": added,
        "unavailable_models": missing,
    }
    return config, summary


def update_analysis_config(path: Path | str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    config = load_analysis_config(config_path)
    if str(config.get("mode") or "single_source") == "multisplit":
        from .multisplit_analysis import update_multisplit_analysis_config

        config, summary = update_multisplit_analysis_config(config_path, config)
    else:
        config, summary = _refresh(config_path, config)
    write_analysis_config(config_path, config)
    return config_path, config, summary


def resolve_analysis_path(config_path: Path | str, value: str | Path) -> Path:
    return _resolve_from_config(Path(config_path).expanduser().resolve(), value)
