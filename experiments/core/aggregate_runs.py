"""Aggregate parseable experiment eval.json artifacts into flat tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .artifacts import atomic_write_json, sha256_file, utc_now_iso

_DUPLICATE_FIELDS = ("experiment_id", "split_id", "model_id", "task_id")


def discover_eval_files(paths: Iterable[Path | str]) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value).expanduser()
        if path.is_file() and path.name == "eval.json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("**/models/*/eval.json")))
    return sorted(set(files))


def rows_from_eval(
    eval_file: Path,
    run_meta: dict[str, Any] | None = None,
    *,
    include_incomplete: bool = False,
) -> list[dict[str, Any]]:
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    run_dir = eval_file.parents[2]
    meta = run_meta if run_meta is not None else _load_run_json(run_dir, include_incomplete=include_incomplete)
    return _rows_from_eval_data(eval_file, data, meta)


def aggregate_paths(
    paths: Iterable[Path | str],
    csv_output: Path | str | None = None,
    json_output: Path | str | None = None,
    *,
    include_incomplete: bool = False,
    approved_run_manifest: Path | str | None = None,
    allow_duplicates: bool = False,
) -> list[dict[str, Any]]:
    input_paths = [Path(path).expanduser() for path in paths]
    manifest = _load_manifest(approved_run_manifest) if approved_run_manifest is not None else None
    discovery_paths = input_paths or _manifest_discovery_paths(manifest)
    if not discovery_paths:
        raise ValueError("At least one run path or an approved-run manifest with run paths is required")

    rows: list[dict[str, Any]] = []
    task_sources: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    included_runs: dict[str, dict[str, Any]] = {}
    skipped_runs: dict[str, dict[str, Any]] = {}
    eval_files = discover_eval_files(discovery_paths)
    for eval_file in eval_files:
        run_dir = eval_file.parents[2]
        run_key = str(run_dir.resolve())
        run_meta = _load_run_json(run_dir, include_incomplete=True)
        if manifest is not None and not _manifest_approves_run(manifest, run_dir, run_meta):
            skipped_runs.setdefault(run_key, _skip_record(run_dir, run_meta, "not_in_approved_manifest"))
            continue
        if run_meta.get("_missing_run_json") and not include_incomplete:
            skipped_runs.setdefault(run_key, _skip_record(run_dir, run_meta, "missing_run_json"))
            continue
        if run_meta.get("status") != "completed" and not include_incomplete:
            skipped_runs.setdefault(run_key, _skip_record(run_dir, run_meta, "run_status_not_completed"))
            continue

        data = json.loads(eval_file.read_text(encoding="utf-8"))
        included_runs.setdefault(run_key, _included_record(run_dir, run_meta))
        rows.extend(_rows_from_eval_data(eval_file, data, run_meta))
        _collect_task_sources(task_sources, eval_file, data, run_meta)

    if not allow_duplicates:
        duplicates = _duplicate_task_sources(task_sources)
        if duplicates:
            raise ValueError(_duplicate_message(duplicates))

    rows = _sort_rows(rows)
    if csv_output is not None:
        write_csv(Path(csv_output), rows)
    if json_output is not None:
        atomic_write_json(
            Path(json_output),
            {
                "schema_version": 1,
                "provenance": _provenance(
                    input_paths,
                    discovery_paths,
                    manifest,
                    included_runs,
                    skipped_runs,
                    len(eval_files),
                    include_incomplete,
                    allow_duplicates,
                ),
                "rows": rows,
            },
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "experiment_id", "run_id", "run_status", "dataset_id", "split_id", "split_seed",
        "task_id", "configuration_id", "training_seed", "model_id", "display_name",
        "model_revision", "finetuning_experiment_id", "finetuning_run_id",
        "checkpoint_kind", "checkpoint_sha256", "selection_mode", "k", "target_pair_completeness", "threshold", "precision",
        "recall", "f0_5", "f1", "f2", "pair_quality", "pair_completeness", "reduction_ratio",
        "query_coverage", "calibration_id", "calibration_dataset_id", "calibration_split_id",
        "calibration_split_seed", "calibration_split_sha256", "calibration_target_pair_completeness",
        "calibration_pair_completeness", "calibration_pair_quality", "calibration_reduction_ratio",
        "calibration_query_coverage",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _rows_from_eval_data(eval_file: Path, data: dict[str, Any], run_meta: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = eval_file.parents[2]
    rows = []
    model_id = data.get("model_id")
    model_storage_key = data.get("model_storage_key")
    model_revision = data.get("model_revision") or _model_revision(run_meta, model_id, model_storage_key)
    display_name = _model_display_name(run_meta, model_id, model_storage_key)
    parent_training = run_meta.get("parent_training") or {}
    split_meta = run_meta.get("split") or {}
    split_seed = split_meta.get("split_seed")
    if split_seed is None:
        split_seed = _snapshot_split_seed(run_dir, split_meta)
    configuration_id = (
        parent_training.get("configuration_id")
        or parent_training.get("experiment_id")
        or model_storage_key
        or model_id
    )
    for task in data.get("tasks", []):
        base = {
            "experiment_id": run_meta.get("experiment_id"),
            "run_id": run_meta.get("run_id", run_dir.name),
            "run_status": run_meta.get("status"),
            "dataset_id": (run_meta.get("dataset") or {}).get("dataset_id"),
            "dataset_source_db_sha256": (run_meta.get("dataset") or {}).get("source_db_sha256"),
            "evaluation_protocol_sha256": (run_meta.get("config") or {}).get("evaluation_protocol_sha256"),
            "split_id": split_meta.get("split_id"),
            "split_seed": split_seed,
            "split_sha256": split_meta.get("split_sha256"),
            "configuration_id": configuration_id,
            "training_seed": parent_training.get("training_seed"),
            "training_split_id": (parent_training.get("training_split") or {}).get("split_id"),
            "training_split_sha256": (parent_training.get("training_split") or {}).get("split_sha256"),
            "training_split_seed": (parent_training.get("training_split") or {}).get("split_seed"),
            "training_protocol": parent_training.get("training_split"),
            "checkpoint_selection": parent_training.get("checkpoint_selection"),
            "best_epoch": parent_training.get("best_epoch"),
            "best_validation_set": parent_training.get("best_validation_set"),
            "model_id": model_id,
            "display_name": display_name,
            "model_revision": model_revision,
            "model_storage_key": model_storage_key,
            "finetuning_experiment_id": parent_training.get("experiment_id"),
            "finetuning_run_id": parent_training.get("run_id"),
            "finetuning_run_dir": parent_training.get("run_dir"),
            "checkpoint_kind": parent_training.get("checkpoint_kind"),
            "checkpoint_path": parent_training.get("checkpoint_path"),
            "checkpoint_sha256": parent_training.get("checkpoint_sha256"),
            "task_id": task.get("task_id"),
            "num_queries": task.get("num_queries"),
            "num_candidates": task.get("num_candidates"),
            "num_cartesian_pairs": task.get("num_cartesian_pairs"),
            "num_possible_pairs": task.get("num_possible_pairs"),
            "num_excluded_pairs": task.get("num_excluded_pairs"),
            "num_positive_pairs": task.get("num_positive_pairs"),
            "query_ids_hash": task.get("query_ids_hash"),
            "candidate_ids_hash": task.get("candidate_ids_hash"),
            "positive_pairs_hash": task.get("positive_pairs_hash"),
            "eval_path": str(eval_file),
            "run_json_path": str(run_dir / "run.json") if (run_dir / "run.json").is_file() else None,
        }
        calibration_source = run_meta.get("calibration_source")
        if isinstance(calibration_source, dict):
            base.update(
                {
                    "calibration_dataset_id": calibration_source.get("dataset_id"),
                    "calibration_split_id": calibration_source.get("split_id"),
                    "calibration_split_seed": calibration_source.get("split_seed"),
                    "calibration_split_sha256": calibration_source.get("split_sha256"),
                }
            )
        for metrics in task.get("metrics", []):
            rows.append({**base, "selection_mode": "top_k", **metrics})
        for metrics in task.get("target_pc_metrics", []):
            rows.append({**base, "selection_mode": "target_pc", "k": None, **metrics})
        for metrics in task.get("calibrated_threshold_metrics", []):
            rows.append({**base, "selection_mode": "calibrated_threshold", "k": None, **metrics})
        for metrics in task.get("calibrated_threshold_top_k_floor_metrics", []):
            rows.append(
                {
                    **base,
                    "selection_mode": "calibrated_threshold_top_k_floor",
                    "k": None,
                    **metrics,
                }
            )
        for metrics in task.get("calibrated_top_k_metrics", []):
            selected_k = metrics.get("k")
            rows.append(
                {
                    **base,
                    "selection_mode": "calibrated_top_k",
                    **metrics,
                    "selected_k": selected_k,
                    "k": None,
                    "selection_attained": bool(metrics.get("calibration_target_attained", False)),
                }
            )
    return rows


def _snapshot_split_seed(run_dir: Path, split_meta: dict[str, Any]) -> int | None:
    raw_path = split_meta.get("split_path") or "split.snapshot.json"
    path = Path(str(raw_path))
    path = path if path.is_absolute() else run_dir / path
    if not path.is_file():
        return None
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
        seed = (split.get("split_strategy") or {}).get("seed")
        return int(seed) if seed is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _collect_task_sources(
    task_sources: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    eval_file: Path,
    data: dict[str, Any],
    run_meta: dict[str, Any],
) -> None:
    run_dir = eval_file.parents[2]
    model_id = data.get("model_id")
    for task in data.get("tasks", []):
        key = (
            str(run_meta.get("experiment_id")),
            str((run_meta.get("split") or {}).get("split_id")),
            str(model_id),
            str(task.get("task_id")),
        )
        task_sources.setdefault(key, []).append(
            {
                "run_id": run_meta.get("run_id", run_dir.name),
                "run_dir": str(run_dir),
                "eval_path": str(eval_file),
                "status": run_meta.get("status"),
            }
        )


def _duplicate_task_sources(
    task_sources: dict[tuple[str, str, str, str], list[dict[str, Any]]]
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    return {key: value for key, value in task_sources.items() if len(value) > 1}


def _duplicate_message(duplicates: dict[tuple[str, str, str, str], list[dict[str, Any]]]) -> str:
    examples = []
    for key, sources in list(duplicates.items())[:5]:
        fields = ", ".join(f"{name}={value!r}" for name, value in zip(_DUPLICATE_FIELDS, key))
        paths = ", ".join(str(source.get("eval_path")) for source in sources[:3])
        examples.append(f"({fields}) in {paths}")
    suffix = "" if len(duplicates) <= 5 else f" and {len(duplicates) - 5} more"
    return "Duplicate aggregate task keys found: " + "; ".join(examples) + suffix + ". Use --allow-duplicates to override."


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_order = {
        "top_k": 0,
        "target_pc": 1,
        "calibrated_threshold": 2,
        "calibrated_threshold_top_k_floor": 3,
        "calibrated_top_k": 4,
    }
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("experiment_id")),
            str(row.get("run_id")),
            str(row.get("task_id")),
            str(row.get("model_id")),
            mode_order.get(str(row.get("selection_mode")), 99),
            int(row.get("k") or 0),
            int(row.get("minimum_top_k") or 0),
            float(
                row.get("target_pair_completeness")
                or row.get("calibration_target_pair_completeness")
                or 0.0
            ),
        ),
    )


def _load_run_json(run_dir: Path, *, include_incomplete: bool = False) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        if include_incomplete:
            return {"run_id": run_dir.name, "status": None, "_missing_run_json": True}
        raise FileNotFoundError(f"run.json not found for aggregate candidate: {run_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not include_incomplete and data.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run_dir} status={data.get('status')!r}")
    return data


def _model_revision(run_meta: dict[str, Any], model_id: Any, model_storage_key: Any) -> str | None:
    for entry in run_meta.get("model_runs", []) or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("model_storage_key") == model_storage_key or entry.get("model_id") == model_id:
            revision = entry.get("model_revision") or entry.get("revision") or entry.get("resolved_revision")
            return str(revision) if revision else None
    return None


def _model_display_name(run_meta: dict[str, Any], model_id: Any, model_storage_key: Any) -> str | None:
    for entry in run_meta.get("model_runs", []) or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("model_storage_key") == model_storage_key or entry.get("model_id") == model_id:
            value = entry.get("display_name")
            return str(value) if value else None
    return None


def _load_manifest(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Approved-run manifest not found: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if manifest_path.suffix.lower() in {".yml", ".yaml"} else json.loads(text)
    entries = data if isinstance(data, list) else (data or {}).get("approved_runs", (data or {}).get("runs"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Approved-run manifest must contain a non-empty 'approved_runs' or 'runs' list")
    approved_paths: set[str] = set()
    approved_ids: set[str] = set()
    for entry in entries:
        _add_manifest_entry(entry, manifest_path.parent, approved_paths, approved_ids)
    return {
        "path": manifest_path,
        "sha256": sha256_file(manifest_path),
        "approved_paths": approved_paths,
        "approved_ids": approved_ids,
        "raw": data,
    }


def _add_manifest_entry(entry: Any, base_dir: Path, approved_paths: set[str], approved_ids: set[str]) -> None:
    if isinstance(entry, str):
        value = entry.strip()
        if not value:
            raise ValueError("Approved-run manifest contains an empty run entry")
        approved_ids.add(Path(value).name)
        approved_paths.add(str(_manifest_path(value, base_dir).resolve()))
        return
    if not isinstance(entry, dict):
        raise ValueError("Approved-run manifest entries must be strings or mappings")
    run_id = entry.get("run_id")
    if run_id:
        approved_ids.add(str(run_id))
    run_path = entry.get("run_dir") or entry.get("path") or entry.get("dir")
    if run_path:
        path = _manifest_path(str(run_path), base_dir).resolve()
        approved_paths.add(str(path))
        approved_ids.add(path.name)
    if not run_id and not run_path:
        raise ValueError("Approved-run manifest mapping entries require run_id or run_dir/path")


def _manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _manifest_discovery_paths(manifest: dict[str, Any] | None) -> list[Path]:
    if manifest is None:
        return []
    return [Path(value) for value in sorted(manifest.get("approved_paths") or [])]


def _manifest_approves_run(manifest: dict[str, Any], run_dir: Path, run_meta: dict[str, Any]) -> bool:
    resolved = str(run_dir.resolve())
    run_id = str(run_meta.get("run_id") or run_dir.name)
    approved_paths = manifest.get("approved_paths") or set()
    approved_ids = manifest.get("approved_ids") or set()
    return resolved in approved_paths or run_id in approved_ids or run_dir.name in approved_ids


def _included_record(run_dir: Path, run_meta: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "run.json"
    return {
        "run_dir": str(run_dir),
        "run_id": run_meta.get("run_id", run_dir.name),
        "experiment_id": run_meta.get("experiment_id"),
        "status": run_meta.get("status"),
        "run_json_path": str(path) if path.is_file() else None,
        "run_json_sha256": sha256_file(path) if path.is_file() else None,
    }


def _skip_record(run_dir: Path, run_meta: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "run_id": run_meta.get("run_id", run_dir.name),
        "experiment_id": run_meta.get("experiment_id"),
        "status": run_meta.get("status"),
        "reason": reason,
    }


def _provenance(
    input_paths: list[Path],
    discovery_paths: list[Path],
    manifest: dict[str, Any] | None,
    included_runs: dict[str, dict[str, Any]],
    skipped_runs: dict[str, dict[str, Any]],
    discovered_eval_files: int,
    include_incomplete: bool,
    allow_duplicates: bool,
) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "input_paths": [str(path) for path in input_paths],
        "discovery_paths": [str(path) for path in discovery_paths],
        "approved_run_manifest": None
        if manifest is None
        else {"path": str(manifest["path"]), "sha256": manifest["sha256"]},
        "include_incomplete": bool(include_incomplete),
        "allow_duplicates": bool(allow_duplicates),
        "duplicate_key_fields": list(_DUPLICATE_FIELDS),
        "discovered_eval_files": int(discovered_eval_files),
        "included_runs": sorted(included_runs.values(), key=lambda item: str(item.get("run_dir"))),
        "skipped_runs": sorted(skipped_runs.values(), key=lambda item: str(item.get("run_dir"))),
    }
