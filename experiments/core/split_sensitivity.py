"""Descriptive sensitivity analysis over predefined Wikidata split realizations.

This module deliberately sits above the ordinary immutable evaluation artifacts.
It never creates splits, recalibrates thresholds, pools pair counts, or treats
split/training repetitions as independent observations. For trainable models it
first averages training repeats within each split, then summarizes the resulting
split-level means with descriptive sample statistics.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from .aggregate_runs import aggregate_paths
from .artifacts import atomic_write_json, sha256_file, utc_now_iso
from .latex_table_output import write_latex_table_with_values
from .latex_table_rendering import latex_escape, latex_label
from .split_schema import load_split

ARTIFACT_TYPE = "split_sensitivity_summary"
SCHEMA_VERSION = 1
DEFAULT_WIKIDATA_SPLIT_SEEDS = (6, 7, 9, 10, 21, 42, 67, 87, 1337, 4711)

# Metrics are summarized independently after each split's training repeats have
# been averaged. Counts are retained in raw/split-level artifacts, never pooled.
METRICS = (
    "precision",
    "recall",
    "f0_5",
    "f1",
    "f2",
    "pair_quality",
    "pair_completeness",
    "reduction_ratio",
    "query_coverage",
    "threshold",
    "candidate_pairs",
    "calibration_error",
    "selected_k",
    "selection_attained",
    "calibration_maximum_pair_completeness",
    "floor_added_candidate_pairs",
    "floor_rescued_true_positives",
    "queries_requiring_floor",
    "floor_activation_rate",
)
POPULATION_FIELDS = (
    "num_queries",
    "num_candidates",
    "num_positive_pairs",
    "num_cartesian_pairs",
    "num_possible_pairs",
    "num_excluded_pairs",
)
TASK_IDENTITY_FIELDS = (
    "query_ids_hash",
    "candidate_ids_hash",
    "positive_pairs_hash",
)
ROW_END = " " + r"\\"


class SplitSensitivityError(ValueError):
    """Raised for an incomplete or semantically unsafe sensitivity analysis."""


def load_split_sensitivity_config(path: Path | str) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Split-sensitivity config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SplitSensitivityError("Split-sensitivity config root must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SplitSensitivityError("Unsupported split-sensitivity schema_version")
    if not data.get("analysis_id"):
        raise SplitSensitivityError("Split-sensitivity config requires analysis_id")
    dataset_id = str(data.get("dataset_id") or "")
    if "wikidata" not in dataset_id.lower():
        raise SplitSensitivityError(
            "Split-sensitivity aggregation is restricted to Wikidata evaluations"
        )
    protocol = data.get("protocol") or {}
    if not isinstance(protocol, dict) or protocol.get("frozen_before_evaluation") is not True:
        raise SplitSensitivityError(
            "protocol.frozen_before_evaluation: true is required; the ten splits must not become a tuning set"
        )
    seeds = data.get("expected_split_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise SplitSensitivityError("expected_split_seeds must be a non-empty list")
    normalized = [int(seed) for seed in seeds]
    if len(normalized) != len(set(normalized)):
        raise SplitSensitivityError("expected_split_seeds contains duplicates")
    data["expected_split_seeds"] = normalized
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SplitSensitivityError("runs must be a non-empty list of run roots")
    static_splits = data.get("static_splits")
    if not isinstance(static_splits, list) or not static_splits:
        raise SplitSensitivityError(
            "static_splits must explicitly reference every versioned Wikidata split file"
        )
    static_seeds: list[int] = []
    for entry in static_splits:
        if not isinstance(entry, dict) or "seed" not in entry or not entry.get("file"):
            raise SplitSensitivityError("Every static_splits entry requires seed and file")
        entry["seed"] = int(entry["seed"])
        static_seeds.append(entry["seed"])
    if sorted(static_seeds) != sorted(normalized):
        raise SplitSensitivityError(
            f"static_splits seeds {sorted(static_seeds)} do not equal expected seeds {sorted(normalized)}"
        )
    configurations = data.get("configurations") or []
    if configurations and not isinstance(configurations, list):
        raise SplitSensitivityError("configurations must be a list")
    seen: set[str] = set()
    for entry in configurations:
        if not isinstance(entry, dict) or not entry.get("configuration_id"):
            raise SplitSensitivityError("Every configuration requires configuration_id")
        configuration_id = str(entry["configuration_id"])
        if configuration_id in seen:
            raise SplitSensitivityError(f"Duplicate configuration_id: {configuration_id}")
        seen.add(configuration_id)
        if "training_seeds" in entry:
            training_seeds = entry["training_seeds"]
            if not isinstance(training_seeds, list) or len(training_seeds) < 2:
                raise SplitSensitivityError(
                    f"{configuration_id}.training_seeds must contain at least two seeds"
                )
            normalized_training_seeds = [int(seed) for seed in training_seeds]
            if len(normalized_training_seeds) != len(set(normalized_training_seeds)):
                raise SplitSensitivityError(
                    f"{configuration_id}.training_seeds must not contain duplicates"
                )
            entry["training_seeds"] = normalized_training_seeds
    return config_path, data


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _validate_static_splits(config_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the exact committed split files before reading any result rows."""
    expected_dataset = str(config["dataset_id"])
    records: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    union_class_counts: set[int] = set()
    union_image_sets: list[set[str]] = []
    for entry in sorted(config["static_splits"], key=lambda item: int(item["seed"])):
        expected_seed = int(entry["seed"])
        path = _resolve(config_path, entry["file"])
        split = load_split(path)
        dataset = split.get("dataset") or {}
        if str(dataset.get("dataset_id")) != expected_dataset:
            raise SplitSensitivityError(
                f"Static split {path} uses dataset {dataset.get('dataset_id')!r}, expected {expected_dataset!r}"
            )
        strategy = split.get("split_strategy") or {}
        if strategy.get("type") != "class_disjoint_role" or int(strategy.get("seed", -1)) != expected_seed:
            raise SplitSensitivityError(
                f"Static split {path} is not class_disjoint_role with seed {expected_seed}"
            )
        ratios = strategy.get("ratios") or {}
        expected_ratios = {"train": 0.0, "val": 0.5, "test": 0.5}
        if any(
            not math.isclose(float(ratios.get(name, -1.0)), value, abs_tol=1e-12)
            for name, value in expected_ratios.items()
        ):
            raise SplitSensitivityError(
                f"Static split {path} must use train=0, val=0.5, test=0.5"
            )
        subsets = split.get("subsets") or {}
        if set(subsets) != {"val", "test"}:
            raise SplitSensitivityError(f"Static split {path} must contain exactly val and test subsets")
        images = split["images"]
        classes_by_subset: dict[str, set[int]] = {}
        ids_by_subset: dict[str, set[str]] = {}
        for subset_name in ("val", "test"):
            roles = subsets[subset_name].get("roles") or {}
            if not {"modern", "historic"}.issubset(roles):
                raise SplitSensitivityError(
                    f"Static split {path} subset {subset_name!r} requires modern and historic roles"
                )
            modern_classes = {int(images[file_id]["class_id"]) for file_id in roles["modern"]}
            historic_classes = {int(images[file_id]["class_id"]) for file_id in roles["historic"]}
            if modern_classes != historic_classes:
                raise SplitSensitivityError(
                    f"Static split {path} subset {subset_name!r} does not have both roles for every class"
                )
            classes_by_subset[subset_name] = modern_classes
            ids_by_subset[subset_name] = {
                str(file_id) for role_ids in roles.values() for file_id in role_ids
            }
        if classes_by_subset["val"] & classes_by_subset["test"]:
            raise SplitSensitivityError(f"Static split {path} leaks classes between val and test")
        if ids_by_subset["val"] & ids_by_subset["test"]:
            raise SplitSensitivityError(f"Static split {path} leaks images between val and test")
        union_classes = classes_by_subset["val"] | classes_by_subset["test"]
        union_images = ids_by_subset["val"] | ids_by_subset["test"]
        union_class_counts.add(len(union_classes))
        union_image_sets.append(union_images)
        source_hash = str(dataset.get("source_db_sha256") or "")
        if source_hash:
            source_hashes.add(source_hash)
        records.append(
            {
                "split_seed": expected_seed,
                "split_id": split["split_id"],
                "split_sha256": sha256_file(path),
                "path": str(path),
                "num_val_classes": len(classes_by_subset["val"]),
                "num_test_classes": len(classes_by_subset["test"]),
                "num_classes": len(union_classes),
                "num_val_images": len(ids_by_subset["val"]),
                "num_test_images": len(ids_by_subset["test"]),
                "num_images": len(union_images),
                "source_db_sha256": source_hash or None,
            }
        )
    if len(source_hashes) > 1:
        raise SplitSensitivityError("Static split files use different Wikidata source databases")
    if len(union_class_counts) != 1 or any(images != union_image_sets[0] for images in union_image_sets[1:]):
        raise SplitSensitivityError(
            "Static split versions do not partition the same eligible class/image population"
        )
    return records


def _configuration_lookup(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    entries: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for raw in config.get("configurations") or []:
        entry = dict(raw)
        configuration_id = str(entry["configuration_id"])
        entries[configuration_id] = entry
        source_ids = entry.get("source_ids") or [configuration_id]
        for source_id in source_ids:
            key = str(source_id)
            previous = aliases.get(key)
            if previous is not None and previous != configuration_id:
                raise SplitSensitivityError(f"Source identity {key!r} maps to multiple configurations")
            aliases[key] = configuration_id
    return entries, aliases


def _row_configuration_id(
    row: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    aliases: dict[str, str],
) -> str | None:
    candidates = (
        row.get("configuration_id"),
        row.get("finetuning_experiment_id"),
        row.get("model_storage_key"),
        row.get("model_id"),
    )
    if entries:
        for candidate in candidates:
            if candidate is not None and str(candidate) in aliases:
                return aliases[str(candidate)]
        return None
    return next((str(value) for value in candidates if value is not None and str(value)), None)


def _operation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("dataset_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("selection_mode") or ""),
        int(row["k"]) if row.get("k") is not None else None,
        _optional_float(row.get("target_pair_completeness")),
        _optional_float(row.get("calibration_target_pair_completeness")),
        str(row.get("calibration_id") or ""),
        int(row["minimum_top_k"]) if row.get("minimum_top_k") is not None else None,
    )


def _operation_fields(key: tuple[Any, ...]) -> dict[str, Any]:
    dataset_id, task_id, mode, k, target, calibration_target, calibration_id, minimum_top_k = key
    return {
        "dataset_id": dataset_id,
        "task_id": task_id,
        "selection_mode": mode,
        "k": k,
        "target_pair_completeness": target,
        "calibration_target_pair_completeness": calibration_target,
        "calibration_id": calibration_id or None,
        "minimum_top_k": minimum_top_k,
    }


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    return _optional_float(row.get(metric))


def _sample_variance(values: list[float]) -> float:
    return float(statistics.variance(values)) if len(values) > 1 else 0.0


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    variance = _sample_variance(values)
    return {
        "mean": float(statistics.fmean(values)),
        "sample_variance_split": variance,
        "sample_sd_split": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _normalize_rows(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entries, aliases = _configuration_lookup(config)
    dataset_id = str(config["dataset_id"])
    normalized: list[dict[str, Any]] = []
    for source in rows:
        if str(source.get("dataset_id")) != dataset_id:
            continue
        row = dict(source)
        configuration_id = _row_configuration_id(row, entries, aliases)
        if configuration_id is None:
            continue
        split_seed = row.get("split_seed")
        if split_seed is None:
            raise SplitSensitivityError(
                f"Run {row.get('run_id')} has no static split seed in run.json/split.snapshot.json"
            )
        row["configuration_id"] = configuration_id
        row["split_seed"] = int(split_seed)
        if row.get("num_cartesian_pairs") is None:
            queries, candidates = row.get("num_queries"), row.get("num_candidates")
            if queries is not None and candidates is not None:
                row["num_cartesian_pairs"] = int(queries) * int(candidates)
        training_seed = row.get("training_seed")
        row["training_seed"] = int(training_seed) if training_seed not in (None, "") else None
        target = row.get("calibration_target_pair_completeness")
        achieved = row.get("pair_completeness")
        row["calibration_error"] = (
            float(achieved) - float(target)
            if target not in (None, "") and achieved not in (None, "")
            else None
        )
        normalized.append(row)
    if not normalized:
        raise SplitSensitivityError(f"No selected rows found for dataset {dataset_id!r}")
    if not entries:
        for configuration_id in sorted({str(row["configuration_id"]) for row in normalized}):
            first = next(row for row in normalized if row["configuration_id"] == configuration_id)
            entries[configuration_id] = {
                "configuration_id": configuration_id,
                "label": first.get("display_name") or configuration_id,
            }
    return normalized, entries


def _validate_frozen_design(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Reject protocol/model drift across the predefined evaluation matrix."""
    protocol_hashes = {
        str(row["evaluation_protocol_sha256"])
        for row in rows
        if row.get("evaluation_protocol_sha256")
    }
    if bool(config.get("require_protocol_hashes", True)):
        if not protocol_hashes:
            raise SplitSensitivityError(
                "Evaluation protocol hashes are missing; rerun with run.json config.evaluation_protocol_sha256"
            )
        if len(protocol_hashes) != 1:
            raise SplitSensitivityError("Evaluation metric/calibration protocol changed across runs")
    dataset_hashes = {
        str(row["dataset_source_db_sha256"])
        for row in rows
        if row.get("dataset_source_db_sha256")
    }
    if len(dataset_hashes) > 1:
        raise SplitSensitivityError("Wikidata source database changed across split evaluations")
    if bool(config.get("require_static_split_hashes", True)) and any(
        not row.get("split_sha256") for row in rows
    ):
        raise SplitSensitivityError("Every split realization must have a recorded static split SHA-256")

    identities: dict[tuple[str, int | None], set[tuple[Any, ...]]] = defaultdict(set)
    for row in rows:
        training_seed = row.get("training_seed")
        identity = (
            row.get("checkpoint_sha256") or None,
            row.get("model_storage_key") or None,
            row.get("model_revision") or None,
        )
        identities[(str(row["configuration_id"]), training_seed)].add(identity)
    for (configuration_id, training_seed), values in identities.items():
        if len(values) != 1:
            raise SplitSensitivityError(
                f"Model/checkpoint identity changed across splits for {configuration_id}, "
                f"training_seed={training_seed}"
            )


def _validate_populations(
    rows: list[dict[str, Any]], expected_split_seeds: list[int]
) -> tuple[list[dict[str, Any]], list[str]]:
    signatures: dict[tuple[int, str], set[tuple[Any, ...]]] = defaultdict(set)
    split_ids: dict[int, set[tuple[str, str | None]]] = defaultdict(set)
    for row in rows:
        split_seed = int(row["split_seed"])
        task_id = str(row.get("task_id") or "")
        signatures[(split_seed, task_id)].add(
            tuple(row.get(field) for field in (*POPULATION_FIELDS, *TASK_IDENTITY_FIELDS))
        )
        split_ids[split_seed].add((str(row.get("split_id") or ""), row.get("split_sha256")))
    for key, values in signatures.items():
        if len(values) != 1:
            raise SplitSensitivityError(
                f"Incompatible task population denominators for split_seed={key[0]} task={key[1]!r}"
            )
    for split_seed, values in split_ids.items():
        if len(values) != 1:
            raise SplitSensitivityError(f"Multiple static split identities found for seed {split_seed}")
    populations: list[dict[str, Any]] = []
    for (split_seed, task_id), values in sorted(signatures.items()):
        signature = next(iter(values))
        split_id, split_sha = next(iter(split_ids[split_seed]))
        populations.append(
            {
                "split_seed": split_seed,
                "split_id": split_id,
                "split_sha256": split_sha,
                "task_id": task_id,
                **dict(zip((*POPULATION_FIELDS, *TASK_IDENTITY_FIELDS), signature)),
            }
        )
    warnings: list[str] = []
    for task_id in sorted({entry["task_id"] for entry in populations}):
        task_rows = [entry for entry in populations if entry["task_id"] == task_id]
        for field in ("num_queries", "num_positive_pairs", "num_candidates"):
            values = [int(entry[field]) for entry in task_rows if entry.get(field) is not None]
            if values and min(values) > 0 and max(values) / min(values) > 1.10:
                warnings.append(
                    f"{task_id}: {field} differs by more than 10% across split realizations "
                    f"({min(values)}-{max(values)})"
                )
    observed = sorted(split_ids)
    if observed != sorted(expected_split_seeds):
        raise SplitSensitivityError(
            f"Observed split seeds {observed} do not equal expected seeds {sorted(expected_split_seeds)}"
        )
    return populations, warnings


def aggregate_split_sensitivity_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build explicit split-summary rows and paired descriptive comparisons."""
    expected_splits = [int(seed) for seed in config["expected_split_seeds"]]
    aggregation = dict(config.get("aggregation") or {})
    attainment_tolerance = float(aggregation.get("attainment_tolerance", 1e-12))
    if not 0.0 <= attainment_tolerance <= 1.0:
        raise SplitSensitivityError(
            "aggregation.attainment_tolerance must be between 0 and 1"
        )
    tie_tolerance = float(aggregation.get("tie_tolerance", 1e-12))
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise SplitSensitivityError(
            "aggregation.tie_tolerance must be a finite non-negative number"
        )
    normalized, entries = _normalize_rows(rows, config)
    _validate_frozen_design(normalized, config)
    populations, warnings = _validate_populations(normalized, expected_splits)

    grouped: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[(str(row["configuration_id"]), _operation_key(row))].append(row)

    operations_by_configuration: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for configuration_id, operation in grouped:
        operations_by_configuration[configuration_id].add(operation)
    if bool(config.get("require_common_operating_points", True)) and operations_by_configuration:
        expected_operations = next(iter(operations_by_configuration.values()))
        for configuration_id, operations in operations_by_configuration.items():
            if operations != expected_operations:
                raise SplitSensitivityError(
                    f"Configuration {configuration_id!r} does not contain the same operating points as its peers"
                )

    summary_rows: list[dict[str, Any]] = []
    split_level_rows: list[dict[str, Any]] = []
    for (configuration_id, operation), operation_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], tuple(str(v) for v in item[0][1]))
    ):
        entry = entries[configuration_id]
        by_split: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in operation_rows:
            by_split[int(row["split_seed"])].append(row)
        observed_splits = sorted(by_split)
        if observed_splits != sorted(expected_splits):
            raise SplitSensitivityError(
                f"{configuration_id} {_operation_fields(operation)} has split seeds {observed_splits}; "
                f"expected {sorted(expected_splits)}"
            )
        expected_training = (
            {int(seed) for seed in entry["training_seeds"]}
            if "training_seeds" in entry
            else None
        )
        split_metric_values: dict[str, list[float]] = defaultdict(list)
        within_split_variances: dict[str, list[float]] = defaultdict(list)
        per_metric_split_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for split_seed in sorted(expected_splits):
            split_rows = by_split[split_seed]
            seeds = [row.get("training_seed") for row in split_rows]
            if len(seeds) != len(set(seeds)):
                raise SplitSensitivityError(
                    f"Duplicate training seed for {configuration_id}, split {split_seed}, {_operation_fields(operation)}"
                )
            observed_training = {int(seed) for seed in seeds if seed is not None}
            if expected_training is not None and observed_training != expected_training:
                raise SplitSensitivityError(
                    f"{configuration_id}, split {split_seed} has training seeds {sorted(observed_training)}; "
                    f"expected {sorted(expected_training)}"
                )
            if expected_training is None and observed_training and len(observed_training) != len(split_rows):
                raise SplitSensitivityError(
                    f"Mixed seeded and unseeded repetitions for {configuration_id}, split {split_seed}"
                )
            split_record = {
                "artifact_type": "split_level_training_mean",
                "configuration_id": configuration_id,
                "display_name": entry.get("label") or operation_rows[0].get("display_name") or configuration_id,
                "split_seed": split_seed,
                "training_seeds": sorted(observed_training),
                "num_training_repeats": len(split_rows),
                **_operation_fields(operation),
            }
            for metric in METRICS:
                values = [value for row in split_rows if (value := _metric_value(row, metric)) is not None]
                if not values:
                    continue
                if len(values) != len(split_rows):
                    raise SplitSensitivityError(
                        f"Metric {metric} is missing from some training repeats for {configuration_id}, split {split_seed}"
                    )
                mean = float(statistics.fmean(values))
                split_record[f"{metric}_mean_over_training_seeds"] = mean
                split_record[f"{metric}_training_seed_values"] = values
                split_metric_values[metric].append(mean)
                per_metric_split_values[metric].append(
                    {"split_seed": split_seed, "mean_over_training_seeds": mean, "training_seed_values": values}
                )
                if len(values) > 1:
                    within_split_variances[metric].append(_sample_variance(values))
            for field in POPULATION_FIELDS:
                split_record[field] = split_rows[0].get(field)
            split_level_rows.append(split_record)

        summary_row = {
            "artifact_type": ARTIFACT_TYPE,
            "configuration_id": configuration_id,
            "display_name": entry.get("label") or operation_rows[0].get("display_name") or configuration_id,
            "color": entry.get("color"),
            "group": entry.get("group"),
            "num_split_realizations": len(expected_splits),
            "split_seeds": sorted(expected_splits),
            "num_training_repeats_per_split": sorted({len(rows_) for rows_ in by_split.values()}),
            **_operation_fields(operation),
        }
        for metric, values in split_metric_values.items():
            for suffix, value in _summary(values).items():
                summary_row[f"{metric}_{suffix}"] = value
            summary_row[f"{metric}_split_values"] = per_metric_split_values[metric]
            variances = within_split_variances.get(metric) or []
            if variances:
                average_variance = float(statistics.fmean(variances))
                summary_row[f"{metric}_average_within_split_training_variance"] = average_variance
                summary_row[f"{metric}_average_within_split_training_sd"] = math.sqrt(average_variance)
        target = summary_row.get("calibration_target_pair_completeness")
        pc_values = split_metric_values.get("pair_completeness") or []
        if summary_row["selection_mode"] in {
            "calibrated_threshold",
            "calibrated_threshold_top_k_floor",
            "calibrated_top_k",
            "calibrated_union",
        } and target is not None and pc_values:
            attained = sum(
                value >= float(target) - attainment_tolerance
                for value in pc_values
            )
            summary_row["target_attainment_tolerance"] = attainment_tolerance
            summary_row["target_attainment_threshold"] = max(
                0.0, float(target) - attainment_tolerance
            )
            summary_row["target_attainment_count"] = attained
            summary_row["target_attainment_rate"] = attained / len(pc_values)
        if summary_row["selection_mode"] == "calibrated_top_k":
            selection_values = split_metric_values.get("selection_attained") or []
            if selection_values:
                selected = sum(value >= 0.5 for value in selection_values)
                summary_row["validation_selection_attainment_count"] = selected
                summary_row["validation_selection_attainment_rate"] = selected / len(selection_values)
        summary_rows.append(summary_row)

    denominator_rows = _denominator_rows(split_level_rows)
    paired = _paired_comparisons(
        summary_rows,
        config.get("comparisons") or [],
        expected_splits,
        tie_tolerance=tie_tolerance,
    )
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "analysis_id": config["analysis_id"],
        "dataset_id": config["dataset_id"],
        "created_at": utc_now_iso(),
        "interpretation": {
            "scope": "descriptive variation observed across predefined overlapping class-disjoint partition realizations",
            "split_results_independent": False,
            "ranges_are_confidence_intervals": False,
            "counts_pooled_across_splits": False,
            "training_repeats_pooled_with_splits": False,
            "split_statistic": "sample variance across equal-weighted split-level means after averaging training repeats",
            "training_statistic": "average within-split sample variance across training repeats; not a formal variance component",
            "inference_or_significance_claimed": False,
        },
        "protocol": dict(config.get("protocol") or {}),
        "aggregation": {
            **aggregation,
            "attainment_tolerance": attainment_tolerance,
            "tie_tolerance": tie_tolerance,
        },
        "expected_split_seeds": sorted(expected_splits),
        "split_populations": populations,
        "warnings": warnings,
        "summary_rows": summary_rows,
        "split_level_rows": split_level_rows,
        "split_denominators": denominator_rows,
        "paired_comparisons": paired,
        "raw_rows": normalized,
    }


def _denominator_rows(split_level_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact audit view of composition and retained-pair counts per split."""
    fields = (
        "configuration_id",
        "display_name",
        "dataset_id",
        "task_id",
        "selection_mode",
        "k",
        "target_pair_completeness",
        "calibration_target_pair_completeness",
        "calibration_id",
        "minimum_top_k",
        "split_seed",
        "training_seeds",
        "num_training_repeats",
        *POPULATION_FIELDS,
        "candidate_pairs_mean_over_training_seeds",
        "candidate_pairs_training_seed_values",
    )
    return [
        {
            "artifact_type": "split_metric_denominators",
            **{field: row.get(field) for field in fields},
        }
        for row in split_level_rows
    ]


def _split_values(row: dict[str, Any], metric: str) -> dict[int, float]:
    values = row.get(f"{metric}_split_values") or []
    return {
        int(entry["split_seed"]): float(entry["mean_over_training_seeds"])
        for entry in values
    }


def _paired_comparisons(
    summary_rows: list[dict[str, Any]],
    comparisons: list[Any],
    expected_splits: list[int],
    *,
    tie_tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    by_configuration_operation = {
        (str(row["configuration_id"]), _operation_key(row)): row for row in summary_rows
    }
    result: list[dict[str, Any]] = []
    for raw in comparisons:
        if not isinstance(raw, dict):
            raise SplitSensitivityError("comparisons entries must be mappings")
        comparison_id = str(raw.get("comparison_id") or "")
        minuend = str(raw.get("minuend") or "")
        subtrahend = str(raw.get("subtrahend") or "")
        if not comparison_id or not minuend or not subtrahend:
            raise SplitSensitivityError("Each comparison requires comparison_id, minuend, and subtrahend")
        minuend_operations = {
            operation for configuration_id, operation in by_configuration_operation if configuration_id == minuend
        }
        subtrahend_operations = {
            operation for configuration_id, operation in by_configuration_operation if configuration_id == subtrahend
        }
        if minuend_operations != subtrahend_operations:
            raise SplitSensitivityError(f"Paired comparison {comparison_id!r} has incompatible operating points")
        for operation in sorted(minuend_operations, key=lambda value: tuple(str(v) for v in value)):
            a = by_configuration_operation[(minuend, operation)]
            b = by_configuration_operation[(subtrahend, operation)]
            paired_row = {
                "artifact_type": "paired_split_sensitivity_summary",
                "comparison_id": comparison_id,
                "minuend": minuend,
                "subtrahend": subtrahend,
                "num_split_realizations": len(expected_splits),
                "tie_tolerance": float(tie_tolerance),
                **_operation_fields(operation),
            }
            for metric in METRICS:
                a_values, b_values = _split_values(a, metric), _split_values(b, metric)
                if not a_values and not b_values:
                    continue
                if sorted(a_values) != sorted(expected_splits) or sorted(b_values) != sorted(expected_splits):
                    raise SplitSensitivityError(
                        f"Comparison {comparison_id!r}, metric {metric} does not cover identical split seeds"
                    )
                deltas = [a_values[seed] - b_values[seed] for seed in sorted(expected_splits)]
                stats = _summary(deltas)
                for suffix, value in stats.items():
                    paired_row[f"{metric}_difference_{suffix}"] = value
                paired_row[f"{metric}_difference_split_values"] = [
                    {"split_seed": seed, "difference": delta}
                    for seed, delta in zip(sorted(expected_splits), deltas)
                ]
                paired_row[f"{metric}_wins"] = sum(
                    delta > tie_tolerance for delta in deltas
                )
                paired_row[f"{metric}_ties"] = sum(
                    abs(delta) <= tie_tolerance for delta in deltas
                )
                paired_row[f"{metric}_losses"] = sum(
                    delta < -tie_tolerance for delta in deltas
                )
            result.append(paired_row)
    return result


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "artifact_type", "configuration_id", "display_name", "comparison_id", "minuend", "subtrahend",
        "dataset_id", "task_id", "selection_mode", "k", "calibration_target_pair_completeness",
        "split_seed", "training_seed", "num_split_realizations", "target_attainment_count",
        "target_attainment_rate",
    ]
    ordered = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    return path


def _fmt_mean_range(row: dict[str, Any], metric: str, digits: int = 3) -> str:
    mean = row.get(f"{metric}_mean")
    lower = row.get(f"{metric}_min")
    upper = row.get(f"{metric}_max")
    if mean is None or lower is None or upper is None:
        return "--"
    return f"{float(mean):.{digits}f} [{float(lower):.{digits}f}, {float(upper):.{digits}f}]"


def _render_fixed_table(rows: list[dict[str, Any]], analysis_id: str) -> str:
    selected = [row for row in rows if row.get("selection_mode") == "top_k"]
    if not selected:
        return ""
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Fixed-$k$ Wikidata split sensitivity. Values are arithmetic means",
        "    across ten predefined split realizations after averaging training repeats within",
        "    each split; brackets show the observed minimum--maximum range, not a confidence",
        "    interval. Models are ranked by means only; ranking does not imply significance.}",
        f"    \\label{{{latex_label(f'tab:{analysis_id}-fixed-sensitivity')}}}",
        "    \\begin{tabular}{llrrr}",
        "        \\toprule",
        "        Model & $k$ & PC mean [range] & PQ mean [range] & RR mean [range]" + ROW_END,
        "        \\midrule",
    ]
    for row in sorted(selected, key=lambda value: (str(value.get("display_name")), int(value.get("k") or 0))):
        lines.append(
            "        " + " & ".join(
                [
                    latex_escape(str(row.get("display_name") or row["configuration_id"])),
                    str(row.get("k")),
                    _fmt_mean_range(row, "pair_completeness"),
                    _fmt_mean_range(row, "pair_quality"),
                    _fmt_mean_range(row, "reduction_ratio", 4),
                ]
            ) + ROW_END
        )
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def _render_calibrated_table(rows: list[dict[str, Any]], analysis_id: str) -> str:
    selected = [row for row in rows if row.get("selection_mode") == "calibrated_threshold"]
    if not selected:
        return ""
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Validation-calibrated Wikidata split sensitivity. Values are arithmetic",
        "    means across ten predefined split realizations after averaging training repeats",
        "    within each split; brackets are observed ranges, not confidence intervals. Target",
        "    met counts test partitions whose achieved PC reaches the calibration target. Training",
        "    SD is the square root of the average within-split training sample variance.}",
        f"    \\label{{{latex_label(f'tab:{analysis_id}-calibrated-sensitivity')}}}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{lrrrrrr}",
        "        \\toprule",
        "        Model & $\\rho$ & PC mean [range] & Target met & PQ mean [range] & RR mean [range] & PC training SD" + ROW_END,
        "        \\midrule",
    ]
    for row in sorted(
        selected,
        key=lambda value: (
            str(value.get("display_name")),
            float(value.get("calibration_target_pair_completeness") or 0.0),
        ),
    ):
        count = row.get("target_attainment_count")
        total = row.get("num_split_realizations")
        lines.append(
            "        " + " & ".join(
                [
                    latex_escape(str(row.get("display_name") or row["configuration_id"])),
                    f"{float(row.get('calibration_target_pair_completeness')):.3f}",
                    _fmt_mean_range(row, "pair_completeness"),
                    f"{count}/{total}" if count is not None else "--",
                    _fmt_mean_range(row, "pair_quality"),
                    _fmt_mean_range(row, "reduction_ratio", 4),
                    (
                        f"{float(row['pair_completeness_average_within_split_training_sd']):.3f}"
                        if row.get("pair_completeness_average_within_split_training_sd") is not None
                        else "--"
                    ),
                ]
            ) + ROW_END
        )
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def _render_paired_table(rows: list[dict[str, Any]], analysis_id: str) -> str:
    if not rows:
        return ""
    calibrated = [row for row in rows if row.get("selection_mode") == "calibrated_threshold"]
    selected = calibrated or rows
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Paired model differences on identical predefined split realizations.",
        "    Brackets show observed difference ranges and wins count strictly positive",
        "    differences. These descriptive comparisons do not imply statistical significance.}",
        f"    \\label{{{latex_label(f'tab:{analysis_id}-paired-sensitivity')}}}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{llrrr}",
        "        \\toprule",
        "        Comparison & Operating point & $\\Delta$PC mean [range]; wins & $\\Delta$PQ mean [range]; wins & $\\Delta$RR mean [range]; wins" + ROW_END,
        "        \\midrule",
    ]
    for row in selected:
        target = row.get("calibration_target_pair_completeness")
        operating = f"$\\rho={float(target):.3f}$" if target is not None else f"$k={row.get('k')}$"
        cells = [latex_escape(str(row["comparison_id"])), operating]
        for metric in ("pair_completeness", "pair_quality", "reduction_ratio"):
            mean = row.get(f"{metric}_difference_mean")
            lower = row.get(f"{metric}_difference_min")
            upper = row.get(f"{metric}_difference_max")
            wins = row.get(f"{metric}_wins")
            total = row.get("num_split_realizations")
            cells.append(
                "--" if mean is None else f"{float(mean):+.3f} [{float(lower):+.3f}, {float(upper):+.3f}]; {wins}/{total}"
            )
        lines.append("        " + " & ".join(cells) + ROW_END)
    lines.extend(["        \\bottomrule", "    \\end{tabular}%", "    }", "\\end{table}"])
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _write_charts(
    rows: list[dict[str, Any]], output_dir: Path, formats: tuple[str, ...]
) -> tuple[list[Path], list[str]]:
    try:
        import matplotlib
    except ModuleNotFoundError:
        return [], ["matplotlib is not installed; split-sensitivity charts were skipped"]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    warnings: list[str] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id") or "task")].append(row)
    for task_id, task_rows in by_task.items():
        fixed = [row for row in task_rows if row.get("selection_mode") == "top_k"]
        if fixed:
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in fixed:
                by_configuration[str(row["configuration_id"])].append(row)
            for configuration_id, model_rows in sorted(by_configuration.items()):
                model_rows.sort(key=lambda row: int(row["k"]))
                ks = [int(row["k"]) for row in model_rows]
                means = [float(row["pair_completeness_mean"]) for row in model_rows]
                lower = [float(row["pair_completeness_min"]) for row in model_rows]
                upper = [float(row["pair_completeness_max"]) for row in model_rows]
                first = model_rows[0]
                color = first.get("color") or None
                label = str(first.get("display_name") or configuration_id)
                line, = ax.plot(ks, means, marker="o", ms=3, lw=1.3, color=color, label=label)
                ax.fill_between(ks, lower, upper, color=line.get_color(), alpha=0.15)
            ax.set_xscale("log")
            ax.set_xlabel(r"Candidate budget $k$")
            ax.set_ylabel(r"Mean pairs completeness PC@$k$")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            paths.extend(_save_formats(fig, output_dir, f"pc_at_k_sensitivity_{_safe_name(task_id)}", formats))
            plt.close(fig)

            for k in sorted({int(row["k"]) for row in fixed}):
                selected = [row for row in fixed if int(row["k"]) == k]
                selected.sort(key=lambda row: float(row["pair_completeness_mean"]), reverse=True)
                fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.42 * len(selected) + 1.2)))
                y = list(range(len(selected)))
                means = [float(row["pair_completeness_mean"]) for row in selected]
                low = [mean - float(row["pair_completeness_min"]) for mean, row in zip(means, selected)]
                high = [float(row["pair_completeness_max"]) - mean for mean, row in zip(means, selected)]
                colors = [row.get("color") or "#4C78A8" for row in selected]
                ax.barh(y, means, color=colors, alpha=0.8)
                ax.errorbar(means, y, xerr=[low, high], fmt="none", ecolor="black", capsize=3, lw=0.8)
                for ypos, row in zip(y, selected):
                    values = _split_values(row, "pair_completeness").values()
                    ax.scatter(list(values), [ypos] * len(list(values)), color="black", s=8, alpha=0.35, zorder=3)
                ax.set_yticks(y, [str(row.get("display_name") or row["configuration_id"]) for row in selected])
                ax.invert_yaxis()
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel(f"Mean PC@{k}; whiskers are observed split range")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                paths.extend(_save_formats(fig, output_dir, f"ranked_pc_at_{k}_sensitivity_{_safe_name(task_id)}", formats))
                plt.close(fig)

        calibrated = [row for row in task_rows if row.get("selection_mode") == "calibrated_threshold"]
        for target in sorted({float(row["calibration_target_pair_completeness"]) for row in calibrated}):
            target_rows = [row for row in calibrated if float(row["calibration_target_pair_completeness"]) == target]
            for metric, label in (
                ("pair_completeness", "PC"),
                ("pair_quality", "PQ"),
                ("reduction_ratio", "RR"),
            ):
                target_rows.sort(key=lambda row: float(row[f"{metric}_mean"]), reverse=True)
                fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.46 * len(target_rows) + 1.2)))
                y = list(range(len(target_rows)))
                means = [float(row[f"{metric}_mean"]) for row in target_rows]
                low = [mean - float(row[f"{metric}_min"]) for mean, row in zip(means, target_rows)]
                high = [float(row[f"{metric}_max"]) - mean for mean, row in zip(means, target_rows)]
                colors = [row.get("color") or "#4C78A8" for row in target_rows]
                ax.barh(y, means, color=colors, alpha=0.8)
                ax.errorbar(means, y, xerr=[low, high], fmt="none", ecolor="black", capsize=3, lw=0.8)
                for ypos, row in zip(y, target_rows):
                    values = list(_split_values(row, metric).values())
                    ax.scatter(values, [ypos] * len(values), color="black", s=8, alpha=0.35, zorder=3)
                    if metric == "pair_completeness":
                        ax.text(
                            min(0.995, float(row[f"{metric}_max"]) + 0.008), ypos,
                            f"{row.get('target_attainment_count', 0)}/{row['num_split_realizations']}",
                            va="center", fontsize=7,
                        )
                ax.set_yticks(y, [str(row.get("display_name") or row["configuration_id"]) for row in target_rows])
                ax.invert_yaxis()
                ax.set_xlim(0.0, 1.0)
                if metric == "pair_completeness":
                    ax.axvline(target, color="black", ls=":", lw=0.8)
                ax.set_xlabel(f"Mean {label} at calibration target {target:g}; whiskers are observed split range")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                tag = f"{target:g}".replace(".", "p")
                paths.extend(_save_formats(fig, output_dir, f"calibrated_{metric}_rho{tag}_{_safe_name(task_id)}", formats))
                plt.close(fig)
    return paths, warnings


def _save_formats(fig: Any, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        paths.append(path)
    return paths


def write_split_sensitivity_analysis(config_path: Path | str) -> dict[str, Any]:
    path, config = load_split_sensitivity_config(config_path)
    static_splits = _validate_static_splits(path, config)
    run_paths = [_resolve(path, value) for value in config["runs"]]
    rows = aggregate_paths(run_paths, allow_duplicates=True)
    artifact = aggregate_split_sensitivity_rows(rows, config)
    static_by_seed = {int(entry["split_seed"]): entry for entry in static_splits}
    for population in artifact["split_populations"]:
        static = static_by_seed[int(population["split_seed"])]
        if population.get("split_id") != static["split_id"]:
            raise SplitSensitivityError(
                f"Run split_id for seed {population['split_seed']} does not match configured static file"
            )
        if population.get("split_sha256") != static["split_sha256"]:
            raise SplitSensitivityError(
                f"Run split SHA-256 for seed {population['split_seed']} does not match configured static file"
            )
    artifact["static_splits"] = static_splits
    output_cfg = config.get("output") or {}
    output_dir = _resolve(path, output_cfg.get("directory") or f"../analyses/{config['analysis_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact["provenance"] = {
        "analysis_config": str(path),
        "analysis_config_sha256": sha256_file(path),
        "run_roots": [str(value) for value in run_paths],
        "source_eval_paths": sorted({str(row.get("eval_path")) for row in artifact["raw_rows"]}),
    }
    summary_json = output_dir / "split_sensitivity_metrics.json"
    atomic_write_json(summary_json, artifact)
    summary_csv = _write_csv(output_dir / "split_sensitivity_metrics.csv", artifact["summary_rows"])
    split_csv = _write_csv(output_dir / "split_level_metrics.csv", artifact["split_level_rows"])
    denominators_csv = _write_csv(
        output_dir / "split_metric_denominators.csv", artifact["split_denominators"]
    )
    raw_csv = _write_csv(output_dir / "raw_metrics.csv", artifact["raw_rows"])
    paired_csv = _write_csv(output_dir / "paired_comparisons.csv", artifact["paired_comparisons"])

    tables_dir = output_dir / "latex_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_paths: list[Path] = []
    for name, content in (
        ("fixed_k_sensitivity.tex", _render_fixed_table(artifact["summary_rows"], config["analysis_id"])),
        ("calibrated_sensitivity.tex", _render_calibrated_table(artifact["summary_rows"], config["analysis_id"])),
        ("paired_sensitivity.tex", _render_paired_table(artifact["paired_comparisons"], config["analysis_id"])),
    ):
        if content:
            table_path = tables_dir / name
            table_paths.extend(write_latex_table_with_values(table_path, content))

    formats = tuple(str(value) for value in output_cfg.get("formats") or ["pdf"])
    invalid = sorted(set(formats) - {"pdf", "svg", "png"})
    if invalid:
        raise SplitSensitivityError(f"Unsupported chart format(s): {', '.join(invalid)}")
    chart_paths, chart_warnings = _write_charts(
        artifact["summary_rows"], output_dir / "figures", formats
    )
    result = {
        "analysis_id": config["analysis_id"],
        "output_dir": str(output_dir),
        "summary_rows": len(artifact["summary_rows"]),
        "split_level_rows": len(artifact["split_level_rows"]),
        "paired_rows": len(artifact["paired_comparisons"]),
        "warnings": [*artifact["warnings"], *chart_warnings],
        "outputs": {
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "split_level_csv": str(split_csv),
            "split_denominators_csv": str(denominators_csv),
            "raw_csv": str(raw_csv),
            "paired_csv": str(paired_csv),
            "tables": [str(value) for value in table_paths],
            "charts": [str(value) for value in chart_paths],
        },
    }
    atomic_write_json(output_dir / "analysis_result.json", result)
    return result
