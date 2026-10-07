"""Zero-margin bar charts for calibration protocols other than a plain threshold.

The threshold protocol already writes its bar charts into ``figures/``. Each
additional protocol writes the same filenames into
``figures/calibration_protocols/<protocol>/``. Safety-margin sweeps are not
part of this output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROTOCOL_DIRECTORY = "calibration_protocols"
DEFAULT_PROTOCOLS = ("calibrated_k", "union")
KNOWN_PROTOCOLS = ("threshold", "calibrated_k", "union")
SELECTION_MODE = {
    "threshold": "calibrated_threshold",
    "calibrated_k": "calibrated_top_k",
    "union": "calibrated_union",
}
PC_AXIS_LABEL = {
    "threshold": "Test PC at validation-selected threshold",
    "calibrated_k": "Test PC at validation-selected k",
    "union": "Test PC at the union of validation-selected k and threshold",
}
_CLEARED_ON_UNION = (
    "precision",
    "recall",
    "f0_5",
    "f1",
    "f2",
    "pair_quality",
    "query_coverage",
    "candidate_pairs",
    "calibration_error",
    "calibration_pair_quality",
    "calibration_reduction_ratio",
    "calibration_query_coverage",
    "selection_attained",
    "calibration_maximum_pair_completeness",
    "floor_added_candidate_pairs",
    "floor_rescued_true_positives",
    "queries_requiring_floor",
    "floor_activation_rate",
    "minimum_top_k",
    "k",
)


class CalibrationProtocolBarsError(ValueError):
    """Raised when a protocol bar-chart config asks for an unsupported sweep."""


def protocol_directory(figures_dir: Path, protocol: str) -> Path:
    return figures_dir / PROTOCOL_DIRECTORY / protocol


def parse_protocol_chart(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the enabled protocol chart, inheriting ``style_from`` settings."""
    charts = config.get("charts") or {}
    chart = charts.get("calibration_protocol_bars")
    if not isinstance(chart, dict) or not bool(chart.get("enabled", False)):
        return None
    if chart.get("safety_margins") is True:
        raise CalibrationProtocolBarsError(
            "charts.calibration_protocol_bars.safety_margins must be false; "
            "safety-margin sweeps are not rendered"
        )
    style_from = chart.get("style_from")
    merged = dict(chart)
    if style_from is not None:
        inherited = charts.get(str(style_from))
        if not isinstance(inherited, dict):
            raise CalibrationProtocolBarsError(
                "charts.calibration_protocol_bars.style_from must name another chart mapping"
            )
        merged = {**inherited, **chart}
    protocols = merged.get("protocols") or list(DEFAULT_PROTOCOLS)
    if not isinstance(protocols, list) or not protocols:
        raise CalibrationProtocolBarsError(
            "charts.calibration_protocol_bars.protocols must be a non-empty list"
        )
    unknown = [str(value) for value in protocols if str(value) not in KNOWN_PROTOCOLS]
    if unknown:
        raise CalibrationProtocolBarsError(
            "charts.calibration_protocol_bars.protocols has unknown entries: "
            + ", ".join(unknown)
        )
    if len(protocols) != len({str(value) for value in protocols}):
        raise CalibrationProtocolBarsError(
            "charts.calibration_protocol_bars.protocols contains duplicates"
        )
    merged["protocols"] = [str(value) for value in protocols]
    merged["safety_margins"] = False
    return merged


def chart_targets(chart: dict[str, Any]) -> list[float]:
    raw = chart.get("calibration_targets", chart.get("target_pc"))
    if raw is None:
        return []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [float(raw)]
    return [float(value) for value in raw]


def zero_margin_union_metrics(document: dict[str, Any]) -> dict[str, float] | None:
    """Return the union cell with no K margin and no threshold margin.

    Other cells in ``calibrated_union.json`` are safety margins and are ignored.
    """
    k_margins = list(document.get("k_margins") or [])
    threshold_margins = list(document.get("threshold_margins") or [])
    k_index = next((index for index, value in enumerate(k_margins) if int(value) == 0), None)
    threshold_index = next(
        (
            index
            for index, value in enumerate(threshold_margins)
            if abs(float(value)) <= 1e-12
        ),
        None,
    )
    if k_index is None or threshold_index is None:
        return None
    completeness = document.get("pair_completeness") or []
    reduction = document.get("reduction_ratio") or []
    applied_k = document.get("applied_k") or []
    applied_threshold = document.get("applied_threshold") or []
    if (
        k_index >= len(completeness)
        or threshold_index >= len(completeness[k_index])
        or k_index >= len(reduction)
        or threshold_index >= len(reduction[k_index])
        or k_index >= len(applied_k)
        or threshold_index >= len(applied_threshold)
    ):
        return None
    target = document.get("target_pc")
    if target is None:
        return None
    return {
        "calibration_target_pair_completeness": float(target),
        "pair_completeness": float(completeness[k_index][threshold_index]),
        "reduction_ratio": float(reduction[k_index][threshold_index]),
        "selected_k": float(applied_k[k_index]),
        "threshold": float(applied_threshold[threshold_index]),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def union_rows_for_batch(batch: list[dict[str, Any]], eval_path: Path) -> list[dict[str, Any]]:
    """Build one zero-margin union row per matching threshold row in ``batch``."""
    document = _load_json(eval_path.parent / "calibrated_union.json")
    if document is None:
        return []
    metrics = zero_margin_union_metrics(document)
    if metrics is None:
        return []
    target = metrics["calibration_target_pair_completeness"]
    calibration_pc = document.get("calibration_pair_completeness")
    rows: list[dict[str, Any]] = []
    for base in batch:
        if base.get("selection_mode") != "calibrated_threshold":
            continue
        observed = base.get("calibration_target_pair_completeness")
        if observed is None or abs(float(observed) - target) > 1e-9:
            continue
        row = dict(base)
        for key in _CLEARED_ON_UNION:
            row[key] = None
        row["selection_mode"] = "calibrated_union"
        row["calibration_target_pair_completeness"] = target
        row["pair_completeness"] = metrics["pair_completeness"]
        row["reduction_ratio"] = metrics["reduction_ratio"]
        row["selected_k"] = metrics["selected_k"]
        row["threshold"] = metrics["threshold"]
        row["calibration_pair_completeness"] = (
            float(calibration_pc) if calibration_pc is not None else None
        )
        rows.append(row)
    return rows


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def protocol_operating_rows(
    task: dict[str, Any] | None,
    protocol: str,
    *,
    union_document: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Stored operating points for one protocol, without safety margins."""
    if protocol == "threshold":
        return list((task or {}).get("calibrated_threshold_metrics") or [])
    if protocol == "calibrated_k":
        return list((task or {}).get("calibrated_top_k_metrics") or [])
    if protocol == "union":
        metrics = zero_margin_union_metrics(union_document or {})
        if metrics is None:
            return []
        row = dict(metrics)
        calibration_pc = (union_document or {}).get("calibration_pair_completeness")
        if calibration_pc is not None:
            row["calibration_pair_completeness"] = float(calibration_pc)
        return [row]
    raise CalibrationProtocolBarsError(f"Unknown calibration protocol {protocol!r}")


def protocol_bars_by_target(
    models: list[Any],
    artifacts: dict[str, dict[str, Any]],
    task_id: str,
    protocol: str,
) -> dict[float, list[dict[str, Any]]]:
    """Test PC and RR at the zero-margin operating point, grouped by target PC."""
    from .run_figures import _eval_task

    by_rho: dict[float, list[dict[str, Any]]] = {}
    for spec in models:
        stored = artifacts.get(spec.storage_key) or {}
        task = _eval_task(stored, task_id)
        union_document = None
        if protocol == "union":
            union_document = _load_json(spec.model_dir / "calibrated_union.json")
        for row in protocol_operating_rows(task, protocol, union_document=union_document):
            rho = _finite(row.get("calibration_target_pair_completeness"))
            pc = _finite(row.get("pair_completeness"))
            rr = _finite(row.get("reduction_ratio"))
            if rho is None or pc is None or rr is None:
                continue
            by_rho.setdefault(rho, []).append(
                {
                    "spec": spec,
                    "pc": pc,
                    "rr": rr,
                    "threshold": _finite(row.get("threshold")),
                    "calibration_pc": _finite(row.get("calibration_pair_completeness")),
                    "calibration_rr": _finite(row.get("calibration_reduction_ratio")),
                }
            )
    for bars in by_rho.values():
        bars.sort(key=lambda entry: (-entry["pc"], entry["spec"].storage_key))
    return by_rho


def protocol_validation_rr_by_target(
    models: list[Any],
    artifacts: dict[str, dict[str, Any]],
    task_id: str,
    protocol: str,
) -> dict[float, list[dict[str, Any]]]:
    """Calibration-population RR when that value is stored. Union does not store it."""
    bars = protocol_bars_by_target(models, artifacts, task_id, protocol)
    by_rho: dict[float, list[dict[str, Any]]] = {}
    for rho, entries in bars.items():
        kept = [entry for entry in entries if entry.get("calibration_rr") is not None]
        if not kept:
            continue
        kept.sort(key=lambda entry: (-float(entry["calibration_rr"]), entry["spec"].storage_key))
        by_rho[rho] = [
            {
                "spec": entry["spec"],
                "rr": float(entry["calibration_rr"]),
                "calibration_pc": entry.get("calibration_pc"),
            }
            for entry in kept
        ]
    return by_rho


def prune_incomplete_union_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop every union row unless each 99% threshold row has a matching union row.

    A partial protocol would change the shared operating-point set across models.
    """
    union_rows = [row for row in rows if row.get("selection_mode") == "calibrated_union"]
    if not union_rows:
        return rows
    targets = {
        float(row["calibration_target_pair_completeness"])
        for row in union_rows
        if row.get("calibration_target_pair_completeness") is not None
    }
    if len(targets) != 1:
        return [row for row in rows if row.get("selection_mode") != "calibrated_union"]
    target = next(iter(targets))

    def _key(row: dict[str, Any]) -> tuple[str, int, str]:
        return (
            str(row.get("configuration_id")),
            int(row["split_seed"]),
            str(row.get("task_id")),
        )

    expected = {
        _key(row)
        for row in rows
        if row.get("selection_mode") == "calibrated_threshold"
        and row.get("split_seed") is not None
        and row.get("calibration_target_pair_completeness") is not None
        and abs(float(row["calibration_target_pair_completeness"]) - target) <= 1e-9
    }
    actual = {_key(row) for row in union_rows if row.get("split_seed") is not None}
    if actual != expected:
        return [row for row in rows if row.get("selection_mode") != "calibrated_union"]
    return rows
