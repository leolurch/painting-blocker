"""Write compact numeric-data sidecars for rendered charts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json


GRAPH_SUFFIXES = {".pdf", ".png", ".svg"}


def json_chart_value(value: Any) -> Any:
    """Convert chart inputs to strict JSON values without changing precision."""
    if isinstance(value, np.ndarray):
        return [json_chart_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_chart_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): json_chart_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [json_chart_value(item) for item in value if item is not None]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Chart JSON sidecars cannot contain non-finite numbers")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported chart JSON value: {type(value).__name__}")


def write_chart_sidecar(output_dir: Path, stem: str, data: dict[str, Any]) -> Path:
    """Write ``<stem>.json`` beside every image format sharing that stem."""
    path = output_dir / f"{stem}.json"
    atomic_write_json(path, json_chart_value(data))
    return path


def require_chart_sidecars(paths: Any) -> None:
    """Fail rendering if any graph produced in this invocation lacks its sidecar."""
    missing = [
        Path(path).with_suffix(".json")
        for path in paths
        if Path(path).suffix.lower() in GRAPH_SUFFIXES
        and not Path(path).with_suffix(".json").is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Generated graphs lack numeric JSON sidecars: {formatted}")
