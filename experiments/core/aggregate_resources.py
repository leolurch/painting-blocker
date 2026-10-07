"""Aggregate resource_metrics.json artifacts into flat resource tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .artifacts import atomic_write_json


def discover_resource_files(paths: Iterable[Path | str]) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value).expanduser()
        if path.is_file():
            if path.name != "resource_metrics.json" and path.parent.name != "resource_metrics":
                raise ValueError(f"Not a resource metrics file: {path}")
            files.append(path)
        elif path.is_dir():
            if path.name == "resource_metrics":
                files.extend(sorted(path.glob("*.json")))
            if (path / "resource_metrics.json").is_file():
                files.append(path / "resource_metrics.json")
            files.extend(sorted(path.glob("**/resource_metrics/*.json")))
            files.extend(sorted(path.glob("**/models/*/resource_metrics.json")))
    return sorted(set(files))


def rows_from_resource_file(resource_file: Path) -> dict[str, Any]:
    data = json.loads(resource_file.read_text(encoding="utf-8"))
    run_dir = _infer_run_dir(resource_file)
    run_meta = _load_run_json(run_dir) if run_dir else {}
    summary = dict(data.get("summary") or {})
    embedding = dict(data.get("embedding") or {})
    retrieval = dict(data.get("retrieval") or {})
    model = dict(data.get("model") or {})
    system = dict(data.get("system") or {})
    num_images = _maybe_int(embedding.get("num_images"))
    embedding_seconds = _maybe_float(embedding.get("wall_time_seconds"))
    retrieval_seconds = _maybe_float(retrieval.get("wall_time_seconds"))
    row = {
        "experiment_id": run_meta.get("experiment_id") or data.get("experiment_id"),
        "run_id": run_meta.get("run_id") or (run_dir.name if run_dir else None),
        "dataset_id": ((run_meta.get("dataset") or {}).get("dataset_id") or data.get("dataset_id")),
        "split_id": (run_meta.get("split") or {}).get("split_id"),
        "model_id": data.get("model_id"),
        "model_revision": data.get("model_revision"),
        "model_storage_key": data.get("model_storage_key"),
        "created_at": data.get("created_at"),
        "resource_path": str(resource_file),
        "num_images": num_images,
        "embedding_dimensionality": _first(
            summary.get("embedding_dimensionality"),
            embedding.get("embedding_dimensionality"),
            model.get("embedding_dim"),
        ),
        "model_size": summary.get("model_size") or model.get("model_size_label"),
        "parameter_count": _first(summary.get("parameter_count"), model.get("parameter_count")),
        "inference_engine": summary.get("inference_engine") or model.get("inference_engine"),
        "embedding_cache_reused_for_run": bool(embedding.get("cache_reused_for_run", False)),
        "embedding_wall_time_seconds": embedding_seconds,
        "embedding_time_per_100_images_seconds": _first(
            embedding.get("time_per_100_images_seconds"), summary.get("embedding_time_per_100_images_seconds")
        ),
        "embedding_images_per_second": _rate(num_images, embedding_seconds),
        "embedding_gpu_peak_used_bytes": _gpu_peak(embedding),
        "retrieval_wall_time_seconds": retrieval_seconds,
        "similarity_time_seconds": _maybe_float(retrieval.get("similarity_time_seconds")),
        "evaluation_time_seconds": _maybe_float(retrieval.get("evaluation_time_seconds")),
        "retrieval_num_queries": _maybe_int(retrieval.get("num_queries")),
        "retrieval_latency_per_query_seconds": _first(
            retrieval.get("latency_per_query_seconds"), summary.get("retrieval_latency_per_query_seconds")
        ),
        "retrieval_gpu_peak_used_bytes": _gpu_peak(retrieval),
        "gpu_memory_peak_used_bytes": _first(
            summary.get("gpu_memory_peak_used_bytes"),
            _max_present(_gpu_peak(embedding), _gpu_peak(retrieval)),
        ),
        "cpu_cores": _first(summary.get("cpu_cores"), system.get("cpu_cores")),
        "gpu_names": _gpu_names(system.get("gpu") or summary.get("gpu")),
        "memory_total_bytes": _first(summary.get("memory_total_bytes"), system.get("memory_total_bytes")),
    }
    return row


def aggregate_resource_paths(
    paths: Iterable[Path | str],
    csv_output: Path | str | None = None,
    json_output: Path | str | None = None,
    include_cached: bool = False,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for resource_file in discover_resource_files(paths):
        row = rows_from_resource_file(resource_file)
        if row["embedding_cache_reused_for_run"] and not include_cached:
            continue
        key = (
            str(row.get("experiment_id")),
            str(row.get("run_id")),
            str(row.get("model_storage_key") or row.get("model_id")),
        )
        if key not in rows_by_key or resource_file.parent.name == "resource_metrics":
            rows_by_key[key] = row
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (str(row.get("experiment_id")), str(row.get("run_id")), str(row.get("model_id"))),
    )
    if csv_output is not None:
        write_csv(Path(csv_output), rows)
    if json_output is not None:
        atomic_write_json(Path(json_output), {"schema_version": 1, "rows": rows})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "experiment_id", "run_id", "dataset_id", "split_id", "model_id", "model_revision", "model_storage_key",
        "num_images", "embedding_dimensionality", "model_size", "parameter_count", "inference_engine",
        "embedding_cache_reused_for_run", "embedding_wall_time_seconds", "embedding_time_per_100_images_seconds",
        "embedding_images_per_second", "retrieval_wall_time_seconds", "similarity_time_seconds",
        "evaluation_time_seconds", "retrieval_num_queries", "retrieval_latency_per_query_seconds",
        "embedding_gpu_peak_used_bytes", "retrieval_gpu_peak_used_bytes", "gpu_memory_peak_used_bytes",
        "cpu_cores", "gpu_names", "memory_total_bytes", "created_at", "resource_path",
    ]
    fieldnames = sorted({key for row in rows for key in row})
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered or preferred)
        writer.writeheader()
        writer.writerows(rows)


def _infer_run_dir(resource_file: Path) -> Path | None:
    if resource_file.parent.name == "resource_metrics":
        return resource_file.parents[1]
    if resource_file.name == "resource_metrics.json" and resource_file.parent.parent.name == "models":
        return resource_file.parents[2]
    for parent in resource_file.parents:
        if (parent / "run.json").is_file():
            return parent
    return None


def _load_run_json(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        return {"run_id": run_dir.name}
    return json.loads(path.read_text(encoding="utf-8"))


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _maybe_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _rate(count: int | None, seconds: float | None) -> float | None:
    return float(count / seconds) if count is not None and seconds and seconds > 0 else None


def _gpu_peak(block: dict[str, Any]) -> int | None:
    gpu = block.get("gpu_memory")
    if not isinstance(gpu, dict) or gpu.get("nvidia_smi_peak_used_bytes") is None:
        return None
    return int(gpu["nvidia_smi_peak_used_bytes"])


def _max_present(*values: int | None) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def _gpu_names(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    names = sorted({str(gpu.get("name")) for gpu in value if isinstance(gpu, dict) and gpu.get("name")})
    return "; ".join(names) if names else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.core.aggregate_resources")
    parser.add_argument("--runs", required=True, nargs="+", type=Path, help="Run dirs or resource_metrics.json files")
    parser.add_argument("--output", required=True, type=Path, help="CSV output path")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--include-cached", action="store_true", help="Include rows where embeddings were reused from cache"
    )
    args = parser.parse_args(argv)
    rows = aggregate_resource_paths(args.runs, args.output, args.json_output, include_cached=args.include_cached)
    print(json.dumps({"status": "ok", "rows": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
