"""Plot pair-quality/pair-completeness curves for one experiment result set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def discover_curve_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file() and path.name == "curves.json":
            files.append(path)
        elif path.is_file() and path.name == "eval.json":
            curve_path = path.with_name("curves.json")
            if curve_path.is_file():
                files.append(curve_path)
        elif path.is_dir():
            files.extend(sorted(path.glob("**/models/*/curves.json")))
    return sorted(set(files))


def load_curve(curve_file: Path, task_id: str) -> tuple[str, list[dict[str, Any]]] | None:
    data = json.loads(curve_file.read_text(encoding="utf-8"))
    for task in data.get("tasks", []):
        if task.get("task_id") == task_id:
            return str(data.get("model_id") or curve_file.parent.name), list(task.get("precision_recall") or [])
    return None


def plot_pq_pc(
    curve_files: list[Path],
    task_id: str,
    output: Path,
    title: str | None = None,
    *,
    styles: dict[Path, dict[str, str]] | None = None,
    pc_limits: tuple[float, float] = (0.0, 1.0),
    pq_limits: tuple[float, float] = (0.0, 1.0),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = 0
    plt.figure(figsize=(8, 5))
    for curve_file in curve_files:
        loaded = load_curve(curve_file, task_id)
        if loaded is None:
            continue
        model_id, rows = loaded
        if not rows:
            continue
        points = sorted(
            ((float(row.get("pair_completeness", row.get("recall", 0.0))), float(row.get("pair_quality", row.get("precision", 0.0)))) for row in rows),
            key=lambda pair: pair[0],
        )
        style = (styles or {}).get(curve_file.resolve(), {})
        plot_kwargs = {
            key: style[key]
            for key in ("color", "linestyle")
            if style.get(key)
        }
        plt.plot(
            [pc for pc, _pq in points],
            [pq for _pc, pq in points],
            label=style.get("label", model_id),
            **plot_kwargs,
        )
        plotted += 1
    if plotted == 0:
        raise ValueError(f"No curves with task_id={task_id!r} found")
    plt.xlabel("Pairs completeness (PC)")
    plt.ylabel("Pair quality (PQ)")
    plt.title(title or f"PQ-PC curve: {task_id}")
    plt.xlim(*pc_limits)
    plt.ylim(*pq_limits)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.core.plot_pq_pc")
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="Run dirs, eval.json, or curves.json files")
    parser.add_argument("--task", default="modern_to_historic")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    curve_files = discover_curve_files(args.runs)
    if not curve_files:
        raise FileNotFoundError("No curves.json files found")
    plot_pq_pc(curve_files, args.task, args.output, args.title)
    print(json.dumps({"status": "ok", "output": str(args.output), "curves": len(curve_files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
