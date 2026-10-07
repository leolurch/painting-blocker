"""Shared LaTeX table rendering helpers for blocking experiment summaries."""

from __future__ import annotations

import math

METRICS = (
    ("reduction_ratio", "RR"),
    ("pair_completeness", "PC"),
    ("pair_quality", "PQ"),
)
ROW_END = " " + r"\\"


def render_metric_table(
    caption_lines: list[str],
    label: str,
    groups: list[tuple[float | int, str]],
    row_specs: list[tuple[tuple[str, str | None], str]],
    values: dict[tuple[tuple[str, str | None], float | int, str], float],
    table_kind: str,
) -> str:
    align = "l" + "r" * (len(groups) * len(METRICS))
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        *_caption_lines(caption_lines),
        f"    \\label{{{latex_label(label)}}}",
    ]
    lines.extend([
        "    \\resizebox{\\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{align}}}",
        "        \\toprule",
    ])
    if groups:
        lines.extend(_header_lines(groups))
    else:
        lines.append("        Model" + ROW_END)
    lines.append("        \\midrule")
    best = _best_values(row_specs, groups, values)
    for index, (row_key, label_text) in enumerate(row_specs):
        lines.extend(_row_lines(row_key, label_text, groups, values, best, table_kind))
        if index != len(row_specs) - 1:
            lines.append("")
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def target_label(target: float) -> str:
    if math.isclose(target, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return "$PC = 1.00$"
    formatted = f"{target:.6f}".rstrip("0").rstrip(".")
    return f"$PC \\geq {formatted}$"


def latex_label(value: str) -> str:
    """Return a command-free LaTeX identifier while preserving underscores."""
    return "".join(
        char if char.isalnum() or char in ":._-" else "-" for char in str(value)
    )


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _caption_lines(lines: list[str]) -> list[str]:
    if not lines:
        return ["    \\caption{}"]
    if len(lines) == 1:
        return [f"    \\caption{{{lines[0]}}}"]
    rendered = [f"    \\caption{{{lines[0]}"]
    rendered.extend(f"    {line}" for line in lines[1:-1])
    rendered.append(f"    {lines[-1]}}}")
    return rendered


def _header_lines(groups: list[tuple[float | int, str]]) -> list[str]:
    lines = ["        \\multirow{2}{*}{Model}"]
    for index, (_key, label) in enumerate(groups):
        suffix = ROW_END if index == len(groups) - 1 else ""
        lines.append(f"        & \\multicolumn{{3}}{{c}}{{{label}}}{suffix}")
    for index, _group in enumerate(groups):
        start = 2 + index * len(METRICS)
        lines.append(f"        \\cmidrule(lr){{{start}-{start + len(METRICS) - 1}}}")
    metric_labels = [label for _key, label in METRICS]
    for index, _group in enumerate(groups):
        suffix = ROW_END if index == len(groups) - 1 else ""
        lines.append(f"        & {' & '.join(metric_labels)}{suffix}")
    return lines


def _row_lines(
    row_key: tuple[str, str | None],
    label_text: str,
    groups: list[tuple[float | int, str]],
    values: dict[tuple[tuple[str, str | None], float | int, str], float],
    best: dict[tuple[float | int, str], float],
    table_kind: str,
) -> list[str]:
    lines = [f"        {latex_escape(label_text)}"]
    if not groups:
        lines[0] += ROW_END
        return lines
    for index, (group_key, _label) in enumerate(groups):
        cells = []
        for metric_key, _metric_label in METRICS:
            value = values.get((row_key, group_key, metric_key))
            cells.append(_format_cell(value, best.get((group_key, metric_key)), metric_key, table_kind))
        suffix = ROW_END if index == len(groups) - 1 else ""
        lines.append(f"        & {' & '.join(cells)}{suffix}")
    return lines


def _best_values(
    row_specs: list[tuple[tuple[str, str | None], str]],
    groups: list[tuple[float | int, str]],
    values: dict[tuple[tuple[str, str | None], float | int, str], float],
) -> dict[tuple[float | int, str], float]:
    row_keys = [row_key for row_key, _label in row_specs]
    best: dict[tuple[float | int, str], float] = {}
    for group_key, _group_label in groups:
        for metric_key, _metric_label in METRICS:
            present = [
                values[(row_key, group_key, metric_key)]
                for row_key in row_keys
                if _finite(values.get((row_key, group_key, metric_key)))
            ]
            if present:
                best[(group_key, metric_key)] = max(present)
    return best


def _format_cell(value: float | None, best_value: float | None, metric_key: str, table_kind: str) -> str:
    if not _finite(value):
        return "--"
    text = _format_metric(float(value), metric_key, table_kind)
    if _finite(best_value) and math.isclose(float(value), float(best_value), rel_tol=1e-12, abs_tol=1e-12):
        return f"\\textbf{{{text}}}"
    return text


def _format_metric(value: float, metric_key: str, table_kind: str) -> str:
    if metric_key == "reduction_ratio":
        return f"{value:.4f}"
    if metric_key == "pair_completeness":
        return f"{value:.3f}"
    if metric_key == "pair_quality" and table_kind == "target_pc":
        return f"{value:.4f}"
    return f"{value:.3f}"


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))
