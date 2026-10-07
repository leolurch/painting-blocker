"""Write generated LaTeX tables with editable headings and separate data rows."""

from __future__ import annotations

from pathlib import Path

from .artifacts import atomic_write_text


LATEX_TABLE_DEPENDENCIES = (
    "% LaTeX dependencies (load in the document preamble):\n"
    "%   \\usepackage{booktabs}\n"
    "%   \\usepackage{multirow}\n"
    "%   \\usepackage{graphicx}\n"
    "% Required highlighting commands:\n"
    "%   \\newcommand{\\best}[1]{\\textbf{#1}}\n"
    "%   \\newcommand{\\secondbest}[1]{\\underline{#1}}\n"
)


def write_latex_table_with_values(path: Path, content: str) -> list[Path]:
    """Write a table and externalize every tabular data body.

    Column headings and all table structure remain in ``path``. Rows between the
    first ``\\midrule`` and the corresponding ``\\bottomrule`` are written to a
    sibling ``*_values.tex`` file and imported by the main table. The values
    file deliberately omits the final row's ``\\\\``; the importing line in the
    main table supplies it, keeping ``\\bottomrule`` in the same input stream as
    the row terminator. A file containing multiple tabulars receives numbered
    values files.
    """
    path = Path(path)
    if not content.startswith("% LaTeX dependencies"):
        content = LATEX_TABLE_DEPENDENCIES + content
    lines = content.splitlines(keepends=True)
    blocks = _data_blocks(lines)
    if not blocks:
        raise ValueError(f"Generated LaTeX table has no midrule/bottomrule data body: {path}")

    _remove_stale_values_files(path)
    value_paths = _value_paths(path, len(blocks))
    block_has_rows: list[bool] = []
    for (_start, _end, body), value_path in zip(blocks, value_paths):
        values, has_rows = _values_without_final_row_end(body)
        atomic_write_text(value_path, values)
        block_has_rows.append(has_rows)

    replacements = zip(blocks, value_paths, block_has_rows)
    for (start, end, body), value_path, has_rows in reversed(list(replacements)):
        indent = _body_indent(body, lines[start - 1] if start else "")
        row_end = r"\\" if has_rows else ""
        input_line = (
            f"{indent}\\input{{{_latex_input_path(value_path)}}}{row_end}\n"
        )
        lines[start:end] = [input_line]

    atomic_write_text(path, "".join(lines))
    return [path, *value_paths]


def _data_blocks(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    blocks: list[tuple[int, int, list[str]]] = []
    tabular_open = False
    data_start: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("\\begin{tabular}"):
            tabular_open = True
            data_start = None
        elif tabular_open and stripped == "\\midrule" and data_start is None:
            data_start = index + 1
        elif tabular_open and stripped == "\\bottomrule":
            if data_start is not None:
                blocks.append((data_start, index, lines[data_start:index]))
            data_start = None
        elif tabular_open and stripped.startswith("\\end{tabular}"):
            tabular_open = False
            data_start = None
    return blocks


def _values_without_final_row_end(body: list[str]) -> tuple[str, bool]:
    lines = "".join(body).strip("\n").splitlines(keepends=True)
    final_index = next(
        (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
        None,
    )
    if final_index is None:
        return "\n", False

    line = lines[final_index]
    newline = "\n" if line.endswith("\n") else ""
    content = line.rstrip("\r\n").rstrip()
    if not content.endswith(r"\\"):
        raise ValueError("Final generated table row must end with \\\\")
    lines[final_index] = content[:-2].rstrip() + newline
    return "".join(lines).strip("\n") + "\n", True


def _value_paths(path: Path, count: int) -> list[Path]:
    if count == 1:
        return [path.with_name(f"{path.stem}_values.tex")]
    return [path.with_name(f"{path.stem}_values_{index}.tex") for index in range(1, count + 1)]


def _remove_stale_values_files(path: Path) -> None:
    for candidate in path.parent.glob(f"{path.stem}_values*.tex"):
        candidate.unlink()


def _body_indent(body: list[str], fallback: str) -> str:
    for line in body:
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return fallback[: len(fallback) - len(fallback.lstrip())] + "    "


def _latex_input_path(path: Path) -> str:
    """Prefer repository-relative generated paths and portable run-local paths."""
    parts = path.parts
    if "figures" in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index("figures")
        return Path(*parts[index:]).as_posix()
    if "latex_tables" in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index("latex_tables")
        return Path(*parts[index:]).as_posix()
    return path.name
