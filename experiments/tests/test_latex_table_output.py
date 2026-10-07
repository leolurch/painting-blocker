from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.core.latex_table_output import write_latex_table_with_values


class LatexTableOutputTest(unittest.TestCase):
    def test_rows_are_externalized_but_column_headings_remain_in_main_file(self):
        source = (
            "\\begin{table}\n"
            "    \\begin{tabular}{lrr}\n"
            "        \\toprule\n"
            "        Model & PC & RR \\\\\n"
            "        \\midrule\n"
            "        Alpha & 0.9 & 0.8 \\\\\n"
            "        \\midrule\n"
            "        Beta & 0.8 & 0.7 \\\\\n"
            "        \\bottomrule\n"
            "    \\end{tabular}\n"
            "\\end{table}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "figures/generated/analysis/latex_tables/blocking_fixed_k.tex"
            )
            written = write_latex_table_with_values(path, source)
            main = path.read_text(encoding="utf-8")
            values_path = path.with_name("blocking_fixed_k_values.tex")
            values = values_path.read_text(encoding="utf-8")

        self.assertEqual(written, [path, values_path])
        self.assertIn("Model & PC & RR", main)
        self.assertIn("\\toprule", main)
        self.assertIn("\\midrule", main)
        self.assertIn("\\bottomrule", main)
        self.assertIn(
            "\\input{figures/generated/analysis/latex_tables/blocking_fixed_k_values.tex}\\\\",
            main,
        )
        self.assertIn("\\usepackage{booktabs}", main)
        self.assertIn("\\usepackage{multirow}", main)
        self.assertIn("\\usepackage{graphicx}", main)
        self.assertIn("\\newcommand{\\best}", main)
        self.assertIn("\\newcommand{\\secondbest}", main)
        self.assertNotIn("Alpha &", main)
        self.assertEqual(
            values,
            "        Alpha & 0.9 & 0.8 \\\\\n"
            "        \\midrule\n"
            "        Beta & 0.8 & 0.7\n",
        )
        self.assertNotIn("Model &", values)
        self.assertEqual(values.count("\\midrule"), 1)

    def test_multiple_tabulars_receive_numbered_values_files(self):
        table = (
            "\\begin{tabular}{lr}\n"
            "\\toprule\n"
            "Name & Value \\\\\n"
            "\\midrule\n"
            "A & 1 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latex_tables/tables.tex"
            written = write_latex_table_with_values(path, table + table)
            main = path.read_text(encoding="utf-8")

        self.assertEqual([value.name for value in written[1:]], ["tables_values_1.tex", "tables_values_2.tex"])
        self.assertIn("\\input{latex_tables/tables_values_1.tex}\\\\", main)
        self.assertIn("\\input{latex_tables/tables_values_2.tex}\\\\", main)


if __name__ == "__main__":
    unittest.main()
