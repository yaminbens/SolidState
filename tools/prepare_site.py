"""Prepare the notebooks for the site build (run in CI only, it edits the notebooks in place).

The notebooks keep a '<div dir="rtl">' wrapper in every markdown cell so that they are right-to-left
in Colab and local Jupyter. MyST turns the content of a raw HTML block into plain HTML and loses the
math, so for the site the wrapper is removed (site.css supplies the RTL direction).
"""
import pathlib, re
import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPEN = re.compile(r'\A<div dir="rtl"[^>]*>\s*')
CLOSE = re.compile(r'\s*</div>\s*\Z')

for path in sorted(ROOT.glob("week*.ipynb")):
    nb = nbformat.read(path, 4)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = CLOSE.sub("", OPEN.sub("", cell.source))
    nbformat.write(nb, path)
    print("prepared", path.name)
