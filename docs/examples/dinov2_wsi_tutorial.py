"""Compatibility launcher for the canonical Sphinx-Gallery example.

The maintained source now lives in ``docs/gallery`` so the HTML page, Python
download, and Jupyter notebook are generated from one file.
"""

from pathlib import Path
from runpy import run_path


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "gallery"
    / "feature_extraction"
    / "plot_dinov2_wsi.py"
)

run_path(str(EXAMPLE), run_name="__main__")
