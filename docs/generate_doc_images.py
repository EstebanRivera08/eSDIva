"""
Generate some images used in the SonDI documentation besides the example ones.

Fill this file with the codes for new figures for the documentation.

Output directories
------------------
ASSETS     = docs/assets/          — focal-law plots, subdivision, ConvexCircular

Run from the repository root:

    uv run docs/generate_doc_images.py

"""

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

# ---------------------------------------------------------------------------
# Output directories (relative to repo root where the script is run)
# ---------------------------------------------------------------------------
ASSETS = Path("docs/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

WIN = (900, 600)  # default window size for PyVista screenshots


print(f"\nDone.\n  docs/assets/           → {ASSETS.resolve()}\n")
