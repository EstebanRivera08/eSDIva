"""
Shared output configuration for all numbered eSDIva examples.

Edit the two variables below to control where figures are saved and whether
examples run interactively or write files.  The environment variable
``ESDIVA_SAVE_FIG=1`` overrides ``SAVE_FIG`` (used to batch-regenerate the
documentation figures without editing this file).
"""

import os
from pathlib import Path

# Set to True to save figures to FIG_FOLDER; False to show interactively.
SAVE_FIG = os.environ.get("ESDIVA_SAVE_FIG", "0") == "1" or False

# Destination folder for saved figures.
# Default: docs/examples/assets — keeps the documentation always up-to-date.
# Switch to  Path(__file__).parent / "assets"  to save locally instead.
FIG_FOLDER = Path(__file__).parent.parent / "docs" / "examples" / "assets"

# Resolution multiplier for saved screenshots (higher = sharper, slower).
SCALE = 3
