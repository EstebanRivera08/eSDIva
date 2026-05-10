"""PyField: acoustic field simulator based on the spatial impulse response method."""

from importlib.metadata import PackageNotFoundError, version

import pyfield.plotting as plotting
import pyfield.psimulation as psimulation
import pyfield.transducers as transducers
import pyfield.utilities as utilities
from pyfield.plotting import plot2D_pressure_slices
from pyfield.psimulation import PyField
from pyfield.utilities import to_dB

# backward-compat alias
plot_pressure_planes = plot2D_pressure_slices

__all__ = [
    "PyField",
    "psimulation",
    "transducers",
    "utilities",
    "plotting",
    "plot2D_pressure_slices",
    "plot_pressure_planes",
    "to_dB",
]

try:
    __version__ = version("pyfield")
except PackageNotFoundError:
    __version__ = "0.0.1"


def main() -> None:
    """Print a greeting message from the pyfield package."""
    print("Hello from pyfield!")
