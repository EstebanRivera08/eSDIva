"""PyField: acoustic field simulator based on the spatial impulse response method."""

from importlib.metadata import PackageNotFoundError, version

import pyfield.psimulation as psimulation
import pyfield.transducers as transducers
import pyfield.utilities as utilities
from pyfield.psimulation import PyField
from pyfield.utilities import (
    plot_pressure_field,
    plot_pressure_planes,
    to_dB,
)

__all__ = [
    "PyField",
    "psimulation",
    "transducers",
    "utilities",
    "plot_pressure_planes",
    "plot_pressure_field",
    "to_dB",
]

try:
    __version__ = version("pyfield")
except PackageNotFoundError:
    __version__ = "0.0.1"


def main() -> None:
    """Print a greeting message from the pyfield package."""
    print("Hello from pyfield!")
