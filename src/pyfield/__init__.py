"""PyField: acoustic field simulator based on the spatial impulse response method."""

from importlib.metadata import PackageNotFoundError, version

import pyfield.plotting as plotting
import pyfield.psimulation as psimulation
import pyfield.transducers as transducers
import pyfield.utilities as utilities
from pyfield.plotting import plot2D_pressure_slices
from pyfield.psimulation import Emission, PyField
from pyfield.utilities import align_to_common_time, to_dB


# backward-compat alias (old signature was x, y, z, p; new is p, x, y, z)
def plot_pressure_planes(x, y, z, pressure_field, **kwargs):
    return plot2D_pressure_slices(pressure_field, x=x, y=y, z=z, **kwargs)


__all__ = [
    "Emission",
    "PyField",
    "psimulation",
    "transducers",
    "utilities",
    "plotting",
    "plot2D_pressure_slices",
    "align_to_common_time",
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
