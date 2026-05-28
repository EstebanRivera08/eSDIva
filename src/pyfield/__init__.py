"""PyField: acoustic field simulator based on the spatial impulse response method."""

from importlib.metadata import PackageNotFoundError, version

import pyfield.beamforming as beamforming
import pyfield.plotting as plotting
import pyfield.psimulation as psimulation
import pyfield.transducers as transducers
import pyfield.utilities as utilities
from pyfield.beamforming import das, envelope_db
from pyfield.plotting import plot2D_pressure_slices
from pyfield.psimulation import Emission, PyField, Reception
from pyfield.utilities import align_to_common_time, to_dB


# backward-compat alias (old signature was x, y, z, p; new is p, x, y, z)
def plot_pressure_planes(x, y, z, pressure_field, **kwargs):
    """Plot pressure planes (deprecated, use `plot2D_pressure_slices`).

    Parameters
    ----------
    x : numpy.ndarray
        Lateral coordinates.
    y : numpy.ndarray
        Elevation coordinates.
    z : numpy.ndarray
        Axial coordinates.
    pressure_field : numpy.ndarray
        Pressure data.
    **kwargs
        Forwarded to `plot2D_pressure_slices`.

    Returns
    -------
    None
        No return value; displays the plot.
    """
    return plot2D_pressure_slices(pressure_field, x=x, y=y, z=z, **kwargs)


__all__ = [
    "Emission",
    "PyField",
    "Reception",
    "beamforming",
    "das",
    "envelope_db",
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
