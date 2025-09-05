import pysonogen.psimulation as psimulation
import pysonogen.transducers as transducers
from pysonogen.functions import (
    add_pressure_to_plotter,
    add_transducer_to_plotter,
    compute_pressure_vol_mesh,
    plot_field_planes,
    plot_pressure_field,
)

__all__ = [
    "psimulation",
    "transducers",
    "plot_field_planes",
    "plot_pressure_field",
    "add_transducer_to_plotter",
    "add_pressure_to_plotter",
    "compute_pressure_vol_mesh",
]

__version__ = "0.1.0"


def main() -> None:
    print("Hello from pysonogen!")
