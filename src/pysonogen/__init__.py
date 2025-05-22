

import pysonogen.transducers as transducers
import pysonogen.pyfield as pyfield
from pysonogen.functions import (plot_field_planes,
                                 plot_pressure_field,
                                 add_transducer_to_plotter,
                                 add_pressure_to_plotter)
                                 

__all__ = ["pyfield", "transducers",
           "plot_field_planes", "plot_pressure_field",
           "add_transducer_to_plotter","add_pressure_to_plotter"]

__version__ = "0.1.0"

def main() -> None:
    print("Hello from pysonogen!")
