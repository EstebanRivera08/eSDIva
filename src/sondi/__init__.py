"""SonDI: acoustic field simulator based on the spatial impulse response method."""

from importlib.metadata import PackageNotFoundError, version

import sondi.attenuation as attenuation
import sondi.beamforming as beamforming
import sondi.emission as emission
import sondi.plotting as plotting
import sondi.reception as reception
import sondi.transducers as transducers
import sondi.utilities as utilities
from sondi.beamforming import DAS_focused_scanline, envelope_db
from sondi.emission import Emission
from sondi.plotting import plot2D_pressure_slices
from sondi.reception import Reception, ReceptionConventional
from sondi.utilities import align_to_common_time, to_dB


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
    "Reception",
    "ReceptionConventional",
    "attenuation",
    "beamforming",
    "DAS_focused_scanline",
    "emission",
    "envelope_db",
    "reception",
    "transducers",
    "utilities",
    "plotting",
    "plot2D_pressure_slices",
    "align_to_common_time",
    "plot_pressure_planes",
    "to_dB",
]

try:
    __version__ = version("sondi")
except PackageNotFoundError:
    __version__ = "0.1.0"


def main() -> None:
    """Greet the user and run a small timed demo field, then show it in 3-D.

    Entry point for ``uv run sondi`` / the ``sondi`` console script. Prints a
    banner, focuses a 17x17 matrix array at 3 mm, computes one monochromatic
    (CW) pressure volume around the focus, reports how long the SIR took, and
    opens a PyVista window with the transducer mesh and the normalized pressure
    volume — a few-second "it works" that shows the 3-D field for new users.
    """
    import time

    import numpy as np
    import pyvista as pv

    from .plotting import add_pressure_vol, add_transducer_mesh, create_3Dvol_mesh

    banner = rf"""
   ____             ____ ___
  / ___|  ___  _ __|  _ \_ _|
  \___ \ / _ \| '_ \ | | | |
   ___) | (_) | | | | |_| | |
  |____/ \___/|_| |_|____/___|   v{__version__}

  The friendly acoustic field simulator.
  sono- (sound) + SDI (Sparse Delta Integration) - fast, exact ultrasound fields.
"""
    print(banner)

    # 17x17 matrix array at 10 MHz, geometrically focused at 3 mm depth.
    probe = transducers.MatrixArrayTransducer(
        n_elements_x=17,
        n_elements_y=17,
        element_width_mm=0.2,
        element_height_mm=0.2,
        kerf_x_mm=0.05,
        kerf_y_mm=0.05,
        no_sub_x=2,
        no_sub_y=2,
        frequency_Hz=10e6,
    )
    focus_mm = np.array([0, 0, 3])
    probe.compute_delays(focus_mm=focus_mm)
    probe.compute_apodization(focus_mm=focus_mm, FoverD=1.0)

    # Coarse grid around the focus (0.5 x 0.5 x 1.5 mm box) — kept coarse so the
    # volume renders in seconds; drop the step sizes for a finer field.
    field_points = {
        "x_extent": [focus_mm[0] - 0.5, focus_mm[0] + 0.5],
        "y_extent": [focus_mm[1] - 0.5, focus_mm[1] + 0.5],
        "z_extent": [focus_mm[2] - 1.5, focus_mm[2] + 1.5],
        "dx": 0.03,
        "dy": 0.03,
        "dz": 0.05,
    }

    print("Simulating a CW field at 10 MHz around the 3 mm focus ...")
    sim = emission.Emission(probe, monochromatic=True)
    t0 = time.perf_counter()
    p, coords = sim(field_points, method="auto")
    dt = time.perf_counter() - t0
    print(f"  done in {dt:.2f} s  (peak |p| = {float(np.abs(p).max()):.3g})")
    print("  opening the 3-D field - close the window to exit.")

    # Build the transducer + normalized-pressure meshes and render both.
    tx_mesh = probe.get_mesh()
    pressure_mesh = create_3Dvol_mesh(
        p / p.max(), coords["x"], coords["y"], coords["z"], scalars="Pressure"
    )
    plotter = pv.Plotter(window_size=(700, 700), notebook=False)
    plotter = add_pressure_vol(pressure_mesh, plotter=plotter, ambient=0.6)
    plotter = add_transducer_mesh(tx_mesh, plotter=plotter, ambient=1.0)
    plotter.add_axes(label_size=(0.1, 0.1))
    plotter.camera.up = (0, 0, -1)
    plotter.show()
