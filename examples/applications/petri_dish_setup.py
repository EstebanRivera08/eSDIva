"""Application — Place a focused transducer's beam inside a lab setup (STL mesh).

A 0.5-inch, 1-inch-focus bowl (F-number = focal length / diameter = 2) at 5 MHz.
Its focal spot is long and thin: about 1.41·λ·F# wide (-6 dB width of the Airy
pattern of a circular aperture) and many times longer than wide. A Petri
dish, loaded from a CAD file (STL) and tilted 45°, is placed around the focus to
check that the beam lands where the sample sits.

You will see: the measured -6 dB focal width and length, then a 3-D
scene with the probe, the focal volume and the dish.
"""

# %% Imports
from pathlib import Path

import numpy as np
import pyvista as pv

from esdiva.emission import Emission
from esdiva.plotting import (
    add_pressure_vol,
    add_stl_mesh,
    add_transducer_mesh,
    create_3Dvol_mesh,
    load_mesh_from_stl,
)
from esdiva.transducers import ConcaveCircularTransducer
from esdiva.utilities import fwhm_mm

FC_HZ = 5e6
INCH_MM = 25.4

# %% Transducer and its 3-D focal field
tx = ConcaveCircularTransducer(
    diameter_mm=0.5 * INCH_MM,
    focus_mm=1.0 * INCH_MM,
    no_sub_diameter=20,
    frequency_Hz=FC_HZ,
)
volume = {
    "x_extent": [-2, 2],
    "y_extent": [-2, 2],
    "z_extent": [18, 35],
    "dx": 0.1,
    "dy": 0.1,
    "dz": 0.25,
}
p, coords = Emission(tx, monochromatic=True)(volume)
p = p / p.max()

# %% Focal size: measured vs estimate
ix, iy, iz = np.unravel_index(p.argmax(), p.shape)  # Indices of the peak.
wavelength_mm, f_number = 1540 / FC_HZ * 1e3, 2.0
print(
    f"Focal width : {fwhm_mm(p[:, iy, iz], coords['x']):.2f} mm "
    f"(estimate 1.41*lambda*F# = {1.41 * wavelength_mm * f_number:.2f} mm)"
)
print(f"Focal length: {fwhm_mm(p[ix, iy, :], coords['z']):.1f} mm")

# %% The dish: loaded from STL, moved to the focus and tilted 45 degrees about x
dish = load_mesh_from_stl(
    Path(__file__).parent / "Petri_dish.stl",
    translation=(0, -10, 25),
    rotation_axis=(1, 0, 0),
    rotation_angle=-45,
)

# %% 3-D scene
plotter = pv.Plotter()
add_transducer_mesh(tx.get_mesh(), plotter=plotter, show_scalar_bar=False)
add_pressure_vol(
    create_3Dvol_mesh(p, coords["x"], coords["y"], coords["z"], scalars="Pressure"),
    plotter=plotter,
    contour_levels=15,
)
add_stl_mesh(dish, plotter=plotter, color="lightgray", opacity=0.3, label="Petri dish")
plotter.camera.up = (0, 0, -1)  # Depth (+z) points down the screen.
plotter.show()
