"""Learn 11 — Grating lobes: periodic vs sparse spiral arrays.

When array elements sit on a regular grid with a pitch d larger than the
wavelength λ, the wavelets also add in phase in other directions, sin(θ) = λ/d:
these are grating lobes, false copies of the focus. Placing the same number of
elements on a spiral breaks the periodicity: the grating lobes disappear and their
energy spreads into a low, diffuse side-lobe floor.

You will see: the focal-plane (XY) pressure in dB of a 64-element grid and of a
64-element spiral of similar size, both focused at 60 mm.
"""

# %% Imports
import numpy as np

from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices
from esdiva.transducers import CustomTransducer, FlatCircularTransducer

FC_HZ = 1e6  # λ = 1.54 mm in water.
FOCUS_MM = [0, 0, 60]
N = 64  # Elements in each array.
RADIUS_MM = 25.0  # Spiral aperture radius.

# %% Element positions (x, y, z) in mm
# Periodic: an 8 x 8 grid with a 5 mm pitch (about 3 λ), so grating lobes are
# expected 60 mm · tan(asin(λ/d)) ≈ 19 mm off axis.
g = np.linspace(-17.5, 17.5, 8)
grid_xy = np.array([[x, y, 0.0] for x in g for y in g])
# Spiral: each element turns by the golden angle and steps out in radius.
k = np.arange(N)
r = RADIUS_MM * np.sqrt((k + 0.5) / N)
angle = k * np.pi * (3 - np.sqrt(5))
spiral_xy = np.column_stack([r * np.cos(angle), r * np.sin(angle), np.zeros(N)])

# %% Build both arrays from the same 3 mm disc element
disc = FlatCircularTransducer(diameter_mm=3.0, no_sub_diameter=6, frequency_Hz=FC_HZ)
focal_plane = {
    "x_extent": [-30, 30],
    "y_extent": [-30, 30],
    "z_extent": [60, 60],
    "dx": 0.3,
    "dy": 0.3,
    "dz": 0,
}

for name, positions in [("Periodic grid", grid_xy), ("Sparse spiral", spiral_xy)]:
    tx = CustomTransducer(
        elements=[disc] * N, positions_mm=positions, frequency_Hz=FC_HZ
    )
    tx.compute_delays(focus_mm=FOCUS_MM)
    tx.show(scalars="Delays")
    p, coords = Emission(tx, monochromatic=True)(focal_plane)
    plot2D_pressure_slices(p, coords=coords, db_scale=True, title=f"{name}: z = 60 mm")
