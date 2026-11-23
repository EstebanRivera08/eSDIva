"""Learn 03 — Electronic focusing: the focal spot width is about λ·F/D.

A linear array focuses by delaying each element so all wavelets arrive at the focus
in phase. Diffraction limits the spot: its lateral width (-6 dB) is about λ·F/D, where
λ is the wavelength and F/D is the focal depth divided by the aperture width.

You will see: the CW pressure map of a focused array in dB, and the measured focal
width printed next to the λ·F/D estimate (the tapered apodization widens it a
little). Try changing F_OVER_D.
"""

# %% Imports
import numpy as np

from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities import fwhm_mm

FC_HZ = 5e6
C = 1540.0  # m/s
FOCUS_MM = [0, 0, 30]
F_OVER_D = 2.0  # Active aperture = 30 mm / 2 = 15 mm.

# %% Transducer and beamforming
tx = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=1,
    no_sub_y=5,
    frequency_Hz=FC_HZ,
)
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=F_OVER_D)

# %% Simulate the CW field on the XZ plane
xz_plane = {
    "x_extent": [-6, 6],
    "y_extent": [0, 0],
    "z_extent": [5, 60],
    "dx": 0.05,
    "dy": 0,
    "dz": 0.25,
}
sim = Emission(tx, c=C, monochromatic=True)
p, coords = sim(xz_plane)  # |P| in Pa, shape (Nx, 1, Nz).

# %% Measure the focal width
iz = np.argmin(np.abs(coords["z"] - FOCUS_MM[2]))  # Row closest to the focal depth.
wavelength_mm = C / FC_HZ * 1e3
print(f"Measured -6 dB width at the focus: {fwhm_mm(p[:, 0, iz], coords['x']):.2f} mm")
print(f"Prediction lambda*F/D:             {wavelength_mm * F_OVER_D:.2f} mm")

plot2D_pressure_slices(
    p, coords=coords, db_scale=True, title="Focused linear array (dB)"
)
