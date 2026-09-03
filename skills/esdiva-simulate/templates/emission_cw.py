"""Continuous-wave beam profile of a focused linear array.

Monochromatic emission returns |H(r, w_c)|: the steady-state pressure amplitude at
the centre frequency, one value per field point. It answers "how strong is the beam
here" (beam width, depth of field, sidelobe level) and carries no time axis, so it
cannot answer time-of-flight questions -- use emission_transient.py for those.

Cells are marked with `# %%`, so this runs as a script and maps one-to-one onto
notebook cells.
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

import esdiva.transducers as transducers
from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices

# %% Probe
# no_sub_x / no_sub_y split each element into patches. Patch size sets the accuracy
# of the spatial impulse response: aim for patches of order lambda/2, which needs
# more subdivision along the tall elevation dimension than across the narrow width.
FC = 5e6  # centre frequency (Hz)
tx = transducers.LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=FC,
)

# %% Focal law
# Geometric focus at 30 mm on the beam axis. F/D = 2 opens the aperture to an
# F-number of 2 and tapers it with a Hanning window, trading focal tightness for
# lower sidelobes.
FOCUS_MM = [0.0, 0.0, 30.0]
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0, apodization_type="hanning")

# %% Field grid
# The XZ plane through the elevation centre (y_extent collapsed to a single plane).
# Keep the near edge off the aperture face: the SIR is singular at z = 0.
field_points = {
    "x_extent": [-8.0, 8.0],
    "y_extent": [0.0, 0.0],
    "z_extent": [2.0, 50.0],
    "dx": 0.2,
    "dy": 0.0,
    "dz": 0.5,
}

# %% Simulate
sim = Emission(tx, monochromatic=True, c=1540.0, verbose=True)
p, coords = sim(field_points, method="auto")  # (Nx, Ny, Nz) CW amplitude at fc
print(f"CW amplitude field: {p.shape}   peak = {p.max():.3g}")

# %% Beam profiles
# The lateral cut at the focus gives the -6 dB beam width; the axial cut gives the
# depth of field. Both are read from the same CW map.
x_mm, z_mm = coords["x"], coords["z"]
iz = int(np.argmin(np.abs(z_mm - FOCUS_MM[2])))
lateral_db = 20 * np.log10(p[:, 0, iz] / p[:, 0, iz].max())
axial_db = 20 * np.log10(p[:, 0, :].max(axis=0) / p.max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(x_mm, lateral_db)
ax1.axhline(-6, ls="--", c="k", lw=0.8)
ax1.set(xlabel="x (mm)", ylabel="dB", title=f"Lateral profile at z = {FOCUS_MM[2]} mm")
ax2.plot(z_mm, axial_db)
ax2.axhline(-6, ls="--", c="k", lw=0.8)
ax2.set(xlabel="z (mm)", ylabel="dB", title="On-axis profile")
fig.tight_layout()

# %% Field map
plot2D_pressure_slices(p, coords=coords, db_scale=True)
plt.show()
