"""Propagating wavefront from a pulsed, steered transmit.

Transient emission convolves the excitation with the spatial impulse response and
returns pressure on a time axis: p.shape = (Nt, Nx, Ny, Nz). Use it for wavefronts,
pulse shape, time of flight, and anything that will feed a reception simulation.

Cells are marked with `# %%`, so this runs as a script and maps one-to-one onto
notebook cells.
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

import esdiva.transducers as transducers
from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices

# %% Probe and transmit
# A virtual source behind the array (z < 0) makes the aperture radiate a diverging
# spherical wave -- the transmit used by ultrafast imaging. A focus at z > 0 would
# converge instead; a steering angle would give a plane wave.
FC = 5e6
FS = 200e6  # the SIR is a train of sharp edges, so sample far above the pulse band
VIRTUAL_SOURCE_MM = [0.0, 0.0, -10.0]

tx = transducers.LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=FC,
)
tx.compute_delays(focus_mm=VIRTUAL_SOURCE_MM)
tx.compute_apodization(focus_mm=VIRTUAL_SOURCE_MM, FoverD=1.0)

# %% Excitation
# Two-cycle Hanning-windowed sine at fc: short enough for good axial resolution,
# smooth enough that its spectrum stays inside the probe band.
CYCLES = 2
t_pulse = np.arange(0.0, CYCLES / FC, 1.0 / FS)
excitation = (np.sin(2 * np.pi * FC * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# %% Field grid (XZ plane)
field_points = {
    "x_extent": [-10.0, 10.0],
    "y_extent": [0.0, 0.0],
    "z_extent": [1.0, 25.0],
    "dx": 0.25,
    "dy": 0.0,
    "dz": 0.25,
}

# %% Simulate
# Pressure follows the time derivative of the surface velocity, so the simulator
# returns rho * d(e (*) ir_tx)/dt (*) h(r, t). With no impulse_response set, the
# bare excitation acts as the normal velocity.
sim = Emission(tx, fs=FS, excitation=excitation, c=1540.0, verbose=True)
p, coords = sim(field_points, method="auto")  # (Nt, Nx, Ny, Nz)

t = coords["t0"] + np.arange(p.shape[0]) * coords["dt"]
print(f"Transient field: {p.shape}   t = [{t[0] * 1e6:.2f}, {t[-1] * 1e6:.2f}] us")
print(f"Phase timings (s): {sim.time_log}")

# %% Waveform on the beam axis
# One field point's pressure over time: the shape the medium actually sees.
ix = int(np.argmin(np.abs(coords["x"] - 0.0)))
iz = int(np.argmin(np.abs(coords["z"] - 15.0)))
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(t * 1e6, p[:, ix, 0, iz])
ax.set(xlabel="time (us)", ylabel="pressure (Pa)", title="On-axis waveform, z = 15 mm")
fig.tight_layout()

# %% Wavefront snapshot
# Peak absolute pressure over time collapses the 4-D field to the familiar beam map;
# plot2D_pressure_slices accepts the transient array directly for animation.
plot2D_pressure_slices(p, coords=coords, db_scale=True)
plt.show()
