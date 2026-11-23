"""Learn 04 — Transient pulse: the delay law shapes the wavefront.

A short pulse (a few cycles at fc) is fired by every element, each with its own
delay. A linear delay ramp, Δt = x·sin(θ)/c, tilts the wavefront into a plane wave
steered by θ. Delays that mimic a point source behind the array (a "virtual source"
at z < 0) give a diverging, circular wavefront that insonifies a wide sector.

You will see: two animations of the pressure magnitude in dB on the XZ plane, a steered plane
wave and a diverging wave, both from the same array and the same pulse.
"""

# %% Imports
import numpy as np

from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices
from esdiva.transducers import LinearArrayExample
from esdiva.utilities import tone_burst

FS = 100e6  # Sampling frequency (Hz) of the simulated pressure signals.
STEERING_DEG = 15
VIRTUAL_SOURCE_MM = [0, 0, -10]  # Behind the array, so the wave diverges.

XZ_PLANE = {
    "x_extent": [-10, 10],
    "y_extent": [0, 0],
    "z_extent": [0.5, 20],
    "dx": 0.05,
    "dy": 0,
    "dz": 0.05,
}

# %% Array and pulse
tx = LinearArrayExample()  # Ready-made probe: 128 elements, 0.11 mm pitch, 12.5 MHz.
pulse = tone_burst(tx.fc, n_cycles=2, fs=FS)  # 2-cycle Hann-windowed sine at fc.

# %% Two delay laws, same array
delay_laws = {
    f"Plane wave steered {STEERING_DEG} deg": {"angle_steering_deg": STEERING_DEG},
    "Diverging wave": {"focus_mm": VIRTUAL_SOURCE_MM},
}
for title, law in delay_laws.items():
    tx.compute_delays(**law)
    sim = Emission(tx, fs=FS, excitation=pulse)
    p, coords = sim(XZ_PLANE)  # Signed pressure (Pa), shape (Nt, Nx, Ny, Nz).
    t = coords["t0"] + np.arange(p.shape[0]) * coords["dt"]  # Time of each frame (s).
    # dB re the peak |p|: a linear scale hides the weak edge waves and the tail.
    plot2D_pressure_slices(p, coords=coords, time_array=t, db_scale=True, title=title)
