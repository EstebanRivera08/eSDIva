"""Learn 06 — Pulse-echo RF: what each receiving element records.

The array transmits a focused pulse; three point scatterers echo it back and every
element records its own RF signal (one "channel"). An echo reaches the elements
closest to the scatterer first, so across the channels each scatterer draws a
hyperbola: its apex points at the scatterer's lateral position, its time at its depth.

You will see: a 3-D preview of the setup, then the RF of all channels as an image.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from esdiva.reception import Reception
from esdiva.transducers import LinearArrayExample
from esdiva.utilities import tone_burst

FS = 100e6
FOCUS_MM = [0, 0, 20]

# %% Transmit focused at 20 mm, receive on every element
tx = LinearArrayExample()  # Ready-made probe: 128 elements, 0.11 mm pitch, 12.5 MHz.
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)
rx = LinearArrayExample()  # Same probe, no receive delays: raw channel data.

# %% Three point scatterers (x, y, z in mm) and their reflectivities
scatterers_mm = np.array([[0.0, 0.0, 20.0], [-3.0, 0.0, 15.0], [3.0, 0.0, 25.0]])
amplitudes = np.array([1.0, 0.7, 0.7])

sim = Reception(tx, rx, fs=FS, excitation=tone_burst(tx.fc, n_cycles=3, fs=FS))
sim.show(scatterers_mm, amplitudes)  # Check the geometry before simulating.

rf, coords = sim.pulse_echo_rf(scatterers_mm, amplitudes)  # Shape (128 channels, Nt).
t_us = (coords["t0"] + np.arange(rf.shape[1]) * coords["dt"]) * 1e6

# %% Plot the channel data
plt.imshow(
    rf.T / np.abs(rf).max(),
    aspect="auto",
    cmap="RdBu",
    extent=[0, rf.shape[0] - 1, t_us[-1], t_us[0]],
)
plt.xlabel("Receive element")
plt.ylabel("Time (µs)")
plt.title("Channel RF: one hyperbola per scatterer")
plt.colorbar(label="Normalised RF")
plt.show()
