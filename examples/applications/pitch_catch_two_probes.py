"""Application — Pitch-catch: one probe transmits, another one receives.

A small piston on the left transmits; a linear array on the right, turned so its
elements run along y, receives. Both are tilted toward a common target 7 mm deep.
Unlike a single probe, an echo's arrival time is the transmit path PLUS the
receive path, which are no longer equal: one scatterer still draws a curve across
the receive channels, but it is set by the receiving probe's position.

You will see: a 3-D preview of both probes and the scatterers, then the received
channel RF.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from esdiva.reception import Reception
from esdiva.transducers import FlatCircularTransducer, LinearArrayExample
from esdiva.utilities import tone_burst

FS = 100e6
TARGET_MM = [0.0, 0.0, 7.0]  # Where both beam axes cross.

# %% Transmit: a 4 mm piston at x = -8 mm, aimed at the target
tx = FlatCircularTransducer(diameter_mm=4.0, no_sub_diameter=25, frequency_Hz=12.5e6)
tx.aim(position_mm=[-8, 0, 0], target_mm=TARGET_MM)

# %% Receive: a linear array at x = +8 mm, elements along y, aimed at the target
rx = LinearArrayExample()
rx.aim(position_mm=[8, 0, 0], target_mm=TARGET_MM, roll_deg=90)

# %% Thirty random scatterers in the XZ plane around the target
rng = np.random.default_rng(19)
scatterers_mm = np.array(TARGET_MM) + rng.uniform(-2, 2, (30, 3)) * [1, 0, 1]
amplitudes = rng.uniform(1, 2, 30)

sim = Reception(tx, rx, fs=FS, excitation=tone_burst(tx.fc, n_cycles=2, fs=FS))
sim.show(scatterers_mm, amplitudes)

# %% Received RF, one column per receive element
rf, coords = sim.pulse_echo_rf(scatterers_mm, amplitudes)
t_us = (coords["t0"] + np.arange(rf.shape[1]) * coords["dt"]) * 1e6
plt.imshow(
    rf.T / np.abs(rf).max(),
    aspect="auto",
    cmap="RdBu",
    extent=[0, rf.shape[0] - 1, t_us[-1], t_us[0]],
)
plt.xlabel("Receive element")
plt.ylabel("Time (µs)")
plt.title("Pitch-catch channel RF")
plt.colorbar(label="Normalised RF")
plt.show()
