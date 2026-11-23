"""Learn 05 — Tissue attenuation: the beam loses α·f^y dB per cm.

Soft tissue absorbs ultrasound: the amplitude drops by α0·f^y dB per cm of path,
with α0 in dB/(cm·MHz^y), f in MHz and y between 1 and 1.3. The loss grows with
depth and with frequency, so the high frequencies of a pulse fade first.

You will see: the on-axis peak pressure of a focused pulse in water and in tissue,
and their ratio in dB next to the straight line -α0·fc^y·z expected at fc. Deep
down the pulse loses a little less than the line: its high frequencies were absorbed
first, so the surviving pulse is centred below fc. (Run with `monochromatic=True`
and the two curves coincide.)
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from esdiva.emission import Emission
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities import tone_burst

FC_HZ = 5e6
FS = 100e6
FOCUS_MM = [0, 0, 30]
ALPHA0 = 0.5  # dB/(cm·MHz^y), soft tissue.
POWER_Y = 1.0  # Power-law exponent y.

# %% Focused array and 2-cycle pulse
tx = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.15,
    element_height_mm=5.0,
    kerf_mm=0.03,
    no_sub_x=1,
    no_sub_y=5,
    frequency_Hz=FC_HZ,
)
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)
pulse = tone_burst(FC_HZ, n_cycles=2, fs=FS)

# Field points: a line along the beam axis (x = y = 0), from 1 to 80 mm deep.
axis_line = {
    "x_extent": [0, 0],
    "y_extent": [0, 0],
    "z_extent": [1, 80],
    "dx": 0,
    "dy": 0,
    "dz": 0.5,
}

# %% Same simulation, without and with attenuation
peaks = {}
for medium, alpha0 in [("water", None), ("tissue", ALPHA0)]:
    sim = Emission(tx, fs=FS, excitation=pulse, alpha0=alpha0, freq_power=POWER_Y)
    p, coords = sim(axis_line)  # Signed pressure, shape (Nt, 1, 1, Nz).
    peaks[medium] = np.abs(p[:, 0, 0, :]).max(axis=0)  # Peak over time at each depth.
z_mm = coords["z"]

# %% Plot
loss_db = 20 * np.log10(peaks["tissue"] / peaks["water"])
expected_db = -ALPHA0 * (FC_HZ / 1e6) ** POWER_Y * z_mm / 10  # z in cm.

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
ax1.plot(z_mm, peaks["water"] / 1e3, label="water")
ax1.plot(z_mm, peaks["tissue"] / 1e3, label=f"tissue, {ALPHA0} dB/(cm MHz)")
ax1.set(ylabel="Peak pressure (kPa)", title="On-axis peak pressure")
ax1.legend()
ax2.plot(z_mm, loss_db, label="simulated: tissue / water")
ax2.plot(z_mm, expected_db, "--", label="-alpha0 * fc^y * z")
ax2.set(xlabel="Depth z (mm)", ylabel="Loss (dB)")
ax2.legend()
plt.show()
