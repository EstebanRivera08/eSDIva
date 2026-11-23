"""Learn 09 — Full matrix capture (FMC): every transmit-receive pair.

Instead of focusing, fire one element at a time and record on all elements. With N
elements this stores N × N RF signals; any focus or steering can then be applied
afterwards, in software (the basis of synthetic-aperture imaging and the total
focusing method). By acoustic reciprocity, firing i and listening on j gives the
same signal as firing j and listening on i, so the matrix is symmetric.

You will see: the channel RF when element 0 fires, and the echo peak of every
(transmit, receive) pair as a symmetric N × N image.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from esdiva.reception import Reception
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities import tone_burst

FC_HZ = 5e6
FS = 100e6

# %% A small 16-element array keeps the 16 × 16 acquisition fast
tx = LinearArrayTransducer(
    n_elements=16,
    element_width_mm=0.3,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=1,
    no_sub_y=4,
    frequency_Hz=FC_HZ,
)
rx = tx.copy()
scatterers_mm = np.array([[0.0, 0.0, 10.0], [-1.5, 0.0, 14.0], [1.5, 0.0, 18.0]])

# %% Acquire: one transmit event per element
sim = Reception(tx, rx, fs=FS, excitation=tone_burst(FC_HZ, n_cycles=2, fs=FS))
# decimation=1 keeps the full sampling rate; countdown=False skips the 10 s
# "abort now" pause meant for long runs.
rf, coords = sim.synthetic_aperture_rf(scatterers_mm, decimation=1, countdown=False)
print(f"FMC data shape: {rf.shape} = (transmit element, receive element, time)")

# %% Plot
t_us = (coords["t0"] + np.arange(rf.shape[2]) * coords["dt"]) * 1e6
peaks = np.abs(rf).max(axis=2)  # Echo peak of each (transmit, receive) pair.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.imshow(
    rf[0].T, aspect="auto", cmap="RdBu", extent=[0, rf.shape[1] - 1, t_us[-1], t_us[0]]
)
ax1.set(xlabel="Receive element", ylabel="Time (µs)", title="Element 0 fires")
ax2.imshow(peaks / peaks.max(), cmap="viridis")
ax2.set(
    xlabel="Receive element",
    ylabel="Transmit element",
    title="Echo peak per pair (symmetric)",
)
plt.show()
