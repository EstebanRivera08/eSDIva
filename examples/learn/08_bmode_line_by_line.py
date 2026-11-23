"""Learn 08 — A B-mode image, one focused line at a time.

A conventional scanner builds its image line by line: for each lateral position it
focuses on transmit and on receive at a fixed depth, sums the channels, and keeps
the envelope. With a single fixed focus the image is sharpest at the focal depth
and the point targets blur above and below it (limited depth of field).

You will see: the B-mode image (60 dB range) of a column of point targets, with
the focal depth marked.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from esdiva.reception import Reception
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities import tone_burst

FC_HZ = 3e6
FS = 100e6
C = 1540.0
FOCUS_DEPTH_MM = 60.0

# %% Array; the 2-cycle pulse models the piezo response on transmit and receive
tx = LinearArrayTransducer(
    n_elements=128,
    element_width_mm=0.41,
    element_height_mm=5.0,
    kerf_mm=0.1,
    no_sub_x=1,
    no_sub_y=4,
    frequency_Hz=FC_HZ,
)
tx.impulse_response = tone_burst(FC_HZ, n_cycles=2, fs=FS)
rx = tx.copy()
sim = Reception(tx, rx, c=C, fs=FS, excitation=tone_burst(FC_HZ, n_cycles=2, fs=FS))

# Point targets on the axis, every 10 mm from 20 to 100 mm deep.
z_targets = np.arange(20, 101, 10.0)
targets_mm = np.column_stack([0 * z_targets, 0 * z_targets, z_targets])

# %% Scan: one focused line every 0.5 mm
x_lines_mm = np.arange(-8, 8.01, 0.5)
depth_mm = np.arange(10, 110, 0.05)  # Common depth axis for all lines.
image = []
for x in x_lines_mm:
    # Focuses TX and RX at (x, 0, 60 mm), sums the channels, returns the envelope.
    envelope, coords = sim.scan_focusline(
        [x, 0, FOCUS_DEPTH_MM], targets_mm, FoverD=2.0, apodization_type="hanning"
    )
    t = coords["t0"] + np.arange(envelope.size) * coords["dt"]
    image.append(np.interp(depth_mm, t * C / 2 * 1e3, envelope))  # Time -> depth.
image = np.array(image).T  # Shape (depth, lines).

# %% Display in dB
bmode_db = 20 * np.log10(np.maximum(image / image.max(), 1e-3))  # 60 dB floor.
plt.figure(figsize=(4, 8))
plt.imshow(
    bmode_db,
    cmap="gray",
    aspect="equal",
    extent=[x_lines_mm[0], x_lines_mm[-1], depth_mm[-1], depth_mm[0]],
)
plt.axhline(FOCUS_DEPTH_MM, color="r", ls="--", lw=0.8, label="focal depth")
plt.xlabel("x (mm)")
plt.ylabel("Depth (mm)")
plt.colorbar(label="dB")
plt.legend(loc="lower right")
plt.show()
