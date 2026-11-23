"""Learn 10 — Speckle: imaging a tissue-mimicking phantom.

Tissue is modelled as many random point scatterers, far more than one per
resolution cell. Their echoes interfere randomly and form speckle, the grainy
texture of every ultrasound image. An echogenicity map scales the scatterer
strengths: 0 inside an anechoic cyst, > 1 inside a bright lesion. How dark the cyst
stays is set by the PSF side lobes, which leak echoes from the surrounding speckle
into it, so cyst contrast is a direct measure of PSF quality.

You will see: a 3-D preview of the scatterer cloud, then the map next to the
B-mode image (33 focused lines, as in Learn 08).
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from esdiva.reception import Reception
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities import make_phantom, tone_burst

FC_HZ = 5e6
FS = 100e6
C = 1540.0
FOCUS_DEPTH_MM = 25.0

# %% Echogenicity map on an x-z grid: background 1, cyst 0, lesion 4
box = {"x_extent": [-12, 12], "y_extent": [-1, 1], "z_extent": [12, 38]}  # mm
x, z = np.meshgrid(np.linspace(-12, 12, 256), np.linspace(12, 38, 256), indexing="ij")
echo_map = np.ones_like(x)
echo_map[(x + 4) ** 2 + (z - 25) ** 2 < 3**2] = 0.0  # Cyst: radius 3 mm at x = -4.
echo_map[(x - 4) ** 2 + (z - 25) ** 2 < 2**2] = 4.0  # Lesion: radius 2 mm at x = +4.

# %% Random scatterers: uniform positions, Gaussian amplitudes scaled by the map
# 25 000 scatterers give ~5 per resolution cell, enough for fully developed speckle.
positions_mm, amplitudes = make_phantom(box, 25000, echogenicity_map=echo_map, seed=1)

# %% Probe (the 2-cycle pulse models the piezo response on transmit and receive)
tx = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=1,
    no_sub_y=4,
    frequency_Hz=FC_HZ,
)
tx.impulse_response = tone_burst(FC_HZ, n_cycles=2, fs=FS)
sim = Reception(tx, tx, c=C, fs=FS, excitation=tone_burst(FC_HZ, n_cycles=2, fs=FS))
sim.show(positions_mm, amplitudes)  # The cyst appears as a hole in the cloud.

# %% Scan 33 focused lines and convert time to depth
x_lines_mm = np.linspace(-8, 8, 33)
depth_mm = np.arange(12, 38, 0.02)
image = []
for x_line in x_lines_mm:
    envelope, coords = sim.scan_focusline(
        [x_line, 0, FOCUS_DEPTH_MM],
        positions_mm,
        amplitudes,
        FoverD=2.0,
        apodization_type="hanning",
    )
    t = coords["t0"] + np.arange(envelope.size) * coords["dt"]
    image.append(np.interp(depth_mm, t * C / 2 * 1e3, envelope))
image = np.array(image).T

# %% Display the map and the B-mode image side by side
bmode_db = 20 * np.log10(np.maximum(image / image.max(), 10 ** (-50 / 20)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
ax1.imshow(echo_map.T, cmap="gray", extent=[-12, 12, 38, 12])
ax1.set(title="Echogenicity map", xlabel="x (mm)", ylabel="Depth (mm)")
ax2.imshow(bmode_db, cmap="gray", extent=[-8, 8, depth_mm[-1], depth_mm[0]])
ax2.set(title="B-mode (50 dB range)", xlabel="x (mm)")
plt.show()
