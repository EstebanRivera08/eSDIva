"""Learn 07 — Point spread function (PSF): the image of a single point.

In pulse-echo, the echo from a point scatterer is shaped by the transmit beam AND
the receive beam, so the PSF is their product: narrower than either one, and
narrowest at the focus. We measure the lateral PSF by moving one scatterer sideways
in front of a focused bowl and recording the echo strength at each position.

You will see: the lateral PSF in dB at the focal depth and twice as deep, where
the beam has spread again, with their -6 dB widths printed.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from esdiva.reception import Reception
from esdiva.transducers import ConcaveCircularTransducer
from esdiva.utilities import fwhm_mm, tone_burst

FC_HZ = 3e6
FS = 100e6
FOCUS_MM = 40.0  # Radius of curvature of the bowl = its geometric focus.
DEPTHS_MM = [FOCUS_MM, 2 * FOCUS_MM]

# %% Focused bowl, used to transmit and to receive
tx = ConcaveCircularTransducer(
    diameter_mm=16.0, focus_mm=FOCUS_MM, no_sub_diameter=16, frequency_Hz=FC_HZ
)
# A real piezo rings for ~2 cycles: give both apertures that impulse response.
tx.impulse_response = tone_burst(FC_HZ, n_cycles=2, fs=FS)
rx = tx.copy()
sim = Reception(tx, rx, fs=FS, excitation=tone_burst(FC_HZ, n_cycles=1, fs=FS))

# %% Move one scatterer sideways at each depth
x_mm = np.arange(-8, 8.01, 0.1)
for z in DEPTHS_MM:
    points = np.column_stack([x_mm, np.zeros_like(x_mm), np.full_like(x_mm, z)])
    # per_scatterer=True: one RF trace per scatterer, shape (P, 1 element, Nt).
    rf, coords = sim.pulse_echo_rf(points, per_scatterer=True)
    envelope = np.abs(hilbert(rf[:, 0, :], axis=-1)).max(axis=-1)  # Echo strength.
    print(f"z = {z:4.0f} mm: lateral -6 dB width = {fwhm_mm(envelope, x_mm):.2f} mm")
    plt.plot(x_mm, 20 * np.log10(envelope / envelope.max()), label=f"z = {z:.0f} mm")

plt.ylim(-40, 1)
plt.xlabel("Lateral position x (mm)")
plt.ylabel("Echo amplitude (dB)")
plt.title("Pulse-echo lateral PSF of a focused bowl")
plt.legend()
plt.show()
