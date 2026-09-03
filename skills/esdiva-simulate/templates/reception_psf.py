"""Point spread function of a focused transmit, from pulse-echo RF.

`pulse_echo_rf(..., per_scatterer=True)` keeps each scatterer's echo separate, so
each slice is that point's own pulse-echo response -- the PSF at that position.
Summing them instead (per_scatterer=False) gives the channel data a real system
would record.

Cells are marked with `# %%`, so this runs as a script and maps one-to-one onto
notebook cells.
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

import esdiva.transducers as transducers
from esdiva.reception import Reception
from esdiva.beamforming import envelope_db

# %% Probe pair
# Transmit and receive are separate objects even for one physical probe: receive
# weights are per channel, and sharing one object would let the transmit focal law
# corrupt the receive traces.
FC = 5e6
FS = 200e6
FOCUS_MM = [0.0, 0.0, 30.0]

tx = transducers.LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=FC,
)
# Copy BEFORE the transmit focal law is applied: reception returns per-element RF
# without summing, so any receive delay or taper would be baked into every channel.
# Keep receive unfocused (zero delays, unit apodization) unless you are deliberately
# modelling receive weighting -- focusing belongs in the beamformer.
rx = tx.copy()
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0, apodization_type="hanning")

# %% Pulse model
# The pulse-echo signal carries three time derivatives of the drive, and a physical
# system supplies them through the transmit and receive impulse responses. Setting
# both (2-cycle burst at fc) and driving with the bare excitation is what makes the
# PSF realistic: without them the aperture's diffraction tails dominate the spectrum
# and the PSF comes out roughly 60 % too wide.
t_pulse = np.arange(0.0, 2 / FC, 1.0 / FS)
burst = (np.sin(2 * np.pi * FC * t_pulse) * np.hanning(len(t_pulse))).astype(np.float32)
excitation = burst.copy()
tx.impulse_response = burst
rx.impulse_response = burst

# %% Point targets
# Three points on the beam axis, straddling the transmit focus, so the axial change
# in the PSF is visible.
scatterers_mm = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 30.0], [0.0, 0.0, 40.0]])

# %% Simulate
sim = Reception(tx, rx, fs=FS, c=1540.0, excitation=excitation, method="spectral")
psf, coords = sim.pulse_echo_rf(scatterers_mm, per_scatterer=True)  # (N, Erx, Nt)
t = coords["t0"] + np.arange(psf.shape[-1]) * coords["dt"]
print(f"PSF: {psf.shape} (scatterers, rx channels, samples)")

# %% Check the timing convention
# coords["t0"] is the BEAMFORMING reference: the two-way pulse lag and the transmit
# bulk delay are already removed, so an echo peaks at its GEOMETRIC round-trip time.
# The check is exact at the transmit focus, where every element's contribution
# arrives in phase and the transmit path is simply |r_focus| (away from the focus
# the transmit wavefront is no longer a sphere centred on the array, so compare
# there only after fitting the wavefront -- see reception_sequence_das.py).
c = 1540.0
centers = rx.element_centers  # (E, 3) in metres
i_focus = 1
r = scatterers_mm[i_focus] * 1e-3
e_peak = int(np.argmax(np.abs(psf[i_focus]).max(axis=1)))
t_geo = (np.linalg.norm(r) + np.linalg.norm(r - centers[e_peak])) / c
t_meas = t[int(np.argmax(np.abs(psf[i_focus, e_peak])))]
print(f"at the focus: geometric {t_geo * 1e6:.3f} us   measured {t_meas * 1e6:.3f} us")
print(f"error = {(t_meas - t_geo) / coords['dt']:.2f} samples (should be well under 1)")

# %% Channel-data view
# The per-channel echo of one point traces the familiar hyperbola in (channel, time).
fig, ax = plt.subplots(figsize=(7, 4))
ax.imshow(
    envelope_db(psf[i_focus].T),  # envelope_db expects (Nt, N_lines)
    aspect="auto",
    vmin=-40,
    vmax=0,
    cmap="gray",
    extent=[0, psf.shape[1], t[-1] * 1e6, t[0] * 1e6],
)
ax.set(
    xlabel="receive channel",
    ylabel="time (us)",
    title=f"Pulse-echo channel data, point at z = {scatterers_mm[i_focus, 2]:.0f} mm",
)
fig.tight_layout()
plt.show()
