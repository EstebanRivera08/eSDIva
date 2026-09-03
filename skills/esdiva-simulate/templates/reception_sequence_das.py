"""Diverging-wave sequence -> RF -> image, with both the built-in and a hand-written
beamformer.

This is the skeleton of an imaging study: build the transmit events, acquire RF for
all of them, then reconstruct. The last cell reconstructs the same data with a dozen
lines of NumPy to show exactly what the RF output contains -- that is the entry
point for a custom beamformer (Fourier migration, model-based, learned, ...).

Sized to run in a couple of minutes. A real study needs a denser phantom (5-10
scatterers per resolution cell), more events, and out_path checkpointing.

Cells are marked with `# %%`, so this runs as a script and maps one-to-one onto
notebook cells.
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

import esdiva.transducers as transducers
from esdiva.reception import Reception
from esdiva.beamforming import das_volume
from esdiva.utilities import make_phantom, to_dB

C = 1540.0
FC = 5e6
FS = 100e6

# %% Probe pair and pulse model
tx = transducers.LinearArrayTransducer(
    n_elements=32,
    element_width_mm=0.30,
    element_height_mm=5.0,
    kerf_mm=0.03,
    no_sub_x=1,
    no_sub_y=3,
    frequency_Hz=FC,
)
rx = tx.copy()  # never share one object: receive weights are per channel

t_pulse = np.arange(0.0, 2 / FC, 1.0 / FS)
burst = (np.sin(2 * np.pi * FC * t_pulse) * np.hanning(len(t_pulse))).astype(np.float32)
excitation = burst.copy()
tx.impulse_response = burst
rx.impulse_response = burst

# %% Transmit events
# A diverging wave is a spherical wavefront from a virtual source BEHIND the array
# (z < 0). Element e simply fires at its travel time from that source, referenced to
# the earliest element. Spread the sources so every corner of the image box lies
# inside every event's cone -- derive that per probe, never copy a layout.
VIRTUAL_SOURCES_MM = np.array([[-6.0, 0.0, -12.0], [6.0, 0.0, -12.0]])
centers = tx.element_centers  # (E, 3) in metres

events = []
for vs_mm in VIRTUAL_SOURCES_MM:
    d = np.linalg.norm(centers - vs_mm * 1e-3, axis=1) / C
    events.append(
        {
            "delays": (d - d.min()).astype(np.float32),
            "apodization": np.ones(len(centers), np.float32),
            # Extra key: das_volume reads the wavefront geometry from it.
            # sequence_rf ignores everything but delays/apodization.
            "virtual_source_mm": np.asarray(vs_mm, float),
        }
    )

# %% Phantom
# Speckle comes from many sub-wavelength scatterers at RANDOM positions; a regular
# lattice would return coherent lattice echoes instead. Amplitudes are N(0,1) scaled
# by the local echogenicity, so a zero region is anechoic.
BOX_MM = {"x_extent": [-6.0, 6.0], "y_extent": [-0.5, 0.5], "z_extent": [10.0, 25.0]}
pos_mm, amp = make_phantom(BOX_MM, n_scatterers=600, seed=0)

# Two bright wires, to have a resolvable target next to the speckle.
wires_mm = np.array([[0.0, 0.0, 15.0], [2.0, 0.0, 20.0]])
pos_mm = np.vstack([pos_mm, wires_mm])
amp = np.concatenate([amp, np.full(len(wires_mm), 3.0)])

# %% Acquire
# Add out_path="rf_store" for a long run: each event is checkpointed on completion,
# a re-run resumes, and a changed configuration refuses instead of mixing data.
sim = Reception(tx, rx, fs=FS, c=C, excitation=excitation, method="spectral")
rf, coords = sim.sequence_rf(pos_mm, amp, events)  # (N_events, Erx, Nt)
t0_ev = np.asarray(coords["t0_per_event"], float)
print(f"RF: {rf.shape} (events, rx channels, samples)   dt = {coords['dt']:.3e} s")

# %% Reconstruct with the built-in beamformer
# Each event is beamformed with ITS OWN t0 (the time grid depends on that event's
# delays), then compounded as complex IQ: summing envelopes would throw away the
# phase and blur the result. t_offset_s stays 0 -- t0 is already the beamforming
# reference.
# One elevation plane. Unlike the emission field grid, das_volume builds its axes
# with np.arange and needs a nonzero step, so collapse an axis with a step larger
# than its extent rather than with dy = 0.
GRID_MM = {
    "x_extent": [-6.0, 6.0],
    "y_extent": [0.0, 0.5],
    "z_extent": [10.0, 25.0],
    "dx": 0.15,
    "dy": 1.0,
    "dz": 0.15,
}
iq = None
for e, event in enumerate(events):
    vol, axes = das_volume(
        rf[e : e + 1],
        {"dt": coords["dt"], "t0_per_event": t0_ev[e : e + 1]},
        [event],
        rx,
        GRID_MM,
        c=C,
        fnum=1.0,
        rx_apodization="rect",  # element directivity already tapers the aperture
    )
    iq_e = hilbert(vol, axis=2)  # analytic signal along the axial axis
    iq = iq_e if iq is None else iq + iq_e

bmode_db = to_dB(np.abs(iq[:, 0, :]))

# %% The same image from scratch -- the custom-beamformer entry point
# Everything a reconstruction needs: rf[event, channel, sample], the per-event time
# origin, dt, and the element positions in metres. The transmit wavefront is fitted
# from the event's own delays, so no delay-reference convention is assumed.
x_mm, z_mm = axes["x_mm"], axes["z_mm"]
X, Z = np.meshgrid(x_mm, z_mm, indexing="ij")
voxels_m = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1) * 1e-3
n_samples = np.arange(rf.shape[-1])

iq_manual = np.zeros(len(voxels_m), complex)
for e, event in enumerate(events):
    tau = event["delays"] - event["delays"].max()  # firing instants, s
    r_vs = event["virtual_source_mm"] * 1e-3
    # Diverging wave: tau_e = t_ref - |r_e - r_vs|/c, so t_ref is the mean residual.
    t_ref = np.mean(tau - np.linalg.norm(centers - r_vs, axis=1) / C)
    t_tx = t_ref + np.linalg.norm(voxels_m - r_vs, axis=1) / C

    line = np.zeros(len(voxels_m))
    analytic = hilbert(rf[e], axis=-1)  # interpolate the analytic signal
    for ch, ce in enumerate(centers):
        t_rx = np.linalg.norm(voxels_m - ce, axis=1) / C
        idx = (t_tx + t_rx - t0_ev[e]) / coords["dt"]  # NO pulse-lag term
        line = line + np.interp(idx, n_samples, analytic[ch].real)
    iq_manual = iq_manual + hilbert(line.reshape(X.shape), axis=1).ravel()

bmode_manual_db = to_dB(np.abs(iq_manual.reshape(X.shape)))

# %% Compare
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
for ax, img, title in (
    (ax1, bmode_db, "das_volume"),
    (ax2, bmode_manual_db, "hand-written DAS"),
):
    ax.imshow(img.T, extent=extent, vmin=-30, vmax=0, cmap="gray", aspect="equal")
    ax.set(xlabel="x (mm)", title=title)
ax1.set_ylabel("z (mm)")
fig.tight_layout()
plt.show()
