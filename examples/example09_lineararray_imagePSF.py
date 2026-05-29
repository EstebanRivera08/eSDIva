"""
Example 09: Linear Array — B-mode PSF Image via DAS Beamforming

Field II parallel: ``fieldiiexamples/example_point_spread_functions.m``

Simulates a PSF phantom (point scatterers at multiple depths and lateral
positions), acquires multi-focus RF data with ``Reception.compute_sequence``,
and beamforms with the ``pyfield.beamforming.das`` delay-and-sum engine to
produce a log-compressed B-mode image.

  1. PSF phantom: scatterers at 3 depths × 5 lateral positions
  2. Multi-focus TX sequence: one event per scan line
  3. DAS beamforming per scan line → ``(Nt,)`` RF line
  4. Hilbert envelope + log-compression → B-mode image

Field II difference: Field II performs beamforming inside ``calc_scat``.
PyField separates simulation and beamforming — the ``pyfield.beamforming``
module provides standalone DAS and envelope functions.

Run with:
    uv run examples/example09_lineararray_imagePSF.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_FOLDER, SAVE_FIG

from pyfield.beamforming import das, envelope_db
from pyfield.reception import ReceptionSDI
from pyfield.transducers import LinearArrayTransducer

# ============================================================================
# CONFIGURATION
# ============================================================================
C = 1540.0  # speed of sound (m/s)
FS = 100e6  # sampling frequency (Hz)
PULSE_CYCLES = 2

Z_FOCUS_MM = 20.0  # TX focal depth for all scan lines
FOVERD = 2.0

# Scan-line grid
X_LINES_MM = np.arange(-5, 5.1, 1.0)  # 11 lateral scan lines

# PSF phantom scatterers
X_SCAT = np.array([-3.0, 0.0, 3.0])  # mm, lateral
Z_SCAT = np.array([12.0, 20.0, 28.0])  # mm, depths
xx, zz = np.meshgrid(X_SCAT, Z_SCAT)
SCATTERER_POS = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(
    np.float32
)
SCATTERER_AMP = np.ones(len(SCATTERER_POS), dtype=np.float32)

print("\n--- Example 09: Linear Array — B-mode PSF Image ---\n")
print("Field II parallel: example_point_spread_functions.m")
print(f"Phantom: {len(SCATTERER_POS)} point scatterers")
print(f"Scan lines: {len(X_LINES_MM)}, focus at z={Z_FOCUS_MM} mm")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LINEAR ARRAY
# ============================================================================
tx = LinearArrayTransducer(
    n_elements=32,
    element_width_mm=0.25,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=1,
    no_sub_y=4,
    frequency_Hz=5e6,
)
rx = tx

fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# ============================================================================
# STEP 2: BUILD TX EVENTS (one per scan line)
# ============================================================================
tx_events = []
for x in X_LINES_MM:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tx.compute_delays(focus_mm=[float(x), 0.0, Z_FOCUS_MM])
        tx.compute_apodization(focus_mm=[float(x), 0.0, Z_FOCUS_MM], FoverD=FOVERD)
    tx_events.append({"delays": tx.delays.copy(), "apodization": tx.apodization.copy()})

# ============================================================================
# STEP 3: ACQUIRE RF DATA
# ============================================================================
sim = ReceptionSDI(tx, rx, c=C, fs=FS, excitation=excitation)
rf_seq, coords = sim.compute_sequence(SCATTERER_POS, SCATTERER_AMP, tx_events)
# rf_seq.shape = (N_lines, Nt, E_rx)

t = coords["t0"] + np.arange(rf_seq.shape[1]) * coords["dt"]
depth_mm = t * C / 2 * 1e3  # convert round-trip time to depth

print(f"\nRF sequence shape: {rf_seq.shape}  (N_lines, Nt, E_rx)")
print(f"Depth range: {depth_mm[0]:.1f} – {depth_mm[-1]:.1f} mm")

# ============================================================================
# STEP 4: DAS BEAMFORMING + ENVELOPE
# ============================================================================
# Restore last focus for the rx element_centers lookup (geometry unchanged).
bmode_lines = []
for i, x in enumerate(X_LINES_MM):
    rf_line = das(
        rf_seq[i],  # (Nt, E_rx)
        coords,
        rx,
        focus_mm=[float(x), 0.0, Z_FOCUS_MM],
        c=C,
    )
    env = envelope_db(rf_line, vmin=10 ** (-60 / 20))  # clip at -60 dB
    bmode_lines.append(env)

bmode = np.stack(bmode_lines, axis=1)  # (Nt, N_lines)

# ============================================================================
# STEP 5: DISPLAY B-MODE IMAGE
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(
    bmode,
    aspect="auto",
    extent=[X_LINES_MM[0], X_LINES_MM[-1], depth_mm[-1], depth_mm[0]],
    cmap="gray",
    vmin=-60,
    vmax=0,
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Depth (mm)")
ax.set_title(
    f"B-mode PSF image — DAS focus at z={Z_FOCUS_MM} mm\n"
    f"{len(X_LINES_MM)} lines × {tx.n_elements} elements"
)

# Mark true scatterer positions
for s in SCATTERER_POS:
    ax.plot(s[0], s[2], "r+", markersize=8, markeredgewidth=1.5)

plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "bmode_psf.png"), dpi=150)
    print(f"\nSaved to {FIG_FOLDER / 'bmode_psf.png'}")

plt.show()
