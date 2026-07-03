"""
Example 07: Linear Array — Focused TX, All-RX Reception

Demonstrates the `Reception` class for pulse-echo RF simulation.  TX focuses
at a single point while all RX elements receive simultaneously.

  1. Focused TX + flat RX (same transducer, different delay laws)
  2. 3-D setup preview with ``sim.show()`` (apertures + scatterer cloud)
  3. RF data shape: ``(E_rx, Nt)`` — one time trace per RX channel
  4. Visualisation: RF waterfall image + single-channel envelope

Run with:
    uv run examples/example07_lineararray_TXfocus_RXall.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from config import FIG_FOLDER, SAVE_FIG

import pyfield.transducers as transducers
from pyfield.reception import ReceptionSDI
from pyfield.utilities import to_dB

# ============================================================================
# CONFIGURATION
# ============================================================================
FOCUS_MM = [0, 0, 20]  # TX focal point in mm
C = 1540.0  # speed of sound (m/s)
FS = 200e6  # sampling frequency (Hz)
PULSE_CYCLES = 3

print("\n--- Example 07: Linear Array — Focused TX, All-RX ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: TRANSDUCER SETUP
# ============================================================================
# TX: focused at FOCUS_MM
tx = transducers.Domino()
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)

# RX: same geometry, no electronic focusing (receive on all elements)
rx = transducers.Domino()

print(f"TX/RX elements: {tx.n_elements}")
print(f"TX focus: {FOCUS_MM} mm")

# ============================================================================
# STEP 2: EXCITATION PULSE
# ============================================================================
fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# ============================================================================
# STEP 3: DEFINE SCATTERERS
# ============================================================================
# Three point scatterers: on-axis at focus, and two off-axis
scatterer_pos = np.array(
    [
        [0.0, 0.0, 20.0],  # on-axis, at focus
        [-3.0, 0.0, 15.0],  # off-axis left, shallow
        [3.0, 0.0, 25.0],  # off-axis right, deep
    ],
    dtype=np.float32,
)
scatterer_amp = np.array([1.0, 0.7, 0.7], dtype=np.float32)

# ============================================================================
# STEP 4: SIMULATE — WITH EXCITATION
# ============================================================================
sim = ReceptionSDI(tx, rx, c=C, fs=FS, excitation=excitation)

# 3-D sanity check of the setup: apertures + scatterers, before simulating.
sim.show(
    scatterer_pos,
    scatterer_amp,
    save_path=str(FIG_FOLDER / "reception_setup.png") if SAVE_FIG else None,
)

rf, coords = sim(scatterer_pos, scatterer_amp)
# rf.shape = (E_rx, Nt)

t = coords["t0"] + np.arange(rf.shape[1]) * coords["dt"]
print(f"\nRF shape: {rf.shape}  (E_rx, Nt)")
print(f"Time range: {t[0] * 1e6:.2f} – {t[-1] * 1e6:.2f} µs")

# ============================================================================
# STEP 5: VISUALISE
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RF waterfall image (all channels)
ax = axes[0]
ax.imshow(
    rf.T / (np.abs(rf).max() + 1e-30),  # (E_rx, Nt) → (Nt, E_rx) for time-on-y
    aspect="auto",
    extent=[0, rf.shape[0] - 1, t[-1] * 1e6, t[0] * 1e6],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
ax.set_xlabel("RX element")
ax.set_ylabel("Time (µs)")
ax.set_title("RF data (all channels)")

# Single channel (centre element) RF trace
ax = axes[1]
ch = rf.shape[0] // 2
ax.plot(t * 1e6, rf[ch, :], "b", linewidth=0.7)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Single channel (element {ch})")
ax.grid(True, alpha=0.3)

# Log-compressed envelope of centre channel
ax = axes[2]
env = np.abs(hilbert(rf[ch, :].astype(np.float64)))
env_db = to_dB(env)
ax.plot(t * 1e6, env_db, "k", linewidth=0.8)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Amplitude (dB)")
ax.set_title(f"Envelope — element {ch}")
ax.set_ylim(-60, 5)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Reception — focused TX at z={FOCUS_MM[2]} mm, {len(scatterer_pos)} scatterers"
)
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "reception_txfocus.png"), dpi=150)

plt.show()
