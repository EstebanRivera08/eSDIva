"""
Example 19: Dual-Probe Pulse-Echo — transform() TX/RX + Reception show()

Pitch-catch configuration used in flow imaging and vector Doppler research:
one linear array transmits a focused beam, a second identical array —
rigidly moved with `transform()` — receives the echoes from an oblique
angle.

  1. TX linear array at the canonical pose, focused on the target
  2. RX array translated and tilted 30° so both beams cross at the target
  3. `sim.show()` — 3-D preview of both apertures + scatterers
  4. Pulse-echo RF on the tilted RX aperture

Run with:
    uv run examples/example19_dualprobe_reception_show.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG

import pyfield.transducers as transducers
from pyfield.reception import ReceptionSDI

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_MM = np.array([0.0, 0.0, 20.0])  # beams cross here
RX_TILT_DEG = 30.0  # RX probe tilted about the y-axis
C = 1540.0
FS = 200e6
PULSE_CYCLES = 2

print("\n--- Example 19: Dual-Probe Pulse-Echo (transform + show) ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: TX PROBE — CANONICAL POSE, FOCUSED ON THE TARGET
# ============================================================================
tx = transducers.Domino()
tx.compute_delays(focus_mm=TARGET_MM)
tx.compute_apodization(focus_mm=TARGET_MM, FoverD=2.0)

# ============================================================================
# STEP 2: RX PROBE — TRANSLATED + TILTED TOWARD THE TARGET
# ============================================================================
# Rotate about y through the target point: the RX face keeps looking at the
# target from RX_TILT_DEG off-axis (pitch-catch geometry).
rx = transducers.Domino()
th = np.deg2rad(RX_TILT_DEG)
R = np.array(
    [
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)],
    ]
)
t_mm = TARGET_MM - R @ TARGET_MM  # rotate about the target, not the origin
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t_mm
rx.transform(T)

print(f"TX at canonical pose, RX tilted {RX_TILT_DEG}° about the target")

# ============================================================================
# STEP 3: SCATTERERS + 3-D SETUP PREVIEW
# ============================================================================
scatterer_pos = np.array(
    [
        [0.0, 0.0, 20.0],  # at the beam crossing
        [-2.0, 0.0, 17.0],
        [2.0, 0.0, 23.0],
    ],
    dtype=np.float32,
)
scatterer_amp = np.array([1.0, 0.6, 0.6], dtype=np.float32)

fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# The simulator snapshots both apertures at construction, AFTER the RX move.
sim = ReceptionSDI(tx, rx, c=C, fs=FS, excitation=excitation)

# 3-D preview: TX (blue), tilted RX (salmon), scatterers faded by amplitude.
sim.show(
    scatterer_pos,
    scatterer_amp,
    save_path=str(FIG_FOLDER / "dualprobe_setup.png") if SAVE_FIG else None,
)

# ============================================================================
# STEP 4: PULSE-ECHO RF ON THE TILTED RX
# ============================================================================
rf, coords = sim(scatterer_pos, scatterer_amp)
t = coords["t0"] + np.arange(rf.shape[1]) * coords["dt"]
print(f"RF shape: {rf.shape}  (E_rx, Nt)")

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(
    rf.T / (np.abs(rf).max() + 1e-30),
    aspect="auto",
    extent=[0, rf.shape[0] - 1, t[-1] * 1e6, t[0] * 1e6],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
plt.colorbar(im, ax=ax, label="RF (norm.)")
ax.set_xlabel("RX element")
ax.set_ylabel("Time (µs)")
ax.set_title(f"Pitch-catch RF — RX tilted {RX_TILT_DEG}°")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "dualprobe_rf.png"), dpi=150)

plt.show()

print("\nDone.")
