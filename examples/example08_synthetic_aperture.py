"""
Example 08: Linear Array — Full Matrix Capture (FMC)

Demonstrates ``ReceptionSDI.synthetic_aperture_rf()`` for Full Matrix Capture (FMC), where
each TX element fires individually while all RX elements record.

Output shape: ``(E_tx, Nt, E_rx)`` — one complete RF dataset per TX element.
FMC is the base for synthetic aperture (SA) and total focusing method (TFM)
imaging.

Run with:
    uv run examples/example08_synthetic_aperture.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from pyfield.reception import ReceptionSDI
from pyfield.transducers import LinearArrayTransducer

# ============================================================================
# CONFIGURATION
# ============================================================================
C = 1540.0  # speed of sound (m/s)
FS = 100e6  # Hz — lower fs keeps the array compact
PULSE_CYCLES = 2

print("\n--- Example 08: Full Matrix Capture (FMC) ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: SMALL LINEAR ARRAY (faster FMC for a demo)
# ============================================================================
tx = LinearArrayTransducer(
    n_elements=16,
    element_width_mm=0.3,
    element_height_mm=5.0,
    kerf_mm=0.05,
    no_sub_x=1,
    no_sub_y=4,
    frequency_Hz=5e6,
)
rx = tx  # same transducer for TX and RX

fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)
tx.excitation = excitation
# NOTE: no piezo impulse_response is set, so the elements are ideally
# broadband (echo = drive convolved with the aperture SIRs only). Fine for an
# API demo; for realistic imaging set tx/rx `.impulse_response` as in example20.

print(f"Array: {tx.n_elements} elements, fc = {fc / 1e6:.1f} MHz")

# ============================================================================
# STEP 2: DEFINE SCATTERERS
# ============================================================================
scatterer_pos = np.array(
    [
        [0.0, 0.0, 10.0],
        [-1.5, 0.0, 14.0],
        [1.5, 0.0, 18.0],
    ],
    dtype=np.float32,
)
scatterer_amp = np.ones(len(scatterer_pos), dtype=np.float32)

print(f"Scatterers: {len(scatterer_pos)} point targets")

# ============================================================================
# STEP 3: FULL MATRIX CAPTURE
# ============================================================================
sim = ReceptionSDI(tx, rx, c=C, fs=FS)
rf_fmc, coords = sim.synthetic_aperture_rf(
    scatterer_pos, scatterer_amp, decimation=1, countdown=False
)
# rf_fmc.shape = (E_tx, E_rx, Nt)

t = coords["t0"] + np.arange(rf_fmc.shape[2]) * coords["dt"]
print(f"\nFMC shape: {rf_fmc.shape}  (E_tx, E_rx, Nt)")
print(f"Time range: {t[0] * 1e6:.2f} – {t[-1] * 1e6:.2f} µs")

# ============================================================================
# STEP 4: VISUALISE
# ============================================================================
E_tx = rf_fmc.shape[0]
E_rx = rf_fmc.shape[1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# A-scan for first TX element (all RX channels)
ax = axes[0]
ax.imshow(
    rf_fmc[0].T / (np.abs(rf_fmc[0]).max() + 1e-30),  # (Erx,Nt) → (Nt,Erx) for display
    aspect="auto",
    extent=[0, E_rx - 1, t[-1] * 1e6, t[0] * 1e6],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
ax.set_xlabel("RX element")
ax.set_ylabel("Time (µs)")
ax.set_title("TX=0: RF data (all RX)")

# A-scan for middle TX element
mid = E_tx // 2
ax = axes[1]
ax.imshow(
    rf_fmc[mid].T / (np.abs(rf_fmc[mid]).max() + 1e-30),  # (Erx,Nt) → (Nt,Erx)
    aspect="auto",
    extent=[0, E_rx - 1, t[-1] * 1e6, t[0] * 1e6],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
ax.set_xlabel("RX element")
ax.set_ylabel("Time (µs)")
ax.set_title(f"TX={mid}: RF data (all RX)")

# FMC matrix: peak amplitude per (TX, RX) pair at a single time sample
peak_matrix = np.max(np.abs(rf_fmc), axis=2)  # (E_tx, E_rx), max over time
ax = axes[2]
im = ax.imshow(peak_matrix, aspect="auto", cmap="viridis")
plt.colorbar(im, ax=ax, label="Peak |RF|")
ax.set_xlabel("RX element")
ax.set_ylabel("TX element")
ax.set_title("FMC — peak amplitude matrix")

plt.suptitle(f"Full Matrix Capture: {E_tx}×{E_rx} = {E_tx * E_rx} (TX, RX) pairs")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "ex08_reception_fmc.png"), dpi=150)

plt.show()
