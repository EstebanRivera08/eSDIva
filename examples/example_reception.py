"""
Linear Array — Reception API: Pulse-Echo RF Simulation

Demonstrates the `Reception` class for computing received RF signals from
point scatterers using the PE SDI (Pulse-Echo Sparse Delta Integration)
kernel:

  1. Single-focus PE   — One TX focus, all RX elements receive
  2. With excitation   — Same setup + excitation pulse
  3. Scan sequence     — Multiple TX events (scan line sweep)

Run with:
    uv run examples/example_reception.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG

import pyfield.transducers as transducers
from pyfield.psimulation import Reception

# ============================================================================
# CONFIGURATION
# ============================================================================
FOCUS_MM = [0, 0, 20]
SPEED_OF_SOUND = 1540.0  # m/s
FS = 200e6  # Hz
PULSE_CYCLES = 3

print("\n--- Linear Array: Reception API ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE TX AND RX TRANSDUCERS
# ============================================================================
tx = transducers.Domino()
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)

# RX: same geometry, no focusing (receive on all elements simultaneously).
rx = transducers.Domino()

print(f"TX elements: {tx.n_elements}")
print(f"RX elements: {rx.n_elements}")

# ============================================================================
# STEP 2: DEFINE SCATTERERS
# ============================================================================
# Three point scatterers along the axial direction (positions in mm).
scatterer_positions = np.array(
    [
        [-2.0, 0.0, 15.0],  # off-axis left, shallow
        [0.0, 0.0, 20.0],  # on-axis, at focus
        [2.0, 0.0, 25.0],  # off-axis right, deep
    ],
    dtype=np.float32,
)
scattering_amplitudes = np.array([1.0, 1.0, 1.0], dtype=np.float32)

print(f"Scatterers: {scatterer_positions.shape[0]}")

# ============================================================================
# STEP 3: EXCITATION PULSE
# ============================================================================
fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)

# ============================================================================
# MODE 1 — SINGLE-FOCUS PE (no excitation, pure PE SIR derivative)
# ============================================================================
print("\nMode 1: PE SIR derivative (no excitation)")
sim = Reception(tx, rx, c=SPEED_OF_SOUND, fs=FS, excitation=None, verbose=True)
rf_raw, coords_raw = sim(scatterer_positions, scattering_amplitudes)
# rf_raw.shape = (Nt, E_rx)

print(f"  RF shape: {rf_raw.shape}")
print(f"  t0 = {coords_raw['t0']:.6e} s, dt = {coords_raw['dt']:.6e} s")

# ============================================================================
# MODE 2 — WITH EXCITATION (Hanning-windowed sine burst)
# ============================================================================
print("\nMode 2: PE with excitation")
sim.set("excitation", excitation)
rf_exc, coords_exc = sim(scatterer_positions, scattering_amplitudes)

print(f"  RF shape: {rf_exc.shape}")

# ============================================================================
# MODE 3 — SCAN SEQUENCE (3 TX events with different focal points)
# ============================================================================
print("\nMode 3: Scan sequence (3 TX events)")
# Create 3 TX events: focus at x = -3, 0, +3 mm.
import warnings

tx_events = []
for focus_x in [-3.0, 0.0, 3.0]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Compute delays for each focal point.
        tx_copy = transducers.Domino()
        tx_copy.compute_delays(focus_mm=[focus_x, 0, 20])
        tx_copy.compute_apodization(focus_mm=[focus_x, 0, 20], FoverD=2.0)
    tx_events.append(
        {"delays": tx_copy.delays.copy(), "apodization": tx_copy.apodization.copy()}
    )

rf_seq, coords_seq = sim.compute_sequence(
    scatterer_positions, scattering_amplitudes, tx_events
)
# rf_seq.shape = (N_events, Nt, E_rx)

print(f"  RF sequence shape: {rf_seq.shape}")

# ============================================================================
# VISUALISATION
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Reconstruct time vector.
t_raw = coords_raw["t0"] + np.arange(rf_raw.shape[0]) * coords_raw["dt"]
t_exc = coords_exc["t0"] + np.arange(rf_exc.shape[0]) * coords_exc["dt"]

# Plot 1: Raw PE SIR derivative — single channel (element 0).
ax = axes[0]
ax.plot(t_raw * 1e6, rf_raw[:, 0], "k", linewidth=0.5)
ax.set_xlabel("Time (us)")
ax.set_ylabel("Amplitude")
ax.set_title("Mode 1: Raw PE SIR (ch 0)")
ax.grid(True, alpha=0.3)

# Plot 2: With excitation — single channel (element 0).
ax = axes[1]
ax.plot(t_exc * 1e6, rf_exc[:, 0], "b", linewidth=0.5)
ax.set_xlabel("Time (us)")
ax.set_ylabel("Amplitude")
ax.set_title("Mode 2: With excitation (ch 0)")
ax.grid(True, alpha=0.3)

# Plot 3: Sequence — B-mode-like RF display for event 1 (focus at x=0).
ax = axes[2]
rf_event1 = rf_seq[1]  # second event (x=0 focus)
t_seq = coords_seq["t0"] + np.arange(rf_event1.shape[0]) * coords_seq["dt"]
extent = [0, rf_event1.shape[1] - 1, t_seq[-1] * 1e6, t_seq[0] * 1e6]
ax.imshow(
    rf_event1 / max(np.abs(rf_event1).max(), 1e-30),
    aspect="auto",
    cmap="RdBu",
    vmin=-1,
    vmax=1,
    extent=extent,
)
ax.set_xlabel("RX element")
ax.set_ylabel("Time (us)")
ax.set_title("Mode 3: Sequence event 2")

plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "reception_overview.png"), dpi=150)
    print(f"\nSaved to {FIG_FOLDER / 'reception_overview.png'}")

plt.show()
