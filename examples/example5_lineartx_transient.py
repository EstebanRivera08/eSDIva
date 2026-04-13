"""
Example 5: Linear Array — Transient (Pulsed) Pressure Field

Demonstrates pulsed ultrasound simulation with a linear array transducer.
Two emission modes are shown:

  * Mode 1 — Focused emission (no user-defined excitation; PyField uses
    default pulsed response).
  * Mode 2 — Steered emission with an explicit excitation signal
    (Hanning-windowed sine burst).

Steps
-----
1. Create a Domino linear-array transducer
2. Apply focused or steered delays
3. Define a pulsed excitation signal
4. Compute the transient (4-D) pressure field
5. Animate the propagating wavefront

Run with:
    uv run examples/example5_lineartx_transient.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyfield.transducers as transducers
from pyfield.psimulation import PyField
from pyfield.utilities import plot_slices_2d

# ============================================================================
# CONFIGURATION
# ============================================================================
SAVE_FIG = True  # Set True to save figures to assets/
FIG_FOLDER = Path(__file__).parent / "assets"
SCALE = 3  # Resolution multiplier when saving

# Emission mode: 1 = pulsed focused, 2 = steered with explicit excitation
EMISSION_TYPE = 1

# Steering configuration (used when EMISSION_TYPE == 2)
STEERING_ANGLE_X_DEG = -10
SPEED_OF_SOUND_MPS = 1540

# Excitation signal
PULSE_CYCLES = 2

# Simulation grid (all in mm)
PLANE_X_EXTENT_MM = [-10, 10]
PLANE_Z_EXTENT_MM = [0, 15]
GRID_SPACING_X_MM = 0.05
GRID_SPACING_Z_MM = 0.05

# Visualisation
FIGURE_SIZE = (9, 5)
CMAP_NAME = "jet"

print("\n --- Example 5: Linear Array — Transient (Pulsed) --- \n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE TRANSDUCER AND APPLY DELAYS
# ============================================================================
tx = transducers.Domino()

if EMISSION_TYPE == 1:
    FOCUS_MM = [0, 0, 8]
    tx.compute_delays(focus_mm=FOCUS_MM)
    tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=1)
else:
    # Beam steering: element delays computed from angle
    steering_rad = np.deg2rad(STEERING_ANGLE_X_DEG)
    element_indices = np.arange(tx.n_elements)
    steered_delays_s = (
        tx.pitch * element_indices * np.sin(steering_rad)
    ) / SPEED_OF_SOUND_MPS
    steered_delays_s -= steered_delays_s.min()
    tx.set_delays(steered_delays_s)

tx.plot_delays_apodization()

# ============================================================================
# STEP 2: DEFINE SIMULATION PLANE
# ============================================================================
plane_config = {
    "x_extent": PLANE_X_EXTENT_MM,
    "y_extent": [0, 0],
    "z_extent": PLANE_Z_EXTENT_MM,
    "dx": GRID_SPACING_X_MM,
    "dy": 0,
    "dz": GRID_SPACING_Z_MM,
}

# ============================================================================
# STEP 3: BUILD EXCITATION SIGNAL
# ============================================================================
simulator = PyField(tx)
center_freq_hz = simulator.fc
sampling_freq_hz = simulator.fs

pulse_duration_s = PULSE_CYCLES / center_freq_hz
time_array_s = np.arange(0, pulse_duration_s, 1 / sampling_freq_hz)
window = np.hanning(len(time_array_s))
excitation_signal = np.sin(2 * np.pi * center_freq_hz * time_array_s) * window

# ============================================================================
# STEP 4: COMPUTE TRANSIENT PRESSURE FIELD
# ============================================================================
if EMISSION_TYPE == 1:
    print("Simulating pulsed focused emission...")
    x, y, z, p_field = simulator(plane_config, monochromatic=False)
else:
    print("Simulating steered emission with excitation signal...")
    # Show the excitation signal
    plt.figure(figsize=(10, 3))
    plt.plot(excitation_signal, "k", linewidth=1.5)
    plt.title("Excitation Signal (Pulsed)")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalised Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(str(FIG_FOLDER / "transient_excitation.png"), dpi=100 * SCALE)
    plt.show()

    x, y, z, p_field = simulator(plane_config, excitation=excitation_signal)

# ============================================================================
# STEP 5: ANIMATE THE PROPAGATING WAVEFRONT
# ============================================================================
n_frames = p_field.shape[0]
time_array_s = np.linspace(0, n_frames / sampling_freq_hz, n_frames)

plot_slices_2d(
    x,
    y,
    z,
    p_field,
    time_array=time_array_s,
    db_scale=True,
    figsize=FIGURE_SIZE,
    cmap=CMAP_NAME,
    vmin=-40,
    vmax=0,
    save_path=str(FIG_FOLDER / "transient_field.gif") if SAVE_FIG else None,
)
