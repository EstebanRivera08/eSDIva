"""
Example 3: Pulsed (Transient) Pressure Field Simulation

This example demonstrates transient ultrasound pressure field simulation using a steered
transducer with a pulsed excitation signal. It shows:
  1. Transducer steering via delayed element excitation
  2. Pulsed excitation signal definition (multi-cycle pulse)
  3. Transient (4D spatio-temporal) pressure field computation
  4. Frame-by-frame visualization of the pressure propagation

"""

import matplotlib.pyplot as plt
import numpy as np

import pyfield
import pyfield.transducers as transducers
from pyfield.psimulation import PyField
from pyfield.utilities import to_dB

print("\n --- Example 3: Pulsed (Transient) Pressure Field --- \n")

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Transducer Selection

SIMU_TYPE = 1  # Options: 2 : "monochromatic", 1: "transient"
Emission_type = 1  # Options: 1: "pulsed focused", 2: "steered with defined excitation"
TRANSDUCER_TYPE = "Domino"  # Options: "Domino", "Zeus_Matrix"
# MyTransducer = transducers.Zeus_Matrix()  # Alternative option

# Steering Configuration
STEERING_ANGLE_X_DEG = -10  # Steering angle in degrees (X-axis)
SPEED_OF_SOUND_MPS = 1540  # Speed of sound in medium (m/s)

# Excitation Signal Definition
PULSE_CYCLES = 2  # Number of RF cycles in the pulse

# Simulation Plane Definition (all in mm)
PLANE_X_EXTENT_MM = [-10, 10]  # Lateral extent
PLANE_Z_EXTENT_MM = [0, 15]  # Depth extent
GRID_SPACING_X_MM = 0.05  # Lateral grid resolution
GRID_SPACING_Z_MM = 0.05  # Depth grid resolution

# Visualization Parameters
FIGURE_SIZE = (9, 5)
CMAP_NAME = "jet"


# ============================================================================
# STEP 1: INITIALIZE TRANSDUCER AND APPLY STEERING DELAYS
# ============================================================================

# Create transducer instance
MyTransducer = transducers.Domino()

if Emission_type == 1:
    focus = [0, 0, 8]  # Focus at 8 mm depth along Z-axis
    MyTransducer.compute_delays(focus_mm=focus)
    MyTransducer.compute_apodization(focus_mm=focus, FoverD=1)
else:
    # Calculate element-wise delays for beam steering
    # Theory: Elements are excited with varying delays to steer the beam
    # delay = (element_position * sin(steering_angle)) / speed_of_sound
    steering_angle_x_rad = np.deg2rad(STEERING_ANGLE_X_DEG)
    element_indices = np.arange(MyTransducer.n_elements)
    pitch_mm = MyTransducer.pitch

    # Compute steering delays for each element
    steered_delays_s = (
        pitch_mm * element_indices * np.sin(steering_angle_x_rad)
    ) / SPEED_OF_SOUND_MPS
    steered_delays_s = steered_delays_s - np.min(
        steered_delays_s
    )  # Normalize to start at
    MyTransducer.set_delays(steered_delays_s)

MyTransducer.plot_delays_apodization()


# ============================================================================
# STEP 2: DEFINE SIMULATION PLANE AND SPATIAL GRID
# ============================================================================

# Define the rectangular plane where the pressure field will be computed
# The plane is perpendicular to the transducer and extends in X-Z directions
# Y is held constant (2D slice of the 3D field)
plane_config = {
    "x_extent": PLANE_X_EXTENT_MM,
    "y_extent": [0, 0],  # 2D plane: fixed at Y=0
    "z_extent": PLANE_Z_EXTENT_MM,
    "dx": GRID_SPACING_X_MM,
    "dy": 0,  # No variation in Y
    "dz": GRID_SPACING_Z_MM,
}

# ============================================================================
# STEP 3: CREATE PYFIELD SIMULATOR AND DEFINE EXCITATION SIGNAL
# ============================================================================

# Wrap the transducer in PyField simulator
simulator = PyField(MyTransducer)

# Extract transducer center frequency and sampling frequency
center_frequency_hz = simulator.fc
sampling_frequency_hz = simulator.fs
time_step_s = 1 / sampling_frequency_hz

# Create time axis for the excitation pulse
# The pulse duration is determined by the number of cycles at center frequency
pulse_duration_s = PULSE_CYCLES / center_frequency_hz
time_array_s = np.arange(0, pulse_duration_s, time_step_s)

window = np.hanning(len(time_array_s))  # Hanning window to shape the pulse envelope
# Define the excitation signal: sine wave modulated by pulse envelope
excitation_signal = np.sin(2 * np.pi * center_frequency_hz * time_array_s) * window

# ============================================================================
# STEP 5: COMPUTE PRESSURE FIELD - TWO MODES AVAILABLE
# ============================================================================

# Mode 1: MONOCHROMATIC (CW) - Continuous wave at center frequency
if SIMU_TYPE == 2:
    # This computes the steady-state field without time variation
    x, y, z, p_field_mono = simulator(plane_config)
# Shape: (Nx, Ny, Nz) - 3D spatial field

# Mode 2: TRANSIENT (PULSED) - Time-varying field with pulse excitation
# This computes the full spatio-temporal field as the pulse propagates
else:
    if Emission_type == 1:
        print("Simulating pulsed focused emission...")
        x, y, z, p_field_transient = simulator(plane_config, monochromatic=False)
    else:
        print("Simulating steered emission with excitation signal...")
        plt.figure(figsize=(10, 3))
        plt.plot(excitation_signal, "k", linewidth=1.5)
        plt.title("Excitation Signal (Pulsed)")
        plt.xlabel("Sample Index")
        plt.ylabel("Normalized Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        x, y, z, p_field_transient = simulator(
            plane_config, excitation=excitation_signal
        )
# Shape: (Nx, Ny, Nz, Nt) - 4D spatio-temporal field

# ============================================================================
# STEP 6: VISUALIZE PRESSURE FIELD
# ============================================================================

from pyfield.utilities import plot_slices_2d

if SIMU_TYPE == 2:
    # Monochromatic: single static figure
    plot_slices_2d(x, y, z, p_field_mono, db_scale=True, figsize=FIGURE_SIZE, vmin=-30)

else:
    # Transient: FuncAnimation — much faster than a manual plt.pause loop.
    # The field is a single XZ plane (Ny=1) so a single-panel animation is shown.
    n_frames = p_field_transient.shape[0]
    time_array_s = np.linspace(0, n_frames / sampling_frequency_hz, n_frames)
    plot_slices_2d(
        x, y, z, p_field_transient,
        time_array=time_array_s,
        db_scale=True,
        figsize=FIGURE_SIZE,
        cmap=CMAP_NAME,
        vmin=-40, vmax=0,
    )
