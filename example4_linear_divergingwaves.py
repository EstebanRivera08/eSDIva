"""
Example 5: Pulsed (Transient) Pressure Field Simulation

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
%load_ext autoreload
%autoreload 2 

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Transducer Selection

SIMU_TYPE = "transient"  # Options: "monochromatic", "transient"
TRANSDUCER_TYPE = "Domino"  # Options: "Domino", "Zeus_Matrix"
# MyTransducer = transducers.Zeus_Matrix()  # Alternative option

# Steering Configuration
VIRTUAL_FOCUS_MM = [0, 0 ,-1]
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
MyTransducer.compute_delays(focus_mm = VIRTUAL_FOCUS_MM)
MyTransducer.compute_apodization(focus_mm = VIRTUAL_FOCUS_MM, FoverD = 1)
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

# Define the excitation signal: sine wave modulated by pulse envelope
window = np.hanning(len(time_array_s))  # Hanning window for smooth pulse envelope
excitation_signal = np.sin(2 * np.pi * center_frequency_hz * time_array_s) * window

# ============================================================================
# STEP 5: COMPUTE PRESSURE FIELD - TWO MODES AVAILABLE
# ============================================================================

# Mode 1: MONOCHROMATIC (CW) - Continuous wave at center frequency
if SIMU_TYPE == "monochromatic":
    # This computes the steady-state field without time variation
    x, y, z, p_field_mono = simulator(plane_config)
# Shape: (Nx, Ny, Nz) - 3D spatial field

# Mode 2: TRANSIENT (PULSED) - Time-varying field with pulse excitation
# This computes the full spatio-temporal field as the pulse propagates
else:
    plt.figure(figsize=(10, 3))
    plt.plot(excitation_signal, "k", linewidth=1.5)
    plt.title("Excitation Signal (Pulsed)")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    x, y, z, p_field_transient = simulator(plane_config, excitation=excitation_signal)
# Shape: (Nx, Ny, Nz, Nt) - 4D spatio-temporal field

# ============================================================================
# STEP 6: VISUALIZE TRANSIENT PRESSURE FIELD OVER TIME
# ============================================================================

# Create an animated visualization showing pressure field evolution
# Each frame shows a 2D slice (XZ plane) at a specific time step
if SIMU_TYPE == "monochromatic":
    from pyfield.utilities import plot_pressure_planes

    plot_pressure_planes(
        x, y, z, p_field_mono, db_scale=True, figsize=FIGURE_SIZE, vmin=-30
    )

else:
    fig = plt.figure(figsize=FIGURE_SIZE)
    video_duration_s = 5
    fps = 30
    initial_time_idx = 0
    num_time_steps = p_field_transient.shape[0]
    total_time_s = np.linspace(
        0, num_time_steps / sampling_frequency_hz, num_time_steps
    )
    step = max(1, num_time_steps // (video_duration_s * fps))

    # p_field_norm = p_field_transient / p_field_transient.max()  # Normalize to max pressure
    p_field_norm = p_field_transient  # Keep raw pressure for dB conversion
    p_max = p_field_transient.max()  # Store max pressure for dB conversion

    # vmin = 0
    # vmax = 1
    vmin = -40  # dB scale minimum for visualization
    vmax = 0  # dB scale maximum (normalized to max pressure)

    plt.xlabel("Lateral Position (mm)")
    plt.ylabel("Depth (mm)")

    for time_idx in range(initial_time_idx, num_time_steps, step):
        # If the fig has been closed, break the loop
        if not plt.fignum_exists(fig.number):
            print("Visualization stopped by user.")
            break
        # Clear previous frame
        plt.clf()

        # Extract pressure at this time step (XZ plane at y=0)
        # pressure_at_t = p_field_norm[time_idx,:, :, :].squeeze()
        pressure_at_t = to_dB(p_field_norm[time_idx, :, :, :].squeeze(), vmax=p_max)

        # Create the 2D image with proper spatial extent
        im = plt.imshow(
            pressure_at_t.T,
            extent=(
                PLANE_X_EXTENT_MM[0],
                PLANE_X_EXTENT_MM[1],
                PLANE_Z_EXTENT_MM[1],
                PLANE_Z_EXTENT_MM[0],
            ),
            aspect="auto",
            cmap=CMAP_NAME,
            origin="upper",
            vmin=vmin,  # Dynamic range for visualization
            vmax=vmax,
        )

        # Add colorbar and labels if figure does not have one
        if im.colorbar is None:
            cbar = plt.colorbar(im, label="Pressure (dB re. max)")
            plt.clim(vmin, vmax)
        current_time_us = total_time_s[time_idx] * 1e6

        plt.title(
            f"Transient Pressure Field - Time = {current_time_us:.3f} µs (frame {time_idx + 1}/{num_time_steps})"
        )
        # Pause to create animation effect
        plt.pause(0.1)
