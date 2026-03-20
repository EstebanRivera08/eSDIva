import matplotlib.pyplot as plt
import numpy as np

import pyfield
import pyfield.transducers as transducers
from pyfield.psimulation import PyField
from pyfield.utilities import to_dB, plot_pressure_planes
%load_ext autoreload
%autoreload 2

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Transducer Selection
SAVE_DIR = "optimization_sequence"
TRANSDUCER_TYPE = "Domino"  # Options: "Domino", "Zeus_Matrix"
# MyTransducer = transducers.Zeus_Matrix()  # Alternative option

# Steering Configuration
STEERING_ANGLES_X_DEG = np.linspace(-10,10,11)  # Steering angle in degrees (X-axis)
SPEED_OF_SOUND_MPS = 1540  # Speed of sound in medium (m/s)

# Excitation Signal Definition
PULSE_CYCLES = 2  # Number of RF cycles in the pulse

# Simulation Plane Definition (all in mm)
PLANE_X_EXTENT_MM = [-10, 10]  # Lateral extent
PLANE_Z_EXTENT_MM = [0, 15]  # Depth extent
GRID_SPACING_X_MM = 0.05  # Lateral grid resolution
GRID_SPACING_Z_MM = 0.05  # Depth grid resolution

# Visualization Parameters
FIGURE_SIZE = (7, 5)

# ============================================================================
# STEP 1: INITIALIZE TRANSDUCER AND APPLY STEERING DELAYS
# ============================================================================

# Create transducer instance
MyTransducer = transducers.Domino()

# Calculate element-wise delays for beam steering
# Theory: Elements are excited with varying delays to steer the beam
# delay = (element_position * sin(steering_angle)) / speed_of_sound
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
window = np.hanning(len(time_array_s))  # Hanning window for smooth envelope
excitation_signal = np.sin(2 * np.pi * center_frequency_hz * time_array_s)*window

plt.figure(figsize=(4, 3))
plt.plot(excitation_signal, "k", linewidth=1.5)
plt.title("Excitation Signal ")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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

max_p = 0
max_p_global = 0  # To track global max pressure for consistent color scaling
pressure_fields = []  # To store pressure fields for each steering angle

for i, angle_deg in enumerate(STEERING_ANGLES_X_DEG):

    steering_angle_x_rad = np.deg2rad(angle_deg) 
    element_indices = np.arange(MyTransducer.n_elements)
    pitch_mm = MyTransducer.pitch

# Compute steering delays for each element
    steered_delays_s = (
        pitch_mm * element_indices * np.sin(steering_angle_x_rad)
    ) / SPEED_OF_SOUND_MPS
    steered_delays_s = steered_delays_s - np.min(steered_delays_s)  # Normalize to start at
    MyTransducer.set_delays(steered_delays_s)
    MyTransducer.plot_delays(figsize = (4,3))

# ============================================================================
# STEP 3: CREATE PYFIELD SIMULATOR AND DEFINE EXCITATION SIGNAL
# ============================================================================
# Wrap the transducer in PyField simulator
    simulator = PyField(MyTransducer)

# ============================================================================
# STEP 5: COMPUTE PRESSURE FIELD - TWO MODES AVAILABLE ============================================================================

# Mode 1: MONOCHROMATIC (CW) - Continuous wave at center frequency
    # This computes the steady-state field without time variation
    x, y, z, p_field_mono = simulator(plane_config)
# Shape: (Nx, Ny, Nz) - 3D spatial field
    if i != 0 or i != len(STEERING_ANGLES_X_DEG) - 1:
        max_p = np.max(p_field_mono)
        if max_p > max_p_global:
            max_p_global = max_p

    pressure_fields.append(p_field_mono)
    

# Create an animated visualization showing pressure field evolution
# Each frame shows a 2D slice (XZ plane) at a specific time step
max_p_global = np.max(pressure_fields[len(STEERING_ANGLES_X_DEG)//2])  # Use the central angle's max pressure 
FIGURE_SIZE = (5, 3)

for angle_deg, p_field_mono in zip(STEERING_ANGLES_X_DEG, pressure_fields):

    plot_pressure_planes(
        x, y, z, p_field_mono/max_p_global, db_scale=False, 
        figsize=FIGURE_SIZE, vmin = 0, vmax = 1, p_max = max_p_global,
        save_dir = SAVE_DIR,
        save_fig_name=f"planewave_steering_{angle_deg}deg.png"
    )

# Summed field
p_field_sum = np.sum(pressure_fields, axis=0)

plot_pressure_planes(
    x, y, z, p_field_sum/np.max(p_field_sum), db_scale=False, 
    figsize=FIGURE_SIZE,vmin= 0, vmax = 1, p_max = max_p_global,
    save_dir = SAVE_DIR,
    save_fig_name=f"planewave_steering_sum{len(STEERING_ANGLES_X_DEG)}.png"
)
