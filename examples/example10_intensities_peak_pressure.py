"""
Example 10: Linear Array — Peak Pressure and Ispta Along Acoustic Axis

Field II parallel: ``fieldiiexamples/example_intensity.m``

Computes the on-axis peak pressure and spatial-peak temporal-average intensity
(Ispta) versus depth for a focused linear array, with and without attenuation.
Directly comparable to the Field II intensity example.

Differences from Field II:
  - PyField uses causal power-law attenuation with Kramers–Kronig dispersion
    instead of non-causal minimum-phase correction.
  - Explicit excitation pulse passed to ``Emission``; no internal ``calc_hp``
    call needed.

Run with:
    uv run examples/example10_intensities_peak_pressure.py
"""

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_FOLDER, SAVE_FIG

import pyfield.transducers as transducers
from pyfield.emission import Emission

# ============================================================================
# CONFIGURATION  (mirrors Field II example_intensity.m)
# ============================================================================
FREQUENCY_HZ = 5e6  # centre frequency
C = 1540.0  # speed of sound (m/s)
FS = 200e6  # sampling frequency (Hz)
PULSE_CYCLES = 2  # number of cycles in the excitation pulse
FOCUS_MM = [0, 0, 30]  # focal point

# Attenuation (brain tissue, 0.5 dB/(cm·MHz^y))
ALPHA0 = 0.5  # dB/(cm·MHz^y)
FREQ_POWER = 1.0  # power-law exponent y

# Axial scan: on-axis points from 1 to 100 mm
Z_MM = np.arange(1, 101, 1, dtype=np.float32)

# Intensity units conversion
RHO = 1000.0  # kg/m³ (water-like medium)
Z_ACOUSTIC = 1.480e6  # characteristic acoustic impedance [kg/(m²·s)]
T_PRF = 1 / 5e3  # pulse repetition period [s]

print("\n--- Example 10: Peak Pressure and Ispta Along Acoustic Axis ---\n")
print("Field II parallel: example_intensity.m")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: TRANSDUCER SETUP (65-element array with elevation focus)
# ============================================================================
lam = C / FREQUENCY_HZ  # wavelength [m]
width_m = lam / 2  # element width = λ/2
kerf_m = lam / 10
tx = transducers.LinearArrayTransducer(
    n_elements=65,
    element_width_mm=width_m * 1e3,
    element_height_mm=5.0,
    kerf_mm=kerf_m * 1e3,
    no_sub_x=1,
    no_sub_y=6,
    elevation_focus_mm=40.0,
    frequency_Hz=FREQUENCY_HZ,
)
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)

# ============================================================================
# STEP 2: EXCITATION PULSE
# ============================================================================
t_pulse = np.arange(0, PULSE_CYCLES / FREQUENCY_HZ, 1.0 / FS)
pulse = (np.sin(2 * np.pi * FREQUENCY_HZ * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)
tp = len(pulse) / FS  # pulse duration [s]

# ============================================================================
# STEP 3: AXIAL FIELD GRID (on-axis points)
# ============================================================================
axial_plane = {
    "x_extent": [0, 0],
    "y_extent": [0, 0],
    "z_extent": [float(Z_MM[0]), float(Z_MM[-1])],
    "dx": 0,
    "dy": 0,
    "dz": float(Z_MM[1] - Z_MM[0]),
}


# ============================================================================
# STEP 4: SIMULATE WITH AND WITHOUT ATTENUATION
# ============================================================================
def run_axial(alpha0):
    sim = Emission(tx, fs=FS, excitation=pulse, rho=RHO, alpha0=alpha0)
    p, coords = sim(axial_plane)  # (Nt, 1, 1, Nz)
    p_axis = p[:, 0, 0, :]  # (Nt, Nz)
    peak = np.max(np.abs(p_axis), axis=0)  # (Nz,)
    ispta = np.sum(p_axis**2, axis=0) / (2 * Z_ACOUSTIC * FS * T_PRF)  # W/m²
    return peak, ispta


print("Simulating without attenuation ...")
peak_no_att, ispta_no_att = run_axial(alpha0=None)

print("Simulating with attenuation (brain tissue) ...")
peak_att, ispta_att = run_axial(alpha0=ALPHA0)

# Convert Ispta from W/m² to mW/cm²
ispta_no_att_mwcm2 = ispta_no_att * 1e3 / 1e4
ispta_att_mwcm2 = ispta_att * 1e3 / 1e4

# ============================================================================
# STEP 5: PLOT
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax = axes[0]
ax.plot(Z_MM, ispta_no_att_mwcm2, "b-", label="No attenuation")
ax.plot(Z_MM, ispta_att_mwcm2, "r--", label=f"α₀={ALPHA0} dB/(cm·MHz), y={FREQ_POWER}")
ax.set_ylabel("Ispta (mW/cm²)")
ax.set_title(
    f"Acoustic axis — {tx.n_elements}-element array, focus at z={FOCUS_MM[2]} mm"
)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(Z_MM, peak_no_att / 1e3, "b-", label="No attenuation")
ax.plot(Z_MM, peak_att / 1e3, "r--", label=f"α₀={ALPHA0} dB/(cm·MHz), y={FREQ_POWER}")
ax.set_xlabel("Axial distance (mm)")
ax.set_ylabel("Peak pressure (kPa)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(
    "Field II parallel: example_intensity.m\n(PyField uses causal K-K attenuation)"
)
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "intensities_peak_pressure.png"), dpi=150)
    print(f"Saved to {FIG_FOLDER / 'intensities_peak_pressure.png'}")

plt.show()
