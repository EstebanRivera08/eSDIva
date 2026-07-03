"""
Example 11: Linear Array — CW Pressure Field with Attenuation

Field II parallel: ``fieldiiexamples/example_lineararray_attenuation_monochromatic_CW.m``

Shows the effect of tissue attenuation on a CW pressure field by comparing
an unattenuated simulation (water) against a brain-tissue model.

Key differences from Field II:
  - Field II uses a linear approximation: α(f) ≈ α₀ + (∂α/∂f)·(f − f₀),
    which is non-causal.
  - PyField implements causal power-law attenuation with Kramers–Kronig
    dispersion (Szabo 1994 / Holm 2019): H_att(f, d) = exp(−α|f|^y·d)
    × exp(−jφ_KK), which is physically exact.

Run with:
    uv run examples/example11_lineararray_attenuations_monochromatic_CW.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG

from pyfield.emission import Emission
from pyfield.plotting import plot2D_pressure_slices
from pyfield.transducers import LinearArrayTransducer

# ============================================================================
# CONFIGURATION  (Domino-like 128-element probe at 10 MHz)
# ============================================================================
FS = 100e6  # Hz
FREQUENCY_HZ = 10e6
C = 1540.0
FOCUS_MM = [0, 0, 8]  # close focus for a short near-field example

# Attenuation: brain tissue (ITIS database values)
ALPHA0_BRAIN = 0.5912  # dB/(cm·MHz^y)  — ITIS whole-brain
FREQ_POWER_BRAIN = 1.3  # power-law exponent y

# XZ simulation plane
PLANE = {
    "x_extent": [-7, 7],
    "y_extent": [0, 0],
    "z_extent": [0.5, 15],
    "dx": 0.05,
    "dy": 0,
    "dz": 0.05,
}

FIGSIZE = (10, 5)

print("\n--- Example 11: CW Pressure Field with Attenuation ---\n")
print("Field II parallel: example_lineararray_attenuation_monochromatic_CW.m")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: TRANSDUCER (Domino-like 128-element array at 10 MHz)
# ============================================================================
tx = LinearArrayTransducer(
    n_elements=128,
    element_width_mm=0.108,
    element_height_mm=1.5,
    kerf_mm=0.002,
    no_sub_x=1,
    no_sub_y=10,
    elevation_focus_mm=8.0,
    frequency_Hz=FREQUENCY_HZ,
)
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=1.0)

print(f"Transducer: {tx.n_elements} elements, fc = {FREQUENCY_HZ / 1e6:.0f} MHz")
print(f"Focus: {FOCUS_MM} mm")

# ============================================================================
# STEP 2: WATER (no attenuation)
# ============================================================================
print("\nSimulating water (no attenuation) ...")
sim_water = Emission(tx, monochromatic=True, fs=FS)
p_water, coords = sim_water(PLANE)

# ============================================================================
# STEP 3: BRAIN TISSUE (causal power-law attenuation)
# ============================================================================
print("Simulating brain tissue (causal K-K attenuation) ...")
sim_brain = Emission(
    tx,
    monochromatic=True,
    fs=FS,
    alpha0=ALPHA0_BRAIN,
    freq_power=FREQ_POWER_BRAIN,
)
p_brain, coords = sim_brain(PLANE)

# ============================================================================
# STEP 4: VISUALISE SIDE-BY-SIDE
# ============================================================================
plot2D_pressure_slices(
    p_water,
    coords=coords,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title="Water — no attenuation",
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="attenuation_water.png",
)

plot2D_pressure_slices(
    p_brain,
    coords=coords,
    db_scale=True,
    figsize=FIGSIZE,
    vmin=-40,
    title=(
        f"Brain tissue — α₀={ALPHA0_BRAIN} dB/(cm·MHz^y), "
        f"y={FREQ_POWER_BRAIN}  (causal K-K)"
    ),
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="attenuation_brain.png",
)

# Difference map (dB attenuation at each spatial point)
p_water_safe = np.where(p_water > 0, p_water, 1e-30)
p_brain_safe = np.where(p_brain > 0, p_brain, 1e-30)
att_map_db = 20 * np.log10(p_brain_safe / p_water_safe)

fig, ax = plt.subplots(figsize=FIGSIZE)
# p shape is (Nx, Ny, Nz); take the XZ slice and transpose for imshow
att_xz = np.squeeze(att_map_db).T  # (Nz, Nx)
extent = [
    coords["x"].min(),
    coords["x"].max(),
    coords["z"].max(),
    coords["z"].min(),
]
im = ax.imshow(
    att_xz,
    aspect="auto",
    extent=extent,
    cmap="RdBu_r",
    vmin=-20,
    vmax=0,
)
plt.colorbar(im, ax=ax, label="Attenuation (dB)")
ax.set_xlabel("Lateral x (mm)")
ax.set_ylabel("Axial z (mm)")
ax.set_title("Attenuation map: brain − water (dB)")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "attenuation_map.png"), dpi=150)

plt.show()

print("\nNote: Field II uses non-causal linear-in-frequency approximation.")
print("PyField causal K-K model preserves correct phase dispersion.")
