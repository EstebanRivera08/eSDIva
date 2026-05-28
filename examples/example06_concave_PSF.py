"""
Example 06: Concave Single-Element Transducer — Pulse-Echo PSF

Field II parallel: ``fieldiiexamples/example_concave_psf.m``

Computes the pulse-echo Point Spread Function (PSF) of a spherically focused
single-element transducer.  Each lateral position is simulated as an
independent point scatterer, and the RF envelope is displayed as a 2-D image
(lateral distance × time/depth).

Field II uses ``calc_hhp`` (pulse-echo SIR at every point).  PyField uses
``Reception(tx, tx)`` with one scatterer at a time — same physics, explicit
Python loop.

Run with:
    uv run examples/example06_concave_PSF.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from scipy.signal import hilbert

from pyfield.psimulation import Reception
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities import to_dB

# ============================================================================
# CONFIGURATION
# ============================================================================
# Transducer (matches Field II example_concave_psf.m)
DIAMETER_MM = 16.0  # radius R = 8 mm → diameter 16 mm
FOCUS_MM = 80.0  # geometric focal depth from rim [mm]
FREQUENCY_HZ = 3e6
FS = 100e6  # sampling frequency [Hz]
C = 1540.0  # speed of sound [m/s]

# Scatterers: lateral line at z = 30 mm
SCATTERER_Z_MM = 30.0
X_SCAT_MM = np.arange(-10, 10.0, 0.2)  # -10..+10 mm in 0.5 mm steps

# Excitation / impulse response: 2-cycle Hanning-windowed sine
PULSE_CYCLES = 2

print("\n--- Example 06: Concave Transducer — Pulse-Echo PSF ---\n")
print("Field II parallel: example_concave_psf.m")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: CREATE TRANSDUCER
# ============================================================================
tx = ConcaveCircularTransducer(
    diameter_mm=DIAMETER_MM,
    focus_mm=FOCUS_MM,
    frequency_Hz=FREQUENCY_HZ,
    refine_factor=1,  # No refinement for this example as FieldII uses a single element
    # with no sub-divisions
    no_sub_diameter=16,  # Match the number of sub-elements in Field II
)
tx.show()

# ============================================================================
# STEP 2: SET IMPULSE RESPONSE AND EXCITATION
# ============================================================================
t_ir = np.arange(0, PULSE_CYCLES / FREQUENCY_HZ, 1.0 / FS)
ir = (np.sin(2 * np.pi * FREQUENCY_HZ * t_ir) * np.hanning(len(t_ir))).astype(
    np.float32
)

tx.impulse_response = ir
tx.excitation = ir  # same pulse for TX excitation

# ============================================================================
# STEP 3: SIMULATE EACH LATERAL POSITION (pulse-echo loop)
# ============================================================================
sim = Reception(tx, tx, fs=FS, c=C, verbose=False)

field_points_mm = np.column_stack(
    [
        X_SCAT_MM,
        np.zeros_like(X_SCAT_MM),
        np.full_like(X_SCAT_MM, SCATTERER_Z_MM),
    ]
).astype(np.float32)

print(f"Simulating {len(X_SCAT_MM)} lateral positions at z={SCATTERER_Z_MM} mm ...")
rf_pts, coords = sim.compute_point_rf(field_points_mm)
# rf_pts.shape = (N_lateral, Nt, 1) — mono-element

# (N_lateral, Nt) → transpose → (Nt, N_lateral)
rf_image = rf_pts[:, :, 0].T

# Reconstruct time vector
t_us = (coords["t0"] + np.arange(rf_image.shape[0]) * coords["dt"]) * 1e6  # µs

# ============================================================================
# STEP 4: ENVELOPE AND LOG-COMPRESSION
# ============================================================================
env = np.abs(hilbert(rf_image, axis=0))
env_db = to_dB(env, vmin=10 ** (-60 / 20))  # clip at -60 dB

# ============================================================================
# STEP 5: DISPLAY
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: raw RF (normalised)
ax = axes[0]
ax.imshow(
    rf_image / (np.abs(rf_image).max() + 1e-30),
    aspect="auto",
    extent=[X_SCAT_MM[0], X_SCAT_MM[-1], t_us[-1], t_us[0]],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
ax.set_xlabel("Lateral distance (mm)")
ax.set_ylabel("Time (µs)")
ax.set_title("Raw RF (normalised)")

# Right: log-compressed envelope
ax = axes[1]
im = ax.imshow(
    env_db,
    aspect="auto",
    extent=[X_SCAT_MM[0], X_SCAT_MM[-1], t_us[-1], t_us[0]],
    cmap="hot",
    vmin=-60,
    vmax=0,
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_xlabel("Lateral distance (mm)")
ax.set_ylabel("Time (µs)")
ax.set_title(f"Pulse-echo PSF — concave Ø{DIAMETER_MM:.0f} mm, focus {FOCUS_MM:.0f} mm")

plt.suptitle(
    f"Field II equivalent: example_concave_psf.m  |  z_scat = {SCATTERER_Z_MM} mm"
)
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "concave_psf.png"), dpi=150)
    print(f"Saved to {FIG_FOLDER / 'concave_psf.png'}")

plt.show()
