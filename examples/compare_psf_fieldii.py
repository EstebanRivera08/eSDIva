"""
PSF Comparison: PyField vs Field II

Loads the reference Field II output from ``examples/rf_concave_psf.mat`` and
runs PyField Reception on the identical geometry.  Displays both PSFs
side-by-side to validate the PE SDI delta placement.

Field II reference geometry (``example_concave_psf.m``):
  - Concave circular, Ø 16 mm, focal depth 80 mm, 3 MHz
  - 2-cycle Hanning-windowed sine as impulse response AND excitation
  - Scatterers: 101 lateral positions, z = 30 mm, fs = 100 MHz

Run with:
    uv run examples/compare_psf_fieldii.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import hilbert

# Allow ``from config import …`` when run from repo root via ``uv run``.
sys.path.insert(0, str(Path(__file__).parent))
from config import FIG_FOLDER, SAVE_FIG

from pyfield.psimulation import Reception
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities import to_dB

# ---------------------------------------------------------------------------
# Geometry / simulation parameters (must match Field II reference)
# ---------------------------------------------------------------------------
DIAMETER_MM = 16.0
FOCUS_MM = 80.0
FREQUENCY_HZ = 3e6
FS = 100e6
C = 1540.0
SCATTERER_Z_MM = 30.0
DB_FLOOR = -60.0

# 101 lateral positions from -10 to +10 mm  (matches Field II mat file)
X_SCAT_MM = np.linspace(-10.0, 10.0, 101)

MAT_FILE = Path(__file__).parent / "rf_concave_psf.mat"

# ---------------------------------------------------------------------------
# Load Field II reference
# ---------------------------------------------------------------------------
_mat = scipy.io.loadmat(str(MAT_FILE), simplify_cells=True)["save_struct"]
fii_env_db: np.ndarray = _mat["rf_env_dB"]   # (Nt_fii=120, N_lat=101)
fii_t0: float = float(_mat["t0"])             # 3.842e-05 s
fii_dt: float = 1.0 / FS
fii_Nt: int = fii_env_db.shape[0]
fii_t_us = (fii_t0 + np.arange(fii_Nt) * fii_dt) * 1e6  # µs

print(f"Field II: Nt={fii_Nt}, N_lat={fii_env_db.shape[1]}")
print(f"  t0 = {fii_t0*1e6:.3f} µs, t_end = {fii_t_us[-1]:.3f} µs")
print(f"  dB range: [{fii_env_db.min():.1f}, {fii_env_db.max():.1f}]")

# ---------------------------------------------------------------------------
# Build transducer and run PyField reception
# ---------------------------------------------------------------------------
tx = ConcaveCircularTransducer(
    diameter_mm=DIAMETER_MM,
    focus_mm=FOCUS_MM,
    frequency_Hz=FREQUENCY_HZ,
    refine_factor=1,
    no_sub_diameter=16,
)

t_ir = np.arange(0, 2.0 / FREQUENCY_HZ, 1.0 / FS)
ir = (np.sin(2 * np.pi * FREQUENCY_HZ * t_ir) * np.hanning(len(t_ir))).astype(
    np.float32
)
tx.impulse_response = ir
tx.excitation = ir

sim = Reception(tx, tx, fs=FS, c=C, verbose=False)

field_points_mm = np.column_stack(
    [
        X_SCAT_MM,
        np.zeros_like(X_SCAT_MM),
        np.full_like(X_SCAT_MM, SCATTERER_Z_MM),
    ]
).astype(np.float32)

print(f"\nRunning PyField Reception for {len(X_SCAT_MM)} scatterers ...")
rf_pts, coords = sim.compute_point_rf(field_points_mm)
# rf_pts: (N_lat, Nt_pf, 1) — mono-element

rf_image = rf_pts[:, :, 0].T          # (Nt_pf, N_lat)
pf_t0: float = float(coords["t0"])
pf_dt: float = float(coords["dt"])
pf_Nt: int = rf_image.shape[0]
pf_t_us = (pf_t0 + np.arange(pf_Nt) * pf_dt) * 1e6  # µs

print(f"PyField:  Nt={pf_Nt}, N_lat={rf_image.shape[1]}")
print(f"  t0 = {pf_t0*1e6:.3f} µs, t_end = {pf_t_us[-1]:.3f} µs")
print(f"  t0 shift vs Field II: {(pf_t0 - fii_t0)*1e9:.1f} ns")

# ---------------------------------------------------------------------------
# Envelope and dB compression
# ---------------------------------------------------------------------------
env_pf = np.abs(hilbert(rf_image, axis=0))
env_pf_db = to_dB(env_pf, vmin=10 ** (DB_FLOOR / 20))

# ---------------------------------------------------------------------------
# Numerical diagnostics
# ---------------------------------------------------------------------------
fii_peak_t_idx, fii_peak_x_idx = np.unravel_index(
    np.argmax(fii_env_db), fii_env_db.shape
)
pf_peak_t_idx, pf_peak_x_idx = np.unravel_index(
    np.argmax(env_pf_db), env_pf_db.shape
)
print(
    f"\nPeak location:"
    f"\n  Field II — t = {fii_t_us[fii_peak_t_idx]:.3f} µs, "
    f"x = {X_SCAT_MM[fii_peak_x_idx]:.2f} mm"
    f"\n  PyField  — t = {pf_t_us[pf_peak_t_idx]:.3f} µs, "
    f"x = {X_SCAT_MM[pf_peak_x_idx]:.2f} mm"
    f"\n  Δt = {(pf_t_us[pf_peak_t_idx] - fii_t_us[fii_peak_t_idx])*1e3:.2f} ns"
)

# ---------------------------------------------------------------------------
# Determine shared extent for common colorbar scale
# ---------------------------------------------------------------------------
extent_fii = [X_SCAT_MM[0], X_SCAT_MM[-1], fii_t_us[-1], fii_t_us[0]]
extent_pf = [X_SCAT_MM[0], X_SCAT_MM[-1], pf_t_us[-1], pf_t_us[0]]

# ---------------------------------------------------------------------------
# Plot: 2×2 grid  (envelope dB top row, raw RF bottom row)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

_im_kw_db = dict(aspect="auto", cmap="hot", vmin=DB_FLOOR, vmax=0)
_im_kw_rf = dict(aspect="auto", cmap="RdBu", vmin=-1, vmax=1)

# --- top-left: Field II envelope dB ---
ax = axes[0, 0]
im = ax.imshow(fii_env_db, extent=extent_fii, **_im_kw_db)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("Field II — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

# --- top-right: PyField envelope dB ---
ax = axes[0, 1]
im = ax.imshow(env_pf_db, extent=extent_pf, **_im_kw_db)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("PyField — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

# --- bottom-left: Field II envelope (linear, for FWHM inspection) ---
ax = axes[1, 0]
fii_env_linear = 10 ** (fii_env_db / 20)
ax.imshow(
    fii_env_linear,
    extent=extent_fii,
    aspect="auto",
    cmap="hot",
    vmin=0,
    vmax=1,
)
ax.set_title("Field II — envelope (linear)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

# --- bottom-right: PyField raw RF (normalised) ---
ax = axes[1, 1]
rf_norm = rf_image / (np.abs(rf_image).max() + 1e-30)
ax.imshow(rf_norm, extent=extent_pf, **_im_kw_rf)
ax.set_title("PyField — raw RF (normalised)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

fig.suptitle(
    f"PSF comparison — concave Ø{DIAMETER_MM:.0f} mm, focus {FOCUS_MM:.0f} mm, "
    f"scatterer z={SCATTERER_Z_MM} mm\n"
    f"Δt0 = {(pf_t0 - fii_t0)*1e9:.1f} ns   "
    f"Δt_peak = {(pf_t_us[pf_peak_t_idx] - fii_t_us[fii_peak_t_idx])*1e3:.2f} ns"
)
plt.tight_layout()

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
    plt.savefig(str(FIG_FOLDER / "compare_psf_fieldii.png"), dpi=150)
    print(f"\nSaved to {FIG_FOLDER / 'compare_psf_fieldii.png'}")

plt.show()
