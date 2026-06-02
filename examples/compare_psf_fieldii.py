"""
PSF Comparison: PyField (naive + SDI) vs Field II

Loads Field II reference from ``examples/rf_concave_psf.mat``, runs both
PyField reception backends on the identical geometry, and displays all three
side-by-side in a 3×3 grid (mirrors ``example06_concave_PSF.py`` layout).

Field II reference geometry (``example_concave_psf.m``):
  - Concave circular, Ø 16 mm, focal depth 80 mm, 3 MHz
  - 2-cycle Hanning-windowed sine as impulse response AND excitation
  - Scatterers: 101 lateral positions, z = 30 mm, fs = 100 MHz

Run with:
    uv run examples/compare_psf_fieldii.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG_FOLDER, SAVE_FIG

from pyfield.reception import Reception, ReceptionSDI
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

# 101 lateral positions from -10 to +10 mm (matches Field II mat file)
X_SCAT_MM = np.linspace(-10.0, 10.0, 101)

MAT_FILE = Path(__file__).parent / "rf_concave_psf.mat"

# ---------------------------------------------------------------------------
# Load Field II reference
# ---------------------------------------------------------------------------
_mat = scipy.io.loadmat(str(MAT_FILE), simplify_cells=True)["save_struct"]
fii_env_db: np.ndarray = _mat["rf_env_dB"]  # (Nt_fii=120, N_lat=101)
fii_t0: float = float(_mat["t0"])
# Mat file stores every-5th-sample envelope (RF_data(1:5:600,:) in Matlab).
fii_dt: float = 5.0 / FS
fii_Nt: int = fii_env_db.shape[0]
fii_t_us = (fii_t0 + np.arange(fii_Nt) * fii_dt) * 1e6

print(f"Field II: Nt={fii_Nt}, t0={fii_t0 * 1e6:.3f} µs, t_end={fii_t_us[-1]:.3f} µs")


# ---------------------------------------------------------------------------
# Transducer factory (fresh instances per simulation to avoid state sharing)
# ---------------------------------------------------------------------------
def _make_transducer(with_excitation: bool = False) -> ConcaveCircularTransducer:
    tx = ConcaveCircularTransducer(
        diameter_mm=DIAMETER_MM,
        focus_mm=FOCUS_MM,
        frequency_Hz=FREQUENCY_HZ,
        refine_factor=2,
        no_sub_diameter=25,
    )
    t_ir = np.arange(0, 2.0 / FREQUENCY_HZ, 1.0 / FS)
    # Match Field II: xdc_impulse uses Hanning-windowed sine on the transducer handle
    # used as both TX and RX, so both IR_tx and IR_rx = Hanning×sin.
    ir = np.sin(2 * np.pi * FREQUENCY_HZ * t_ir) * np.hanning(len(t_ir))
    tx.impulse_response = ir
    if with_excitation:
        # Match Field II: xdc_excitation uses plain sine (no Hanning window).
        tx.excitation = np.sin(2 * np.pi * FREQUENCY_HZ * t_ir)
    return tx


tx = _make_transducer()
tx.show()

# ConcaveCircularTransducer now follows the Field II / datasheet convention:
# focus_mm is the focal length (= radius of curvature) and the bowl apex sits at
# z=0, matching xdc_concave. Field points use the same apex-origin depth, so no
# coordinate shift is needed. (Residual ~0.15 us vs Field II is the (jw)^3
# derivative-model difference, not geometry.)
field_points_mm = np.column_stack(
    [
        X_SCAT_MM,
        np.zeros_like(X_SCAT_MM),
        np.full_like(X_SCAT_MM, SCATTERER_Z_MM),
    ]
).astype(np.float32)

# ---------------------------------------------------------------------------
# Run both PyField backends
# ---------------------------------------------------------------------------
print(f"\nSimulating {len(X_SCAT_MM)} lateral positions at z={SCATTERER_Z_MM} mm ...")

print("\n  [1/2] Reception(method='sdi') ...")
t_start = time.time()
sim_naive = Reception(
    _make_transducer(with_excitation=True),
    _make_transducer(),
    fs=FS,
    c=C,
    method="sdi",
    verbose=False,
)
rf_naive, coords_naive = sim_naive.pulse_echo_response(
    field_points_mm, per_scatterer=True
)
t_naive = time.time() - t_start
print(f"  Done in {t_naive:.2f} s")

print("\n  [2/2] ReceptionSDI() ...")
t_start = time.time()
sim_sdi = ReceptionSDI(
    _make_transducer(with_excitation=True),
    _make_transducer(),
    fs=FS,
    c=C,
    verbose=False,
)
rf_sdi, coords_sdi = sim_sdi.pulse_echo_response(field_points_mm, per_scatterer=True)
t_sdi = time.time() - t_start
print(f"  Done in {t_sdi:.2f} s")

# ---------------------------------------------------------------------------
# Prepare RF images: (N_lat, Nt, 1) → (Nt, N_lat)
# ---------------------------------------------------------------------------
rf_naive_img = rf_naive[:, :, 0].T / rf_naive.max()
rf_sdi_img = rf_sdi[:, :, 0].T / rf_sdi.max()

t_us_naive = (
    coords_naive["t0"] + np.arange(rf_naive_img.shape[0]) * coords_naive["dt"]
) * 1e6
t_us_sdi = (coords_sdi["t0"] + np.arange(rf_sdi_img.shape[0]) * coords_sdi["dt"]) * 1e6

peak_rf = max(np.abs(rf_naive_img).max(), np.abs(rf_sdi_img).max()) + 1e-30

env_naive = np.abs(hilbert(rf_naive_img, axis=0))
env_sdi = np.abs(hilbert(rf_sdi_img, axis=0))

peak_env = max(env_naive.max(), env_sdi.max()) + 1e-30
env_naive_db = to_dB(env_naive / peak_env, vmin=10 ** (DB_FLOOR / 20))
env_sdi_db = to_dB(env_sdi / peak_env, vmin=10 ** (DB_FLOOR / 20))

fii_env_linear = 10 ** (fii_env_db / 20)

# ---------------------------------------------------------------------------
# Interpolate PyField envelopes onto Field II time grid for error analysis.
# FII grid is coarser (dt=50 ns vs 10 ns), so PyField is the source.
# ---------------------------------------------------------------------------
eps = 10 ** (DB_FLOOR / 20)
env_naive_on_fii = np.zeros_like(fii_env_linear)
env_sdi_on_fii = np.zeros_like(fii_env_linear)

for i in range(len(X_SCAT_MM)):
    env_naive_on_fii[:, i] = np.interp(
        fii_t_us, t_us_naive, env_naive[:, i] / peak_env, left=0.0, right=0.0
    )
    env_sdi_on_fii[:, i] = np.interp(
        fii_t_us, t_us_sdi, env_sdi[:, i] / peak_env, left=0.0, right=0.0
    )

env_naive_on_fii_db = to_dB(env_naive_on_fii, vmin=eps)
env_sdi_on_fii_db = to_dB(env_sdi_on_fii, vmin=eps)
diff_fii_sdi_db = env_sdi_on_fii_db - fii_env_db
diff_fii_sdi = env_sdi_on_fii - fii_env_linear
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
fii_peak_t_idx = int(np.unravel_index(np.argmax(fii_env_db), fii_env_db.shape)[0])
naive_peak_t_idx = int(np.unravel_index(np.argmax(env_naive_db), env_naive_db.shape)[0])
sdi_peak_t_idx = int(np.unravel_index(np.argmax(env_sdi_db), env_sdi_db.shape)[0])

print(f"\nPeak time:")
print(f"  Field II : {fii_t_us[fii_peak_t_idx]:.3f} µs")
print(
    f"  Naive    : {t_us_naive[naive_peak_t_idx]:.3f} µs  (Δ = {t_us_naive[naive_peak_t_idx] - fii_t_us[fii_peak_t_idx]:.3f} µs)"
)
print(
    f"  SDI      : {t_us_sdi[sdi_peak_t_idx]:.3f} µs  (Δ = {t_us_sdi[sdi_peak_t_idx] - fii_t_us[fii_peak_t_idx]:.3f} µs)"
)

# Max dB error on FII grid
center_idx = len(X_SCAT_MM) // 2
print(f"\nMax |dB error| vs Field II (on-axis):")
print(
    f"  Naive: {np.abs(env_naive_on_fii_db[:, center_idx] - fii_env_db[:, center_idx]).max():.2f} dB"
)
print(
    f"  SDI  : {np.abs(env_sdi_on_fii_db[:, center_idx] - fii_env_db[:, center_idx]).max():.2f} dB"
)

# ---------------------------------------------------------------------------
# Plot 3×3: naive | SDI | Field II
# ---------------------------------------------------------------------------
extent_naive = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_naive[-1], t_us_naive[0]]
extent_sdi = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_sdi[-1], t_us_sdi[0]]
extent_fii = [X_SCAT_MM[0], X_SCAT_MM[-1], fii_t_us[-1], fii_t_us[0]]

peak_row_naive = int(np.argmax(env_naive[:, center_idx]))
peak_row_sdi = int(np.argmax(env_sdi[:, center_idx]))
peak_row_fii = int(np.argmax(fii_env_linear[:, center_idx]))

_kw_db = dict(aspect="auto", cmap="hot", vmin=DB_FLOOR, vmax=0)
_kw_rf = dict(aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
_kw_lin = dict(aspect="auto", cmap="hot", vmin=0, vmax=1)

fig, axes = plt.subplots(3, 4, figsize=(16, 7))

# --- Row 0: envelope dB (PyField) / Field II envelope dB ---
ax = axes[0, 0]
im = ax.imshow(env_naive_db, extent=extent_naive, **_kw_db)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("Reception naive — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[0, 1]
im = ax.imshow(env_sdi_db, extent=extent_sdi, **_kw_db)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("ReceptionSDI — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[0, 2]
im = ax.imshow(fii_env_db, extent=extent_fii, **_kw_db)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("Field II — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[0, 3]
im = ax.imshow(diff_fii_sdi_db, extent=extent_fii, aspect="auto")
plt.colorbar(im, ax=ax)
ax.set_title("Field II — SDI")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")


# --- Row 1: raw RF (PyField) / |Field II-SDI| ---
ax = axes[1, 0]
ax.imshow(env_naive, extent=extent_naive, **_kw_lin)
ax.set_title(f"Reception naive — raw RF  ({t_naive:.1f} s)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[1, 1]
ax.imshow(env_sdi, extent=extent_sdi, **_kw_lin)
ax.set_title(f"ReceptionSDI — raw RF  ({t_sdi:.1f} s)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[1, 2]
ax.imshow(fii_env_linear, extent=extent_fii, **_kw_lin)
ax.set_title("Field II — raw RF envelope (linear)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[1, 3]
im = ax.imshow(diff_fii_sdi, extent=extent_fii, aspect="auto")
plt.colorbar(im, ax=ax)
ax.set_title("Field II — SDI")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

# --- Row 2: on-axis envelope overlay / lateral profile / dB error vs FII ---
ax = axes[2, 0]
ax.plot(t_us_naive, env_naive[:, center_idx] / peak_env, label="naive", lw=1.2)
ax.plot(t_us_sdi, env_sdi[:, center_idx] / peak_env, "--", label="SDI", lw=1.2)
ax.plot(fii_t_us, fii_env_linear[:, center_idx], ":", ms=3, label="Field II", lw=1.5)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Normalised envelope")
ax.set_title(f"On-axis envelope  (x = {X_SCAT_MM[center_idx]:.1f} mm)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2, 1]
ax.plot(X_SCAT_MM, env_naive_db[peak_row_naive, :], label="naive", lw=1.2)
ax.plot(X_SCAT_MM, env_sdi_db[peak_row_sdi, :], "--", label="SDI", lw=1.2)
ax.plot(X_SCAT_MM, fii_env_db[peak_row_fii, :], ":", label="Field II", lw=1.5)
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Envelope (dB)")
ax.set_title("Lateral profile at envelope peak")
ax.legend()
ax.grid(alpha=0.3)

err_naive_ax = np.abs(env_naive_on_fii_db[:, center_idx] - fii_env_db[:, center_idx])
err_sdi_ax = np.abs(env_sdi_on_fii_db[:, center_idx] - fii_env_db[:, center_idx])

ax = axes[2, 2]
ax.plot(fii_t_us, err_naive_ax, label="|naive − FII|", lw=1.2)
ax.plot(fii_t_us, err_sdi_ax, "--", label="|SDI − FII|", lw=1.2)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Envelope error (dB)")
ax.set_title("On-axis |dB error| vs Field II")
ax.legend()
ax.grid(alpha=0.3)

# Put every panel on the SAME time window. PyField's longer SIR tail otherwise
# stretches its panels vertically, making the (98%-correlated) identical
# structure look spatially different next to Field II.
_t_lo, _t_hi = fii_t_us[0], fii_t_us[-1]
for _a in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]):
    _a.set_ylim(_t_hi, _t_lo)  # time increases downward
for _a in (axes[2, 0], axes[2, 2]):
    _a.set_xlim(_t_lo, _t_hi)

fig.suptitle(
    f"PSF comparison — concave Ø{DIAMETER_MM:.0f} mm, focus {FOCUS_MM:.0f} mm, "
    f"scatterer z={SCATTERER_Z_MM} mm\n"
    f"FII t0={fii_t0 * 1e6:.3f} µs  ·  "
    f"naive Δt0={t_us_naive[naive_peak_t_idx] - fii_t_us[fii_peak_t_idx]:.3f} µs  ·  "
    f"SDI Δt0={t_us_sdi[sdi_peak_t_idx] - fii_t_us[fii_peak_t_idx]:.3f} µs"
)
plt.tight_layout()

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
    plt.savefig(str(FIG_FOLDER / "compare_psf_fieldii.png"), dpi=150)
    print(f"\nSaved to {FIG_FOLDER / 'compare_psf_fieldii.png'}")

plt.show()
