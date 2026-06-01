"""
Example 06: Concave Single-Element Transducer — Pulse-Echo PSF

Field II parallel: ``fieldiiexamples/example_concave_psf.m``

Computes the pulse-echo Point Spread Function (PSF) of a spherically focused
single-element transducer and compares two simulation backends:

- ``Reception(method="naive")`` — conventional FieldII-style: h_tx ⊛ h_rx,
  three temporal derivatives applied to excitation via (jω)³.
- ``ReceptionSDI()`` — combined PE SDI: 16 deltas per (TX, RX) patch pair,
  derivatives absorbed into Dh_pe, one cumsum.

Run with:
    uv run examples/example06_concave_PSF.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from scipy.signal import hilbert

from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities import to_dB

# ============================================================================
# CONFIGURATION
# ============================================================================
DIAMETER_MM = 16.0
FOCUS_MM = 80.0
FREQUENCY_HZ = 3e6
FS = 100e6
C = 1540.0

SCATTERER_Z_MM = 30.0
X_SCAT_MM = np.arange(-10, 10.0, 0.2)
PULSE_CYCLES = 2

print("\n--- Example 06: Concave Transducer — Pulse-Echo PSF (naive vs SDI) ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: TRANSDUCER + EXCITATION
# ============================================================================
tx = ConcaveCircularTransducer(
    diameter_mm=DIAMETER_MM,
    focus_mm=FOCUS_MM,
    frequency_Hz=FREQUENCY_HZ,
    refine_factor=1,
    no_sub_diameter=16,
)
# tx.show()

t_ir = np.arange(0, PULSE_CYCLES / FREQUENCY_HZ, 1.0 / FS)
ir = (np.sin(2 * np.pi * FREQUENCY_HZ * t_ir) * np.hanning(len(t_ir))).astype(
    np.float32
)
tx.impulse_response = ir
tx.excitation = ir

rx = ConcaveCircularTransducer(
    diameter_mm=DIAMETER_MM,
    focus_mm=FOCUS_MM,
    frequency_Hz=FREQUENCY_HZ,
    refine_factor=1,
    no_sub_diameter=16,
)

field_points_mm = np.column_stack(
    [
        X_SCAT_MM,
        np.zeros_like(X_SCAT_MM),
        np.full_like(X_SCAT_MM, SCATTERER_Z_MM),
    ]
).astype(np.float32)

# ============================================================================
# STEP 2: SIMULATE — BOTH BACKENDS
# ============================================================================
print(f"Simulating {len(X_SCAT_MM)} lateral positions at z={SCATTERER_Z_MM} mm ...")

print("\n  [1/2] Reception(method='naive') ...")
t_start = time.time()
sim_naive = Reception(tx, rx, fs=FS, c=C, method="sdi", verbose=False)
rf_naive, coords_naive = sim_naive.scattered_rf(field_points_mm, per_scatterer=True)
t_naive = time.time() - t_start
print(f" Done in {t_naive:.2f} s")

print("\n  [2/2] ReceptionSDI() ...")
t_start = time.time()
sim_sdi = ReceptionSDI(tx, rx, fs=FS, c=C, verbose=False)
rf_sdi, coords_sdi = sim_sdi.scattered_rf(field_points_mm, per_scatterer=True)
t_sdi = time.time() - t_start
print(f" Done in {t_sdi:.2f} s")

# ============================================================================
# STEP 3: PREPARE IMAGES
# ============================================================================
# rf_*.shape = (N_lateral, Nt, 1) — mono-element → (Nt, N_lateral)
rf_naive_img = rf_naive[:, :, 0].T / rf_naive[:, :, 0].max()
rf_sdi_img = rf_sdi[:, :, 0].T / rf_sdi[:, :, 0].max()

t_us_naive = (
    coords_naive["t0"] + np.arange(rf_naive_img.shape[0]) * coords_naive["dt"]
) * 1e6
t_us_sdi = (coords_sdi["t0"] + np.arange(rf_sdi_img.shape[0]) * coords_sdi["dt"]) * 1e6

env_naive = np.abs(hilbert(rf_naive_img, axis=0))
env_sdi = np.abs(hilbert(rf_sdi_img, axis=0))

peak = env_naive.max() + 1e-30
env_naive_db = to_dB(env_naive / peak, vmin=10 ** (-60 / 20))
env_sdi_db = to_dB(env_sdi / peak, vmin=10 ** (-60 / 20))

# Difference on shortest common time axis.
# Compare the shapes
print(f"\n  rf_naive shape: {rf_naive_img.shape}, t_us_naive length: {len(t_us_naive)}")
print(f"  rf_sdi shape: {rf_sdi_img.shape}, t_us_sdi length: {len(t_us_sdi)}")
T_min = min(rf_naive_img.shape[0], rf_sdi_img.shape[0])
diff_abs = np.abs(rf_naive_img[:T_min] - rf_sdi_img[:T_min])
diff_pct = diff_abs / (np.abs(rf_naive_img[:T_min]).max() + 1e-30) * 100.0
t_us_diff = t_us_naive[:T_min]

max_err_pct = float(diff_pct.max())
print(f"\n  Max |naive − SDI| / peak naive: {max_err_pct:.2f} %")

# ============================================================================
# STEP 4: DISPLAY
# ============================================================================
extent_naive = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_naive[-1], t_us_naive[0]]
extent_sdi = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_sdi[-1], t_us_sdi[0]]
extent_diff = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_diff[-1], t_us_diff[0]]

center_idx = len(X_SCAT_MM) // 2
peak_rf = np.abs(rf_naive_img).max() + 1e-30
peak_row = int(np.argmax(env_naive[:, center_idx]))
peak_row_sdi = min(peak_row, env_sdi_db.shape[0] - 1)

fig, axes = plt.subplots(3, 3, figsize=(16, 8))

# Row 0: raw RF images + raw difference
ax = axes[0, 0]
ax.imshow(
    rf_naive_img / peak_rf,
    aspect="auto",
    extent=extent_naive,
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
ax.set_title(f"Reception naive — raw RF  ({t_naive:.1f} s)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[0, 1]
ax.imshow(
    rf_sdi_img / peak_rf,
    aspect="auto",
    extent=extent_sdi,
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
ax.set_title(f"ReceptionSDI — raw RF  ({t_sdi:.1f} s)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[0, 2]
im = ax.imshow(diff_pct, aspect="auto", extent=extent_diff, cmap="viridis", vmin=0)
plt.colorbar(im, ax=ax, label="%")
ax.set_title(f"|naive − SDI| / peak  (max {max_err_pct:.2f} %)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

# Row 1: envelope dB images + envelope difference
env_diff_db = env_naive_db[:T_min] - env_sdi_db[:T_min]

ax = axes[1, 0]
im = ax.imshow(
    env_naive_db, aspect="auto", extent=extent_naive, cmap="hot", vmin=-60, vmax=0
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("Reception naive — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[1, 1]
im = ax.imshow(
    env_sdi_db, aspect="auto", extent=extent_sdi, cmap="hot", vmin=-60, vmax=0
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("ReceptionSDI — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[1, 2]
vmax_ediff = np.abs(env_diff_db).max()
im = ax.imshow(
    env_diff_db,
    aspect="auto",
    extent=extent_diff,
    cmap="RdBu",
    vmin=-vmax_ediff,
    vmax=vmax_ediff,
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("Envelope difference (naive − SDI, dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

# Row 2: on-axis RF overlay + lateral profile + error vs time
ax = axes[2, 0]
ax.plot(t_us_naive, rf_naive_img[:, center_idx] / peak_rf, label="naive", lw=1.2)
ax.plot(t_us_sdi, rf_sdi_img[:, center_idx] / peak_rf, "--", label="SDI", lw=1.2)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Normalized RF")
ax.set_title(f"On-axis RF  (x = {X_SCAT_MM[center_idx]:.1f} mm)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2, 1]
ax.plot(X_SCAT_MM, env_naive_db[peak_row, :], label="naive", lw=1.2)
ax.plot(X_SCAT_MM, env_sdi_db[peak_row_sdi, :], "--", label="SDI", lw=1.2)
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Envelope (dB)")
ax.set_title("Lateral profile at envelope peak")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2, 2]
ax.semilogy(t_us_diff, diff_pct[:, center_idx] + 1e-6, lw=1.2)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Relative error (%)")
ax.set_title("On-axis relative error vs time")
ax.grid(alpha=0.3)

plt.suptitle(
    f"PSF comparison — concave Ø{DIAMETER_MM:.0f} mm, focus {FOCUS_MM:.0f} mm, "
    f"z_scat = {SCATTERER_Z_MM} mm"
)
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "concave_psf_comparison.png"), dpi=150)
    print(f"Saved to {FIG_FOLDER / 'concave_psf_comparison.png'}")

plt.show()
