"""
Example 06: Concave Single-Element Transducer — Pulse-Echo PSF

Field II parallel: ``fieldiiexamples/example_concave_psf.m``

Computes the pulse-echo Point Spread Function (PSF) of a spherically focused
single-element transducer and compares two simulation backends:

- ``Reception(method="fst")`` — conventional Field II-style: sample both
  one-way SIRs, convolve h_tx ⊛ h_rx with the excitation (the pulse-echo ∂³
  is already baked into the band-limited excitation/impulse responses).
- ``Reception()`` — default pulse-echo SDI ("spectral"): the two-way SIR is
  built from the closed-form one-way SIR spectra; the four integrations are
  applied in the Fourier domain (no cumsum), which keeps both methods
  sample-aligned.

Both are the same `Reception` class — only the ``method`` selector differs.

Run with:
    uv run examples/example06_concave_PSF.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from scipy.signal import hilbert

from sondi.reception import Reception
from sondi.transducers import ConcaveCircularTransducer
from sondi.utilities import to_dB

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

print("\n--- Example 06: Concave Transducer — Pulse-Echo PSF (FST vs SDI) ---\n")

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
# The receiving piezo band-passes the echo a second time (drive ⊛ h_tx ⊛ h_rx);
# both backends get the same chain, so the comparison stays exact.
rx.impulse_response = ir

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

print("\n  [1/2] Reception(method='fst') ...")
t_start = time.time()
sim_FST = Reception(tx, rx, fs=FS, c=C, method="fst", verbose=False)
rf_FST, coords_FST = sim_FST.pulse_echo_rf(field_points_mm, per_scatterer=True)
t_FST = time.time() - t_start
print(f" Done in {t_FST:.2f} s")

print("\n  [2/2] Reception() [spectral] ...")
t_start = time.time()
sim_sdi = Reception(tx, rx, fs=FS, c=C, verbose=False)
rf_sdi, coords_sdi = sim_sdi.pulse_echo_rf(field_points_mm, per_scatterer=True)
t_sdi = time.time() - t_start
print(f" Done in {t_sdi:.2f} s")

# ============================================================================
# STEP 3: PREPARE IMAGES
# ============================================================================
# rf_*.shape = (N_lateral, Erx, Nt) — mono-element → (Nt, N_lateral)
rf_FST_img = rf_FST[:, 0, :].T
rf_sdi_img = rf_sdi[:, 0, :].T

time_array = (
    coords_sdi["t0"] + np.arange(rf_FST_img.shape[0]) * coords_sdi["dt"]
) * 1e6  # µs

env_FST = np.abs(hilbert(rf_FST_img, axis=0))
env_sdi = np.abs(hilbert(rf_sdi_img, axis=0))

peak = env_FST.max() + 1e-30
env_FST_db = to_dB(env_FST / peak, vmin=10 ** (-60 / 20))
env_sdi_db = to_dB(env_sdi / peak, vmin=10 ** (-60 / 20))

# Difference on shortest common time axis.
# Compare the shapes
print(f"\n  rf_FST shape: {rf_FST_img.shape}")
print(f"  rf_sdi shape: {rf_sdi_img.shape}")
diff_abs = rf_FST_img - rf_sdi_img
diff_pct = diff_abs / (peak) * 100.0
pos_max_diff = np.unravel_index(np.abs(diff_pct).argmax(), diff_pct.shape)
diff_max = diff_abs[pos_max_diff]
print(f"  Max absolute difference: {diff_max:.2f}, and maxpeak = {peak:.2f}")
print(f"  Max absolute difference at (row, col) = {pos_max_diff}")

max_err_pct = float(diff_pct.max())

# Envelope error is the honest headline: shift-tolerant, reflects PSF agreement
# rather than sub-sample edge jitter in the raw RF.
env_err_pct = float(np.abs(env_FST - env_sdi).max() / (env_FST.max() + 1e-30) * 100.0)
print(f"\n  Max |FST - SDI| / peak (raw, native): {max_err_pct:.2f} %")
print(f"  Max |FST - SDI| / peak (envelope)     : {env_err_pct:.2f} %")

# ============================================================================
# STEP 4: DISPLAY
# ============================================================================
t_us_FST = t_us_sdi = t_us_diff = time_array
extent_FST = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_FST[-1], t_us_FST[0]]
extent_sdi = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_sdi[-1], t_us_sdi[0]]
extent_diff = [X_SCAT_MM[0], X_SCAT_MM[-1], t_us_diff[-1], t_us_diff[0]]

peak_rf = np.abs(rf_FST_img).max() + 1e-30
peak_row = env_FST_db.argmax(axis=0).max()
peak_row_sdi = env_sdi_db.argmax(axis=0).max()
center_idx = len(X_SCAT_MM) // 2
print(f"\n  Peak RF at row {peak_row} (FST), {peak_row_sdi} (SDI)")

fig, axes = plt.subplots(3, 3, figsize=(16, 8))

# Row 0: raw RF images + raw difference
ax = axes[0, 0]

ax.imshow(
    rf_FST_img / peak_rf,
    aspect="auto",
    extent=extent_FST,
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
ax.set_title(f"Reception FST — raw RF  ({t_FST:.1f} s)")
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
ax.set_title(f"Reception [spectral] — raw RF  ({t_sdi:.1f} s)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")
ax.sharex(axes[0, 0])
ax.sharey(axes[0, 0])

ax = axes[0, 2]
im = ax.imshow(diff_pct, aspect="auto", extent=extent_diff, cmap="viridis", vmin=0)
plt.colorbar(im, ax=ax, label="%")
ax.set_title(f"|FST − SDI| / peak, native  (max {max_err_pct:.2f} %)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")
ax.sharex(axes[0, 0])
ax.sharey(axes[0, 0])

# Row 1: envelope dB images + envelope difference
env_diff_db = env_FST_db - env_sdi_db

ax = axes[1, 0]
im = ax.imshow(
    env_FST_db, aspect="auto", extent=extent_FST, cmap="hot", vmin=-60, vmax=0
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("Reception FST — envelope (dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

ax = axes[1, 1]
im = ax.imshow(
    env_sdi_db, aspect="auto", extent=extent_sdi, cmap="hot", vmin=-60, vmax=0
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("Reception [spectral] — envelope (dB)")
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
ax.set_title("Envelope difference (FST − SDI, dB)")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Time (µs)")

# Row 2: on-axis RF overlay + lateral profile + error vs time
ax = axes[2, 0]
ax.plot(t_us_FST, rf_FST_img[:, center_idx], label="FST", lw=1.2)
ax.plot(t_us_sdi, rf_sdi_img[:, center_idx], "--", label="SDI", lw=1.2)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("RF")
ax.set_title(f"On-axis RF  (x = {X_SCAT_MM[center_idx]:.1f} mm)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2, 1]
ax.plot(X_SCAT_MM, env_FST_db[peak_row, :], label="FST", lw=1.2)
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
    f"z_scat = {SCATTERER_Z_MM} mm  "
    f"(raw {max_err_pct:.1f} % | envelope {env_err_pct:.1f} %)"
)
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "ex06_concave_psf_comparison.png"), dpi=150)
    print(f"Saved to {FIG_FOLDER / 'concave_psf_comparison.png'}")

plt.show()
