"""
Example 09: Linear Array — B-mode PSF Image via DAS Beamforming

Field II parallel: ``fieldiiexamples/linear_psf_example/`` (sesr.m scenario)

Simulates a PSF phantom (20 on-axis point scatterers at 5 mm intervals,
z = 15–110 mm) and builds a B-mode line by line with
``ReceptionSDI.scan_focusline`` (one focused scan line per lateral position),
then log-compresses to a B-mode image.

  1. PSF phantom: 20 on-axis scatterers, z = 15, 20, …, 110 mm
  2. 128-element linear array, 3 MHz, λ-pitch, Hanning-windowed IR on TX and RX
  3. Per line: scan_focusline recomputes TX focus + apodization at z = 60 mm,
     simulates the pulse-echo RF, DAS-beamforms (RX focus = same point), and
     returns the Hilbert envelope.
  4. Map each line's time axis to display depth → global log-compression.

Field II parallel: this is the conventional line-by-line acquisition Field II's
psf example performs with ``calc_scat`` filling ``image_data(:,i)``.

Run with:
    uv run examples/example09_lineararray_imagePSF.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG

from pyfield.reception import ReceptionSDI
from pyfield.transducers import LinearArrayTransducer

# ============================================================================
# CONFIGURATION  (matches fieldiiexamples/linear_psf_example/field.m)
# ============================================================================
C = 1540.0  # speed of sound (m/s)
FS = 100e6  # sampling frequency (Hz)
F0 = 3e6  # centre frequency (Hz)
PULSE_CYCLES = 2
SAVE_FIG = True
FOVERD = 2
APOD_TYPE = "hanning"

N_ELEMENTS = 128  # total physical elements
N_ACTIVE = 64  # active aperture per scan line

LAMBDA = C / F0  # wavelength (m)
ELEMENT_WIDTH_MM = LAMBDA * 1e3  # ≈ 0.513 mm (λ-pitch element width)
ELEMENT_HEIGHT_MM = 5.0  # mm
KERF_MM = 0.1  # mm
PITCH_MM = ELEMENT_WIDTH_MM + KERF_MM

Z_FOCUS_MM = 60.0  # fixed TX/RX focal depth (mm)

# Scan-line grid: 20 lines over ±10 mm (matches field.m no_lines/image_width)
NO_LINES = 20
IMAGE_WIDTH_MM = 20.0
X_LINES_MM = np.linspace(-IMAGE_WIDTH_MM / 2, IMAGE_WIDTH_MM / 2, NO_LINES)

# PSF phantom: 20 on-axis scatterers at z = 15, 20, …, 110 mm (pts_pha.m)
N_SCAT = 20
SCATTERER_POS = np.column_stack(
    [
        np.zeros(N_SCAT),
        np.zeros(N_SCAT),
        (np.arange(1, N_SCAT + 1) * 5.0 + 10.0),  # z = 15…110 mm
    ]
).astype(np.float32)
SCATTERER_AMP = np.ones(N_SCAT, dtype=np.float32)

print("\n--- Example 09: Linear Array — B-mode PSF Image ---\n")
print("Field II parallel: linear_psf_example/sesr.m")
print(
    f"Phantom: {N_SCAT} on-axis scatterers, z = {SCATTERER_POS[:, 2].min():.0f}–"
    f"{SCATTERER_POS[:, 2].max():.0f} mm"
)
print(f"Scan lines: {NO_LINES}, focus at z = {Z_FOCUS_MM} mm")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LINEAR ARRAY (128 elements, λ-pitch, 3 MHz)
# ============================================================================
tx = LinearArrayTransducer(
    n_elements=N_ELEMENTS,
    element_width_mm=ELEMENT_WIDTH_MM,
    element_height_mm=ELEMENT_HEIGHT_MM,
    kerf_mm=KERF_MM,
    no_sub_x=1,
    no_sub_y=4,
    frequency_Hz=F0,
)
rx = tx.copy()

# 2-cycle Hanning-windowed pulse — used as IR for TX and RX, and as TX excitation
t_pulse = np.arange(0, PULSE_CYCLES / F0, 1.0 / FS)
pulse = (np.sin(2 * np.pi * F0 * t_pulse) * np.hanning(len(t_pulse))).astype(np.float32)

tx.impulse_response = pulse
rx.impulse_response = pulse
excitation = pulse.copy()

# ============================================================================
# STEP 2-4: SCAN — one focused line per lateral position via scan_focusline
# ============================================================================
# `scan_focusline` recomputes the TX focus + apodization for each focal point,
# simulates the pulse-echo RF, DAS-beamforms the line (RX focus = same point) and
# returns its envelope — the conventional line-by-line acquisition, exactly as a
# scanner builds a B-mode (cf. Field II's psf example filling image_data(:,i)).
# `coords["t0"]` is beam-axis referenced inside scan_focusline, so no manual bulk
# correction is needed; we only map each line's time axis to display depth.
sim = ReceptionSDI(tx, rx, c=C, fs=FS, excitation=excitation)
common_depth_mm = np.arange(9.0, 120.0 + 0.05, 0.05)  # shared display axis
env_lines = []
for x in X_LINES_MM:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env_line, coords = sim.scan_focusline(
            [float(x), 0.0, Z_FOCUS_MM],
            SCATTERER_POS,
            SCATTERER_AMP,
            FoverD=FOVERD,
            apodization_type=APOD_TYPE,
        )
    depth_line_mm = (
        coords["t0"] + np.arange(len(env_line)) * coords["dt"]
    ) * C / 2 * 1e3
    env_lines.append(
        np.interp(common_depth_mm, depth_line_mm, env_line, left=0, right=0)
    )

# Global log-compression (60 dB dynamic range, like mk_img.m)
env_matrix = np.stack(env_lines, axis=1)  # (N_depth, N_lines)
global_max = env_matrix.max()
bmode = 20.0 * np.log10(np.maximum(env_matrix / global_max, 10 ** (-60 / 20)))

# ============================================================================
# STEP 5: DISPLAY B-MODE IMAGE (depth 9–120 mm, matching mk_img.m axis limits)
# ============================================================================
fig, ax = plt.subplots(figsize=(4, 10))
im = ax.imshow(
    bmode,
    aspect="equal",
    extent=[X_LINES_MM[0], X_LINES_MM[-1], common_depth_mm[-1], common_depth_mm[0]],
    cmap="gray",
    vmin=-60,
    vmax=0,
)
plt.colorbar(im, ax=ax, label="dB")
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Depth (mm)")
ax.set_title(
    "B-mode PSF image (mask) \n"
    f"DAS focus at z={Z_FOCUS_MM} mm\n"
    f"{NO_LINES} lines × {N_ACTIVE}/{N_ELEMENTS} act. elems.\n"
    f"3 MHz with apod = '{APOD_TYPE}'"
)

# Mark true scatterer positions
for s in SCATTERER_POS:
    if common_depth_mm[0] <= s[2] <= common_depth_mm[-1]:
        ax.plot(s[0], s[2], "r+", markersize=8, markeredgewidth=1.5)

plt.tight_layout()

if SAVE_FIG:
    figname = f"bmode_psf_FD{FOVERD}_{APOD_TYPE}.png"
    plt.savefig(str(FIG_FOLDER / figname), dpi=150)
    print(f"\nSaved to {FIG_FOLDER / figname}")

plt.show()
