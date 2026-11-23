"""
Advanced — A cross-amplitude-modulation (xAM) sound sheet from a row-column probe.

Nonlinear sound-sheet microscopy (Heiles et al., Science 2025) images microbubbles
and gas vesicles in a thin "sound sheet" swept through a volume by a row-column
addressed (RCA) probe. Its transmit sequence is built only from linear pressure
fields and arithmetic on them, which is exactly what `Field` does:

1. An RCA keeps 2N channels instead of N² by crossing N long "column" bars with N
   "row" bars on one face (`RowColumnAddressedTransducer`). Here the columns fire.
2. A sub-aperture is split in two halves that emit plane waves tilted toward each
   other at +alpha and -alpha (an "X-wave"). They cross along the bisector plane.
3. xAM fires three pulses: TX1 both halves, TX2 the left half only, TX3 the right
   half only. Tissue scatters linearly, so its echoes obey TX1 - TX2 - TX3 = 0 and
   cancel. A bubble's echo also grows with the SQUARE of the local pressure, so
   (p_L + p_R)² - p_L² - p_R² = 2·p_L·p_R survives — and only where BOTH plane-wave
   pulses are present at the same instant: a thin sheet around the bisector.

eSDIva simulates the three LINEAR pressure fields. The quadratic step is a MODEL of
the bubble response (the leading nonlinear term), applied point by point with
`Field` arithmetic — no nonlinear propagation is simulated. Maps are RMS over time
(√⟨·²⟩): sampled above Nyquist, the time-integrated energy is exact even when the
peak of a 2f product would fall between samples.

Then the sheet is swept the way the paper does it, by sliding the sub-aperture.
The medium is homogeneous and the bars identical, so a slid aperture radiates the
same field translated: `Field.transform` moves the simulated sheet instead of
re-simulating it.

Probe and sequence follow the paper's order of magnitude (pitch ≈ lambda at
15.6 MHz, 2-cycle bursts, alpha ≈ 20°); the sub-aperture size is chosen for a
quick run.

Run with:
    uv run examples/advanced/rca_xam_sound_sheet.py
"""

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_FOLDER, SAVE_FIG
from esdiva import Emission, Field
from esdiva.transducers import RowColumnAddressedTransducer
from esdiva.utilities import to_dB, tone_burst

C, FC, FS = 1540.0, 15.6e6, 100e6
ALPHA_DEG = 20.0
N_HALF = 24  # elements per half-aperture
SHIFT = 8  # offset of the sub-aperture from the array centre, in elements

rca = RowColumnAddressedTransducer(
    n_elements=64,
    element_width_mm=0.09,
    kerf_mm=0.01,  # pitch 0.1 mm ≈ lambda (0.099 mm)
    no_sub_x=1,
    no_sub_y=32,  # 6.4 mm bars cut into 0.2 mm patches (far field from ~1 mm)
    frequency_Hz=FC,
)
cols = rca.columns
x_el = cols.element_centers[:, 0]

burst = tone_burst(FC, n_cycles=2, fs=FS)

# XZ plane through the bar centres (y = 0). dx = 0.02 mm resolves a ~0.1 mm sheet
# and makes the slid aperture (8 x 0.1 mm) an exact number of grid steps.
grid = {
    "x_extent": [-2, 2],
    "y_extent": [0, 0],
    "z_extent": [0.5, 8],
    "dx": 0.02,
    "dy": 0,
    "dz": 0.05,
}


def xwave(first):
    """Delays and half-aperture masks of an X-wave from elements first..first+2N-1.

    Each half is a plane wave tilted toward the bisector (delay ∝ the element's
    projection on the propagation direction, x·sin(±alpha)/c). The halves are
    mirror images, so their wavefronts meet ON the bisector together.
    """
    left = np.zeros(cols.n_elements, bool)
    right = np.zeros(cols.n_elements, bool)
    left[first : first + N_HALF] = True
    right[first + N_HALF : first + 2 * N_HALF] = True
    s = np.sin(np.deg2rad(ALPHA_DEG))
    delays = np.zeros(cols.n_elements)
    delays[left] = x_el[left] * s / C  # left half steers toward +x
    delays[right] = -x_el[right] * s / C  # right half steers toward -x
    delays[left] -= delays[left].min()
    delays[right] -= delays[right].min()
    return delays, left, right


def emit(delays, mask):
    """Signed transient pressure of the elements in ``mask`` as a `Field`.

    Delays are the same for every pulse of one X-wave and only the apodization
    changes, so all pulses share one time axis and subtract sample by sample.
    """
    cols.delays = delays
    cols.apodization = mask.astype(float)
    p, coords = Emission(cols, fs=FS, c=C, excitation=burst, verbose=False)(grid)
    return Field(p, coords)


def rms(field):
    """RMS over time of a transient `Field` → one (Nx, 1, Nz) map."""
    return Field(np.sqrt(np.mean(field.data**2, axis=0)), field.coords)


# ── xAM: three linear pulses, combined point by point ─────────────────────────
first = (cols.n_elements - 2 * N_HALF) // 2 - SHIFT // 2
delays, left, right = xwave(first)
tx1 = emit(delays, left | right)  # both halves
tx2 = emit(delays, left)
tx3 = emit(delays, right)
assert tx1.coords["t0"] == tx2.coords["t0"] == tx3.coords["t0"]

linear_residual = tx1 - tx2 - tx3  # tissue: cancels by superposition
xam = tx1**2 - tx2**2 - tx3**2  # quadratic scatterer: 2·p_L·p_R survives
res = np.abs(linear_residual.data).max() / np.abs(tx1.data).max()
print(f"linear residual TX1-TX2-TX3: {res:.1e} of peak (float32 round-off)")

lin_map, xam_map = rms(tx1), rms(xam)
x, z = lin_map.coords["x"], lin_map.coords["z"]
x_mid = 0.5 * (x_el[first + N_HALF - 1] + x_el[first + N_HALF]) * 1e3  # bisector, mm


off = np.abs(x - x_mid) > 0.6  # outside the crossing zone at mid depth
iz = np.argmin(np.abs(z - 3.0))
lin_off, xam_off = (
    20 * np.log10(m.data[off, 0, iz].max() / m.data[:, 0, iz].max())
    for m in (lin_map, xam_map)
)
print(
    f"off-sheet peak at 3 mm (|x - bisector| > 0.6 mm): linear {lin_off:.1f} dB, "
    f"xAM {xam_off:.1f} dB"
)

# ── Sweep the sheet: move it instead of re-simulating the slid aperture ───────


def shift_x(mm):
    T = np.eye(4)
    T[0, 3] = mm
    return T


# Sliding sweep, 4 elements per step. Sheets are fired apart, so they add as POWER.
sweep = sum(
    (xam_map**2).transform(shift_x(k * 4 * rca.pitch_mm)).resample(xam_map)
    for k in range(-3, 4)
)

# ── Figure ────────────────────────────────────────────────────────────────────
ext = [x[0], x[-1], z[-1], z[0]]
panels = [
    ("Linear TX1, RMS |p| (dB)", to_dB(lin_map.data[:, 0, :])),
    ("xAM 2·p_L·p_R, RMS (dB)", to_dB(xam_map.data[:, 0, :])),
    ("xAM sheet swept, 7 positions (dB)", to_dB(sweep.data[:, 0, :]) / 2),
]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
for ax, (title, img) in zip(axes, panels):
    im = ax.imshow(img.T, extent=ext, aspect="auto", cmap="inferno", vmin=-30, vmax=0)
    ax.set(title=title, xlabel="x (mm)", ylabel="z (mm)")
    fig.colorbar(im, ax=ax, shrink=0.8)
for m, label in ((lin_map, "linear TX1"), (xam_map, "xAM")):
    axes[3].plot(x, to_dB(m.data[:, 0, iz]), label=label)
axes[3].set(
    title="Lateral profile at z = 3 mm", xlabel="x (mm)", ylabel="dB", ylim=(-40, 1)
)
axes[3].legend()
axes[3].grid(alpha=0.3)

if SAVE_FIG:
    FIG_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_FOLDER / "rca_xam_sound_sheet.png", dpi=150)
else:
    plt.show()
