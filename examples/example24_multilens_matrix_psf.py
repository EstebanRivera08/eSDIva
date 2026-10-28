"""
Example 24: Point PSF of the paper's diverging-multilens matrix.

Example 23 fitted a single rigid cap to the diverging multilens element of Nardone et al.
(Nat. Commun. 2025, s41467-025-64911-z). That paper builds a large 3-D imaging probe by
tiling 252 of these elements on an 18x14 grid at 5.48 mm pitch, driven at 1 MHz, and shows
the point-spread function of the whole matrix (their Fig. 2). This example assembles that
matrix in eSDIva with `CustomTransducer` — 252 copies of the example-23 free-aspheric cap —
focuses it on a single point scatterer, simulates the pulse-echo RF, and delay-and-sums it
back into a 3-D PSF. Because each element diverges, the matrix sees a wide solid angle, so
a modest-count array still forms a tight focal spot.

The diverging cap is reused straight from example 23 (same fit), so a change to the lens
there flows through here. 252 curved caps is heavy: keep `no_sub_diameter` small, and
preview the geometry with `sim.show()` before the simulation.

Run with:
    uv run python examples/example24_multilens_matrix_psf.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from example23_optimize_radial_element import optimized_aspheric
from scipy.signal import hilbert

from esdiva.beamforming import das_volume
from esdiva.reception import Reception
from esdiva.transducers import CustomTransducer

C0, FC, FS = 1540.0, 1e6, 50e6
N_X, N_Y, PITCH_MM = 18, 14, 5.48  # paper's 18x14 matrix on a 5.48 mm grid
FOCUS_Z_MM = 40.0  # transmit focal depth of the point PSF
NO_SUB = 12  # patches per element diameter — small: 252 curved caps is heavy

# One diverging cap, reused from example 23 (the free-aspheric fit), then tiled.
element = optimized_aspheric(no_sub_diameter=NO_SUB)
xs = (np.arange(N_X) - (N_X - 1) / 2) * PITCH_MM
ys = (np.arange(N_Y) - (N_Y - 1) / 2) * PITCH_MM
gx, gy = np.meshgrid(xs, ys)
positions_mm = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
print(
    f"Matrix: {positions_mm.shape[0]} elements "
    f"({N_X}x{N_Y}), aperture {N_X * PITCH_MM:.0f} x {N_Y * PITCH_MM:.0f} mm, {FC / 1e6:.0f} MHz"
)

# Separate TX and RX arrays (same geometry). Only TX is focused; reception stays unfocused
# (per-element RF), so the receive focal law lives in the beamformer, not the aperture.
tx = CustomTransducer([element] * positions_mm.shape[0], positions_mm, frequency_Hz=FC)
rx = CustomTransducer([element] * positions_mm.shape[0], positions_mm, frequency_Hz=FC)

# Transmit and receive impulse responses: a 2-cycle 1 MHz burst (bare drive, no explicit
# derivative — the band-limited pulse carries the pulse-echo shaping). Skipping these would
# widen the PSF and raise sidelobe clutter.
t_ir = np.arange(0, 2 / FC, 1 / FS)
burst = (np.sin(2 * np.pi * FC * t_ir) * np.hanning(len(t_ir))).astype(np.float32)
tx.impulse_response = burst
rx.impulse_response = burst
tx.excitation = burst

tx.compute_delays(focus_mm=[0, 0, FOCUS_Z_MM])
tx.compute_apodization(focus_mm=[0, 0, FOCUS_Z_MM])

# Single on-axis point target at the transmit focus — the PSF probe.
scat_mm = np.array([[0.0, 0.0, FOCUS_Z_MM]], dtype=np.float32)
scat_amp = np.array([1.0], dtype=np.float32)

sim = Reception(tx, rx, fs=FS, c=C0, verbose=False)

# Preview the geometry before the heavy run (headless screenshot when saving figures).
setup_path = str(FIG_FOLDER / "ex24_setup.png") if SAVE_FIG else None
if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
plotter = sim.show(scat_mm, scat_amp, off_screen=SAVE_FIG, save_path=setup_path)
if not SAVE_FIG:
    plotter.show()

print("Simulating pulse-echo RF (252 TX x 252 RX) ...")
rf, coords = sim.pulse_echo_rf(scat_mm, scat_amp)  # (Erx, Nt)

# Delay-and-sum the one focused event into a 3-D PSF volume around the focus. The focused
# transmit is a converging-then-diverging wavefront → a virtual source AT the focus (z>0).
event = {
    "delays": tx.delays,
    "apodization": tx.apodization,
    "virtual_source_mm": [0.0, 0.0, FOCUS_Z_MM],
}
grid_mm = {
    "x_extent": [-6, 6],
    "y_extent": [-6, 6],
    "z_extent": [FOCUS_Z_MM - 12, FOCUS_Z_MM + 12],
    "dx": 0.15,
    "dy": 0.15,
    "dz": 0.15,
}
vol, axes = das_volume(rf[None], coords, [event], tx, grid_mm, c=C0, fnum=0.8)

# Envelope-detect along depth (z = axis 2), log-compress to dB (peak = 0).
env = np.abs(hilbert(vol, axis=2))
db = 20 * np.log10(np.clip(env / env.max(), 1e-4, None))
x_mm, y_mm, z_mm = axes["x_mm"], axes["y_mm"], axes["z_mm"]
ix0, iy0 = len(x_mm) // 2, len(y_mm) // 2
iz0 = int(np.argmin(np.abs(z_mm - FOCUS_Z_MM)))


def fwhm_6db(axis_mm, line_db):
    """−6 dB width (mm) of a PSF cut, by linear interpolation around the peak."""
    peak = int(np.argmax(line_db))
    below = line_db <= -6.0
    left = np.where(below[:peak])[0]
    right = np.where(below[peak:])[0]
    if not left.size or not right.size:
        return float("nan")
    return float(axis_mm[peak + right[0]] - axis_mm[left[-1]])


lat = fwhm_6db(x_mm, db[:, iy0, iz0])
ele = fwhm_6db(y_mm, db[ix0, :, iz0])
axi = fwhm_6db(z_mm, db[ix0, iy0, :])
print(
    f"PSF −6 dB widths: lateral {lat:.2f} mm, elevation {ele:.2f} mm, axial {axi:.2f} mm"
)

# Two orthogonal slices through the focus: axial-lateral (xz) and lateral-elevation (xy).
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
m0 = ax[0].pcolormesh(
    x_mm, z_mm, db[:, iy0, :].T, cmap="inferno", shading="auto", vmin=-40, vmax=0
)
ax[0].set(xlabel="x (mm)", ylabel="z (mm)", title=f"xz PSF (lateral {lat:.2f} mm)")
ax[0].invert_yaxis()
fig.colorbar(m0, ax=ax[0], label="dB")
m1 = ax[1].pcolormesh(
    x_mm, y_mm, db[:, :, iz0].T, cmap="inferno", shading="auto", vmin=-40, vmax=0
)
ax[1].set(
    xlabel="x (mm)",
    ylabel="y (mm)",
    title=f"xy PSF at z={FOCUS_Z_MM:.0f} mm (elev {ele:.2f} mm)",
)
ax[1].set_aspect("equal")
fig.colorbar(m1, ax=ax[1], label="dB")
fig.suptitle(
    f"Diverging-multilens matrix ({N_X}x{N_Y}, {FC / 1e6:.0f} MHz) — point PSF at {FOCUS_Z_MM:.0f} mm"
)
fig.tight_layout()

if SAVE_FIG:
    save_path = str(FIG_FOLDER / "ex24_matrix_psf.png")
    fig.savefig(save_path, dpi=130)
    print(f"  Saved: {save_path}")
else:
    plt.show()
