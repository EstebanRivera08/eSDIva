"""
Example 24: 3-D point PSF of the paper's diverging-multilens matrix (their Fig. 2b).

Example 23 fitted a single rigid cap to the diverging multilens element of Nardone et al.
(Nat. Commun. 2025, s41467-025-64911-z). That paper builds a large 3-D imaging probe by
tiling 252 of these elements on an 18x14 grid at 5.48 mm pitch, driven at 1 MHz, and shows
the volumetric point-spread function measured in water (their Fig. 2b). The PSF there is
NOT a focused transmit: it is the ultrafast localization-microscopy sequence — 16 single
central elements each fire a diverging wave in turn, and the 16 echoes are compounded on
receive. Because every element diverges, a single element already floods the whole volume,
so 16 of them synthesise a tight focus everywhere.

This example reproduces that: it assembles the matrix with `CustomTransducer` (252 copies
of the example-23 free-aspheric cap), fires the 16-source diverging sequence at one point
target with `sequence_rf`, delay-and-sums the compounded echoes into a 3-D volume with
`das_volume`, and renders it the way the paper does — grayscale, log-compressed (dB), with
a sigmoid opacity transfer function so the main lobe is opaque and the background fades out.
The 5.48 mm pitch is 3.6 lambda at 1 MHz, far above lambda/2, so grating lobes are a real,
expected feature of this sparse multilens design, not a simulation artefact. The render box
widens the paper's PSF volume to 100 x 100 mm laterally (0-60 mm depth, scatterer at
(0, 0, 40)), wide enough to show the full grating-lobe ring.

What is reproduced faithfully is the array-level signature: the peak grating lobe here sits
at about -23 dB, matching the paper's measured -22.4 dB (this level is set by the sparse
geometry and is independent of depth). The -6 dB main lobe here is about 2.5 mm, roughly
twice the paper's quoted 1.06 mm. That gap is NOT a knob this example gets wrong: sweeping
the receive F-number, the receive apodization (rect vs Hann), the element subdivision (and
hence its directivity), the transmit source placement (central cluster vs spread across the
aperture), and the target depth all leave the main lobe near 2.5 mm. Its near-constancy with
depth shows it is set by the single-element diverging transmit and the element directivity,
not by classic array-aperture diffraction. The residual factor of two lives in the paper's
"Beamforming processing", which is not given in the open-access text, so this example
reproduces the measurable array physics (grating-lobe level and geometry) without claiming
to match the raw main-lobe width.

The diverging cap is reused straight from example 23 (same fit), so a change to the lens
there flows through here. 252 curved caps x 16 transmits is heavy: keep `no_sub_diameter`
small, and preview the geometry with `sim.show()` before the simulation.

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
TARGET_Z_MM = 40.0  # depth of the point target whose PSF we measure
N_SOURCES = 16  # paper's ultrafast sequence: 16 central single-element diverging waves
NO_SUB = 20  # patches per element diameter: 10 under-resolves the cap (its -6 dB directivity
# half-angle comes out 28 deg instead of the paper's 33 deg), 20 recovers the 33 deg
# directivity. It barely changes the PSF width here, but keeps the element physically faithful.
# 252 curved caps x 16 TX is heavy, so this is the smallest subdivision that stays faithful.
DB_FLOOR = -40.0  # display floor (grating lobes sit near -22 dB in the paper)

# One diverging cap, reused from example 23 (the free-aspheric fit), then tiled.
element = optimized_aspheric(no_sub_diameter=NO_SUB)
xs = (np.arange(N_X) - (N_X - 1) / 2) * PITCH_MM
ys = (np.arange(N_Y) - (N_Y - 1) / 2) * PITCH_MM
gx, gy = np.meshgrid(xs, ys)
positions_mm = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
n_el = positions_mm.shape[0]
print(
    f"Matrix: {n_el} elements ({N_X}x{N_Y}), aperture "
    f"{N_X * PITCH_MM:.0f} x {N_Y * PITCH_MM:.0f} mm, {FC / 1e6:.0f} MHz"
)

# Separate TX and RX arrays (same geometry): each event drives ONE TX element (one-hot
# apodization), while every RX element records. sequence_rf applies the event's TX weights,
# so TX and RX must be distinct objects or the one-hot would also gate the receive traces.
tx = CustomTransducer([element] * n_el, positions_mm, frequency_Hz=FC)
rx = CustomTransducer([element] * n_el, positions_mm, frequency_Hz=FC)

# Transmit and receive impulse responses: a 2-cycle 1 MHz burst (bare drive, no explicit
# derivative — the band-limited pulse carries the pulse-echo shaping). Skipping these would
# widen the PSF and raise sidelobe clutter.
t_ir = np.arange(0, 2 / FC, 1 / FS)
burst = (np.sin(2 * np.pi * FC * t_ir) * np.hanning(len(t_ir))).astype(np.float32)
tx.impulse_response = burst
rx.impulse_response = burst
tx.excitation = burst

# The 16 most central elements, one single-element diverging wave each. A single small
# element radiates a spherical wave from its own centre, so its virtual source sits ON the
# aperture (z = 0). Compounding the 16 echoes synthesises the focus.
central = np.argsort(np.hypot(positions_mm[:, 0], positions_mm[:, 1]))[:N_SOURCES]
tx_events = []
for e in central:
    apod = np.zeros(n_el, dtype=np.float32)
    apod[e] = 1.0
    tx_events.append(
        {
            "delays": np.zeros(n_el, dtype=np.float32),
            "apodization": apod,
            "virtual_source_mm": [positions_mm[e, 0], positions_mm[e, 1], 0.0],
        }
    )

# Single on-axis point target — the PSF probe.
scat_mm = np.array([[0.0, 0.0, TARGET_Z_MM]], dtype=np.float32)
scat_amp = np.array([1.0], dtype=np.float32)

sim = Reception(tx, rx, fs=FS, c=C0, verbose=False)

# Preview the geometry before the heavy run (headless screenshot when saving figures).
setup_path = str(FIG_FOLDER / "ex24_setup.png") if SAVE_FIG else None
if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
plotter = sim.show(scat_mm, scat_amp, off_screen=SAVE_FIG, save_path=setup_path)
if not SAVE_FIG:
    plotter.show()

print(f"Simulating {N_SOURCES}-source diverging sequence ({n_el} RX each) ...")
rf, coords = sim.sequence_rf(scat_mm, scat_amp, tx_events)  # (N_SOURCES, Erx, Nt)


def fwhm_6db(axis_mm, line_db):
    """-6 dB width (mm) of a PSF cut, by linear interpolation around the peak."""
    peak = int(np.argmax(line_db))
    below = line_db <= -6.0
    left = np.where(below[:peak])[0]
    right = np.where(below[peak:])[0]
    if not left.size or not right.size:
        return float("nan")

    def cross(i, j):  # linear interp of the -6 dB level between samples i and j
        f = (-6.0 - line_db[i]) / (line_db[j] - line_db[i])
        return axis_mm[i] + f * (axis_mm[j] - axis_mm[i])

    il = left[-1]  # last sample below -6 dB left of the peak
    ir = peak + right[0]  # first sample below -6 dB right of the peak
    return float(cross(ir, ir - 1) - cross(il, il + 1))


# Delay-and-sum the 16 single-element events coherently. The receive aperture uses a RECT
# (boxcar) weighting, not a Hann taper: each multilens element already has a broad but
# tapering directivity, so the echoes from the aperture edges are attenuated by the element
# itself. A Hann taper on top would taper twice and widen the main lobe ~2x; rect lets the
# physical element directivity set the effective aperture, matching the paper's processing.
# fnum 0.5 opens the aperture to +-z (near the full 99 mm array at 40 mm depth), i.e. the
# tightest focus the matrix supports.
RX_APOD, FNUM = "rect", 0.5

# Big render volume: the paper's PSF box widened to 100 x 100 mm laterally (full 0-60 mm
# depth, scatterer at (0, 0, 40)) so the grating-lobe ring is fully inside the frame. Cost is
# dominated by this volume (voxels x RX channels x events), so 1 mm voxels keep it ~1 s: the
# ray-cast renderer interpolates them smooth, and the ~1 mm-scale main lobe is measured on
# the fine 0.1 mm grid below, not here.
grid_mm = {
    "x_extent": [-50, 50],
    "y_extent": [-50, 50],
    "z_extent": [0, 60],
    "dx": 1.0,
    "dy": 1.0,
    "dz": 1.0,
}
vol, axes = das_volume(
    rf, coords, tx_events, tx, grid_mm, c=C0, fnum=FNUM, rx_apodization=RX_APOD
)
env = np.abs(hilbert(vol, axis=2))  # envelope along depth (z = axis 2)
db = 20 * np.log10(np.clip(env / env.max(), 10 ** (DB_FLOOR / 20), None))
x_mm, y_mm, z_mm = axes["x_mm"], axes["y_mm"], axes["z_mm"]
iz0 = int(np.argmin(np.abs(z_mm - TARGET_Z_MM)))

# Fine xz plane through the target (0.1 mm voxels, y = 0): the main lobe is ~1 mm, far below
# the render grid, so resolution and sidelobe levels are read here. Envelope-detected along
# depth; the lateral cut at z = 40 mm gives the lateral PSF (and the full grating-lobe
# ladder out to +-50 mm), the axial cut at x = 0 gives the axial PSF.
fine_mm = {
    "x_extent": [-50, 50],
    "y_extent": [0.0, 0.1],
    "z_extent": [TARGET_Z_MM - 8, TARGET_Z_MM + 8],
    "dx": 0.1,
    "dy": 0.1,
    "dz": 0.1,
}
fvol, fax = das_volume(
    rf, coords, tx_events, tx, fine_mm, c=C0, fnum=FNUM, rx_apodization=RX_APOD
)
fenv = np.abs(hilbert(fvol, axis=2))
fdb = 20 * np.log10(np.clip(fenv / fenv.max(), 10 ** (DB_FLOOR / 20), None))
xf, zf = fax["x_mm"], fax["z_mm"]
jz = int(np.argmin(np.abs(zf - TARGET_Z_MM)))
ix = int(np.argmin(np.abs(xf - 0.0)))
lat_db = fdb[:, 0, jz]  # lateral PSF profile at the target depth
axi_db = fdb[ix, 0, :]  # axial PSF profile on-axis
lat = fwhm_6db(xf, lat_db)
axi = fwhm_6db(zf, axi_db)

# Peak grating-/side-lobe level: the strongest point outside the main lobe (|x| > 3 mm).
gl = float(lat_db[np.abs(xf) > 3.0].max())
print(f"PSF -6 dB widths: lateral {lat:.2f} mm, axial {axi:.2f} mm")
print(f"Peak grating/side lobe: {gl:.1f} dB   (paper: 1.06 mm lateral, -22.4 dB lobes)")

# Fig. 2b-style render: grayscale, dB, volume-rendered with a sigmoid opacity transfer
# function so the main lobe is opaque and the -40 dB background fades to transparent. Dark
# theme (black background, white box/ruler) with the z axis pointing DOWN, the ultrasound
# depth convention, so the probe is at the top and the target deeper below it.
import pyvista as pv  # noqa: E402 — deferred so a headless import failure is obvious

img = pv.ImageData(
    dimensions=db.shape,
    spacing=(grid_mm["dx"], grid_mm["dy"], grid_mm["dz"]),
    origin=(float(x_mm[0]), float(y_mm[0]), float(z_mm[0])),
)
img["dB"] = db.flatten(order="F")  # ImageData point data is x-fastest (Fortran order)
pl = pv.Plotter(off_screen=SAVE_FIG, window_size=(900, 800))
pl.set_background("black")
# "smart" mapper picks a ray-cast volume renderer (smooth) over the pixelated fixed-point
# one; outline + mm ruler make the 100 x 100 x 60 mm box size readable.
pl.add_volume(
    img,
    scalars="dB",
    cmap="gray",
    opacity="sigmoid",
    clim=[DB_FLOOR, 0.0],
    mapper="smart",
    scalar_bar_args={"title": "dB", "color": "white"},  # colorbar text white on black
)
pl.add_mesh(img.outline(), color="white", line_width=2)
pl.show_grid(xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)", color="white")
pl.add_axes(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
pl.camera_position = "yz"
pl.camera.azimuth = 35
pl.camera.elevation = 20
pl.camera.up = (0.0, 0.0, -1.0)  # z points down (depth), probe at top

# Quantitative view: a fine xz slice, the xy grating-lobe ring, and the PSF profiles with
# the -6 dB resolution marked.
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].pcolormesh(
    xf, zf, fdb[:, 0, :].T, cmap="gray", shading="gouraud", vmin=DB_FLOOR, vmax=0
)
ax[0].set(xlabel="x (mm)", ylabel="z (mm)", title="xz slice")
ax[0].set_xlim(-25, 25)
ax[0].invert_yaxis()
ax[0].set_aspect("equal")
m1 = ax[1].pcolormesh(
    x_mm, y_mm, db[:, :, iz0].T, cmap="gray", shading="gouraud", vmin=DB_FLOOR, vmax=0
)
ax[1].set(
    xlabel="x (mm)", ylabel="y (mm)", title=f"xy slice at z = {TARGET_Z_MM:.0f} mm"
)
ax[1].set_aspect("equal")
fig.colorbar(m1, ax=ax[1], label="dB")
ax[2].plot(xf, lat_db, label=f"lateral ({lat:.2f} mm)")
ax[2].plot(zf - TARGET_Z_MM, axi_db, label=f"axial ({axi:.2f} mm)")
ax[2].axhline(-6.0, color="0.6", ls="--", lw=1)
ax[2].set(
    xlabel="offset from target (mm)",
    ylabel="dB",
    title="intensity profiles",
    xlim=(-25, 25),
    ylim=(DB_FLOOR, 2),
)
ax[2].legend(loc="upper right")
fig.tight_layout()

if SAVE_FIG:
    render_path = str(FIG_FOLDER / "ex24_matrix_psf.png")
    pl.screenshot(render_path)
    pl.close()
    slices_path = str(FIG_FOLDER / "ex24_psf_slices.png")
    fig.savefig(slices_path, dpi=130)
    print(f"  Saved: {render_path}")
    print(f"  Saved: {slices_path}")
else:
    pl.show()
    plt.show()
