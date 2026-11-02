"""
Example 24: depth-resolved 3-D point PSF of the diverging-multilens matrix (Fig. 2).

Example 23 fitted a single rigid cap to the diverging multilens element of Haidour et al.
(Nat. Commun. 2025, s41467-025-64911-z). That paper builds a large 3-D imaging probe by
tiling 252 of these elements on an 18x14 grid at 5.48 mm pitch, driven at 1 MHz, and
characterises its point-spread function versus depth (their Fig. 2 / Supplementary Fig. 2).
The sequence is a SYNTHETIC-APERTURE acquisition: the 16 elements of the central 4x4 block
each fire one diverging wave in turn, and EVERY element of the whole probe receives, with
uniform unit apodization (a fixed full aperture, not a focused F-number). The 16 single-
element echoes are then compounded coherently on receive to synthesise a focus everywhere.

The decisive modelling detail is where each lens element sits for the beamformer. A rigid
diverging cap has its apex at z = 0 and its rim recessed to z < 0, but it radiates a
spherical wave whose centre of curvature — the element's VIRTUAL ACOUSTIC SOURCE — lies
BEHIND the aperture (here about -1.8 mm). `RadialProfileTransducer.virtual_source_m`
returns that point. Referencing the delay-and-sum to the aperture plane (z = 0) instead
mis-times every echo by the lens's own curvature and defocuses the 16 compounded transmits,
widening the main lobe. This example references BOTH the transmit virtual source and the
receive element position to that measured virtual-source depth, and removes the constant
`2|z_vs|/c` round-trip pedestal it introduces with `t_offset_s` so the targets still
beamform to their true depth. A small sweep over {aperture plane, rim, virtual source}
prints the resulting main-lobe width, so the choice is evidenced by the simulated field
rather than asserted.

The phantom follows the paper: three point scatterers at 30, 45 and 60 mm depth, at the
centre of a 104 x 70 x 100 mm (x,y,z) volume, so the PSF is measured as a function of depth.
The 5.48 mm pitch is 3.6 lambda at 1 MHz, far above lambda/2, so grating lobes are a real,
expected feature of this sparse multilens design, not a simulation artefact.

252 curved caps x 16 transmits is heavy; if it is too slow, lower `NO_SUB`, coarsen the
render grid, or preview the geometry with `sim.show()` before the simulation.

Run with:
    uv run python examples/example24_multilens_matrix_psf.py
"""

import types

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from esdiva.beamforming import das_volume
from esdiva.reception import Reception
from esdiva.transducers import CustomTransducer
from example23_optimize_radial_element import optimized_aspheric
from scipy.signal import hilbert

C0, FC, FS = 1540.0, 1e6, 50e6
N_X, N_Y, PITCH_MM = 18, 14, 5.48  # paper's 18x14 matrix on a 5.48 mm grid
N_SOURCES = 16  # the central 4x4 block, one diverging transmit each
DEPTHS_MM = [30.0, 45.0, 60.0]  # paper's three scatterer depths
NO_SUB = 20  # patches per element diameter: 10 under-resolves the cap directivity
# (-6 dB half-angle 28 deg vs the paper's 33 deg); 20 recovers it. 252 caps x 16 TX is
# heavy, so this is the smallest subdivision that keeps the element physically faithful.
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

# Lens virtual-source depth (mm, signed; negative = behind the aperture). The cap radiates
# from its centre of curvature, not the z=0 plane, and the beamformer must reference the
# delays there. Also keep the rim depth for the datum sweep below.
Z_VS_MM = float(element.virtual_source_m[2]) * 1e3
rim_mm = float(
    np.array(element.sub_quad_verts).mean(axis=1)[:, 2].min() * 1e3
)  # deepest patch centre
print(f"Lens virtual source z = {Z_VS_MM:.2f} mm (rim at {rim_mm:.2f} mm)")

# Separate TX and RX arrays (same geometry): each event drives ONE TX element (one-hot
# apodization) while every RX element records. sequence_rf applies the event's TX weights,
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

# The 16 central elements = the middle 4x4 block. On an 18x14 grid the 16 centres nearest
# the axis ARE exactly that block: its farthest corner is 2.12 pitches out, the next element
# beyond is 2.55 pitches, so the nearest-16 selection cannot leak outside the 4x4.
central = np.argsort(np.hypot(positions_mm[:, 0], positions_mm[:, 1]))[:N_SOURCES]


def make_events(datum_z_mm):
    """16 one-hot diverging transmits, each referenced to the element's acoustic datum.

    A single small element radiates a spherical wave; `virtual_source_mm` places that
    wave's origin at `datum_z_mm` behind the element (the lens virtual source), so DAS
    models the transmit curvature correctly. `delays`/`apodization` drive the simulation
    (one element at a time); the virtual source is read only by the beamformer."""
    events = []
    for e in central:
        apod = np.zeros(n_el, dtype=np.float32)
        apod[e] = 1.0
        events.append(
            {
                "delays": np.zeros(n_el, dtype=np.float32),
                "apodization": apod,
                "virtual_source_mm": [
                    positions_mm[e, 0],
                    positions_mm[e, 1],
                    datum_z_mm,
                ],
            }
        )
    return events


def bf_geom(datum_z_mm):
    """Receive geometry whose element centres sit at the acoustic datum `datum_z_mm`.

    das_volume reads only `element_centers`, so a lightweight shim lets the beamformer
    place the elements at the lens virtual source without disturbing the simulated
    `tx`/`rx`, whose patch geometry must stay physical."""
    el = positions_mm.copy()
    el[:, 2] = datum_z_mm
    return types.SimpleNamespace(element_centers=el * 1e-3)


def datum_offset_s(datum_z_mm):
    """Remove the constant round-trip pedestal of a datum behind the surface.

    Placing both TX source and RX element at `z_vs < 0` lengthens every path by |z_vs|
    per leg relative to the z=0 surface the RF time axis is referenced to. That 2|z_vs|/c
    pedestal is the same for every element, so subtracting it here restores the true
    depth while keeping the (correct) lens curvature in the lateral focusing."""
    return -2.0 * abs(datum_z_mm) * 1e-3 / C0


# Receive on the WHOLE probe with uniform unit apodization (the paper's sequence): a fixed
# full aperture, every element every event. das_volume gates elements beyond z/(2*fnum) of
# the voxel, so fnum = 0.2 (aperture half-width >= 75 mm at the 30 mm target, beyond the
# 58.6 mm array corner) keeps every element in at every depth. RECT weighting: no focused
# F-number taper, the broad element directivity sets the effective aperture.
RX_APOD, FNUM = "rect", 0.2

# Three unit point targets at the paper's depths, on the axis.
scat_mm = np.array([[0.0, 0.0, z] for z in DEPTHS_MM], dtype=np.float32)
scat_amp = np.ones(len(DEPTHS_MM), dtype=np.float32)

sim = Reception(tx, rx, fs=FS, c=C0, verbose=False)

# Preview the geometry before the heavy run (headless screenshot when saving figures).
setup_path = str(FIG_FOLDER / "ex24_setup.png") if SAVE_FIG else None
if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)
plotter = sim.show(scat_mm, scat_amp, off_screen=SAVE_FIG, save_path=setup_path)
if not SAVE_FIG:
    plotter.show()

print(f"Simulating {N_SOURCES}-source diverging sequence ({n_el} RX each) ...")
# The RF is independent of the beamforming datum (the simulation uses delays/apodization,
# not the virtual source), so it is computed once and re-beamformed for each datum below.
rf, coords = sim.sequence_rf(scat_mm, scat_amp, make_events(Z_VS_MM))


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


def fine_plane(depth_mm, datum_z_mm):
    """Envelope (dB) of a fine 0.1 mm xz plane through one target, for a given datum.

    Returns the lateral axis (mm), the axial axis (mm) and the dB image (Nx, Nz). The
    main lobe is ~1 mm, far below the render grid, so resolution is read on this fine
    grid, which also carries the full grating-lobe ladder out to +-50 mm."""
    fine = {
        "x_extent": [-50, 50],
        "y_extent": [0.0, 0.1],
        "z_extent": [depth_mm - 8, depth_mm + 8],
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.1,
    }
    fvol, fax = das_volume(
        rf,
        coords,
        make_events(datum_z_mm),
        bf_geom(datum_z_mm),
        fine,
        c=C0,
        fnum=FNUM,
        rx_apodization=RX_APOD,
        t_offset_s=datum_offset_s(datum_z_mm),
    )
    fenv = np.abs(hilbert(fvol, axis=2))
    # Clip to -60 dB (below the -40 dB maps) so the lateral profile plot resolves the
    # full side/grating-lobe ladder down to its -59 dB axis; widths/lobe levels unaffected.
    fdb = 20 * np.log10(np.clip(fenv / fenv.max(), 10 ** (-60.0 / 20), None))
    return fax["x_mm"], fax["z_mm"], fdb[:, 0, :]


# Datum sweep: lateral -6 dB width at each depth for the aperture plane, the rim, and the
# virtual source. The virtual-source datum should give the tightest lobe if the lens
# curvature is what defocuses the compound; reported, not assumed.
print("\nLateral -6 dB width (mm) vs beamforming datum:")
print(f"  {'depth':>6} | {'z=0':>7} | {'rim':>7} | {'virt.src':>8}")
for z in DEPTHS_MM:
    widths = []
    for datum in (0.0, rim_mm, Z_VS_MM):
        xf, zf, fdb = fine_plane(z, datum)
        jz = int(np.argmin(np.abs(zf - z)))
        widths.append(fwhm_6db(xf, fdb[:, jz]))
    print(f"  {z:>5.0f} | {widths[0]:>7.2f} | {widths[1]:>7.2f} | {widths[2]:>8.2f}")

# Full per-depth metrics with the chosen virtual-source datum (used for the figure too).
lat_lines, axi_lines, metrics = {}, {}, {}
for z in DEPTHS_MM:
    xf, zf, fdb = fine_plane(z, Z_VS_MM)
    jz = int(np.argmin(np.abs(zf - z)))
    ix = int(np.argmin(np.abs(xf - 0.0)))
    lat_db, axi_db = fdb[:, jz], fdb[ix, :]
    lat_lines[z], axi_lines[z] = (xf, lat_db), (zf, axi_db)
    gl = float(lat_db[np.abs(xf) > 3.0].max())  # peak grating/side lobe
    metrics[z] = (fwhm_6db(xf, lat_db), fwhm_6db(zf, axi_db), gl)
print("\nVirtual-source datum, per depth:")
for z in DEPTHS_MM:
    lat, axi, gl = metrics[z]
    print(
        f"  z={z:>4.0f} mm: lateral {lat:.2f} mm, axial {axi:.2f} mm, lobe {gl:.1f} dB"
    )
print("  (paper: 1.06 mm lateral, grating lobes -22.4 dB)")

# Render volume: the paper's 104 x 70 mm lateral box, framed around the three targets. The
# main lobe is ~1 mm, so 1 mm voxels render it ~1 voxel wide (blocky — the ray-cast mapper
# can only interpolate between the samples it is given). 0.5 mm in x/z puts 2-3 samples across
# the lobe so it renders smooth; elevation (y) stays 1 mm since the spots sit at y~0, so the
# cost (voxels x RX channels x events) is ~4x the 1 mm grid, not 8x.
grid_mm = {
    "x_extent": [-52, 52],
    "y_extent": [-35, 35],
    "z_extent": [20, 70],
    "dx": 0.5,
    "dy": 1.0,  # elevation kept coarse: the spots sit at y~0, so halve the voxel count
    "dz": 0.5,
}
vol, axes = das_volume(
    rf,
    coords,
    make_events(Z_VS_MM),
    bf_geom(Z_VS_MM),
    grid_mm,
    c=C0,
    fnum=FNUM,
    rx_apodization=RX_APOD,
    t_offset_s=datum_offset_s(Z_VS_MM),
)
env = np.abs(hilbert(vol, axis=2))  # envelope along depth (z = axis 2)
db = 20 * np.log10(np.clip(env / env.max(), 10 ** (DB_FLOOR / 20), None))
x_mm, y_mm, z_mm = axes["x_mm"], axes["y_mm"], axes["z_mm"]
iy0 = int(np.argmin(np.abs(y_mm - 0.0)))
iz_mid = int(np.argmin(np.abs(z_mm - DEPTHS_MM[1])))

# Fig. 2-style render: grayscale, dB, volume-rendered with a SIGMOID opacity transfer
# function over a -30 dB display window (lobes opaque, the -30 dB background transparent).
# Dark theme (black background, white box/ruler/colorbar) with the z axis pointing DOWN, the
# ultrasound depth convention, so the probe is at the top and the three targets below it.
RENDER_FLOOR = -30.0  # display window for the volume render (tighter than the -40 dB data)
import pyvista as pv  # noqa: E402 — deferred so a headless import failure is obvious

img = pv.ImageData(
    dimensions=db.shape,
    spacing=(grid_mm["dx"], grid_mm["dy"], grid_mm["dz"]),
    origin=(float(x_mm[0]), float(y_mm[0]), float(z_mm[0])),
)
img["dB"] = db.flatten(order="F")  # ImageData point data is x-fastest (Fortran order)
pl = pv.Plotter(off_screen=SAVE_FIG, window_size=(900, 800))
pl.set_background("black")
pl.add_volume(
    img,
    scalars="dB",
    cmap="gray",
    opacity="sigmoid",
    clim=[RENDER_FLOOR, 0.0],
    mapper="smart",
    scalar_bar_args={"title": "dB", "color": "white"},  # colorbar text white on black
)
pl.add_mesh(img.outline(), color="white", line_width=2)
pl.show_grid(
    xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)", color="white", n_zlabels=4
)
pl.add_axes(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
pl.camera_position = "yz"
pl.camera.azimuth = 35
pl.camera.elevation = 20
pl.camera.up = (0.0, 0.0, -1.0)  # z points down (depth), probe at top

# Quantitative view: the xz slice through the three targets, the lateral PSF profiles at
# each depth (with the -6 dB resolution), and the xy grating-lobe star at the mid target.
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].pcolormesh(
    x_mm, z_mm, db[:, iy0, :].T, cmap="gray", shading="gouraud", vmin=DB_FLOOR, vmax=0
)
ax[0].set(xlabel="x (mm)", ylabel="z (mm)", title="xz slice (y = 0)")
ax[0].set_xlim(-25, 25)
ax[0].invert_yaxis()
ax[0].set_aspect("equal")
for z in DEPTHS_MM:
    xf, lat_db = lat_lines[z]
    lat = metrics[z][0]
    ax[1].plot(xf, lat_db, label=f"z={z:.0f} mm ({lat:.2f} mm)")
ax[1].axhline(-6.0, color="0.6", ls="--", lw=1)
ax[1].set(
    xlabel="x (mm)",
    ylabel="dB",
    title="lateral PSF vs depth",
    xlim=(-50, 50),
    ylim=(-59, 0),
)
ax[1].legend(loc="upper right")
m1 = ax[2].pcolormesh(
    x_mm,
    y_mm,
    db[:, :, iz_mid].T,
    cmap="gray",
    shading="gouraud",
    vmin=DB_FLOOR,
    vmax=0,
)
ax[2].set(
    xlabel="x (mm)", ylabel="y (mm)", title=f"xy slice at z = {DEPTHS_MM[1]:.0f} mm"
)
ax[2].set_aspect("equal")
fig.colorbar(m1, ax=ax[2], label="dB")
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
