"""
3-D visualization of the beamformed volume (any scenario from step 1).

Loads the compounded complex IQ volume saved by step 3
(``out/<scenario>/IQ/iq_volume.npz``) — no beamforming happens here, so the
figures re-render in seconds. Three figures, saved to ``figures/<scenario>/``
when ``SAVE_FIG`` is on (interactive windows otherwise). The acquisition
setup (probe + cloud + virtual sources) lives in ``preview_phantom.py``.

1. ``ex21_<scenario>_volume_3d.png`` — the beamformed B-mode volume rendered in
                                  gray with sigmoid opacity (bright speckle
                                  opaque, cyst hollow)
2. ``ex21_<scenario>_mpr_3d.png``    — orthogonal cut planes through the targets +
                                  a −6 dB iso-surface of the bright bodies
3. ``ex21_<scenario>_slices.png``    — phantom-truth vs image slice pairs (wires
                                  marked red), one shared dB colorbar

Run with (pick the scenario in step 1 or via the SCENARIO env var):
    uv run examples/example21_3Dphantom_volume/visualize_beamformed_volume.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from config import SAVE_FIG
from step1_define_phantom_TX_RX import (
    FIG_DIR,
    IQ_DIR,
    SC,
    SCENARIO,
    phantom_map,
)

from pyfield.plotting.plotting_pyvista import add_transducer_mesh
from pyfield.plotting.pyvista_functions import create_3Dvol_mesh
from pyfield.utilities import to_dB

GRID = SC["grid"]
CUT = (SC["tubes"][0][0][0], 0.0, SC["tier_z"])  # MPR planes through the cyst


def _add_volume_box(plotter, x_mm, y_mm, z_mm, color="black"):
    """Wireframe cube marking the extent of the beamformed volume."""
    box = pv.Box(bounds=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]))
    plotter.add_mesh(box, style="wireframe", color=color, line_width=scale)
    del box


SCALE = 4
SVG_FIG = False  # save vector graphics (SVG) instead of PNG (docs embed PNG)
# SAVE_FIG = True
DB_RANGE = 30
# Volume ray-casting cost scales with voxel count; the full grid (~1 M+ voxels)
# is stepped down for the semi-transparent render only (slices stay full-res).
VOL_STRIDE = 2
# High-resolution screenshots: multiply window size + font sizes by SCALE when
# saving (SCALE=1 keeps interactive windows light).
scale = SCALE if SAVE_FIG else 1
if SAVE_FIG:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

FONTSIZE = 19
plt.rcParams.update({"font.size": FONTSIZE})


def _finish(plotter, fname):
    """Save at high resolution (SAVE_FIG) or open the interactive window."""
    if SAVE_FIG:
        plotter.show(screenshot=str(FIG_DIR / fname))
        print(f"saved {FIG_DIR / fname}")
    else:
        plotter.show()

    print(plotter.camera_position)


# ============================================================================
# LOAD THE COMPOUNDED IQ VOLUME (step 3's output) + speckle TGC — shared by 1–3
# ============================================================================
probe = SC["make_probe"](SC["fc"])  # mesh only (probe drawn in the 3-D scenes)
with np.load(IQ_DIR / "iq_volume.npz") as d:
    iq = d["iq"]
    x_mm, y_mm, z_mm = d["x_mm"], d["y_mm"], d["z_mm"]
# Depth-only TGC from the lateral median envelope, wire columns masked out
# (a wire spans the full elevation and biases the median at its depths; the
# tubes cover well under half the width, which the median ignores). Same
# recipe as step 3; the wire mask scales with the lateral PSF λ·z/D.
psf_mm = 1540.0 / SC["fc"] * 1e3 * SC["tier_z"] / SC["aperture_mm"]
no_wire_cols = np.abs(x_mm - SC["wire_x"]) > max(0.5, 1.5 * psf_mm)
prof = np.median(np.abs(iq[no_wire_cols]), axis=(0, 1))
tgc = prof.max() / gaussian_filter1d(prof, sigma=2.0 / GRID["dz"]).clip(
    min=prof.max() * 1e-3
)
env_db = to_dB(np.abs(iq) * tgc[None, None, :])

# ============================================================================
# FIGURE 1 — the beamformed B-mode volume, gray + sigmoid opacity
# ============================================================================
s = VOL_STRIDE
vol_mesh = create_3Dvol_mesh(
    np.clip(env_db[::s, ::s, ::s], -DB_RANGE, 0.0),
    x_mm[::s],
    y_mm[::s],
    z_mm[::s],
    scalars="Envelope (dB)",
)
plotter = pv.Plotter(window_size=[850 * scale, 950 * scale], off_screen=SAVE_FIG)
add_transducer_mesh(probe.get_mesh(), plotter=plotter, color="blue", scale=scale)
_add_volume_box(plotter, x_mm, y_mm, z_mm)
plotter.add_volume(
    vol_mesh,
    scalars="Envelope (dB)",
    mapper="smart",
    cmap="binary",
    opacity="sigmoid_7",
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "dB",
        "title_font_size": 20 * scale,
        "label_font_size": 20 * scale,
        "position_x": 0.3,
        "position_y": 0.05,
    },
)
# plotter.show_grid(
#     xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", font_size=13 * scale
# )
plotter.camera_position = "yz"
plotter.camera.azimuth = 40
plotter.camera.elevation = -25
plotter.camera.up = (0.0, 0.0, -1.0)  # ty: ignore[unresolved-attribute]
plotter.camera_position = [
    (13.378040491070358, 44.161316083419806, -10.470568079647542),
    (-0.5859471497780473, 1.4651943972742822, 9.805476223963389),
    (-0.12900800212517702, -0.3906497673463856, -0.9114547134443298),
]
_finish(plotter, f"ex21_{SCENARIO}_volume_3d.png")

# ============================================================================
# FIGURE 2 — MPR: cut planes through the targets + −6 dB iso-surface
# ============================================================================
igrid = pv.ImageData(
    dimensions=env_db.shape,
    spacing=(x_mm[1] - x_mm[0], y_mm[1] - y_mm[0], z_mm[1] - z_mm[0]),
    origin=(x_mm[0], y_mm[0], z_mm[0]),
)
igrid.point_data["dB"] = np.clip(env_db, -DB_RANGE, 0.0).flatten(order="F")
plotter = pv.Plotter(window_size=[750 * scale, 950 * scale], off_screen=SAVE_FIG)
add_transducer_mesh(probe.get_mesh(), plotter=plotter, color="blue", scale=scale)
_add_volume_box(plotter, x_mm, y_mm, z_mm)
sl = igrid.slice_orthogonal(x=CUT[0], y=CUT[1], z=CUT[2])
plotter.add_mesh(
    sl,
    cmap="gray",
    clim=[-DB_RANGE, 0],
    scalar_bar_args={
        "title": "dB",
        "title_font_size": 16 * scale,
        "label_font_size": 14 * scale,
        "position_x": 0.3,
        "position_y": 0.05,
    },
)
plotter.add_mesh(igrid.contour([-6.0], scalars="dB"), color="gold", opacity=0.35)
plotter.show_grid(
    xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", font_size=13 * scale
)
plotter.camera_position = "yz"
plotter.camera.azimuth = 35
plotter.camera.elevation = -20
plotter.camera.up = (0.0, 0.0, -1.0)  # ty: ignore[unresolved-attribute]
plotter.camera_position = [
    (49.257390450467156, 45.65136551981425, -16.808795747374276),
    (0.5175521500055128, 1.2820914515172719, 13.170394487026305),
    (-0.3007284715333519, -0.2846701998928786, -0.9102336313838126),
]
_finish(plotter, f"ex21_{SCENARIO}_mpr_3d.png")

# ============================================================================
# FIGURE 3 — phantom truth vs image: y=0, elevation slice, C-plane
# ============================================================================
truth = phantom_map(
    {k: GRID[k] for k in ("x_extent", "y_extent", "z_extent")},
    (x_mm.size, y_mm.size, z_mm.size),
    SC["tubes"],
    SC["spheres"],
)
iy0 = np.argmin(np.abs(y_mm))
iys = np.argmin(np.abs(y_mm - SC["slice_y"]))
iz0 = np.argmin(np.abs(z_mm - CUT[2]))


def _mark_wires(ax, plane):
    """Overlay the PSF wires in red — the lateral wires run along y at
    x=wire_x, so they appear as points in an x–z panel and as a vertical
    line in the x–y C-plane; the axial wire (along z at y=0) is the dotted
    vertical line in the y=0 panel only."""
    if plane.startswith("xz"):
        for zw in SC["wire_z"]:
            ax.plot(SC["wire_x"], zw, "r+", ms=12, mew=2)
    else:  # xy C-plane
        ax.axvline(SC["wire_x"], color="r", lw=1.5, ls=":")


ext_xz = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
ext_xy = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
cuts = [
    ("y = 0", lambda v: v[:, iy0, :].T, ext_xz, "x (mm)", "z (mm)", "upper", "xz0"),
    (
        f"y = {SC['slice_y']:+.0f} mm",
        lambda v: v[:, iys, :].T,
        ext_xz,
        "x (mm)",
        "z (mm)",
        "upper",
        "xz",
    ),
    (
        f"z = {CUT[2]:.0f} mm",
        lambda v: v[:, :, iz0].T,
        ext_xy,
        "x (mm)",
        "y (mm)",
        "lower",
        "xy",
    ),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
for j, (title, cut, ext, xl, yl, orig, plane) in enumerate(cuts):
    axs[0, j].imshow(
        cut(truth), origin=orig, cmap="gray", vmin=0, vmax=2, extent=ext, aspect="equal"
    )
    axs[0, j].set(title=f"phantom · {title}", xlabel=xl, ylabel=yl)
    m = axs[1, j].imshow(
        cut(env_db),
        origin=orig,
        cmap="gray",
        vmin=-DB_RANGE,
        vmax=0,
        extent=ext,
        aspect="equal",
    )
    axs[1, j].set(title=f"image · {title}", xlabel=xl, ylabel=yl)
    _mark_wires(axs[0, j], plane)
    # _mark_wires(axs[1, j], plane)
# One shared dB colorbar (the image row); the phantom row is a 0–2 truth map,
# so a single dB bar to the right is enough to read the whole figure.
fig.colorbar(
    m,
    ax=axs.ravel().tolist(),
    location="right",
    shrink=0.6,
    label="dB",
)
if SAVE_FIG:
    if SVG_FIG:
        plt.savefig(
            str(FIG_DIR / f"ex21_{SCENARIO}_slices.svg"),
            dpi=150 * scale,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            str(FIG_DIR / f"ex21_{SCENARIO}_slices.png"), dpi=150, bbox_inches="tight"
        )
    plt.close()
    ext = "svg" if SVG_FIG else "png"
    print(f"saved {FIG_DIR / f'ex21_{SCENARIO}_slices.{ext}'}")
else:
    plt.show()


del plotter, igrid, sl, vol_mesh
