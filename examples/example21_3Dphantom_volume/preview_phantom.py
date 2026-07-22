"""Preview the selected scenario's phantom BEFORE spending simulation hours.

Left/middle: echogenicity truth at y=0 (tubes + wire positions marked) and at
the sphere's elevation. Right: C-plane at the tier depth. Then two 3-D scenes:

1. the setup — probe drawn in the transmit colour + scatterer cloud (faded by
   amplitude) + the virtual sources (same first view as
   ``visualize_beamformed_volume``), to check the acquisition geometry;
     "shapes" → the true target geometry as translucent solids (cyan =
                anechoic void, gold = x4 hyperechoic, red = the PSF wires).

Run with (pick the scenario in step 1 or via the SCENARIO env var):
    uv run examples/example21_rca_volume/preview_phantom.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from step1_define_phantom_TX_RX import (
    FIG_DIR,
    FS,
    SC,
    SCENARIO,
    C,
    build_phantom,
    excitation,
    phantom_map,
)

from pyfield.plotting.plotting_pyvista import add_transducer_mesh
from pyfield.reception import ReceptionSDI

# "bmode"  = phantom truth as a gray volume (sigmoid opacity, whole volume).
# "shapes" = the target geometry drawn as translucent solids.
SAVE_FIG = True
SCALE = 5
# High-resolution screenshots: scale window size + fonts by SCALE when saving.
scale = SCALE if SAVE_FIG else 1
FIG_DIR.mkdir(parents=True, exist_ok=True)
VOLUME = SC["volume"]
DB_RANGE = 25  # display range of the gray "bmode" phantom volume
VOL_STRIDE = 2  # step down the volume for the semi-transparent render only
(sx, sy, sz), sr, _ = SC["spheres"][0]

# --- truth slices ---------------------------------------------------------------
shape = (220, 140, 100)
truth = phantom_map(VOLUME, shape, SC["tubes"], SC["spheres"])
x = np.linspace(*VOLUME["x_extent"], shape[0])
y = np.linspace(*VOLUME["y_extent"], shape[1])
z = np.linspace(*VOLUME["z_extent"], shape[2])
iy0 = np.argmin(np.abs(y))
iys = np.argmin(np.abs(y - sy))
iz0 = np.argmin(np.abs(z - SC["tier_z"]))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
ext_xz = [x[0], x[-1], z[-1], z[0]]
for ax, iy, title in [
    (axs[0], iy0, "y = 0 (tubes + wires)"),
    (axs[1], iys, f"y = {sy:+.0f} mm (sphere centred)"),
]:
    ax.imshow(
        truth[:, iy, :].T,
        origin="upper",
        cmap="gray",
        vmin=0,
        vmax=2,
        extent=ext_xz,
        aspect="equal",
    )
    for zw in SC["wire_z"]:
        ax.plot(SC["wire_x"], zw, "r+", ms=12, mew=2)
    ax.set(title=title, xlabel="x (mm)", ylabel="z (mm)")
# The axial wire (along z at y=0) crosses the lateral wires: dotted line in
# the y=0 panel only.
# axs[0].axvline(SC["wire_x"], color="r", lw=0.8, ls=":")
axs[2].imshow(
    truth[:, :, iz0].T,
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=2,
    extent=[x[0], x[-1], y[0], y[-1]],
    aspect="equal",
)
# axs[2].plot(SC["wire_x"], 0, "r+", ms=12, mew=2)
axs[2].axvline(SC["wire_x"], color="r", lw=0.6, ls=":")
axs[2].set(title=f"C-plane z = {SC['tier_z']:g} mm", xlabel="x (mm)", ylabel="y (mm)")
fig.suptitle(f"'{SCENARIO}' contrast-ladder phantom (wires marked red)")
plt.savefig(
    str(FIG_DIR / f"ex21_{SCENARIO}_phantom_preview.png"),
    dpi=150 * scale,
    bbox_inches="tight",
)
print(f"saved {FIG_DIR / f'ex21_{SCENARIO}_phantom_preview.png'}")
if not SAVE_FIG:
    plt.show()

# --- 3-D scene 1: setup — probe (TX colour) + cloud + virtual sources ------------
pos, amp = build_phantom(SC)
print(f"{pos.shape[0]:,d} scatterers")
probe = SC["make_probe"](SC["fc"])
sim = ReceptionSDI(
    probe,
    SC["make_probe"](SC["fc"]),
    c=C,
    fs=FS,
    excitation=excitation(SC["fc"]),
    verbose=False,
)
a = amp / amp.max()  # normalise to 1 for the opacity mapping

plotter = sim.show(
    pos,
    a,
    TX_color="blue",
    RX_color="blue",
    legend=False,
    point_size=4 * scale,
    clim=[-DB_RANGE, 0],
    window_size=[450, 850],  # sim.show scales this by `scale` internally
    scale=scale,
    off_screen=SAVE_FIG,
    return_plotter=True,
    show_scalar_bar=True,
)
plotter.show_grid(
    xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", font_size=10 * scale
)
# The virtual sources are geometric points behind the aperture (z < 0) from
# which each transmitted wavefront diverges — drawn solid to check the layout.
for vs in SC["vs_mm"]:
    plotter.add_mesh(pv.Sphere(radius=0.4, center=vs), color="crimson")
plotter.view_xz()
plotter.camera.up = (0.0, 0.0, -1.0)
plotter.camera.azimuth = 30
plotter.camera.elevation = -15
# position for zeus
plotter.camera_position = [
    (40.63782866542591, 131.19717103289818, 49.756366861746656),
    (-0.8574287778690786, 2.7018981147572134, -1.231694034516682),
    (-0.0002597194219193105, 0.36890467737773347, -0.9294671976754493),
]

if SAVE_FIG:
    plotter.show(screenshot=str(FIG_DIR / f"ex21_{SCENARIO}_phantom_setup.png"))
    print(f"saved {FIG_DIR / f'ex21_{SCENARIO}_phantom_setup.png'}")
else:
    plotter.show()
    print(plotter.camera_position)


# --- 3-D scene 2: the phantom, "bmode" volume or "shapes" solids -----------------
plotter = pv.Plotter(window_size=[380 * scale, 780 * scale], off_screen=SAVE_FIG)
add_transducer_mesh(probe.get_mesh(), plotter=plotter, color="blue", scale=scale)

# Wireframe cube marking the extent of the beamformed volume.
plotter.add_mesh(
    pv.Box(bounds=(*VOLUME["x_extent"], *VOLUME["y_extent"], *VOLUME["z_extent"])),
    style="wireframe",
    color="black",
    line_width=scale,
)

# The true target geometry as translucent solids over a faint scatterer
# cloud: cyan = anechoic void, gold = x4 hyperechoic, red = the PSF wires.
# plotter.add_mesh(
#     pv.PolyData(pos),
#     color="white",
#     point_size=1.5 * scale,
#     opacity=0.06,
#     show_scalar_bar=False,
# )
y_len = np.ptp(VOLUME["y_extent"])
for (cx, cz), r, gain in SC["tubes"]:
    cyl = pv.Cylinder(center=(cx, 0, cz), direction=(0, 1, 0), radius=r, height=y_len)
    plotter.add_mesh(
        cyl,
        color="cyan" if gain == 0 else "gold",
        opacity=0.25 if gain == 0 else 0.5,
    )
plotter.add_mesh(pv.Sphere(radius=sr, center=(sx, sy, sz)), color="gold", opacity=0.7)
for zw in SC["wire_z"]:
    plotter.add_mesh(
        pv.Cylinder(
            center=(SC["wire_x"], 0, zw),
            direction=(0, 1, 0),
            radius=0.08,
            height=y_len,
        ),
        color="red",
    )
# The virtual sources are geometric points behind the aperture (z < 0) from
# which each transmitted wavefront diverges — drawn solid to check the layout.
for vs in SC["vs_mm"]:
    plotter.add_mesh(pv.Sphere(radius=0.4, center=vs), color="tab:orange")
out_name = f"ex21_{SCENARIO}_phantom_cloud.png"
camera_position = [
    (101.5990890436286, 87.60383065018787, -66.54554346152524),
    (-5.145975499997907, -1.7513725932338637, -7.797357875559309),
    (-0.266881958903478, -0.2862124408688282, -0.9202480419450778),
]

plotter.show_grid(
    xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", font_size=10 * scale
)
# plotter.camera_position = "yz"
plotter.camera.azimuth = 35
plotter.camera.elevation = -20
plotter.camera.up = (0.0, 0.0, -1.0)  # ty: ignore[unresolved-attribute]
plotter.camera_position = camera_position
if SAVE_FIG:
    plotter.show(screenshot=str(FIG_DIR / out_name))
    print(f"saved {FIG_DIR / out_name}")
else:
    plotter.show()
    print(plotter.camera_position)

del plotter, cyl
