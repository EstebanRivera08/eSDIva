"""
Example 25: move and combine simulated volumes on one canvas with `Field`.

The medium is homogeneous, so a beam simulated once can be moved with its probe
instead of re-simulated. `Field` wraps any ``(data, coords)`` volume — a pressure
field, a beamformed PSF, an envelope — and the recommended steps are always the same:

1. wrap each result:            ``Field(p, coords)``
2. pose it with the probe's 4x4: ``.transform(T)``
3. build one common grid:        ``Field.canvas(grid)`` (the same dict `Emission` takes)
4. bring every view onto it:     ``.resample(canvas)``
5. combine point by point:       ``a + b``, ``a - b``, ``a * b``, ``sum(views)``

Three cases, each a question a researcher actually asks:

A. Compound three tilted views. The probe fires at -15, 0, +15 deg at different times,
   so the views add INCOHERENTLY: sum intensities |P|^2, not amplitudes.
B. Check that moving a field equals re-simulating the moved probe — a difference
   map (moved - resimulated) that should be ~0 up to interpolation error.
C. Two-way CW beam. In pulse-echo the transmit and receive apertures act in
   sequence, so the two-way amplitude response is the PRODUCT |P_tx| * |P_rx|:
   narrower main lobe and lower sidelobes than either one alone.

Monochromatic XZ planes keep it to a few seconds. A PSF from `das_volume` drops in
the same way: ``Field(envelope, axes)`` (its ``x_mm``/``y_mm``/``z_mm`` axes are read
as-is).

Run with:
    uv run python examples/example25_field_canvas.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG
from esdiva import Emission, Field
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities import to_dB

FC = 5e6
FOCUS_MM = [0, 0, 30]


def rot_y(deg):
    """4x4 rotation about the elevation (y) axis through the probe face centre."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    T = np.eye(4)
    T[[0, 0, 2, 2], [0, 2, 0, 2]] = [c, s, -s, c]
    return T


def make_probe(FoverD=4.0):
    probe = LinearArrayTransducer(
        n_elements=32,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=2,
        no_sub_y=4,
        frequency_Hz=FC,
    )
    probe.compute_delays(focus_mm=FOCUS_MM)
    probe.compute_apodization(focus_mm=FOCUS_MM, FoverD=FoverD)
    return probe


# One XZ plane (y = 0): rotations about y keep every view in this plane.
grid = {
    "x_extent": [-15, 15],
    "y_extent": [0, 0],
    "z_extent": [2, 45],
    "dx": 0.1,
    "dy": 0,
    "dz": 0.1,
}
canvas = Field.canvas(grid)
# Simulate the source WIDER than the canvas: a tilted view must still cover every
# canvas point, or resample leaves zeros where the rotated grid does not reach.
source_grid = {**grid, "x_extent": [-25, 25], "z_extent": [0.1, 50]}

tx = make_probe()
p, coords = Emission(tx, monochromatic=True)(source_grid)  # |P(r, fc)| in Pa
beam = Field(p, coords)

# A. Incoherent compounding of three tilted views (intensity adds).
views = [
    Field(p**2, coords, scalars="Intensity").transform(rot_y(a)) for a in (-15, 0, 15)
]
compound = sum(v.resample(canvas) for v in views)

# B. Moved field vs re-simulated moved probe.
moved_probe = make_probe()
moved_probe.transform(rot_y(15))
p_resim, c_resim = Emission(moved_probe, monochromatic=True)(grid)
diff = beam.transform(rot_y(15)).resample(canvas) - Field(p_resim, c_resim)
# Compare only where the moved field has data: resampling a field of ones gives
# its coverage mask (canvas points behind the source grid's aperture plane get 0).
covered = Field(np.ones_like(p), coords).transform(rot_y(15)).resample(canvas)
err = np.where(covered.data > 0.999, np.abs(diff.data), np.nan) / p_resim.max()
far = canvas.coords["z"] > 5.0
# Residual = trilinear interpolation on a 0.1 mm (~lambda/3) grid: small in the
# beam, larger in the rapidly oscillating near field; refine dx/dz to shrink it.
print(
    f"B. max |moved - resimulated|: {np.nanmax(err[..., far]):.2%} of peak beyond "
    f"5 mm, {np.nanmax(err[..., ~far]):.2%} in the near field"
)

# C. Two-way beam: TX (F/4, 7.5 mm aperture) times RX (full 9.6 mm aperture), both
# focused at 30 mm, on the canvas grid.
tx_c = beam.resample(canvas)
rx = make_probe(FoverD=1.0)
p_rx, _ = Emission(rx, monochromatic=True)(grid)
two_way = tx_c * Field(p_rx, canvas.coords)


def width_mm(data, z_mm=30.0):
    """-6 dB lateral width of an amplitude plane (Nx, 1, Nz) at depth z."""
    x_mm, z_axis = canvas.coords["x"], canvas.coords["z"]
    iz = np.argmin(np.abs(z_axis - z_mm))
    prof = data[:, 0, iz] / data[:, 0, iz].max()
    return np.ptp(x_mm[prof >= 0.5])


# The two-way response IS the product: its -6 dB lobe is narrower than either alone.
print(
    f"C. -6 dB width at 30 mm: TX {width_mm(tx_c.data):.2f} mm, RX {width_mm(p_rx):.2f} mm, "
    f"two-way {width_mm(two_way.data):.2f} mm"
)

# ── Figure ────────────────────────────────────────────────────────────────────
x, z = canvas.coords["x"], canvas.coords["z"]
ext = [x[0], x[-1], z[-1], z[0]]
panels = [
    ("Single view |P| (dB)", to_dB(tx_c.data[:, 0, :]), (-40, 0)),
    ("A. 3-view intensity compound (dB)", to_dB(compound.data[:, 0, :]) / 2, (-40, 0)),
    ("B. |moved − resimulated| (% peak)", 100 * err[:, 0, :], (0, 3)),
    ("C. two-way |P_tx|·|P_rx| (dB)", to_dB(two_way.data[:, 0, :]), (-60, 0)),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
for ax, (title, img, (lo, hi)) in zip(axes, panels):
    cmap = "viridis" if title.startswith("B") else "inferno"
    im = ax.imshow(img.T, extent=ext, aspect="equal", cmap=cmap, vmin=lo, vmax=hi)
    ax.set(title=title, xlabel="x (mm)", ylabel="z (mm)")
    fig.colorbar(im, ax=ax, shrink=0.8)

if SAVE_FIG:
    fig.savefig(FIG_FOLDER / "example25_field_canvas.png", dpi=150)
else:
    plt.show()
