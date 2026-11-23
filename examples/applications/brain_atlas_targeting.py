"""Application — Aim a focused beam at a rat brain region.

A linear array focuses 8 mm deep; a 3-D rat brain atlas (Waxholm Space rat, from
BrainGlobe) is placed under the probe so you can check which regions the focal spot
reaches: here the primary motor (M1) and hind-limb somatosensory (S1-hl) cortex.

Needs the optional atlas packages: `uv sync --extra atlas` (or
`pip install "esdiva[atlas]"`). The first run downloads the atlas (a few hundred MB).
"""

# %% Imports
import pyvista as pv

from esdiva.emission import Emission
from esdiva.plotting import (
    add_pressure_vol,
    add_regions_mesh,
    add_transducer_mesh,
    create_3Dvol_mesh,
)
from esdiva.transducers import LinearArrayExample
from esdiva.utilities import BG_Atlas

FOCUS_MM = [-1, 0, 8]

# %% Focused field around the target (a small 3-D volume)
tx = LinearArrayExample()  # 128 elements, 12.5 MHz.
tx.compute_delays(focus_mm=FOCUS_MM)
tx.compute_apodization(focus_mm=FOCUS_MM, FoverD=1, apodization_type="rect")

volume = {
    "x_extent": [-1.25, -0.75],
    "y_extent": [-0.5, 0.5],
    "z_extent": [7, 9],
    "dx": 0.0125,
    "dy": 0.025,
    "dz": 0.05,
}
p, coords = Emission(tx, monochromatic=True)(volume)

# %% Rat brain, top of the brain 4.5 mm under the probe
atlas = BG_Atlas("whs_sd_rat_39um", region_names=("root", "M1", "S1-hl"))
atlas.place_under_probe(scale_mm=8.0, gap_mm=4.5, shift_mm=(2.0, -2.0))

# %% 3-D scene: brain regions, probe (colored by delay), focal spot
plotter = pv.Plotter()
add_regions_mesh(
    atlas.pv_mesh,
    plotter=plotter,
    kwargs_dict={
        "root": {"color": "lightgray", "opacity": 0.4},
        "M1": {"color": "green", "opacity": 0.3},
        "S1-hl": {"color": "yellow", "opacity": 0.3},
    },
)
add_transducer_mesh(tx.get_mesh(), plotter=plotter, scalars="Delays")
add_pressure_vol(
    create_3Dvol_mesh(
        p / p.max(), coords["x"], coords["y"], coords["z"], scalars="Pressure"
    ),
    plotter=plotter,
    plot_focal_spot=True,
)
plotter.add_legend(loc="upper left")
plotter.camera.up = (0, 0, -1)  # Depth (+z) points down the screen.
plotter.show()
