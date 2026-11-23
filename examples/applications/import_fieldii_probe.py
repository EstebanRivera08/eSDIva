"""Application — Reuse a Field II probe in eSDIva.

Field II describes an aperture as a list of small rectangles. In MATLAB,
`xdc_get(Th, 'rect')` exports them; `from_fieldii_rect_data` turns that table into an
eSDIva transducer, so a probe already defined in Field II needs no rewriting. The
export used here (`fieldii/domino_fieldii.mat`, made by the `.m` script next to it)
is the same 128-element linear array as eSDIva's `LinearArrayExample`.

You will see: both apertures overlaid (they coincide), then the CW field of the
imported probe.
"""

# %% Imports
from pathlib import Path

import pyvista as pv
import scipy.io

from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices
from esdiva.transducers import LinearArrayExample, from_fieldii_rect_data

# %% Load the Field II export and import it
data = scipy.io.loadmat(
    Path(__file__).parent / "fieldii" / "domino_fieldii.mat", simplify_cells=True
)
# The rectangle table has no lens radius, so pass the elevation focus explicitly.
imported = from_fieldii_rect_data(
    data["rect"], frequency_hz=float(data["f0"]), elevation_focus_mm=8.0
)
native = LinearArrayExample()

# %% Overlay: native (blue) and imported (orange wireframe)
plotter = pv.Plotter()
plotter.add_mesh(native.get_mesh(), color="royalblue", show_edges=True, label="eSDIva")
plotter.add_mesh(
    imported.get_mesh(), color="orange", style="wireframe", label="Field II"
)
plotter.add_legend()
plotter.show()

# %% Simulate the imported probe like any other eSDIva transducer
xz_plane = {
    "x_extent": [-8, 8],
    "y_extent": [0, 0],
    "z_extent": [5, 40],
    "dx": 0.15,
    "dy": 0,
    "dz": 0.25,
}
p, coords = Emission(imported, monochromatic=True)(xz_plane)
plot2D_pressure_slices(
    p, coords=coords, db_scale=True, title="Imported Field II probe, CW"
)
