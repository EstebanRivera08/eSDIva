"""
Example 17: Import a Field II Transducer

Loads a probe exported from MATLAB Field II and turns it into a native
PyField transducer.  The imported aperture behaves like any other PyField
transducer: it can be focused, visualised, and passed to `Emission` /
`ReceptionSDI`.

MATLAB export (one line after building the aperture)::

    Th   = xdc_linear_array(128, width, height, kerf, 1, 1, [0 0 60]/1000);
    geom = xdc_get(Th, 'rect');
    save('my_probe.mat', 'geom', 'f0', ...);

Steps
-----
1. Inspect the .mat structure with `explore_mat`
2. Import the geometry with `from_fieldii_rect_data`
3. Visualise the imported aperture (delays after electronic focusing)
4. Simulate the CW pressure field of the imported probe

Run with:
    uv run examples/example17_import_fieldii_transducer.py
"""

from pathlib import Path

import scipy.io
from config import FIG_FOLDER, SAVE_FIG

from pyfield.emission import Emission
from pyfield.plotting import plot2D_pressure_slices
from pyfield.transducers import from_fieldii_rect_data
from pyfield.utilities import explore_mat

# ============================================================================
# CONFIGURATION
# ============================================================================
MAT_FILE = (
    Path(__file__).parent
    / "fieldiiexamples"
    / "linear_psf_example"
    / "linear_psf_fieldii.mat"
)
FOCUS_MM = [0, 0, 60]  # electronic focus for the imported array

# XZ simulation plane.  Imported patches keep Field II's subdivision (here
# one 0.5 × 5 mm rectangle per element), so the far-field SIR is only valid
# beyond ~12 mm — start the grid there.
PLANE = {
    "x_extent": [-15, 15],
    "y_extent": [0, 0],
    "z_extent": [15, 100],
    "dx": 0.25,
    "dy": 0,
    "dz": 0.5,
}

print("\n--- Example 17: Import a Field II Transducer ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND INSPECT THE .MAT FILE
# ============================================================================
data = scipy.io.loadmat(MAT_FILE, simplify_cells=True)
explore_mat(data, name=MAT_FILE.name)

f0 = float(data["f0"])
geom = data["geom"]  # (26, M) xdc_get(Th, 'rect') matrix
print(f"\nProbe: {geom.shape[1]} mathematical elements at f0 = {f0 / 1e6:.1f} MHz")

# ============================================================================
# STEP 2: IMPORT THE GEOMETRY
# ============================================================================
# Each Field II mathematical element becomes one PyField patch, so per-element
# delays and apodization from the export are preserved exactly.
tx = from_fieldii_rect_data(geom, frequency_hz=f0)
print(f"Imported: {tx}")

# ============================================================================
# STEP 3: FOCUS AND VISUALISE
# ============================================================================
# The imported transducer is a standard PyField transducer — electronic
# focusing works on the imported element centres.
tx.compute_delays(focus_mm=FOCUS_MM)

if SAVE_FIG:
    # Screenshot path: reuse the reception preview machinery via PyVista.
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.add_mesh(tx.get_mesh(), scalars="Delays", cmap="rainbow", show_edges=True)
    pl.add_axes()
    pl.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
    pl.screenshot(str(FIG_FOLDER / "fieldii_import_mesh.png"))
    pl.close()
else:
    tx.show(scalars="Delays")

# ============================================================================
# STEP 4: SIMULATE THE CW PRESSURE FIELD
# ============================================================================
sim = Emission(tx, monochromatic=True)
p, coords = sim(PLANE, method="auto")

plot2D_pressure_slices(
    p,
    coords=coords,
    db_scale=True,
    vmin=-40,
    title=f"Imported Field II probe — CW field, focus at z = {FOCUS_MM[2]} mm",
    save_path=str(FIG_FOLDER) if SAVE_FIG else None,
    file_name="fieldii_import_cw.png",
)

print("\nDone.")
