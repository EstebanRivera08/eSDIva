"""
Example 17: Import a Field II Transducer and check it matches PyField

Builds the native PyField ``Domino`` probe and the SAME probe exported from
MATLAB Field II, then overlays the two apertures to confirm PyField imports
Field II geometry patch-for-patch.

Generate the Field II export first (needs MATLAB + the Field II toolbox)::

    >> run examples/fieldiiexamples/domino_fieldii.m   % writes domino_fieldii.mat

That script rebuilds the Domino probe with ``xdc_focused_array`` and saves
``rect = xdc_get(Th, 'rect')``.  This example loads it with
``from_fieldii_rect_data`` and compares the imported patch mosaic against
``pyfield.transducers.Domino()``.

Steps
-----
1. Build the native PyField Domino probe
2. Load + import the Field II export (skipped with a hint if the .mat is absent)
3. Compare the two patch mosaics numerically (sorted patch centres)
4. Overlay both apertures in PyVista + simulate the imported probe's CW field

Run with:
    uv run examples/example17_import_fieldii_transducer.py
"""

import gc
from pathlib import Path

import numpy as np
import scipy.io
from config import FIG_FOLDER, SAVE_FIG

from pyfield.emission import Emission
from pyfield.plotting import plot2D_pressure_slices
from pyfield.transducers import Domino, from_fieldii_rect_data
from pyfield.utilities import explore_mat

# ============================================================================
# CONFIGURATION
# ============================================================================
MAT_FILE = Path(__file__).parent / "fieldiiexamples" / "domino_fieldii.mat"
ELEVATION_FOCUS_MM = 8.0  # Domino cylindrical-lens focal length (Field II Rfocus)

# XZ plane for the CW field of the imported probe. Domino patches are 0.108 x
# 0.15 mm, so the far-field SIR is valid within a few mm — start the grid at 5.
PLANE = {
    "x_extent": [-8, 8],
    "y_extent": [0, 0],
    "z_extent": [5, 40],
    "dx": 0.15,
    "dy": 0,
    "dz": 0.25,
}


def patch_centers(t) -> np.ndarray:
    """Return the (M, 3) centres of every rectangular patch, sorted for compare.

    Each patch is a flat rectangle tiling the aperture; its centre is the mean
    of the four corner vertices.  Sorting lexicographically makes the array
    independent of element/patch ordering so two transducers built by
    different tools can be compared row-by-row.
    """
    c = np.array([q.mean(axis=0) for q in t.sub_quad_verts])
    return c[np.lexsort((c[:, 2], c[:, 1], c[:, 0]))]


print("\n--- Example 17: Import a Field II Transducer ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: NATIVE PYFIELD DOMINO
# ============================================================================
native = Domino()
print(f"Native PyField : {native}")

# ============================================================================
# STEP 2: IMPORT THE FIELD II EXPORT
# ============================================================================
if not MAT_FILE.exists():
    print(
        f"\n{MAT_FILE.name} not found. Generate it in MATLAB first:\n"
        f"    >> run {MAT_FILE.parent / 'domino_fieldii.m'}\n"
        "Showing the native Domino aperture only.\n"
    )
    imported = None
else:
    data = scipy.io.loadmat(MAT_FILE, simplify_cells=True)
    explore_mat(data, name=MAT_FILE.name)
    f0 = float(data["f0"])
    imported = from_fieldii_rect_data(
        data["rect"], frequency_hz=f0, elevation_focus_mm=ELEVATION_FOCUS_MM
    )
    print(f"Imported Field II: {imported}")

    # ------------------------------------------------------------------
    # STEP 3: NUMERIC COMPARISON — do the two patch mosaics coincide?
    # ------------------------------------------------------------------
    cn, ci = patch_centers(native), patch_centers(imported)
    if cn.shape != ci.shape:
        print(
            f"\nPatch count differs: native {cn.shape[0]} vs Field II "
            f"{ci.shape[0]} — check no_sub_x/no_sub_y in domino_fieldii.m."
        )
    else:
        max_diff_mm = np.abs(cn - ci).max() * 1e3
        print(f"\nMax patch-centre difference: {max_diff_mm:.4f} mm")
        assert max_diff_mm < 0.2, (
            f"Imported aperture deviates {max_diff_mm:.3f} mm from native Domino."
        )
        print("PASS: imported Field II aperture matches native Domino.")

# ============================================================================
# STEP 4: OVERLAY THE TWO APERTURES + SIMULATE THE IMPORTED CW FIELD
# ============================================================================
import pyvista as pv  # noqa: E402

pl = pv.Plotter(off_screen=SAVE_FIG, window_size=(1600, 1200))
pl.add_mesh(native.get_mesh(), color="royalblue", show_edges=True, label="PyField")
if imported is not None:
    pl.add_mesh(
        imported.get_mesh(),
        color="orange",
        style="wireframe",
        line_width=2,
        label="Field II",
    )
pl.add_legend()
pl.add_axes()
pl.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
if SAVE_FIG:
    pl.screenshot(str(FIG_FOLDER / "ex17_fieldii_import_mesh.png"))
else:
    pl.show()
pl.close()

if imported is not None:
    sim = Emission(imported, monochromatic=True)
    p, coords = sim(PLANE, method="auto")
    plot2D_pressure_slices(
        p,
        coords=coords,
        db_scale=True,
        vmin=-40,
        title="Imported Domino (Field II) — CW field",
        save_path=str(FIG_FOLDER) if SAVE_FIG else None,
        file_name="ex17_fieldii_import_cw.png",
    )

# Finalise every PyVista object while the interpreter is still alive: close the
# render windows, drop the last references and force a collection now, so VTK's
# __del__ does not fire during shutdown ("Exception ignored ... meta_path is None").
pv.close_all()
del pl
gc.collect()

print("\nDone.")
