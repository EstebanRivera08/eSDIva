---
icon: lucide/library
---

# API Reference

---

## `pyfield.transducers`

Transducer geometry classes. All inherit from `TransducerBase` and expose the same patch-based interface.

**Array classes** — `LinearArrayTransducer` · `ConvexArrayTransducer` · `MatrixArrayTransducer`

**Mono-element classes** — `FlatCircularTransducer` · `ConcaveCircularTransducer` · `ConvexCircularTransducer` · `FocusedCircularTransducer`

**Composite** — `CustomTransducer`

**Common methods**

```python
tx.compute_delays(focus_mm=[0, 0, 30])
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)
tx.set_apodization(weights)           # manual patch-wise weights
tx.show(scalars="Apodization")        # interactive 3-D preview
mesh = tx.get_mesh()                  # PyVista PolyData
```

**Key attributes** — `patch_frames` · `delays` · `apodization` · `fc` · `n_elements` · `sub_area`

[Full reference →](transducers.md){ .md-button }

---

## `pyfield.psimulation`

Core simulator. Single callable interface for monochromatic and transient pressure fields.

**Class** — `PyField(transducer, c=1540, rho=1.0, fs=200e6, alpha0=0)`

```python
from pyfield.psimulation import PyField

sim = PyField(tx)

# Monochromatic — returns (x, y, z, p) with p.shape == (Nx, Ny, Nz)
x, y, z, p = sim(field_points, method="auto")

# Transient + excitation — returns (x, y, z, t, p) with p.shape == (Nt, Nx, Ny, Nz)
x, y, z, t, p = sim(field_points, method="auto", excitation=pulse)
```

**`__call__` parameters** — `field_points_mm` · `method` (`"auto"` / `"naive"` / `"sdi"`) · `excitation` · `monochromatic`

[Full reference →](simulation.md){ .md-button }

---

## `pyfield.plotting` — Visualization

Matplotlib (2-D static) and PyVista (3-D interactive) helpers. All PyVista functions accept `plotter=` to compose multiple objects in one scene.

**Matplotlib** — `plot2D_pressure_slices` · `plot2D_planes` · `plot2D_pressure_plane`

**PyVista** — `plot3D_pressure_vol` · `plot3D_pressure_slices` · `add_pressure_vol` · `add_transducer_mesh` · `create_3Dvol_mesh` · `add_regions_mesh` · `add_markers`

```python
from pyfield.plotting import plot2D_pressure_slices, add_pressure_vol, create_3Dvol_mesh

# 2-D orthogonal slices (monochromatic)
plot2D_pressure_slices(p, x=x, y=y, z=z, db_scale=True, vmin=-40)

# Compose a 3-D scene
import pyvista as pv
pl = pv.Plotter()
pl = add_pressure_vol(create_3Dvol_mesh(x, y, z, p), plotter=pl)
pl = add_transducer_mesh(tx.get_mesh(), plotter=pl)
pl.show()
```

[Full reference →](visualization.md){ .md-button }

---

## `pyfield.utilities` — Brain Atlas

BrainGlobe-based atlas integration. Downloads atlas data on first use and registers it into the lab coordinate frame.

**Class** — `BG_Atlas(atlas_name, region_names=None, ...)`

**Key methods** — `transform(T_matrix, inplace)` · `reset_mesh()` · `summary()` · `show_atlases()`

**Key attributes** — `pv_mesh` · `bgatlasToBrain`

```python
from pyfield.utilities import BG_Atlas

atlas = BG_Atlas("whs_sd_rat_39um", region_names=["root", "M1", "S1-hl"])
atlas.transform(T_matrix=T, inplace=True)

# pv_mesh is a dict of region_name → pv.PolyData
from pyfield.plotting import add_regions_mesh
pl = add_regions_mesh(atlas.pv_mesh, plotter=pl, opacity=0.35)
```

**Supported atlases** — `whs_sd_rat_39um` (rat, 39 µm) · `allen_mouse_25um` (mouse, 25 µm)

[Full reference →](brain_atlas.md){ .md-button }
