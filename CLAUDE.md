# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Anytime a modification is performed in the project update this document in case the
logic changes.

## Project Overview

PyField is a Python acoustic field simulator based on the Tupholme–Stepanishen Spatial 
Impulse Response (SIR) method. It models arbitrary transducer geometries as collections 
of rectangular patches and computes pressure fields via convolution with excitation pulses.

Guidelines are loaded automatically from `.claude/rules/`:
- **coding-guidelines** — code style, testing, commits (always loaded)
- **physics-context** — SIR/SDI theory (loaded when touching `h_sir/`, `psimulation/`, `transducers/`)
- **transducers** — geometry conventions, subdivision, z-convention (loaded when touching `transducers/`)

When the user says **"documentation"** or **"the docs"**, they mean the `docs/` folder (MkDocs site). Update the relevant `.md` file there whenever the corresponding code changes.

## Documentation System

Framework: **Zensical** (MkDocs-based). Config: `zensical.toml` (root).

### Commands
```bash
just serve-docs        # build + serve locally (hot-reload)
just docs              # build only → output in site/
just clean-docs        # remove site/ and .cache/
# or directly:
uv run zensical serve
uv run zensical build
```

### Key files
| File | Purpose |
|------|---------|
| `zensical.toml` | Site config: nav, theme, palette, features |
| `docs/index.md` | Landing page |
| `docs/user-guide/*.md` | Conceptual guides |
| `docs/api/*.md` | API reference |
| `docs/examples/*.md` | Worked examples |
| `docs/contributing.md` | Contributor guide |
| `CHANGELOG.md` | Version history (add to nav in `zensical.toml` if needed) |

### Adding a new page
1. Create `docs/<section>/newpage.md`
2. Add entry to `nav` in `zensical.toml`

### Page frontmatter
```yaml
---
icon: lucide/<icon-name>   # lucide icon shown in nav
---
```

### Theme / styling
Configured under `[project.theme]` in `zensical.toml`:
- **Palette**: `[[project.theme.palette]]` blocks — set `primary`, `accent` colors per scheme
- **Schemes**: `"default"` (light) and `"slate"` (dark)
- **Features**: list of MkDocs Material feature strings

## Development Commands

### Package Management
This project uses `uv` for dependency management:
```bash
# Sync dependencies
uv sync

# Run Python scripts
uv run <script.py>

# Add new dependencies
uv add <package>
```
## Architecture

### Module Structure (Subject to change as project evolves)

The codebase follows a modular architecture with clear separation of concerns. 

1) **`src/pyfield/h_sir/`** — Spatial Impulse Response computation

2) **`src/pyfield/transducers/`** — Transducer geometry definitions

3) **`src/pyfield/psimulation/`** — Pressure field simulation

4) **`src/pyfield/utilities/`** — Helper functions, surface subdivision, brain-atlas integration

5) **`src/pyfield/plotting/`** — Visualization (2D Matplotlib and 3D PyVista)

6) **`src/pyfield/cache/`** — Internal/experimental tools (TorchField, DopplerScan, coordinate transforms)

7) **`src/pyfield/scans/`** — Scanning sequence utilities

IMPORTANT NOTES: 
- The module 1) Is the core computation engine, be careful with modifications. 
- Module 2) will be a module under constant development since new transducers can be created and added,
  so think in generalization since backward compatibility might be important.
- Module 3) will have the principal class used for the API. Must be intuitive, 
consistent, and predictable, minimizing friction for adoption and being robust over versions. 
- The scans module is for personal use, keep it independent of the
rest of the project. 
- Anything labelled or using TorchField is under development and will not be release 
soon, so keep independent and secret.

### Simulation Workflow

1. **Create Transducer**:
   ```python
   from pyfield.transducers import LinearArrayTransducer
   tx = LinearArrayTransducer(
       n_elements=64,
       element_width_mm=0.25,
       element_height_mm=12.0,
       kerf_mm=0.05,
       no_sub_x=2,  # Patch subdivisions in x
       no_sub_y=4,  # Patch subdivisions in y
       frequency_Hz=5e6,
   )
   ```

2. **Configure Delays and Apodization (just for multielement transducers)**:
   ```python
   tx.compute_delays(focus_mm=[0, 0, 30])
   tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)
   ```

3. **Define Field Grid** (all distances in mm):
   ```python
   field_points = {
       "x_extent": [-5, 5],
       "y_extent": [-0.5, 0.5],
       "z_extent": [5, 55],
       "dx": 0.1,
       "dy": 1.0,
       "dz": 0.2,
   }
   ```

4. **Run Simulation**:

   `Emission` is the primary simulation class. `PyField` is a deprecated alias
   (emits `DeprecationWarning`, defaults `monochromatic=True` for backward compat).

   ```python
   from pyfield.psimulation import Emission

   # Mode 1 — Monochromatic CW (returns spatial amplitude at fc)
   sim = Emission(tx, monochromatic=True)
   p, coords = sim(field_points, method="auto")
   # p.shape = (Nx, Ny, Nz), coords = {"x": ..., "y": ..., "z": ...}

   # Mode 2 — Pulsed transient (raw SIR, no excitation)
   sim = Emission(tx)
   p, coords = sim(field_points)
   # p.shape = (Nt, Nx, Ny, Nz), coords includes "t0", "dt"

   # Mode 3 — Global excitation (same pulse for all elements)
   import numpy as np
   f_s = 200e6
   fc = 5e6
   t_exc = np.arange(0, 2 / fc, 1 / f_s)
   excitation = np.sin(2 * np.pi * fc * t_exc)
   sim = Emission(tx, fs=f_s, excitation=excitation)
   p, coords = sim(field_points)

   # Mode 4 — Per-element excitation (shape (L, E))
   exc_per_elem = np.stack([excitation * w for w in weights], axis=1)  # (L, E)
   sim = Emission(tx, fs=f_s, excitation=exc_per_elem)
   p, coords = sim(field_points)
   ```

   **Constructor parameters** (keyword-only after transducer):
   - `c=1540.0` — speed of sound (m/s)
   - `rho=1.0` — medium density (kg/m³)
   - `fs=200e6` — sampling frequency (Hz)
   - `alpha0=None` — attenuation in dB/(MHz^y·cm). `None` = no attenuation.
   - `freq_power=1.0` — power-law exponent y
   - `excitation=None` — `None` / `(L,)` / `(L, E)` float32 array
   - `transfer_function=None` — callable `TF(freq) -> array`, applied in freq domain
     multiplicatively alongside excitation in modes 3 and 4
   - `monochromatic=False` — if True, return CW amplitude at fc
   - `fast_attenuation=False` — if True with alpha0 set, use TX-center distance
     (fast approximation); if False (default), per-element loop uses element-center
     distances (accurate near-field model)
   - `verbose=True`

   **Per-element loop trigger**: activated when
   `(alpha0 is not None and not fast_attenuation) or excitation.ndim == 2`.
   Per-element loop computes h_sir independently per element and accumulates
   in freq domain. Peak memory is O(batch_P × nfft) independent of E.

   **Reconstruct time vector**: `t = coords["t0"] + np.arange(p.shape[0]) * coords["dt"]`

   **Runtime update**:
   ```python
   sim.set("alpha0", 0.5)        # enable attenuation
   sim.set("excitation", pulse)  # change excitation
   sim.set("monochromatic", True)
   ```

   **Return convention**: `Emission.__call__` always returns `(pressure, coords)`.
   - `coords` keys `"x"`, `"y"`, `"z"` for structured grid (dict input); also
     `"t0"`, `"dt"` for transient modes.
   - Structured grid: Monochromatic `p.shape = (Nx, Ny, Nz)`,
     Transient `p.shape = (Nt, Nx, Ny, Nz)`.
   - Raw point array input: Monochromatic `p.shape = (N_points,)`,
     Transient `p.shape = (Nt, N_points)`.

   **Time alignment** for multiple transient simulations:
   ```python
   from pyfield.utilities import align_to_common_time
   pxz, cxz = sim(plane_xz_dict)
   pyz, cyz = sim(plane_yz_dict)
   # Default: pads shorter fields with zeros to cover full time range
   common_t, [pxz_a, pyz_a] = align_to_common_time([(pxz, cxz), (pyz, cyz)])
   # Truncate to overlapping interval only
   common_t, [pxz_a, pyz_a] = align_to_common_time(
       [(pxz, cxz), (pyz, cyz)], align_to_shorter=True
   )
   ```

5. **Visualize**:
   ```python
   from pyfield.plotting import plot2D_pressure_slices
   # works for both monochromatic (3D) and transient (4D) pressure fields
   plot2D_pressure_slices(p, coords=coords, db_scale=True, vmin=-40)
   # or explicit: plot2D_pressure_slices(p, x=coords["x"], y=coords["y"], z=coords["z"])
   ```

   **Transient plane plotting** accepts three input formats:
   ```python
   from pyfield.plotting import plot2D_transient_slices

   # Format 1: old dict (backward compatible)
   plot2D_transient_slices({"xz": pxz_a, "yz": pyz_a}, coords=coords)

   # Format 2: list of dicts with translation (new primary format)
   plot2D_transient_slices([
       {"plane": "xz", "data": pxz_a, "translation": (0, 5.0, 0)},
       {"plane": "yz", "data": pyz_a, "translation": (0, 0, 0)},
   ], coords=coords)

   # Format 3: full 4D volume (slices extracted at center)
   plot2D_transient_slices(p_4d, coords=coords)
   ```

### Key Design Patterns

**Patch-Based Discretization**: All transducers are decomposed into small rectangular 
patches (sub-elements). The `no_sub_x` and `no_sub_y` parameters control subdivision 
density and simulation accuracy.

**Lazy Geometry Loading**: `TransducerBase` uses lazy-loaded properties for geometry 
(element centers, patch vertices) to defer computation until needed.

**Method Selection**: The SIR computation supports three methods:
- `"naive"`: Sample-piece-wise-looping  (accurate but slow, reference implementation)
- `"sdi"`: Sparse Delta Integration (new method, faster for large grids, may have
  numerical inaccuracies)
- `"auto"`: Automatically selects between naive and SDI based on grid properties

**Unit Convention**: User-facing APIs use millimeters (`_mm` suffix), but internal computations use SI units (meters, seconds).

**Monochromatic vs Transient**:
- Monochromatic: Returns `(p, coords)` where `p.shape = (Nx, Ny, Nz)` for continuous wave
- Transient: Returns `(p, coords)` where `p.shape = (Nt, Nx, Ny, Nz)` with time info in `coords["t0"]` and `coords["dt"]`

## Important Implementation Details

### Coordinate System
- X-axis: Lateral (across array elements)
- Y-axis: Elevation (perpendicular to imaging plane)
- Z-axis: Axial (beam propagation direction, depth)

### Medium Properties
Default physical parameters (can be overridden in `Emission` constructor):
- Speed of sound `c`: 1540 m/s
- Density `rho`: 1.0 kg/m³
- Sampling frequency `fs`: 200 MHz
- Attenuation `alpha0`: `None` (disabled by default; set to a float to enable)

### Transducer State Management
Each transducer stores:
- Geometry: element centers, patch subdivisions, normals
- Beamforming: delays (seconds), apodization (dimensionless)
- Configuration: frequency, element dimensions

Delays and apodization can be recomputed for different focal points without recreating the transducer.

### Transducer Details
See `.claude/rules/transducers.md` — loaded automatically when touching transducer code.
Covers: mono vs multi-element, focus_mm, z-convention, subdivision methods.

### Brain Atlas Integration
Uses BrainGlobe API to map acoustic fields onto anatomical structures. Requires downloading atlas data (e.g., rat, mouse atlases) on first use.

## Common Modifications

**Adding a New Transducer Type**:
1. Create new class inheriting from `TransducerBase` in appropriate file
2. Implement `_compute_element_centers()` to define element positions
3. Implement `_build_subdivisions()` to generate rectangular patches
4. Export in `src/pyfield/transducers/__init__.py`

**Modifying SIR Computation**:
- Core implementation: `src/pyfield/h_sir/farfield_rect_patch.py`
- Uses Numba JIT compilation for performance
- Parallelized over field points (not patches)

**Adding Visualization Methods**:
- Plane metadata & parsing: `src/pyfield/plotting/plane_utils.py` (`PLANE_META`, `PlaneSpec`, `parse_planes`)
- 2D/Matplotlib: Add to `src/pyfield/plotting/plotting2D.py`
- 3D/PyVista helpers: Add to `src/pyfield/plotting/plotting_pyvista.py`
- 3D/PyVista high-level: Add to `src/pyfield/plotting/plotting3D.py`
- Export new functions from `src/pyfield/plotting/__init__.py`

**Save/Export Convention** (defined in `src/pyfield/plotting/export_utils.py`):
- `save_path` = **always a directory** (or `None` to skip saving)
- `file_name` = **always includes extension** (e.g. `"field.png"`, `"slices.mp4"`)
- Use helpers: `save_matplotlib_animation`, `save_pyvista_screenshot`, `save_pyvista_movie`
- Directory creation handled by helpers via `_resolve_export_path`, not by callers
