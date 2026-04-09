# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Anytime a modification is performed in the project update this document in case the
logic changes.

## Project Overview

PyField is a Python acoustic field simulator based on the Tupholme–Stepanishen Spatial 
Impulse Response (SIR) method. It models arbitrary transducer geometries as collections 
of rectangular patches and computes pressure fields via convolution with excitation pulses.

IMPORTANT: For programming guidelines see @AGENTS.md. !

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

4) **`src/pyfield/utilities/`** — Plotting and helper functions

5) **`src/pyfield/brain_atlas/`** — Brain atlas integration (Bonus for neuroscience
applications)

6) **`src/pyfield/scans/`** — Scanning sequence utilities 

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

4. **Run Simulation (Default is monochromatic)**:
   ```python
   from pyfield.psimulation import PyField
   sim = PyField(tx)
   x, y, z, p = sim(field_points, method="auto")
   ```
if no excitation is given, the transient simulation is pulsed and `monochromatic =
    False` needs to me added to sim(). 

If performing transient simulations and excitation could be given :
  ```python
  import numpy as np
  from pyfield.psimulation import PyField

  f_s = 200e6  # Sampling frequency
  fc = 5e6  # Center frequency of the pulse
  n_cycles = 2  # Number of cycles in the pulse
  time = np.arange(0, n_cycles/fc, 1/f_s)  # Time vector for the pulse
  excitation = np.sin(2*pi*fc*time)  # Example pulse
    sim = PyField(tx, fs=f_s)
    x, y, z, t, p = sim(field_points, method="auto", excitation=excitation)
  ```

5. **Visualize**:
   ```python
   from pyfield.utilities import plot_pressure_planes #if monochromatic (pressure field is 3D)
   plot_pressure_planes(x, y, z, p, db_scale=True, vmin=-40)
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
- Monochromatic: Returns spatial pressure field `p(x,y,z)` for continuous wave
- Transient: Returns spatio-temporal pressure `p(x,y,z,t)` with defined excitation pulse

## Important Implementation Details

### Coordinate System
- X-axis: Lateral (across array elements)
- Y-axis: Elevation (perpendicular to imaging plane)
- Z-axis: Axial (beam propagation direction, depth)

### Medium Properties
Default physical parameters (can be overridden in `PyField` constructor):
- Speed of sound `c`: 1540 m/s
- Density `rho`: 1.0 kg/m³
- Sampling frequency `fs`: 200 MHz
- Attenuation `alpha0`: 0 dB/(MHz cm)

### Transducer State Management
Each transducer stores:
- Geometry: element centers, patch subdivisions, normals
- Beamforming: delays (seconds), apodization (dimensionless)
- Configuration: frequency, element dimensions

Delays and apodization can be recomputed for different focal points without recreating the transducer.

### Mono-Element vs Multi-Element Transducers
`compute_delays()` and `compute_apodization()` are designed for element-level
beamforming (multi-element arrays like linear, convex, and matrix transducers).

For **mono-element transducers** (all circular types: flat, concave, convex, focused):
- `compute_delays()` is ignored with a warning — the entire aperture surface acts
  simultaneously, so element-level delays do not apply.
- `compute_apodization()` returns uniform weights with a warning. However, **patch-wise
  apodization** is physically meaningful (e.g. apodizing the aperture edge) and can be
  set directly via `set_apodization()`.
- Geometric focusing for mono-element transducers is achieved by the physical curvature
  of the surface, not by electronic delays.

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
- 2D/Matplotlib: Add to `src/pyfield/utilities/plotting.py`
- 3D/PyVista: Add to `src/pyfield/utilities/plotting_pyvista.py`
- Ensure compatibility with both `"dark"` and `"light"` themes
