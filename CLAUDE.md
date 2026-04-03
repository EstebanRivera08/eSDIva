# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyField is a Python acoustic field simulator based on the Tupholme–Stepanishen Spatial Impulse Response (SIR) method. It models arbitrary transducer geometries as collections of rectangular patches and computes pressure fields via convolution with excitation pulses.

**Warning**: PyField is currently under development. The API is subject to change.

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

### Running Examples
Run bundled examples to test functionality:
```bash
uv run example1_monochrom_focus.py
uv run example2_ratbrainzones_focus.py
uv run example3_transient_focusing.py
uv run example4_linear_divergingwaves.py
uv run example5_transducer_gallery.py
uv run example6_circular_transducer.py
```

### Running Tests
Run the test file:
```bash
uv run test.py
```

### Code Formatting
The project uses `ruff` for linting (configured in `pyproject.toml` with line-length=88):
```bash
uv run ruff check src/
uv run ruff format src/
```

### Documentation Generation
Generate documentation images:
```bash
uv run test.py  # Contains doc generation code
```

## Architecture

### Module Structure

The codebase follows a modular architecture with clear separation of concerns:

**`src/pyfield/transducers/`** — Transducer geometry definitions
- `base.py`: Abstract `TransducerBase` class. All transducers inherit from this and implement `_compute_element_centers()` and `_build_subdivisions()`.
- `linear.py`: `LinearArrayTransducer`, `ConvexArrayTransducer`
- `matrix.py`: `MatrixArrayTransducer`
- `circular.py`: `FlatCircularTransducer`, `ConcaveCircularTransducer`, `ConvexCircularTransducer`, `FocusedCircularTransducer`
- `custom.py`: `CustomTransducer` for assembling arbitrary multi-element configurations
- `saved_transducers.py`: Pre-defined transducers (`Domino`, `Zeus_Matrix`)
- `geometry_utils.py`: Geometric computation utilities
- `validators.py`: Input validation

**`src/pyfield/h_sir/`** — Spatial Impulse Response computation
- `h_sir.py`: Main `h_sir` class that orchestrates SIR computation
- `farfield_rect_patch.py`: Core `compute_h_sir()` function implementing Tupholme-Stepanishen formulation
- `hsir_SDI.py`: Sparse Delta Integration (SDI) optimized method
- Methods: `"naive"` (brute-force), `"sdi"` (optimized), `"auto"` (automatic selection)

**`src/pyfield/psimulation/`** — Pressure field simulation
- `PyField.py`: Main `PyField` class that converts SIR to pressure
- `sir_to_pressure.py`: Convolution of SIR with excitation pulses
- Supports both monochromatic (spatial-only) and transient (spatio-temporal) simulations

**`src/pyfield/brain_atlas/`** — Brain atlas integration
- `bg_atlas.py`: Integration with BrainGlobe atlas API
- `transformations.py`: Coordinate transformations for brain mapping

**`src/pyfield/utilities/`** — Plotting and helper functions
- `plotting.py`: Matplotlib-based visualization (2D slices)
- `plotting_pyvista.py`: PyVista-based 3D visualization
- `helper_functions.py`: Field point validation, grid creation, time grid computation
- `surface_subdivision.py`: Patch subdivision utilities
- `transformation_functions.py`: Coordinate transformations

**`src/pyfield/scans/`** — Scanning sequence utilities
- `dopplerscan.py`: Doppler scanning implementations

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

2. **Configure Delays and Apodization**:
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
   ```python
   from pyfield.psimulation import PyField
   sim = PyField(tx, monochromatic=True)
   x, y, z, p = sim(field_points, method="auto")
   ```

5. **Visualize**:
   ```python
   from pyfield.utilities import plot_slices_2d
   plot_slices_2d(x, y, z, p, db_scale=True, vmin=-40)
   ```

### Key Design Patterns

**Patch-Based Discretization**: All transducers are decomposed into small rectangular patches (sub-elements). The `no_sub_x` and `no_sub_y` parameters control subdivision density and simulation accuracy.

**Lazy Geometry Loading**: `TransducerBase` uses lazy-loaded properties for geometry (element centers, patch vertices) to defer computation until needed.

**Method Selection**: The SIR computation supports three methods:
- `"naive"`: Direct summation over all patches (slow but accurate)
- `"sdi"`: Sparse Delta Integration (fast, automatic range detection)
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

### Brain Atlas Integration
Uses BrainGlobe API to map acoustic fields onto anatomical structures. Requires downloading atlas data (e.g., rat, mouse atlases) on first use.

## File Organization

- `src/pyfield/`: Main source code
- `example*.py`: Standalone example scripts demonstrating key features
- `tutorials/`: Jupyter notebooks and comparison with Field II
- `others/`: Experimental code, analysis scripts, learning materials
- `docs/`: Markdown documentation (rendered with zensical)
- `test.py`: Test script and documentation figure generation

## Testing Strategy

Currently, testing is done through example scripts rather than formal unit tests. When adding new features:
1. Create a standalone example demonstrating the functionality
2. Verify output visually or against known reference (e.g., Field II)
3. Add the example to the root directory as `example*.py`

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
