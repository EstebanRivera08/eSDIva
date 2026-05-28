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

3) **`src/pyfield/psimulation/`** — Pressure field simulation (Emission + Reception)

4) **`src/pyfield/utilities/`** — Helper functions, surface subdivision, brain-atlas integration

5) **`src/pyfield/plotting/`** — Visualization (2D Matplotlib and 3D PyVista)

IMPORTANT NOTES:
- The module 1) Is the core computation engine, be careful with modifications.
- Module 2) will be a module under constant development since new transducers can be created and added,
  so think in generalization since backward compatibility might be important.
- Module 3) will have the principal class used for the API. Must be intuitive,
consistent, and predictable, minimizing friction for adoption and being robust over versions.

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
   All modes scale pressure by `rho` (unified exit path).
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

5. **Reception (Pulse-Echo RF Simulation)**:

   `Reception` computes received RF signals via the PE SDI kernel, which places
   16 deltas per (TX patch, RX patch) pair for efficient pulse-echo SIR computation.

   ```python
   from pyfield.psimulation import Reception

   # TX and RX can be same or different transducers
   tx = LinearArrayTransducer(...)
   tx.compute_delays(focus_mm=[0, 0, 30])
   tx.impulse_response = ir_pulse     # optional electromechanical IR
   tx.excitation = excitation_pulse   # TX excitation

   rx = LinearArrayTransducer(...)    # or same as tx
   rx.impulse_response = ir_pulse     # optional RX IR

   sim = Reception(tx, rx, fs=200e6, c=1540)

   # Define scatterers
   scatterer_pos = np.array([[0, 0, 30], [1, 0, 35]])  # mm
   scatterer_amp = np.array([1.0, 0.5])

   # Single-focus RF
   rf, coords = sim(scatterer_pos, scatterer_amp)
   # rf.shape = (Nt, E_rx), coords = {"t0": float, "dt": float}

   # Multi-line (sweep TX focus)
   tx_events = [
       {"delays": delays_line1, "apodization": apod_line1},
       {"delays": delays_line2, "apodization": apod_line2},
   ]
   rf_multi, coords = sim.compute_sequence(scatterer_pos, scatterer_amp, tx_events)
   # rf_multi.shape = (N_events, Nt, E_rx)

   # Full matrix capture
   rf_fmc, coords = sim.compute_all(scatterer_pos, scatterer_amp)
   # rf_fmc.shape = (E_tx, Nt, E_rx)
   ```

   **Constructor parameters** (keyword-only after tx, rx):
   - `c=1540.0` — speed of sound (m/s)
   - `rho=1.0` — medium density (kg/m^3)
   - `fs=200e6` — sampling frequency (Hz)
   - `alpha0=None` — attenuation in dB/(MHz^y·cm)
   - `freq_power=1.0` — power-law exponent y
   - `excitation=None` — TX excitation `(L,)` float32 (or uses `tx.excitation`)
   - `verbose=True`

   **Key differences from Emission**:
   - Takes separate TX and RX transducers
   - Uses PE SDI kernel (`compute_pe_sdi`) — no `jw` multiplication (derivatives
     already in Dh_pe)
   - Returns per-element RF data `(Nt, E_rx)`, not spatial pressure fields
   - Scatterer positions instead of field grid

6. **Visualize**:
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

---

## Mathematical Foundations

### 1. Spatial Impulse Response (SIR) — Tupholme-Stepanishen

Pressure from vibrating aperture S at field point r_p:

    p_m(r_p, t) = rho_0 * dv_n/dt *_t h_m(r_p, t)

Where h_m is the SIR:

    h_m(r_p, t) = (1/2pi) * integral_S [ delta(t - |r_m - r_p|/c_0) / |r_m - r_p| ] dS

### 2. Far-Field Trapezoidal SIR (Rectangular Patch)

For patch m with dimensions (w_mx, w_my), center r_m, field point r_p:
- Distance: `l = |r_p - r_m|`
- Unit vector: `u = (r_p - r_m) / l`

Trapezoid parameters:
- `dt1 = min(w_mx*|u_x|, w_my*|u_y|) / c_0`  (shorter side crossing)
- `dt2 = max(w_mx*|u_x|, w_my*|u_y|) / c_0`  (longer side crossing)
- `t1 = l/c_0 - (dt1 + dt2)/2`  (first corner TOF)
- `t2 = t1 + dt1`, `t3 = t1 + dt2`, `t4 = t1 + dt1 + dt2`
- `h_max = w_mx * w_my / (2*pi * dt2 * l)` (plateau amplitude)
- `slope = h_max / dt1`

**Far-field validity**: `w << sqrt(4*l*c_0/f)` — controls required subdivision density.

### 3. SDI Method (Sparse Delta Integration)

Key insight: 2nd derivative of trapezoid = 4 weighted Dirac deltas:

    d2h/dt2 = slope * [delta(t-t1) - delta(t-t2) - delta(t-t3) + delta(t-t4)]

Discrete: 8 sample writes per trapezoid (2 per corner via linear interpolation).
Recover h by double cumsum. SDI wins when `avg_dk >> 8 + 2T/M`.

### 4. Transducer = Sum of M Patches

    h_tx(r_p, t) = sum_{m=1}^{M} a_m * h_m(r_p, t - tau_m)

Where a_m = apodization, tau_m = delay per patch.

### 5. Emission Signal Chain

    p_e(r, t) = rho_0 * v_n(t) *_t dh(r, t)/dt

For monochromatic CW: `p_e,cw(r) = |H(r, omega_c)|` (SIR Fourier transform at fc).

### 6. PE SDI (Pulse-Echo Combined SDI)

Instead of computing dh_tx and d2h_rx separately then FFT-convolving:

    zeta_pe = d2h^e *_t d2h^r = 16 Dirac deltas per (m_e, m_r) pair

Each TX corner (4) × each RX corner (4) = 16 delta events.
32 sample writes per pair (16 × 2 bins via interpolation), then 1 cumsum.

    Dh_pe = integral(zeta_pe)

Received signal (Born approximation):

    p_r(t) = (rho_0/2c_0^2) * f_m(r) *_r [(E_m * v) *_t Dh_pe(r, t)]

Note: no derivative on v — derivatives already absorbed into Dh_pe.

### 7. Causal Power-Law Attenuation

General case (y != 1), Szabo (1994), Holm (2019):

    H_att(w, d) = exp(-alpha0*|w|^y*d) * exp(-j*alpha0*|w|^y*tan(y*pi/2)*d)

Special case (y = 1), O'Donnell (1981):

    H_att(w, d) = exp(-alpha0*|w|*d) * exp(-j*(2*alpha0/pi)*w*ln(|w|/w0)*d)

Unit conversion: `alpha0_neper = alpha0_dB * 100 / (20*log10(e) * 1e6^y)`

Always causal (both absorption + K-K dispersion terms). Non-causal produces precursors.

### 8. Plane-Wave Steering Delays

    n = [sin(theta_x), sin(theta_y), sqrt(1 - sin^2(theta_x) - sin^2(theta_y))]
    d_e = element_centers @ n
    delays = (d_max - d_e) / c

Physical: element with maximum projection fires first (zero delay).
Constraint: `sin^2(theta_x) + sin^2(theta_y) <= 1`.

---

## Emission Workflow

### Mode Decision Tree

```
Emission.__call__(field_points_mm)
    |
    +-- monochromatic=True
    |       use_per_element? -- False --> [A] Mono Global
    |                        -- True  --> [B] Mono Per-Element
    |
    +-- monochromatic=False
            use_per_element? -- True  --> [E] Per-Element Transient
            exc=None, alpha0=None ------> [C] Pulsed Pure
            exc=(L,) or fast_att -------> [D] Global FFT

    use_per_element = (alpha0 is not None and not fast_attenuation) OR exc.ndim == 2
```

### Shared Preamble (every call)

1. `create_3D_spatial_grid_from_points(field_points_mm)` → x, y, z, points_m
2. Compute `per_elem_exc`, `use_per_element` flags
3. `compute_time_grid(P, M, points_m, ...)` → time_grid, t0, dt, T
4. If alpha0 and global path: `compute_attenuation_distances` → distances_m (P,)
5. If exc and tx.impulse_response: convolve `exc * ir_tx`

### [A] Mono Global

`_compute_sir` → `compute_h_sir` (Numba) → `from_sir_to_monochromatic_pressure` (single FFT bin at fc).
If alpha0: `|H_att(fc, d)|` scalar multiply per point.

### [B] Mono Per-Element

Loop over E elements. Each: `compute_h_sir(M/E patches)` → dot product `h_e @ exp(-j2pi*fc*t)` → accumulate + per-element attenuation at fc.

### [C] Pulsed Pure

`_compute_sir` → `compute_h_sir` → return h directly. Fastest mode — no FFT.

### [D] Global FFT

Per P-batch: `compute_h_sir` → `rfft` → multiply `fft_exc * TF * H_att` → `irfft`.

### [E] Per-Element Transient

P-outer, E-inner double loop. Pre-allocated `h_pad_buf = zeros((batch_P, nfft), float32)` **once** outside all loops. Per (batch, element): `compute_h_sir(M/E patches)` → write into h_pad → `rfft` (no scipy internal buffer since already nfft-length) → multiply `fft_exc[e] * TF * H_att_e` → accumulate into `acc_H`. One `irfft` per P-batch (not per element). Freq-domain accumulation preserves inter-element interference.

### Attenuation Integration

- SIR kernels stay lossless — attenuation is **always** post-hoc in frequency domain.
- `P_att(r, f) = P_lossless(r, f) * H_att(f, d)` — one complex multiply per point per freq bin.
- Per-element path: `H_att_e (cols, N_freq)` computed inside E-loop using element-center distances.
- Global path: `H_att (P, N_freq)` pre-computed using TX-center distances.
- `alpha0=None` → no attenuation ops, bit-identical to no-attenuation baseline.

---

## Reception Workflow

### Data Flow

```
Per RX element e_rx:
  compute_pe_sdi(pts, tx_all_patches, rx_elem_patches) → Dh_pe (P, T)
  FFT(Dh_pe) × FFT(v) × FFT(ir_tx) × FFT(ir_rx) × H_att(f, d)
  IFFT → weight by f_m(r) → sum over scatterers → rf[:, e_rx]
  Scale by rho / (2 * c^2)
```

- All convolutions become element-wise freq-domain multiplies.
- IR_tx, IR_rx, V, H_att precomputed once. Only Dh_pe varies per RX element.
- When IR is None, corresponding FFT term = 1 (identity).
- Attenuation distance: two-path model `d_total(s, e) = |r_s - r_tx_center| + |r_s - r_rx_e|`.
- Memory: O(P × nfft) per RX element iteration — E_rx-independent.

### compute_sequence

Loop over TX events (different delays/apodization per event), call `__call__` each time.
Returns `(N_events, Nt, E_rx)`. TX state restored after all events.

### compute_all (Full Matrix Capture)

Each TX element transmits, all RX receive. Returns `(E_tx, Nt, E_rx)`.

---

## Field II Correspondence

| Field II | PyField | Notes |
|----------|---------|-------|
| `xdc_impulse(Th, ir)` | `transducer.impulse_response = ir` | Per-transducer electromechanical IR |
| `xdc_excitation(Th, exc)` | `transducer.excitation = exc` / `Emission(excitation=...)` | TX only |
| `xdc_focus(Th, ...)` | `transducer.compute_delays(focus_mm=...)` | Existing |
| `xdc_apodization(Th, ...)` | `transducer.compute_apodization(...)` | Existing |
| `calc_hp(Th, pts)` | `Emission(tx)(field_points)` | Emitted pressure |
| `calc_scat_multi(tx, rx, pos, amp)` | `Reception(tx, rx)(pos, amp)` | Per-element RF |
| `calc_scat_all(tx, rx, pos, amp)` | `Reception.compute_all(...)` | Full matrix capture |
| `set_field('att', ...)` | `Emission/Reception(alpha0=..., freq_power=...)` | Attenuation |
| `calc_scat(tx, rx, pos, amp)` | **Not implemented** | Beamforming is user's job |
| `xdc_baffle(Th, soft)` | Future extension | Not yet |
| `xdc_dynamic_focus(...)` | Future extension | Requires timeline system |

**Improvements over Field II**:
- Causal power-law attenuation with K-K dispersion (Field II uses non-causal minimum-phase)
- Explicit per-element excitation support via shape dispatch `(L,)` vs `(L, E)`
- Python/NumPy ecosystem

---

## Performance Bottlenecks

### Benchmark Reference (LinearArrayTransducer E=128, M=1280, P=60501, nfft=8192, 12.5 MHz)

| Mode | Time | Primary Bottleneck |
|------|------|--------------------|
| [A] Mono Global | ~29 s | `compute_h_sir` (Numba, all M patches) |
| [B] Mono Per-Element | ~29 s | Same Numba kernel × E (M/E patches each) |
| [C] Pulsed Pure | ~11 s | `compute_h_sir` only — no FFT |
| [D] Global FFT | ~24 s | Numba + scipy FFT interleaved per batch |
| [E] Per-Element | ~17 min | `rfft` × E × n_batches (FFT-bound) |

### Per-Element Mode Breakdown

Total rfft calls: `E × n_batches = 128 × 15 = 1920` at ~0.53 s each.
Total irfft calls: `n_batches = 15` (one per batch, not per element).
Irreducible on CPU: `E × n_batches × t_rfft ≈ 1020 s`.

### Memory Architecture (Per-Element Mode)

```
WRONG (earlier): zeros((cols, nfft), float32) inside E-loop
  → E × n_batches × 140 MB = 268 GB allocation traffic → OS swap

CORRECT (current): h_pad_buf = zeros((batch_P, nfft), float32) ONCE
  → rfft receives already-nfft input → no scipy internal buffer
  → ONE allocation, reused E × n_batches times
```

### Key Performance Rules

- `scipy.fft.rfft/irfft(workers=-1)` for multithreaded FFT (2-3× over numpy.fft)
- All kernel outputs + freq-domain ops stay float32 → complex64 (half memory vs float64)
- Batch size: `batch_P = 400 MB / (nfft × 4 + 2 × N_freq × 8)` bytes
- Freq-domain accumulation: `irfft(sum_e H_e) = sum_e irfft(H_e)` (linearity) — one irfft per batch preserves interference

### Potential Optimizations (not yet implemented)

| Optimization | Target Mode | Expected Gain |
|-------------|-------------|---------------|
| GPU FFT (cupy/torch) | [E] per-element | 10-50× on rfft |
| Coarser grid | all | P ↓ → fewer batches |
| Async Numba + FFT pipeline | [D][E] | Overlap SIR compute and FFT |

---

## Risky Implementations (Validate Physics/Math)

### 1. SDI Tail Artifact — float32 Cumsum Cancellation

**Location**: `sir_derivatives.py` cumsum functions, `farfield_rect_patch.py` integration.

d2h events are ~4e10 in magnitude. Float32 ULP at that scale = 4096. When large positive/negative events cancel, residual is ±4096 (1 ULP), not the true ~±2048. This leaves a DC offset in dh that becomes a linear ramp in h after double cumsum.

**Mitigation**: float64 accumulator + float32 write-back in all `_cumsum_2d`/`_cumsum_3d`:
```python
acc = np.float64(0.0)           # MANDATORY float64
acc += np.float64(arr[i, k])    # promote to float64
out[i, k] = np.float32(acc)     # write back float32
```

**Residual after fix**: ~0.004% of SIR peak — physically negligible but breaks exact comparison.
**Test tolerance**: `rtol=0.005, atol=0.005 × peak` for all SIR comparisons.

### 2. d2h_all ≠ d2h_per_element.sum() — Float32 Non-Associativity

`compute_d2h` (all patches) vs `compute_d2h_per_element` (per-element grouping) accumulate in different orders. Float32 addition is not associative. Difference = up to 1 ULP of event magnitude (4096 at 4e10 scale). After cumsum, persistent constant offset in tail.

**Impact**: relative error ~5e-8. Physically zero. Never compare with atol=0.

### 3. PE SDI vs FFT-Conv Reference — Offset Convention

PE SDI places combined deltas with `+ 2.0` bin offset (not `+1`). The reason:
- Single SDI uses `+1` so that after 1 cumsum, `dh_e` step is at bin `Ne = floor((t_corner-t0)*fs) + 1`.
- Reference `dh_e * d2h_r` first nonzero (from FFT conv) is at `Ne + Nr = natural_e + natural_r + 2`.
- PE SDI must match this: `floor((t_corner_e + t_corner_r - pe_t0)*fs + offset) = Ne + Nr` → **offset = 2**.

Using `+1` (wrong) shifts all Dh_pe events 1 sample early relative to reference. At 100 MHz = 10 ns axial shift.

**Test strategy**: compare after excitation convolution, not raw deltas. Peak ratio < 5%, correlation > 0.95.

### 4. Attenuation y=1 Continuity

`tan(y*pi/2)` diverges as y→1. Cannot test y=1 continuity by approaching from y=1.001 (phase ~636× larger). Test y=1 branch independently via `|H(f)| = exp(-alpha0_nep * f * d)`.

### 5. Global vs Per-Element Excitation Consistency

Global path tiles `(L,) → (L, E)` and calls same element-loop as per-element path. Previous approaches (global SDI dh vs per-element dh → sum) produced different intermediate cumsums that propagated as 6144-magnitude constant offsets through FFT convolution, causing 150× relative differences near zero crossings.

**Current solution**: both paths use identical per-element dh computation. Trade-off: global path does E separate cumsums instead of 1.

### 6. Numba Cache Staleness

After editing Numba kernels, `.nbi`/`.nbc` cache files may retain old compiled versions. Symptom: fix "has no effect." Clear cache:
```powershell
Get-ChildItem -Path "src\pyfield\h_sir\__pycache__" -Filter "*.nb?" | Remove-Item -Force
```

### 7. h_sir.__call__ Is Broken (Pre-existing)

`h_sir.__call__` unpacks 4 values from `check_valid_field_points` which returns 1 value. Do NOT call it — use `compute_derivative` method or `Emission`/`Reception` classes directly.

### 8. `from_sir_to_pressure` Attenuation with No Excitation

When `excitation=None`, `from_sir_to_pressure` returns h_sir directly — no IRFFT step. Attenuation parameter is silently ignored. Callers relying on attenuation must provide excitation.
