# PyField Architecture & Internals

Deep reference for understanding PyField's internals — for contributors and anyone
who wants to know *how* the engine works, plus agents editing core code. The
operational quickstart and physics context live in `CLAUDE.md`. User-facing
conceptual guides live in `docs/` (Zensical site).

Sections:
1. [Full API Reference](#full-api-reference) — Emission & Reception parameters, all modes
2. [Common Modifications](#common-modifications) — extending transducers, SIR, plotting
3. [Emission Workflow](#emission-workflow) — mode decision tree, per-mode breakdown
4. [Reception Workflow](#reception-workflow) — data flow, sequence_rf, synthetic_aperture_rf
5. [Field II Correspondence](#field-ii-correspondence) — API mapping table
6. [Performance Bottlenecks](#performance-bottlenecks) — benchmarks, memory architecture
7. [Risky Implementations](#risky-implementations) — physics/math validation hazards

---

## Full API Reference

### Emission

`Emission` is the primary simulation class. `PyField` is a deprecated alias
(emits `DeprecationWarning`, defaults `monochromatic=True` for backward compat).

```python
from pyfield.emission import Emission

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

### Reception (Pulse-Echo RF Simulation)

Two reception classes are available:
- `Reception` — conventional FieldII-style: `h_pe = h_tx ⊛ h_rx` (each SIR built
  separately, convolved by FFT). Depth-binned post-processing (see
  [Pulse-Echo Post-Processing](#pulse-echo-post-processing--depth-binning)) makes it
  **the fast choice for real arrays** (many patches): cost ~`O(P·M + P·log nfft)`.
  Beats Field II `calc_scat_multi` for `N_scat ≥ 100` (e.g. 2× at `N_scat=10⁴`).
- `ReceptionSDI` — PE SDI kernel: builds the two-way SIR from the *product* of the
  two delta trains (16 deltas per TX/RX patch pair, 1 cumsum). Cost ~`O(P·M_tx·M_rx)`
  — **quadratic in patch count**, so faster only when `M` is small (few-patch /
  monoelement transducers). Not depth-binned.

Both give the same RF; pick by patch count.

**Public API** (axis order `[emission, reception, Nt]` — channels before time;
`coords["t0"]` beam-axis referenced). The pulse-echo signal physically carries the
3rd derivative of the excitation (`v_pe = ρ₀/2c₀² · E_m ⊛ ∂³v/∂t³`), but in practice
that ∂³ is baked into the band-limited excitation + impulse responses
(`∝ e ⊛ h_e ⊛ h_r`), so no method applies an explicit extra ∂/∂t. Field II uses the
same convention (`calc_scat` ≡ `calc_hhp` for a point), so all methods coincide with
it:
| Method | Field II | Output |
|--------|----------|--------|
| `pulse_echo_rf(pts, amp)` = `__call__` | `calc_scat`/`calc_hhp` | `(Erx, Nt)` summed · `(P, Erx, Nt)` `per_scatterer` (PSF) |
| `sequence_rf(pts, amp, events)` | — | `(Nev, Erx, Nt)` (PW/DW emission basis) |
| `synthetic_aperture_rf(pts, amp)` | `calc_scat_all` | `(Ntx_grp, Erx, Nt)` decimated (per-element/group DW basis) |
| `scan_focusline(focus_mm, pts, amp)` | `calc_scat` (beamformed) | `(Nt,)` one focused scan-line envelope |

`pulse_echo_rf` is the core engine; `sequence_rf` loops it; `synthetic_aperture_rf`
= sequence with auto per-element/group events (zero delay, unit apod) + anti-aliased
decimation + size guard; `scan_focusline` recomputes TX **and** RX focus/apod (RX
mirrors TX by default; `rx_FoverD`/`rx_apodization_type` override) then beamforms on
receive INSIDE the SIR kernel (`focused_sum=True`: all RX patches summed in one
`compute_pe_sdi`/`compute_h_sir` call → one FFT pair, no external DAS). These shared
methods plus all common state (`set`, patch extraction, validation) live in
`ReceptionBase` (`reception/base.py`); each subclass adds only its constructor,
time-grid helper, `_compute_rf_inner`, and the convention wrappers.

The 3 physical derivatives are carried by the exc/IR chain, so neither class applies
an explicit ∂³: `Reception` uses `(jω)⁰=1`; `ReceptionSDI`'s kernel places the 3 on
the SIR for speed, then divides by `(jω)³` in the frequency domain (no group delay →
sample-aligned with `Reception`) to relocate them onto the exc/IR chain. Field II
shares this convention, so both classes match it corr≈1.0000 at the RF level
(per-element RF verified 0.997 vs `calc_scat_multi`).

```python
from pyfield.reception import ReceptionSDI  # or Reception for conventional

tx = LinearArrayTransducer(...)
tx.impulse_response = ir_pulse
tx.excitation = excitation_pulse
rx = tx.copy()
rx.impulse_response = ir_pulse
sim = ReceptionSDI(tx, rx, fs=200e6, c=1540)

scatterer_pos = np.array([[0, 0, 30], [1, 0, 35]])  # mm
scatterer_amp = np.array([1.0, 0.5])

# Core pulse-echo RF.  sim(...) == sim.pulse_echo_rf(...).
rf, coords = sim.pulse_echo_rf(scatterer_pos, scatterer_amp)
# rf.shape = (E_rx, Nt), coords = {"t0": float, "dt": float}
psf, coords = sim.pulse_echo_rf(scatterer_pos, per_scatterer=True)  # (P, E_rx, Nt)

# Sweep emission basis (PW/DW events)
tx_events = [{"delays": d1, "apodization": a1}, {"delays": d2, "apodization": a2}]
rf_multi, coords = sim.sequence_rf(scatterer_pos, scatterer_amp, tx_events)  # (Nev,Erx,Nt)

# Synthetic aperture / FMC (per-element diverging-wave basis)
rf_sa, coords = sim.synthetic_aperture_rf(scatterer_pos, scatterer_amp)  # (Ntx,Erx,Nt)

# One conventional focused scan line (loop focus_mm to build a B-mode)
env, coords = sim.scan_focusline([0, 0, 30], scatterer_pos, scatterer_amp,
                                 FoverD=2.0, apodization_type="hanning")  # (Nt,)
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
- `ReceptionSDI` uses the PE SDI kernel (`compute_pe_sdi`); `pulse_echo_rf` divides by `(jω)³` in freq domain to relocate the kernel's 3 derivatives onto the exc/IR chain (which physically carries them)
- `Reception` uses conventional FFT-based `h_pe = h_tx ⊛ h_rx` with `(jω)⁰=1` (no explicit extra ∂/∂t — exc/IR carry the physical derivatives)
- Returns per-element RF data `(Erx, Nt)`, not spatial pressure fields
- Scatterer positions instead of field grid

### Visualization

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

---

## Common Modifications

**Adding a New Transducer Type**:
1. Create new class inheriting from `TransducerBase` in appropriate file
2. Implement `_compute_element_centers()` to define element positions
3. Implement `_build_subdivisions()` to generate rectangular patches
4. Export in `src/pyfield/transducers/__init__.py`

**Importing Field II Transducer Geometry**:
- `FieldIITransducer` (`src/pyfield/transducers/fieldii_compat.py`) — transducer built from raw patches
- `from_fieldii_xdc_data(data)` — parse MATLAB `xdc_get(Th,'all')` struct → `FieldIITransducer`
- `from_fieldii_patch_arrays(centres, u, v, hw, hh)` — explicit patch arrays → `FieldIITransducer`
- Treats each Field II mathematical element as one PyField element (n_elements = N_patches)
- For monostatic reception, sum RF channels after simulation (all channels belong to the single aperture)
- Scale convention: PyField uses `rho/(2c²)`, Field II uses `rho/2`. Raw amplitude differs by `c²≈2.37e6`. Normalised PSF unaffected.

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

### Pulse-Echo Post-Processing & Depth Binning

This is why `Reception` (conventional) is fast. After the SIR kernels run (which are
*not* the bottleneck — ~6% of runtime), the work is the per-element convolution
`h_pe = h_tx ⊛ h_rx`, done by FFT. Three layers cut its cost:

**1. Sum scatterers before the IFFT.** The recorded RF sums over scatterers, and the
excitation/IR filters `F` are the same for all of them. By linearity
`Σ_p a_p · irfft(H_p · F) = irfft((Σ_p a_p H_p) · F)`, so the per-element loop does
**one IFFT per element**, not one per scatterer. (`per_scatterer=True` keeps each
scatterer, so it skips this.)

**2. float32 FFTs.** The SIR is float32; FFTing in float32 → complex64 halves the
dominant forward-FFT cost vs float64, with no meaningful accuracy loss.

**3. Depth binning (`_fast_rf_binned`, `_auto_depth_bins`).** `nfft` is set by the
time grid, which spans the *whole* scatterer cloud — so a scatterer at the focus
(short SIR, ~200 samples) is padded to the same `nfft` as the farthest one (~2048).
Fix: sort scatterers by depth and split into bins. Each bin spans a narrow depth
range → short time grid → small `nfft`. Scatterers are still summed in frequency
within a bin (layer 1), so it stays one IFFT per element per bin.

Bins share **one sample lattice**: each bin's grid starts at `t0_global + n0·dt`
(`_lattice_grid`), so every bin's per-element result drops into the global RF buffer
at the exact integer sample offset `n0` — no interpolation, no sub-sample drift.
(Skip this and the bins misalign by a fraction of a sample, giving ~25% error.)

**Choosing the bin count** (`_auto_depth_bins`) balances two effects:
- *Grid length*: more bins → shorter grids → smaller `nfft`. Diminishing once a bin
  is as short as a single scatterer's SIR (≈ arrival_spread / 128 bins).
- *Cache*: each bin's `H_tx`/`H_rx` FFT batch is reused across all RX elements; when
  it fits in CPU cache the element loop is much faster. That caps the bin size at
  ≈ `_SCATTERERS_PER_BIN` (~200) scatterers/bin — the dominant effect at high
  `N_scat`. The auto rule is `n_bins ≈ max(spread/128, P/200)`, ≥128 scatterers/bin.

Override with the `n_depth_bins` constructor arg (or `sim.set("n_depth_bins", N)`)
to tune for a different CPU or `N_scat ≫ 10⁴`. The optimum is machine-dependent but
broad; binning off (`n_depth_bins=1`) recovers the single-grid path.

Result (Domino linear, E=128, M=1280, vs Field II `calc_scat_multi` time): N=100
1.1×, N=1000 2.0×, N=10⁴ 2.1×. The 3 layers preserve the RF to ~4e-4 (binned vs
unbinned) — within float/grid-snap tolerance. Attenuation, `per_scatterer`, and
`focused_sum` keep the non-binned path. (`ReceptionSDI` is not binned — its cost is
the `O(M²)` kernel, not the FFT.)

### sequence_rf

Loop over TX events (different delays/apodization per event), call `pulse_echo_rf`
each time. Returns `(N_events, Erx, Nt)`. TX state restored after all events.
Warns + suggests `downsampling=` if the output would be large.

### synthetic_aperture_rf (Full Matrix Capture / synthetic aperture)

Each TX element/group fires flat (zero delay, unit apod — overrides TX state), all
RX receive. Returns `(Ntx_grp, Erx, Nt)`, anti-aliased-decimated (`decimation=10`
default). `tx_groups` = `"element"` (FMC) / `int N` (sub-aperture) / custom groups.
Estimates output size first; warns + auto-decimates (or streams to `out_path` memmap)
if it would exceed RAM, after a 10 s abortable countdown.

### scan_focusline

One conventional focused scan line: recompute TX focus+apod from `focus_mm`,
`pulse_echo_rf`, DAS, Hilbert envelope. Returns `(Nt,)`; loop `focus_mm` for a B-mode.

---

## Field II Correspondence

| Field II | PyField | Notes |
|----------|---------|-------|
| `xdc_impulse(Th, ir)` | `transducer.impulse_response = ir` | Per-transducer electromechanical IR |
| `xdc_excitation(Th, exc)` | `transducer.excitation = exc` / `Emission(excitation=...)` | TX only |
| `xdc_focus(Th, ...)` | `transducer.compute_delays(focus_mm=...)` | Existing |
| `xdc_apodization(Th, ...)` | `transducer.compute_apodization(...)` | Existing |
| `calc_hp(Th, pts)` | `Emission(tx)(field_points)` | Emitted pressure (1 derivative) |
| `calc_hhp(tx, rx, pts)` | `(Reception\|ReceptionSDI)(tx, rx).pulse_echo_rf(pts, per_scatterer=True)` | Pulse-echo response / PSF (0 derivatives; bare exc⊛ir⊛ir⊛h) |
| `calc_scat_multi(tx, rx, pos, amp)` | `pulse_echo_rf(pos, amp)` | Per-element scattered RF (0 derivatives, = amp-weighted calc_hhp) |
| `calc_scat_all(tx, rx, pos, amp)` | `synthetic_aperture_rf(...)` | Full matrix capture / synthetic aperture |
| `set_field('att', ...)` | `Emission/Reception(alpha0=..., freq_power=...)` | Attenuation |
| `calc_scat(tx, rx, pos, amp)` (beamformed line) | `scan_focusline(focus_mm, pos, amp)` | Conventional focused scan line |
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

### 3. PE SDI Delta Placement — `k_shift = 0`

PE SDI places combined deltas at `floor((t_corner_e + t_corner_r - pe_t0)*fs)` with
**`k_shift = 0`** (in `_place_pe_sdi_deltas`, `transducer_sir_pe.py`).

Rationale: `zeta = d2h_e ⊛ d2h_r`, one cumsum → `Dh_pe`. Discrete convolution adds
indices, so an event at TX bin `floor((t_e-tx_t0)*fs)` and RX bin
`floor((t_r-rx_t0)*fs)` lands at their sum. Since `pe_t0 = tx_t0 + rx_t0`, that sum
equals `floor((t_e+t_r-pe_t0)*fs)` (± the floor-of-sum vs sum-of-floors ULP). This
matches the naive `h_tx ⊛ h_rx` onset exactly — single-SDI `h` is already correctly
timed (no net +1), so no extra shift is needed.

**History**: was `k_shift = 2.0`, derived on the wrong premise that single-SDI added
+1 per side. Empirically that placed `ReceptionSDI` 2 samples late vs
`Reception(method="naive"|"sdi")` (= 20 ns at 100 MHz). Verified fix: on-axis lag of
naive vs PE-SDI is now 0 (Emission and `Reception(method="sdi")` were already lag-0).

**Test gap**: `test_pe_sdi.py` checks only peak ratio (<5%) + correlation (>0.90),
both lag-insensitive — they did **not** catch the 2-sample shift. `example06` now
asserts on-axis lag == 0 as a phase-regression guard. Residual raw-RF difference
(~10%) vs conventional is FFT-`(jω)³` Gibbs ringing, not timing (envelope ~1.4%).

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
