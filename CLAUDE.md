# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo. Keep this file in
sync when code logic changes.

> **Deep reference lives in [`ARCHITECTURE.md`](ARCHITECTURE.md)** — full API
> parameter tables, all emission/reception modes, internal workflows (mode trees,
> data flow), Field II correspondence, performance benchmarks, and the full
> "risky implementations" hazard list. This file holds only what's needed every
> session: overview, commands, module map, a quickstart, physics, and key gotchas.

## Project Overview

PyField is a Python acoustic field simulator based on the Tupholme–Stepanishen Spatial
Impulse Response (SIR) method. It models arbitrary transducer geometries as collections
of rectangular patches and computes pressure fields via convolution with excitation pulses.

> **Audience-first documentation (package philosophy).** PyField is written for
> ultrasound researchers and students, **not for programmers**. Every docstring and
> comment — public *and* private — must be concise, clear, and **self-sufficient**:
> a physicist must understand the acoustics being computed (and why) without opening
> another file. Explain the physics first, inline. **Never cite markdown files**
> (`ARCHITECTURE.md`, `PE_SDI_kernel_analysis.md`, papers, "see module docstring") from
> a docstring or comment — they drift and the reader may not have them; the code must
> explain itself. Full rules: "Documentation & Comment Philosophy" at the top of
> `.claude/rules/coding-guidelines.md`.

Guidelines load automatically from `.claude/rules/`:
- **coding-guidelines** — code style, testing, commits (always loaded)
- **physics-context** — SIR/SDI theory (loaded when touching `hsir/`, `emission/`, `reception/`, `transducers/`)
- **transducers** — geometry conventions, subdivision, z-convention (loaded when touching `transducers/`)
- **attenuation** — power-law attenuation (loaded when touching attenuation code)

When the user says **"documentation"** or **"the docs"**, they mean the `docs/`
folder (Zensical/MkDocs site). Update the relevant `.md` there whenever the
corresponding code changes.

## Development Commands

`uv` for dependencies, `just` as command runner.

```bash
uv sync                # install + sync venv
uv run <script.py>     # run a script
uv add <package>       # add dependency
just test              # run tests with coverage (alias: just t)
just pre-commit        # ruff-check, ruff-format, ty, codespell, numpydoc (alias: just pc)
just serve-docs        # build + serve Zensical docs locally (hot-reload)
just docs              # build docs only → site/
```

See `.claude/rules/coding-guidelines.md` for the full command list, code style,
testing philosophy, and commit conventions.

### Documentation System

Framework: **Zensical** (MkDocs-based). Config: `zensical.toml` (root). Pages under
`docs/` (`index.md`, `user-guide/*.md`, `api/*.md`, `examples/*.md`). Add a new page
by creating `docs/<section>/newpage.md` and adding it to `nav` in `zensical.toml`.
Page frontmatter sets the nav icon (`icon: lucide/<name>`). Theme/palette configured
under `[project.theme]`.

## Architecture

### Module Structure (subject to change as project evolves)

1. **`src/pyfield/hsir/`** — Spatial Impulse Response computation. **Core engine — modify carefully.**
2. **`src/pyfield/transducers/`** — Transducer geometry. Under active development; prioritize generalization, keep backward compatibility.
3. **`src/pyfield/emission/`** — Emission simulation (`Emission`, deprecated `PyField` alias). **Primary API — keep intuitive/consistent.**
4. **`src/pyfield/reception/`** — Reception simulation (`ReceptionSDI`, `Reception`). **Primary API.**
5. **`src/pyfield/attenuation/`** — Power-law attenuation transfer functions.
6. **`src/pyfield/utilities/`** — Helpers, surface subdivision, brain-atlas integration.
7. **`src/pyfield/plotting/`** — Visualization (2D Matplotlib, 3D PyVista).

### Key Design Patterns

- **Patch-based discretization**: transducers decompose into small rectangular patches; `no_sub_x`/`no_sub_y` control subdivision density and accuracy.
- **Lazy geometry loading**: `TransducerBase` defers element-center/patch-vertex computation until needed.
- **SIR method selection**: `"naive"` (slow reference), `"sdi"` (Sparse Delta Integration, faster on large grids), `"auto"` (picks based on grid properties).
- **Unit convention**: user-facing APIs use mm (`_mm` suffix); internals use SI (m, s).
- **Monochromatic vs transient**: mono returns `p.shape = (Nx, Ny, Nz)` (CW); transient returns `(Nt, Nx, Ny, Nz)` with `coords["t0"]`/`coords["dt"]`.

### Coordinate System
- X: lateral (across array elements) · Y: elevation (perpendicular to imaging plane) · Z: axial (beam propagation, depth)

### Medium Properties (override in constructor)
- `c=1540` m/s · `rho=1.0` kg/m³ · `fs=200e6` Hz · `alpha0=None` (attenuation disabled by default)

### Transducer State
Each transducer stores geometry (element centers, patch subdivisions, normals),
beamforming (delays in s, apodization dimensionless), and configuration (frequency,
element dims). Delays/apodization recompute for new focal points without recreating
the transducer. See `.claude/rules/transducers.md` for mono vs multi-element,
`focus_mm`, z-convention, subdivision methods.

### Brain Atlas Integration
Uses BrainGlobe API to map acoustic fields onto anatomical structures. Requires
downloading atlas data (rat, mouse, ...) on first use.

## Simulation Quickstart

Full parameter tables, all modes, return conventions, and visualization options:
see [`ARCHITECTURE.md` § Full API Reference](ARCHITECTURE.md#full-api-reference).

```python
from pyfield.transducers import LinearArrayTransducer
from pyfield.emission import Emission

# 1. Transducer (mm units; no_sub_x/no_sub_y are required, keyword-only)
tx = LinearArrayTransducer(
    n_elements=64, element_width_mm=0.25, element_height_mm=12.0,
    kerf_mm=0.05, no_sub_x=2, no_sub_y=4, frequency_Hz=5e6,
)

# 2. Beamforming (multi-element only)
tx.compute_delays(focus_mm=[0, 0, 30])
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)

# 3. Field grid (mm)
field_points = {"x_extent": [-5, 5], "y_extent": [-0.5, 0.5],
                "z_extent": [5, 55], "dx": 0.1, "dy": 1.0, "dz": 0.2}

# 4. Emission — 4 modes via constructor flags:
sim = Emission(tx, monochromatic=True)            # CW amplitude at fc → (Nx,Ny,Nz)
sim = Emission(tx)                                 # pulsed transient (raw SIR) → (Nt,...)
sim = Emission(tx, fs=200e6, excitation=exc)       # global excitation (L,)
sim = Emission(tx, fs=200e6, excitation=exc_LE)    # per-element excitation (L,E)
p, coords = sim(field_points, method="auto")       # always returns (pressure, coords)
```

**Reception** (pulse-echo RF): two classes — `ReceptionSDI` (fast PE-SDI kernel,
default) and `Reception` (conventional Tupholme-Stepanishen), same API, sharing
`ReceptionBase` (`base.py`). Physics: the pulse-echo signal carries the 3rd
derivative of the excitation (`v_pe = ρ₀/2c₀² · E_m ⊛ ∂³v/∂t³`); in practice that
∂³ is **baked into** the band-limited excitation + TX/RX impulse responses
(`E_m ⊛ ∂³v/∂t³ ∝ e ⊛ h_e ⊛ h_r`), so neither class applies an explicit ∂³.
`ReceptionSDI` places the two-way delta train `Δδ_pe = D²h_tx ⊛ D²h_rx` (16
deltas/pair, **no cumsum**) and recovers the two-way SIR via `I⁴ = ÷(jω)⁴` in Fourier;
`Reception` builds `h_tx ⊛ h_rx` by FFT directly. Both equal `v_pe ⊛ (h_tx ⊛ h_rx)`.
(`ReceptionSDI` is the *truncated* SDI form — see `PE_SDI_kernel_analysis.md` for the
conventional/truncated/complete taxonomy.) Field II shares the convention
(`calc_scat`≡`calc_hhp`, no explicit ∂³), so both coincide with it — adoption
parallel, not justification. Four methods (axis `[emission, reception,
Nt]`): `pulse_echo_rf` (core, =`__call__`; `per_scatterer=True` gives the PSF),
`sequence_rf` (PW/DW event sweep), `synthetic_aperture_rf` (FMC/`calc_scat_all`,
per-element DW basis, decimated), `scan_focusline` (one focused B-mode line, RX
summed in-kernel). Takes separate TX/RX transducers + scatterer positions; returns
per-element RF `(Erx, Nt)`. `coords["t0"]` is beam-axis referenced. Full details in
`ARCHITECTURE.md`.

```python
from pyfield.reception import ReceptionSDI
sim = ReceptionSDI(tx, rx, fs=200e6, c=1540)
rf, coords = sim.pulse_echo_rf(scatterer_pos_mm, scatterer_amp)        # (Erx, Nt)
psf, coords = sim.pulse_echo_rf(pts, per_scatterer=True)               # (P, Erx, Nt) PSF
env, coords = sim.scan_focusline([0, 0, 30], pts, amp, FoverD=2.0,
                                 apodization_type="hanning")           # (Nt,) one B-mode line
```

**Visualize**: `plot2D_pressure_slices(p, coords=coords, db_scale=True)` (mono 3D or
transient 4D); `plot2D_transient_slices(...)` for transient planes.

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

The pulse-echo signal physically carries the 3rd derivative of the excitation:

    v_pe(t) = (rho_0/2c_0^2) * E_m(t) *_t d3v/dt3   ∝   e(t) *_t h_e(t) *_t h_r(t)

The ∂³ is baked into the band-limited excitation e and TX/RX impulse responses h_e,
h_r (never formed explicitly), same as Field II. So the pulse-echo RF is just
`p_r = v_pe *_t (h_tx *_t h_rx)`.

PE-SDI builds the two-way SIR from the delta product. Each one-way SIR is the double
integral of its corner deltas (h = I² d²h), so:

    Δδ_pe = d2h^e *_t d2h^r = 16 Dirac deltas per (m_e, m_r) pair   (deltas ⊛ deltas)
    h_tx *_t h_rx = I⁴ Δδ_pe

Each TX corner (4) × each RX corner (4) = 16 events; 32 sample writes per pair
(16 × 2 bins via interpolation). `compute_pe_sdi` returns the RAW Δδ_pe — **no cumsum**.
The public RF applies I⁴ entirely in Fourier as ÷(jω)⁴ (with a ×fs: Δδ_pe holds delta
*areas*, ÷(jω) weights each sample by dt), folded into the exc/IR multiply:

    p_r(t) = (rho_0/2c_0^2) * f_m(r) *_r [(E_m * v) *_t (I⁴ Δδ_pe)(r, t)]

This recovers exactly `h_tx *_t h_rx`, so ReceptionSDI ≡ conventional Reception ≡
Field II. ReceptionSDI is the *truncated* SDI form (integrates the delta product); the
*complete* form moves I⁴ onto the velocity (w = I⁴ v_pe). Full taxonomy and complexity:
`PE_SDI_kernel_analysis.md`.

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

## Key Gotchas (read before editing core)

Quick checklist — full rationale, locations, and history in
[`ARCHITECTURE.md` § Risky Implementations](ARCHITECTURE.md#risky-implementations).

1. **SDI float32 cumsum cancellation** — all `_cumsum_*` must use float64 accumulator + float32 write-back. Residual ~0.004% of peak. SIR test tolerance: `rtol=0.005, atol=0.005×peak`.
2. **d2h_all ≠ d2h_per_element.sum()** — float32 non-associativity (~5e-8). Never compare with `atol=0`.
3. **PE SDI delta placement uses `k_shift = 0`** (in `transducer_sir_pe.py`). Was wrongly 2.0 → 2-sample lag. `example06` asserts on-axis lag == 0 as regression guard.
4. **Attenuation y=1 continuity** — `tan(y*pi/2)` diverges near y=1; test the y=1 branch independently.
5. **Global vs per-element excitation** — both paths must use identical per-element dh; divergent cumsums caused 150× near-zero errors.
6. **Numba cache staleness** — after editing kernels, clear `.nb?` cache or fixes "have no effect":
   ```powershell
   Get-ChildItem -Path "src\pyfield\h_sir\__pycache__" -Filter "*.nb?" | Remove-Item -Force
   ```
7. **`h_sir.__call__` is broken** (pre-existing) — do NOT call; use `compute_derivative` or `Emission`/`Reception`.
8. **`from_sir_to_pressure` ignores attenuation when `excitation=None`** — provide excitation if attenuation must apply.

## graphify

Knowledge graph at `graphify-out/` (god nodes, communities, cross-file relationships).

- Codebase questions: run `graphify query "<question>"` first when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships, `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, smaller than `GRAPH_REPORT.md` or raw grep.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
