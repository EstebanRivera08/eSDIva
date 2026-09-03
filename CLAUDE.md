# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo. Keep this file in
sync when code logic changes.

> **Deep reference lives in [`ARCHITECTURE.md`](ARCHITECTURE.md)** — full API
> parameter tables, all emission/reception modes, internal workflows (mode trees,
> data flow), Field II correspondence, performance benchmarks, and the full
> "risky implementations" hazard list. This file holds only what's needed every
> session: overview, commands, module map, a quickstart, physics, and key gotchas.

## Project Overview

eSDIva is a Python acoustic field simulator based on the Tupholme–Stepanishen Spatial
Impulse Response (SIR) method. It models arbitrary transducer geometries as collections
of rectangular patches and computes pressure fields via convolution with excitation pulses.

> **Audience-first documentation (package philosophy).** eSDIva is written for
> ultrasound researchers and students, **not for programmers**. Every docstring and
> comment — public *and* private — must be concise, clear, and **self-sufficient**:
> a physicist must understand the acoustics being computed (and why) without opening
> another file. Explain the physics first, inline. **Never cite markdown files**
> (`ARCHITECTURE.md`, `PE_SDI_kernel_analysis.md`, papers, "see module docstring") from
> a docstring or comment — they drift and the reader may not have them; the code must
> explain itself. Full rules: "Documentation & Comment Philosophy" at the top of
> `.claude/rules/coding-guidelines.md`.

Guidelines load automatically from `.claude/rules/`:
- **coding-guidelines** — code style, testing, commits, **doubt-driven development**
  (never assert an untested physical cause; run the doubt cycle on non-trivial
  physics/diagnosis claims) (always loaded)
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

## User-Facing Agent Skills (`skills/`)

`skills/` holds portable Agent Skills for **users** of the package (distinct from
`.claude/skills/`, which are maintainer skills for work inside the repo). Plain
`SKILL.md` + `name`/`description` front matter, so Claude Code, Codex (`.agents/skills/`)
and OpenCode all read them unchanged; the repo root doubles as a Claude Code plugin
(`.claude-plugin/`). Two skills: `esdiva-simulate` (SKILL.md + `references/`
transducers, emission, reception, visualization, physics + four `# %%`-celled
templates) and `esdiva-contribute` (issue + small-PR workflow).
`tests/integration/test_skill_templates.py` executes every template, so a public API
change fails CI instead of a user's first session. Keep the references in sync when
a public API, convention or default changes — they are documentation users read
without the code. Install matrix: `skills/README.md`. Root `AGENTS.md` points
non-Claude agents at both this file and `skills/`.

## Architecture

### Module Structure (subject to change as project evolves)

1. **`src/esdiva/hsir/`** — Spatial Impulse Response computation. **Core engine — modify carefully.**
2. **`src/esdiva/transducers/`** — Transducer geometry. Under active development; prioritize generalization, keep backward compatibility.
3. **`src/esdiva/emission/`** — Emission simulation (`Emission`, deprecated `eSDIva` alias). **Primary API — keep intuitive/consistent.**
4. **`src/esdiva/reception/`** — Reception simulation (`Reception`, backed by `ReceptionConventional`). **Primary API.**
5. **`src/esdiva/attenuation/`** — Power-law attenuation transfer functions.
6. **`src/esdiva/utilities/`** — Helpers, surface subdivision, brain-atlas integration.
7. **`src/esdiva/plotting/`** — Visualization (2D Matplotlib, 3D PyVista).
8. **`src/esdiva/beamforming/`** — RF post-processing: `DAS_focused_scanline` (one line), `das_rca_volume` (numba 3-D DAS for row-column plane-wave sequences), `das_volume` (general numba 3-D DAS, TX=RX: each event dict carries `delays`/`apodization` + `virtual_source_mm` (DW z<0 / focused z>0 / synthetic-aperture z≈0) or `angles_deg` (PW/multiplane (θx,θy)); the TX time origin is recovered from the event's own delays, so no min/max delay-reference convention is assumed; `coherence_weight=True` multiplies each voxel by its aperture coherence factor to suppress incoherent clutter), `envelope_db`. (Diverging waves use `das_volume` with `virtual_source_mm` z<0 — the former `das_dw_volume` was a redundant subset, removed.)
9. **`src/esdiva/io/`** — `RFDataset`: checkpointed on-disk RF store (one compressed `.npz` per TX event + `contents.json` with a config fingerprint; atomic writes, resume skips completed events, changed config refuses with a diff; `load_all` sums chunk groups when written with `checkpoint_chunks > 1`). `save_rf_hdf5(path, rf, coords, ...)` / `RFDataset.to_hdf5(path)` export one self-describing HDF5 file (channel data + timing, UFF-compatible field names: `sampling_frequency`/`initial_time`/`sound_speed`) for MATLAB/USTB interchange — the `.npz` store stays the internal checkpoint format.

### Key Design Patterns

- **Patch-based discretization**: transducers decompose into small rectangular patches; `no_sub_x`/`no_sub_y` control subdivision density and accuracy.
- **Lazy geometry loading**: `TransducerBase` defers element-center/patch-vertex computation until needed.
- **SIR method selection**: `"FST"` (slow reference), `"sdi"` (Sparse Delta Integration, faster on large grids), `"auto"` (picks based on grid properties).
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
the transducer. `transform(T_matrix)` (4×4 homogeneous, translation in mm)
rigidly moves ALL computed geometry — quads, patch frames, element centers — so
simulation and visualization both follow; simulators snapshot geometry at
construction, so call `sim.set("transducer"/"tx"/"rx", t)` after transforming
(`clean()` reverts to the canonical pose). See `.claude/rules/transducers.md`
for mono vs multi-element, `focus_mm`, z-convention, subdivision methods.

### Brain Atlas Integration
Uses BrainGlobe API to map acoustic fields onto anatomical structures. Requires
downloading atlas data (rat, mouse, ...) on first use.

## Simulation Quickstart

Full parameter tables, all modes, return conventions, and visualization options:
see [`ARCHITECTURE.md` § Full API Reference](ARCHITECTURE.md#full-api-reference).

```python
from esdiva.transducers import LinearArrayTransducer
from esdiva.emission import Emission

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

**Reception** (pulse-echo RF): one public class — `Reception` (the fast PE-SDI kernel),
with `ReceptionConventional` (conventional Tupholme-Stepanishen) as the sampled-convolution
backend it delegates to. Both share `ReceptionBase` (`base.py`). Physics: the pulse-echo
signal carries the 3rd derivative of the excitation (`v_pe = ρ₀/2c₀² · E_m ⊛ ∂³v/∂t³`); in
practice that ∂³ is **baked into** the band-limited excitation + TX/RX impulse responses
(`E_m ⊛ ∂³v/∂t³ ∝ e ⊛ h_e ⊛ h_r`), so neither applies an explicit ∂³.
`Reception` selects how `v_pe ⊛ (h_tx ⊛ h_rx)` is evaluated via `method=` (default
`spectral`): **`spectral`** (closed-form one-way spectra `Σ_TX·Σ_RX = F{Δδ_pe}`, **no
forward FFT**, cost ∝ M, exact, band-limited bins only, every RX element's spectrum built in
one batched kernel call, supports per-patch one-way attenuation); **`fst` / `sdi` / `auto`**
(sample both SIRs and FFT-convolve — delegates to `ReceptionConventional`; the string is its
SIR-sampling kernel, `auto` lets it choose per grid); **`paired`** (pedagogic reference
only — the two-way delta train `Δδ_pe = D²h_tx ⊛ D²h_rx`, 16 deltas/pair; pushes `I⁴` onto
the drive `w = I⁴ v_pe` once and splats a copy of `w` per corner event — **no FFT, no
cumsum**, exact but cost ∝ M²·len(w), so far slower than `spectral` and **warns on
selection**). Field II shares the convention
(`calc_scat`≡`calc_hhp`, no explicit ∂³), so both coincide with it — adoption
parallel, not justification. Four methods (axis `[emission, reception,
Nt]`): `pulse_echo_rf` (core, =`__call__`; `per_scatterer=True` gives the PSF),
`sequence_rf` (PW/DW event sweep; `out_path=` checkpoints each event to an
`RFDataset` folder — crash-safe, resumable, refuses a changed config;
`checkpoint_chunks=N` splits each event into N scatterer chunks checkpointed
separately — zero-amplitude grid-sentinel points pin one time grid per event so
the chunk RFs sum exactly),
`synthetic_aperture_rf` (FMC/`calc_scat_all`,
per-element DW basis, decimated), `scan_focusline` (one focused B-mode line, RX
summed in-kernel). `pulse_echo_rf` and `synthetic_aperture_rf` accept the same
`out_path=`/`checkpoint_chunks=` (both route through `sequence_rf`:
`pulse_echo_rf` wraps its current TX focus into a one-event sequence so the
fingerprint covers it; `synthetic_aperture_rf` turns its groups into events —
its `out_path` is an `RFDataset` folder now, no longer a raw `.npy` memmap). `show(scatterer_positions_mm, amplitudes)` previews the setup
in 3-D (TX/RX meshes + scatterers coloured/faded by amplitude). Takes separate TX/RX transducers + scatterer positions; returns
per-element RF `(Erx, Nt)`. Scatterer positions may also be an Emission-style grid
dict (`x_extent`/`dx`…) → regular lattice of unit point targets (PSF maps; NOT a
phantom — periodic lattices give coherent echoes, not speckle). For phantoms use
`esdiva.utilities.make_phantom(extents_mm, n, echogenicity_map)` → random positions
+ `N(0,1)·map(r)` amplitudes (see `example20`). `coords["t0"]` is beam-axis referenced. Full details in
`ARCHITECTURE.md`.

```python
from esdiva.reception import Reception
sim = Reception(tx, rx, fs=200e6, c=1540)
rf, coords = sim.pulse_echo_rf(scatterer_pos_mm, scatterer_amp)        # (Erx, Nt)
psf, coords = sim.pulse_echo_rf(pts, per_scatterer=True)               # (P, Erx, Nt) PSF
env, coords = sim.scan_focusline([0, 0, 30], pts, amp, FoverD=2.0,
                                 apodization_type="hanning")           # (Nt,) one B-mode line
```

**Imaging-study checklist** (full recipe + measured evidence:
[`ARCHITECTURE.md` § Imaging Simulation Recipe](ARCHITECTURE.md#imaging-simulation-recipe-phantom--sequence-studies)):

1. Set `tx.impulse_response` AND `rx.impulse_response` (2-cycle burst at fc), bare
   drive — skipping IRs widens the PSF ~60 % and raises sidelobe clutter (aperture
   diffraction tails dominate the spectrum). RF-checkpoint fingerprint does NOT
   cover IRs: delete the RF folder after changing the pulse model.
2. Derive PW/DW virtual sources per probe from the coverage rule (every volume
   corner inside every event's cone) — never copy a VS layout between probes.
3. Phantom: ≥5–10 scatterers per resolution cell (cell ~λ³), anechoic targets
   ≥3 PSF radii, wires dim (+10 dB) and far from contrast targets.
4. Preview (`sim.show`) + one-event speckle check BEFORE the long run.
5. Beamform: `coords["t0"]` is the **beamforming reference**, not the instant of
   the first sample — it is set so an echo peaks at its geometric round-trip time
   (the two-way pulse lag is already subtracted), matching what USTB calls
   `initial_time` and what MUST's `dasmtx` assumes. So **any** beamformer, built-in
   or custom, reads the sample at `(t_tx + t_rx − t0)·fs` with no lag term;
   `t_offset_s` defaults to `0.0` and exists for foreign RF (raw Field II
   `calc_scat`, which still carries the lag) or a system delay. `das_volume`
   additionally recovers each event's delay reference. Rect RX apodization
   (element directivity already tapers); RCA bars → `das_rca_volume`.
6. Metrics: TGC from speckle-only, PSF-scaled ROIs/margins (λz/D units, not mm),
   plain DAS numbers (CF only as ceiling), ~30 dB display window.

**Visualize**: `plot2D_pressure_slices(p, coords=coords, db_scale=True)` (mono 3D or
transient 4D); `plot2D_transient_slices(...)` for transient planes.

---

## Mathematical Foundations

All SIR/SDI equations — trapezoidal SIR params, the SDI delta train, the sum
over M patches, the emission/pulse-echo signal chains, the three PE-SDI
evaluations (`spectral`/`paired`/conventional), causal power-law attenuation, and
plane-wave steering delays — live in `.claude/rules/physics-context.md`, which
auto-loads whenever you touch `hsir/`, `emission/`, `reception/`, or
`transducers/`. Attenuation *implementation* rules: `.claude/rules/attenuation.md`.
Kept in one place so the physics can't drift between two copies.

---

## Key Gotchas (read before editing core)

Quick checklist — full rationale, locations, and history in
[`ARCHITECTURE.md` § Risky Implementations](ARCHITECTURE.md#risky-implementations).

1. **SDI float32 cumsum cancellation** — the inline double cumsum in `compute_parallelized_sir_optimized` (`farfield_rect_patch.py`) accumulates in a float64 scalar (`acc`/`acc2`) and writes back to the float32 `d2h`/`h_out`. The delta placement in `_place_sir_sdi_deltas` casts each split write with `np.float32(...)` before the `+=` (matches the cumsum's rounding). Residual ~0.004% of peak. SIR test tolerance: `rtol=0.005, atol=0.005×peak`.
2. **d2h_all ≠ d2h_per_element.sum()** — float32 non-associativity (~5e-8). Never compare with `atol=0`.
3. **PE SDI on-axis lag must be 0** — the delta placement in `transducer_sir_pe_sdi.py` was once a 2-sample lag bug; `example06` asserts on-axis lag == 0 as the regression guard.
4. **Attenuation y=1 continuity** — `tan(y*pi/2)` diverges near y=1; test the y=1 branch independently.
5. **Global vs per-element excitation** — both paths must use identical per-element dh; divergent cumsums caused 150× near-zero errors.
6. **Numba cache staleness** — after editing kernels, clear `.nb?` cache or fixes "have no effect":
   ```powershell
   Get-ChildItem -Path "src\esdiva\hsir\__pycache__" -Filter "*.nb?" | Remove-Item -Force
   ```
7. **`from_sir_to_pressure` ignores attenuation when `excitation=None`** — provide excitation if attenuation must apply.

## graphify

Knowledge graph at `graphify-out/` (god nodes, communities, cross-file relationships).

- Codebase questions: run `graphify query "<question>"` first when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships, `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, smaller than `GRAPH_REPORT.md` or raw grep.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
