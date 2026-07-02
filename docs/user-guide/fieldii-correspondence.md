---
icon: lucide/arrow-right-left
---

# Field II Correspondence

PyField deliberately mirrors Field II's conventions so a Field II user can
transition (and cross-validate) with minimal friction. This page collects the
correspondences in one place.

## Transducer classes ↔ `xdc_*` functions

| Field II | PyField | z-datum |
|---|---|---|
| `xdc_linear_array` | `LinearArrayTransducer` | flat face at z = 0 |
| `xdc_focused_array` | `LinearArrayTransducer(elevation_focus_mm=...)` | element face (rim) at z = 0, lens dished back to −sag |
| `xdc_convex_array` | `ConvexArrayTransducer` | centre element at z = 0, arc centre at z = −R |
| `xdc_convex_focused_array` | `ConvexArrayTransducer(elevation_focus_mm=...)` | as above + rim-referenced lens |
| `xdc_2d_array` / `xdc_rectangles` | `MatrixArrayTransducer` | flat face at z = 0 |
| `xdc_piston` | `FlatCircularTransducer` | face at z = 0 |
| `xdc_concave` | `ConcaveCircularTransducer` | apex at z = 0, rim at z = +sag |
| — (convex bowl) | `ConvexCircularTransducer` | apex at z = 0, rim at z = −sag |
| `xdc_focused_array` (1 element, cylindrical lens) | `FocusedCircularTransducer` | face (curved-axis rim) at z = 0, centre line at −sag |
| any `Th` via `xdc_get(Th, 'all')` | `from_fieldii_xdc_data` → `FieldIITransducer` | as exported |

`sag = R − √(R² − (D/2)²)` in every case.

Lens tiles of the native lensed arrays are sampled **equal-arc** in the lens
angle θ (nodes at `y = R·sin θ`), exactly matching Field II's tiling of
`xdc_focused_array`.

## Simulation calls

| Field II | PyField |
|---|---|
| `calc_h` | `Emission(tx)` — pulsed mode (returns `ρ₀·h`; identical arrays at `rho=1`) |
| `calc_hp` | `Emission(tx, fs=..., excitation=e)` with `xdc_impulse` ↔ `tx.set_impulse_response` |
| `calc_scat` | `ReceptionSDI(tx, rx).scan_focusline(...)` (focused, apodized, summed on receive) |
| `calc_hhp` / `calc_scat` (unit point) | `ReceptionSDI(tx, rx).pulse_echo_rf(...)` per-element RF |
| `calc_scat_all` | `ReceptionSDI(tx, rx).synthetic_aperture_rf(...)` (FMC) |

The pulse-echo derivative convention is shared: the physical `∂³v/∂t³` is
carried by the band-limited excitation and TX/RX impulse responses — neither
simulator applies an explicit derivative, so `calc_scat` for a unit point
equals `calc_hhp` and PyField's RF coincides with both (correlation ≈ 1.0000).

## Time origin (`t0`)

Field II reports absolute time from the excitation start. PyField returns each
result with `coords["t0"]`, referenced to the **beam axis**: the TX (and RX)
focusing bulk `delays.max()` is subtracted so downstream beamforming needs no
per-line correction.

For a lens-focused aperture, PyField's time grid is referenced to the
first-arriving rim, but the focused elevation echo peaks one lens transit
later; reception therefore adds `elevation_lens_sag / c` once per aperture
(TX and RX). With this, a native elevation-focused linear array matches
Field II pulse-echo RF at lag 0.

**Imported probes:** the lens curvature is already in the imported patches
(the RF *shape* matches without help), but the `t0` transit correction needs
the lens focal length — pass `elevation_focus_mm=` (the Field II `Rfocus`) to
`from_fieldii_xdc_data` / `FieldIITransducer`, or set
`tx.elevation_lens_sag` (metres) directly.

## Amplitude scale

PyField scales the pulse-echo RF by `ρ₀ / 2c₀²` (the physical scattering
prefactor); Field II uses approximately `ρ₀ / 2`. For a unit-amplitude
scatterer the raw RF amplitudes therefore differ by a factor `c₀²`
(≈ 2.37 × 10⁶ at 1540 m/s). Normalised comparisons (envelope / peak, PSF,
correlation) are unaffected.

## Units

Field II is SI everywhere (metres, seconds). PyField's user-facing API is
**mm** (`_mm` suffixes) with SI internals — the one deliberate departure.
`FieldIITransducer` takes its patch geometry in metres, exactly as exported
by `xdc_get`.
