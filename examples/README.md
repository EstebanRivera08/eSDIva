# PyField Examples

Complete example suite demonstrating every PyField capability.
Run any example with `uv run examples/<script>.py`.

---

## Visualization Tools

| Script | Description |
|--------|-------------|
| `visualization_trapezoid_SDI_vs_FWT.py` | Interactive comparison of trapezoid integration methods (SDI vs FWT) with parameter sliders. |

---

## Core Examples (01 – 11)

| # | Script | Description | Field II parallel |
|---|--------|-------------|:-----------------:|
| 01 | `example01_transducer_gallery.py` | 3-D gallery of all 7 transducer types with geometry, delays, and apodization visualized in PyVista. | |
| 02 | `example02_monoelements_monochromatic_CW.py` | CW pressure fields for all circular mono-element transducers (flat, concave, focused, convex). | |
| 03 | `example03_multielements_monochromatic_CW.py` | Focused linear array CW field with `Emission(monochromatic=True)`; runtime update via `sim.set()`. | |
| 04 | `example04_lineararray_excitation_DW.py` | Diverging-wave transient emission with Hanning-windowed excitation; animated wavefront propagation. | |
| 05 | `example05_matrixarray_pulsed_steeredPW.py` | Steered plane-wave pulsed emission on a matrix array; 3-D PyVista animation via `plot3D_transient_slices`. | |
| 06 | `example06_concave_PSF.py` | Pulse-echo PSF of a concave single-element transducer; lateral × time RF image. | ✓ `example_concave_psf.m` |
| 07 | `example07_lineararray_TXfocus_RXall.py` | Focused TX with all-RX reception; RF waterfall and envelope of centre channel. | |
| 08 | `example08_anotherreceptionexample.py` | Full Matrix Capture (FMC) with `ReceptionSDI.rf_matrix()`; peak amplitude matrix visualization. | |
| 09 | `example09_lineararray_imagePSF.py` | Multi-focus RF acquisition + DAS beamforming (`pyfield.beamforming`) → B-mode PSF image. | ✓ `example_point_spread_functions.m` |
| 10 | `example10_intensities_peak_pressure.py` | On-axis peak pressure and Ispta vs depth with and without tissue attenuation. | ✓ `example_intensity.m` |
| 11 | `example11_lineararray_attenuations_monochromatic_CW.py` | CW pressure field comparison: water vs brain tissue (causal K-K attenuation). | ✓ `example_lineararray_attenuation_monochromatic_CW.m` |

> **Field II notes**
> - Field II uses a non-causal linear-frequency attenuation approximation.
>   PyField implements causal power-law attenuation with Kramers–Kronig dispersion
>   (Szabo 1994 / Holm 2019).
> - Field II `calc_hhp` (pulse-echo response / PSF, 1 derivative) ↔
>   `Reception.pulse_echo_response()`. Field II `calc_scat` (scattered echo,
>   3 derivatives) ↔ `Reception.scattered_rf()` / `ReceptionSDI.scattered_rf()`.

---

## Extras (12 – 16)

Advanced or application-specific examples.

| # | Script | Description |
|---|--------|-------------|
| 12 | `example12_txconcave_mousebrain.py` | Focused ultrasound targeting mouse brain anatomy (BrainGlobe atlas + PyVista). |
| 13 | `example13_txlinear_ratbrainzones.py` | Linear array focused on specific rat-brain zones (M1, S1-hl) using WHS atlas. |
| 14 | `example14_importstl_petri_dish.py` | Import and visualise STL meshes — transformations, multiple objects, materials. |
| 15 | `example15_monoelement_petridish.py` | Combine STL experimental setup (Petri dish) with acoustic simulation in 3-D scene. |
| 16 | `example16_subdivide_parametric_surface.py` | Parametric surface subdivision utility demo using an ellipsoidal cap. |

---

## Configuration

`config.py` (in this directory) controls figure output:

```python
SAVE_FIG = False            # True → save figures to FIG_FOLDER
FIG_FOLDER = ...            # destination directory (default: docs/examples/assets)
SCALE = 3                   # DPI multiplier for saved screenshots
```

---

## API Coverage

| Feature | Example(s) |
|---------|-----------|
| `Emission(monochromatic=True)` | 02, 03, 11 |
| `Emission(excitation=...)` | 04, 05, 10 |
| `Emission` per-element excitation `(L, E)` | — (see CLAUDE.md) |
| `Emission.set()` runtime update | 03 |
| `Reception.scattered_rf()` (calc_scat) | 06, 07 |
| `Reception.pulse_echo_response()` (calc_hhp PSF) | — |
| `ReceptionSDI.rf_sequence()` | 09 |
| `ReceptionSDI.rf_matrix()` FMC | 08 |
| Attenuation (`alpha0`, `freq_power`) | 10, 11 |
| `pyfield.beamforming.das` | 09 |
| `pyfield.beamforming.envelope_db` | 09 |
| `plot2D_pressure_slices` | 03, 04, 11 |
| `plot2D_transient_slices` | — |
| `plot3D_transient_slices` | 05 |
| `align_to_common_time` | — |
| Brain atlas integration | 12, 13 |
| STL mesh import | 14, 15 |
| Surface subdivision | 16 |
