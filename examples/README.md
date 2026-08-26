# eSDIva Examples

Complete example suite demonstrating every eSDIva capability.
Run any example with `uv run examples/<script>.py`.

---

## Visualization Tools

| Script | Description |
|--------|-------------|
| `visualization_trapezoid_SDI_vs_FWT.py` | Interactive comparison of trapezoid integration methods (SDI vs FWT) with parameter sliders. |

---

## Transducers & Emission (01 – 05)

| # | Script | Description | Field II parallel |
|---|--------|-------------|:-----------------:|
| 01 | `example01_transducer_gallery.py` | 3-D gallery of all 7 transducer types with geometry, delays, and apodization visualized in PyVista. | |
| 02 | `example02_monoelements_monochromatic_CW.py` | CW pressure fields for all circular mono-element transducers (flat, concave, focused, convex). | |
| 03 | `example03_multielements_monochromatic_CW.py` | Focused linear and matrix array CW fields rendered in 3-D with `Emission(monochromatic=True)`. | |
| 04 | `example04_lineararray_excitation_DW.py` | Diverging-wave transient emission with Hanning-windowed excitation; animated wavefront propagation. | |
| 05 | `example05_matrixarray_pulsed_steeredPW.py` | Steered plane-wave pulsed emission on a 32×32 matrix array; 3-D PyVista animation via `plot3D_transient_slices`. | |

## Reception & Imaging (06 – 09)

| # | Script | Description | Field II parallel |
|---|--------|-------------|:-----------------:|
| 06 | `example06_concave_PSF.py` | Pulse-echo PSF of a concave single-element transducer; conventional vs PE-SDI backend comparison. | ✓ `example_concave_psf.m` |
| 07 | `example07_lineararray_TXfocus_RXall.py` | Focused TX with all-RX reception; `sim.show()` setup preview, RF waterfall and envelope. | |
| 08 | `example08_synthetic_aperture.py` | Full Matrix Capture (FMC) with `synthetic_aperture_rf()`; peak amplitude matrix visualization. | |
| 09 | `example09_lineararray_imagePSF.py` | Line-by-line B-mode via `scan_focusline()` (like Field II `calc_scat`) → B-mode PSF image. | ✓ `linear_psf_example/` |

## Attenuation & Safety Metrics (10 – 11)

| # | Script | Description | Field II parallel |
|---|--------|-------------|:-----------------:|
| 10 | `example10_intensities_peak_pressure.py` | On-axis peak pressure and Ispta vs depth with and without tissue attenuation. | ✓ `example_intensity.m` |
| 11 | `example11_lineararray_attenuations_monochromatic_CW.py` | CW pressure field comparison: water vs brain tissue (causal K-K attenuation). | ✓ `example_lineararray_attenuation_monochromatic_CW.m` |

> **Field II notes**
> - Field II uses a non-causal linear-frequency attenuation approximation.
>   eSDIva implements causal power-law attenuation with Kramers–Kronig dispersion
>   (Szabo 1994 / Holm 2019).
> - Field II `calc_hhp` ≡ `calc_scat` (unit point) ↔ `Reception.pulse_echo_rf()`
>   (any `method`) — zero explicit temporal derivatives; all pulse shaping lives in
>   excitation + impulse responses. RF correlation ≈ 0.997.

## Applications (12 – 19)

| # | Script | Description |
|---|--------|-------------|
| 12 | `example12_txconcave_mousebrain.py` | Focused ultrasound targeting mouse brain anatomy (BrainGlobe atlas + PyVista). |
| 13 | `example13_txlinear_ratbrainzones.py` | Linear array focused on specific rat-brain zones (M1, S1-hl) using WHS atlas. |
| 14 | `example14_importstl_petri_dish.py` | Import and visualise STL meshes — transformations, multiple objects, materials. |
| 15 | `example15_monoelement_petridish.py` | Combine STL experimental setup (Petri dish) with acoustic simulation in 3-D scene. |
| 16 | `example16_subdivide_parametric_surface.py` | Parametric surface subdivision utility demo using an ellipsoidal cap. |
| 17 | `example17_import_fieldii_transducer.py` | Import a Field II probe (`xdc_get(Th,'rect')` export) with `from_fieldii_rect_data`. |
| 18 | `example18_customtransducer_3Dplanes.py` | Sparse spiral `CustomTransducer`, `transform()` repositioning, 3-D plane-slice visualization. |
| 19 | `example19_dualprobe_reception_show.py` | Pitch-catch pulse-echo: RX array tilted with `transform()`; `sim.show()` 3-D preview. |
| 20 | `example20_phantom_simulation.py` | Speckle phantom via `make_phantom()` (cyst + lesion), piezo impulse response set, focused B-mode with `scan_focusline()`. |
| 21 | `example21_3Dphantom_volume/` | Full volumetric case study: shared phantom, diverging-wave sequence, checkpointed `sequence_rf` acquisition, `das_volume` IQ compounding, honest metrics. See its `README.md` (incl. "Design notes & pitfalls"). |

---

## Configuration

`config.py` (in this directory) controls figure output:

```python
SAVE_FIG = False            # True → save figures to FIG_FOLDER
FIG_FOLDER = ...            # destination directory (default: docs/examples/assets)
SCALE = 3                   # DPI multiplier for saved screenshots
```

`ESDIVA_SAVE_FIG=1` in the environment overrides `SAVE_FIG` (used to
batch-regenerate documentation figures). On Windows also set `PYTHONUTF8=1`
so Unicode symbols print correctly.

---

## API Coverage

| Feature | Example(s) |
|---------|-----------|
| `Emission(monochromatic=True)` | 02, 03, 11, 17, 18 |
| `Emission(excitation=...)` | 04, 05, 10 |
| `Emission` per-element excitation `(L, E)` | — (see CLAUDE.md) |
| `Emission.set()` runtime update | 18 |
| `pulse_echo_rf()` (calc_scat ≡ calc_hhp) | 06, 07, 19 |
| `pulse_echo_rf(per_scatterer=True)` (PSF) | 06 |
| `scan_focusline()` (conventional B-mode line) | 09, 20 |
| `synthetic_aperture_rf()` FMC | 08 |
| `sequence_rf()` (PW/DW event sweep, checkpointed) | 21 |
| `Reception.show()` 3-D setup preview | 07, 19, 20 |
| `transform()` rigid aperture motion | 18, 19 |
| `CustomTransducer` | 01, 18 |
| `from_fieldii_rect_data` (Field II import) | 17 |
| Attenuation (`alpha0`, `freq_power`) | 10, 11 |
| `esdiva.beamforming.das` | 09 |
| `das_volume` (general 3-D DAS) + IQ compounding | 21 |
| `make_phantom` (speckle phantom) | 20, 21 |
| `plot2D_pressure_slices` | 02, 04, 11, 17 |
| `plot3D_transient_slices` | 05 |
| `create_2Dimage_mesh` / `add_2D_image` | 18 |
| `align_to_common_time` | 05 |
| `explore_mat` (MATLAB utilities) | 17 |
| Brain atlas integration | 12, 13 |
| STL mesh import | 14, 15 |
| Surface subdivision | 16 |
