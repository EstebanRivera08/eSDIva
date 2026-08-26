---
icon: lucide/book-open-check
---

# Examples

Worked examples that progressively introduce eSDIva's features — from basic transducer geometry to pulse-echo imaging, brain-atlas integration and STL mesh simulation.

Run any example directly:

```bash
uv run examples/exampleN_name.py
```

Output figures are controlled by `examples/config.py`. Set `FIG_FOLDER` to choose where images are saved and toggle `SAVE_FIG` (or set `ESDIVA_SAVE_FIG=1`) to switch between saving and interactive display.

---

## Transducers & Emission

<div class="grid cards" markdown>

-   :lucide-shapes: **[1. Transducer Gallery](example01_transducer_gallery.md)**

    ---

    Every transducer type in one script: linear, convex, matrix arrays and all four circular mono-element geometries, plus a custom TUS helmet.

-   :lucide-circle: **[2. Mono-element Fields](example02_monoelements_monochromatic_CW.md)**

    ---

    Monochromatic CW pressure fields for all four circular types: flat piston, concave bowl, cylindrical line-focus, and convex dome.

-   :lucide-box: **[3. Multi-element 3-D](example03_multielements_monochromatic_CW.md)**

    ---

    Focused field for a linear and matrix array rendered in 3-D with PyVista. Transducer geometry and pressure volume in one scene.

-   :lucide-waves: **[4. Diverging Wave (Transient)](example04_lineararray_excitation_DW.md)**

    ---

    Pulsed diverging-wave emission with a virtual source behind the array. Animated GIF of the propagating wavefront.

-   :lucide-move-3d: **[5. Steered Plane Wave (3-D)](example05_matrixarray_pulsed_steeredPW.md)**

    ---

    Matrix-array plane wave steered off-axis; transient field animated in 3-D on two orthogonal planes with the transducer mesh.

</div>

## Reception & Imaging

<div class="grid cards" markdown>

-   :lucide-crosshair: **[6. Concave Pulse-Echo PSF](example06_concave_PSF.md)**

    ---

    Point spread function of a focused single-element transducer; conventional vs PE-SDI backends agree to a fraction of a percent.

-   :lucide-radio: **[7. Focused TX, All-RX](example07_lineararray_TXfocus_RXall.md)**

    ---

    Single-line B-mode acquisition: focused transmit, all channels receive. Includes the `sim.show()` 3-D setup preview.

-   :lucide-grid-3x3: **[8. Full Matrix Capture](example08_synthetic_aperture_FMC.md)**

    ---

    Every TX element fires individually while all RX record — the dataset behind synthetic aperture and TFM imaging.

-   :lucide-scan-line: **[9. B-mode PSF Image](example09_lineararray_imagePSF.md)**

    ---

    The classic Field II PSF phantom imaged line by line with `scan_focusline` and log-compressed to a B-mode image.

</div>

## Attenuation & Safety Metrics

<div class="grid cards" markdown>

-   :lucide-activity: **[10. Peak Pressure & Ispta](example10_intensities_peak_pressure.md)**

    ---

    On-axis peak pressure and Ispta vs depth, with and without tissue attenuation — the quantities behind acoustic-output safety metrics.

-   :lucide-thermometer: **[11. CW Field with Attenuation](example11_lineararray_attenuations.md)**

    ---

    Water vs brain tissue: causal power-law attenuation with Kramers–Kronig dispersion reshaping a 10 MHz focused beam.

</div>

## Applications

<div class="grid cards" markdown>

-   :lucide-brain: **[12. Mouse Brain Atlas](example12_txconcave_mousebrain.md)**

    ---

    Concave bowl with focal spot overlaid on a BrainGlobe mouse atlas. Anatomy, transducer, and pressure in one frame.

-   :lucide-target: **[13. Rat Brain Targeting](example13_txlinear_ratbrainzones.md)**

    ---

    FUS targeting of motor and somatosensory regions. Atlas scaling by lambda–bregma distance and registration to the transducer frame.

-   :lucide-file-code: **[14. STL Meshes](example14_importstl_petri_dish.md)**

    ---

    Loading and visualising STL files of experimental components. Transforms, multiple objects in one scene, and custom lighting.

-   :lucide-flask-conical: **[15. STL + Simulation](example15_monoelement_petridish.md)**

    ---

    End-to-end: concave bowl transducer, pressure field, and petri-dish STL composed in a single PyVista 3-D scene.

-   :lucide-grid: **[16. Surface Subdivision](example16_subdivide_parametric_surface.md)**

    ---

    Direct use of `subdivide_parametric_surface` on an ellipsoidal cap. Patch distribution, coverage, and flat-patch accuracy.

-   :lucide-import: **[17. Import a Field II Probe](example17_import_fieldii_transducer.md)**

    ---

    One-line MATLAB export → `from_fieldii_rect_data` → a native eSDIva transducer with the original delays and apodization.

-   :lucide-spline: **[18. Custom Sparse Array](example18_customtransducer_3Dplanes.md)**

    ---

    64-element spiral array assembled with `CustomTransducer`, repositioned with `transform()`, and visualised on 3-D plane slices.

-   :lucide-git-compare-arrows: **[19. Dual-Probe Pulse-Echo](example19_dualprobe_reception_show.md)**

    ---

    Pitch-catch: a second array tilted 30° with `transform()` receives from an oblique angle. Setup previewed with `sim.show()`.

-   :lucide-circle-dot-dashed: **[20. Speckle Phantom B-mode](example20_phantom_simulation.md)**

    ---

    `make_phantom` cyst phantom, piezo impulse response set, focused B-mode line-by-line with `scan_focusline`.

-   :lucide-box: **[21. Volumetric Case Study](example21_rca_volume.md)**

    ---

    Full pipeline: shared phantom, diverging-wave sequence, checkpointed `sequence_rf` acquisition, `das_volume` IQ compounding, honest metrics.

</div>

---

## Prerequisites

| Examples | Requirements |
|----------|-------------|
| 1–11, 16–21 | Core eSDIva installation |
| 12–13 | `brainglobe-atlasapi` (atlas data downloaded on first run) |
| 14–15 | `Petri_dish.stl` placed in `examples/` |
