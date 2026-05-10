---
icon: lucide/book-open-check
---

# Examples

Worked examples that progressively introduce PyField's features — from basic transducer geometry to brain-atlas integration and STL mesh simulation.

Run any example directly:

```bash
uv run examples/exampleN_name.py
```

Output figures are controlled by `examples/config.py`. Set `FIG_FOLDER` to choose where images are saved and toggle `SAVE_FIG` to switch between saving and interactive display.

---

<div class="grid cards" markdown>

-   :lucide-shapes: **[1. Transducer Gallery](example1_transducer_gallery.md)**

    ---

    Every transducer type in one script: linear, convex, matrix arrays and all four circular mono-element geometries, plus a custom TUS helmet.

-   :lucide-circle: **[2. Mono-element Fields](example2_monoelement_transducers.md)**

    ---

    Monochromatic CW pressure fields for all four circular types: flat piston, concave bowl, cylindrical line-focus, and convex dome.

-   :lucide-layout-list: **[3. Linear Array (CW)](example3_lineartx_monochromatic.md)**

    ---

    Continuous-wave simulation with a Domino linear array using a diverging-wave transmit strategy. XZ pressure plane in dB scale.

-   :lucide-box: **[4. Multi-element 3-D](example4_multielement_transducers.md)**

    ---

    Focused field for a linear and matrix array rendered in 3-D with PyVista. Transducer geometry and pressure volume in one scene.

-   :lucide-waves: **[5. Transient Simulation](example5_lineartx_transient.md)**

    ---

    Pulsed emission with Hanning-windowed excitation. Animated GIF of the propagating wavefront through the medium.

-   :lucide-brain: **[6. Mouse Brain Atlas](example6_monoelement_mouse.md)**

    ---

    Concave bowl with focal spot overlaid on a BrainGlobe mouse atlas (`allen_mouse_25um`). Anatomy, transducer, and pressure in one frame.

-   :lucide-target: **[7. Rat Brain Targeting](example7_ratbrainzones_focus.md)**

    ---

    FUS targeting of motor and somatosensory regions. Atlas scaling by lambda–bregma distance and affine registration to the transducer frame.

-   :lucide-file-code: **[8. STL Meshes](example8_importstl_petridish.md)**

    ---

    Loading and visualising STL files of experimental components. Geometric transforms, multiple objects in one scene, and custom lighting.

-   :lucide-flask-conical: **[9. STL + Simulation](example9_monoelement_petridish.md)**

    ---

    End-to-end: concave bowl transducer, pressure field, and petri-dish STL composed in a single PyVista 3-D scene.

-   :lucide-grid: **[10. Surface Subdivision](example10_subdivide_parametric_surface.md)**

    ---

    Direct use of `subdivide_parametric_surface` on an ellipsoidal cap. Patch distribution, coverage, and flat-patch approximation accuracy.

</div>

---

## Prerequisites

| Examples | Requirements |
|----------|-------------|
| 1–5, 10 | Core PyField installation |
| 6–7 | `brainglobe-atlasapi` (atlas data downloaded on first run) |
| 8–9 | `Petri_dish.stl` placed in `examples/` |
