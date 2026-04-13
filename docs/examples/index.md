# Examples

PyField ships with a set of worked examples that progressively introduce the
library's features — from basic transducer geometry all the way to brain-atlas
integration and STL mesh import.

Every example script lives in the `examples/` folder at the repository root and
can be run directly:

```bash
uv run examples/exampleN_name.py
```

Set `SAVE_FIG = True` at the top of any script to save output figures to
`examples/assets/` instead of opening interactive windows.

## Learning path

| # | Example | What you will learn |
|---|---------|---------------------|
| 1 | [Transducer Gallery](example1_transducer_gallery.md) | Meet every transducer type available in PyField |
| 2 | [Mono-element Pressure Fields](example2_monoelement_transducers.md) | Simplest CW simulation with circular transducers |
| 3 | [Linear Array (CW)](example3_lineartx_monochromatic.md) | Multi-element monochromatic diverging-wave field |
| 4 | [Multi-element 3-D](example4_multielement_transducers.md) | Linear + matrix array comparison with 3-D rendering |
| 5 | [Transient Simulation](example5_lineartx_transient.md) | Pulsed excitation and wavefront animation |
| 6 | [Mouse Brain Atlas](example6_monoelement_mouse.md) | Anatomy integration with BrainGlobe (mouse) |
| 7 | [Rat Brain Targeting](example7_ratbrainzones_focus.md) | Coordinate transforms and region targeting (rat) |
| 8 | [STL Meshes](example8_importstl_petridish.md) | Loading experimental geometry from STL files |
| 9 | [STL + Simulation](example9_monoelement_petridish.md) | Complete experimental setup visualisation |

## Prerequisites

Examples 1–5 only require the core PyField installation.  Examples 6–7 additionally
need the BrainGlobe atlas packages (`brainglobe-atlasapi`).  Examples 8–9 require an
STL file (`Petri_dish.stl`) placed in the `examples/` folder.
