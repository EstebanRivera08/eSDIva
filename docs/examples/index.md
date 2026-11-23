---
icon: lucide/book-open-check
---

# Examples

Three tiers, from first contact to full research studies. Every script is short,
runs on its own, and explains its physics in its first lines.

```bash
uv run examples/learn/01_transducers.py
```

Each file is split into `# %%` cells, so you can also run it cell by cell in VS Code,
Spyder or Jupyter.

## Learn — start here

One physical idea per script, 40–80 lines, in this order:

| # | Example | You learn |
|---|---------|-----------|
| 01 | [Transducers](learn/01_transducers.md) | an aperture is a set of small patches; delays and apodization |
| 02 | [Piston & bowl (CW)](learn/02_piston_and_bowl_cw.md) | near field, far field, geometric focusing |
| 03 | [Array focusing (CW)](learn/03_array_focusing_cw.md) | focal width ≈ λ·F/D |
| 04 | [Plane vs diverging wave](learn/04_plane_vs_diverging_wave.md) | the delay law shapes the transient wavefront |
| 05 | [Attenuation](learn/05_attenuation.md) | α·f^y loss with depth, and why a pulse loses less |
| 06 | [Pulse-echo RF](learn/06_pulse_echo_rf.md) | channel data: one hyperbola per scatterer |
| 07 | [Point spread function](learn/07_psf.md) | PSF = transmit beam × receive beam |
| 08 | [B-mode, line by line](learn/08_bmode_line_by_line.md) | depth of field of a fixed focus |
| 09 | [Full matrix capture](learn/09_full_matrix_capture.md) | N × N signals, reciprocity |
| 10 | [Speckle phantom](learn/10_speckle_phantom.md) | speckle, cyst contrast |
| 11 | [Grating lobes & sparse arrays](learn/11_sparse_spiral_array.md) | sin θ = λ/d, and how a spiral removes them |
| 12 | [Why SDI](learn/12_why_sdi.md) | the trick inside eSDIva's SIR engine |

## Applications

Real setups built from the same pieces.

- [Brain atlas targeting](applications/brain_atlas_targeting.md): focus on a rat brain region (needs the `atlas` extra).
- [STL lab setup](applications/petri_dish_setup.md): a focused beam inside a Petri dish loaded from CAD.
- [Import a Field II probe](applications/import_fieldii_probe.md): reuse an `xdc_get(Th, 'rect')` export.
- [Matrix steered plane wave (3-D)](applications/matrix_steered_pw_3d.md): 2-D steering, 3-D animation.
- [Pitch-catch, two probes](applications/pitch_catch_two_probes.md): place probes anywhere with `aim`.
- [Move & refocus a probe](applications/transform_and_refocus.md): delays follow the new pose.
- [Custom curved element](applications/custom_surface_element.md): any `z(r)` surface, and how fine to tile it.

## Advanced case studies

Full studies and paper reproductions. They run for minutes to hours and assume the
Learn tier.

- [3-D phantom volume](advanced/3d_phantom_volume.md): diverging-wave matrix acquisition, `das_volume`, compounding, metrics.
- [Ultrafast compound Doppler](advanced/flow_doppler.md): Bercoff 2011 replicated on a moving phantom.
- [Multilens element fit](advanced/multilens_element_fit.md) and [multilens matrix PSF](advanced/multilens_matrix_psf.md).
- [RCA xAM sound sheet](advanced/rca_xam_sound_sheet.md): nonlinear sound-sheet sequence with `Field` arithmetic.

Figures on these pages are regenerated with `just figures`.
