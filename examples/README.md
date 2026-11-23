# eSDIva examples

Run any script with `uv run examples/<tier>/<script>.py`. Each one explains its
physics in its first lines and is split into `# %%` cells (run cell by cell in
VS Code, Spyder or Jupyter).

## `learn/` — start here, in order

| # | Script | Physics you learn |
|---|--------|-------------------|
| 01 | `01_transducers.py` | apertures are flat patches; delays and apodization |
| 02 | `02_piston_and_bowl_cw.py` | near field, far field, geometric focus |
| 03 | `03_array_focusing_cw.py` | focal width ≈ λ·F/D |
| 04 | `04_plane_vs_diverging_wave.py` | the delay law shapes the wavefront |
| 05 | `05_attenuation.py` | α·f^y loss with depth |
| 06 | `06_pulse_echo_rf.py` | channel RF: one hyperbola per scatterer |
| 07 | `07_psf.py` | PSF = transmit beam × receive beam |
| 08 | `08_bmode_line_by_line.py` | depth of field of a fixed focus |
| 09 | `09_full_matrix_capture.py` | N × N signals, reciprocity |
| 10 | `10_speckle_phantom.py` | speckle and cyst contrast |
| 11 | `11_sparse_spiral_array.py` | grating lobes, sin θ = λ/d |
| 12 | `12_why_sdi.py` | the SDI trick inside eSDIva |

## `applications/` — real setups

`brain_atlas_targeting.py` (needs `uv sync --extra atlas`), `petri_dish_setup.py`,
`import_fieldii_probe.py`, `matrix_steered_pw_3d.py`, `pitch_catch_two_probes.py`,
`transform_and_refocus.py`, `custom_surface_element.py`.

## `advanced/` — case studies (minutes to hours)

`3d_phantom_volume/` and `flow_doppler/` (multi-step studies, each with its own
README), `multilens_element_fit.py`, `multilens_matrix_psf.py`,
`rca_xam_sound_sheet.py`. Their figure output is set in `advanced/config.py`.

## Figures and tests

`just figures` reruns `learn/` and `applications/` headless and saves every figure
to `docs/examples/assets/`; `just test-examples` checks they all still run.
Developer scripts (interactive SDI slider, Doppler bias budget, video maker) live in
`tools/`.
