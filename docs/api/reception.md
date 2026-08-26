---
icon: lucide/activity
---

# Reception

Pulse-echo RF simulator. See the [Reception user guide](../user-guide/reception.md)
for the method taxonomy, PSF, and phantom recipes.

## Reception

The single pulse-echo class. Its `method` selector chooses how the two-way SIR is
evaluated — all methods give the same RF, they trade speed only:

- `"spectral"` (default) — fast sparse-delta kernel via closed-form one-way SIR spectra.
- `"fst"` / `"sdi"` / `"auto"` — sampled two-way SIR convolution (delegated to the
  conventional `ReceptionConventional` backend; the string names its SIR-sampling kernel).
- `"paired"` — exact but slow pedagogic reference (warns on selection).

::: esdiva.reception.Reception
    options:
      members:
        - pulse_echo_rf
        - sequence_rf
        - synthetic_aperture_rf
        - scan_focusline
        - show
        - set

## ReceptionConventional (backend)

The conventional Tupholme-Stepanishen sampled-convolution backend that `Reception`
delegates to for `method="fst"/"sdi"/"auto"`. Same API; normally reached through
`Reception`, documented here for reference.

::: esdiva.reception.ReceptionConventional
    options:
      members:
        - pulse_echo_rf
        - sequence_rf
        - synthetic_aperture_rf
        - scan_focusline
        - show
