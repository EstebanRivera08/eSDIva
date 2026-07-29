---
icon: lucide/activity
---

# Reception

Pulse-echo RF simulators. See the [Reception user guide](../user-guide/reception.md)
for the method taxonomy, PSF, and phantom recipes.

## ReceptionSDI

Fast sparse-delta pulse-echo kernel (default choice).

::: pyfield.reception.ReceptionSDI
    options:
      members:
        - pulse_echo_rf
        - sequence_rf
        - synthetic_aperture_rf
        - scan_focusline
        - show
        - set

## Reception

Conventional Tupholme-Stepanishen reference implementation (same API).

::: pyfield.reception.Reception
    options:
      members:
        - pulse_echo_rf
        - sequence_rf
        - synthetic_aperture_rf
        - scan_focusline
        - show
