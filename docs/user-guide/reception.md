---
icon: lucide/activity
---

# Reception (RF)

Reception computes the **pulse-echo RF** scattered by point targets back onto the
receive elements — the raw channel data a scanner would digitise. It is the basis
for PSF, phantom, FMC, and sequence (PW/DW) imaging studies, and matches Field II
to correlation ~1.0 while running **>20× faster** on large apertures.

Two classes share one API:

- **`ReceptionSDI`** — fast sparse-delta kernel (default choice).
- **`Reception`** — conventional Tupholme-Stepanishen reference.

```python
from pyfield.reception import ReceptionSDI

sim = ReceptionSDI(tx, rx, fs=200e6, c=1540)          # separate TX / RX transducers
rf, coords = sim.pulse_echo_rf(scatterer_pos_mm, scatterer_amp)   # (Erx, Nt)
```

The physics: the pulse-echo signal is `v_pe ⊛ h_tx ⊛ h_rx`, with the excitation
and TX/RX impulse responses carrying the band-limited pulse (same convention as
Field II — no explicit derivative applied).

## Algorithmic flow

```mermaid
flowchart LR
    TX[TX transducer] --> R[ReceptionSDI<br/>fs · c · method]
    RX[RX transducer] --> R
    R --> M{method}
    M -->|spectral| SP[Closed-form one-way<br/>spectra · no FFT]
    M -->|paired| PA[16 corner deltas / pair<br/>splat drive]
    M -->|conventional| CV[sample SIRs + FFT]
    SP --> API
    PA --> API
    CV --> API
    subgraph API[entry methods]
      P1[pulse_echo_rf]
      P2[sequence_rf · PW/DW]
      P3[synthetic_aperture_rf · FMC]
      P4[scan_focusline]
    end
    API --> RF[rf Erx×Nt, coords]
```

## Entry methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `pulse_echo_rf` | Single transmit; core call. `per_scatterer=True` → PSF | `(Erx, Nt)` / `(P, Erx, Nt)` |
| `sequence_rf` | PW/DW event sweep; `out_path=` checkpoints each event | `(Nevt, Erx, Nt)` |
| `synthetic_aperture_rf` | FMC — per-element transmit basis | `(Etx, Erx, Nt)` |
| `scan_focusline` | One focused B-mode line, RX summed in-kernel | `(Nt,)` |

The `method=` flag (`auto` / `spectral` / `paired` / `conventional`) only trades
speed — all produce the same RF.

## Scatterers, PSF, and phantoms

- **PSF** — pass field points and `per_scatterer=True` to get each target's
  point-spread response. A grid dict gives a regular lattice of unit targets.
- **Phantoms** — `pyfield.utilities.make_phantom(extents_mm, n, echogenicity_map)`
  returns random positions with `N(0,1)·map(r)` amplitudes → realistic speckle.

```python
from pyfield.utilities import make_phantom

pos, amp = make_phantom(extents_mm, n=20000, echogenicity_map=my_map)
rf, coords = sim.pulse_echo_rf(pos, amp)
```

!!! warning "Lattice ≠ phantom"
    A periodic grid of scatterers gives *coherent* echoes (PSF maps), not speckle.
    Use `make_phantom` for speckle statistics.

## Preview the setup

`sim.show(scatterer_positions_mm, amplitudes)` renders TX/RX meshes and the
scatterer cloud in 3-D — run it before a long simulation.

![Dual-probe pulse-echo setup](../examples/assets/ex19_dualprobe_setup.png)

## Examples

| Figure | Study |
|--------|-------|
| ![Concave PSF](../examples/assets/ex06_concave_psf_comparison.png) | [PSF vs Field II](../examples/example06_concave_PSF.md) |
| ![FMC](../examples/assets/ex08_reception_fmc.png) | [Full Matrix Capture](../examples/example08_synthetic_aperture_FMC.md) |
| ![Phantom B-mode](../examples/assets/ex20_phantom_bmode.png) | [Speckle phantom](../examples/example20_phantom_simulation.md) |

## Beamforming note

The RF is referenced to the **geometric** round-trip time. The built-in DAS
beamformers (`das_volume`, `das_rca_volume`) auto-apply `coords["pulse_center_lag_s"]`
(the half-pulse envelope delay); a custom beamformer must add it itself.

Full signatures: [API → Reception](../api/reception.md) · [API → Beamforming](../api/beamforming.md).
