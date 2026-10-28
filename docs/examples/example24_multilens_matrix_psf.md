---
icon: lucide/grid-3x3
---

# Example 24: Diverging-Multilens Matrix Point PSF

[Example 23](example23_optimize_radial_element.md) fitted a single rigid cap to the
diverging multilens element of Nardone et al. (*Nat. Commun.* 2025, s41467-025-64911-z).
The paper builds a large 3-D imaging probe by tiling **252** of these elements on an
**18×14** grid at **5.48 mm** pitch, driven at **1 MHz**, and shows the point-spread
function of the whole matrix (their Fig. 2). This example assembles that matrix in eSDIva
with [`CustomTransducer`][esdiva.transducers.CustomTransducer] — 252 copies of the
example-23 free-aspheric cap — focuses it on one point scatterer, simulates the pulse-echo
RF, and delay-and-sums it back into a 3-D PSF.

[Source on GitHub](https://github.com/EstebanRivera08/eSDIva/blob/main/examples/example24_multilens_matrix_psf.py)

---

## Assembling the matrix

The diverging cap is reused straight from example 23 (same fit), so a change to the lens
there flows through here. 252 curved caps is heavy — keep `no_sub_diameter` small, and
preview the geometry with `sim.show()` before the simulation.

```python
from example23_optimize_radial_element import optimized_aspheric
from esdiva.transducers import CustomTransducer

element = optimized_aspheric(no_sub_diameter=12)   # one diverging cap, small patch count
# 18×14 centres on a 5.48 mm grid → positions_mm (252, 3)
tx = CustomTransducer([element] * 252, positions_mm, frequency_Hz=1e6)
rx = CustomTransducer([element] * 252, positions_mm, frequency_Hz=1e6)
```

TX and RX are separate copies of the same geometry; only TX is focused
(`tx.compute_delays(focus_mm=[0, 0, 40])`). Reception returns per-element RF, so the
receive focal law lives in the beamformer, not the aperture. Both apertures get a 2-cycle
1 MHz impulse response (bare drive) so the pulse-echo spectrum is band-limited.

![Setup preview](assets/ex24_setup.png)

---

## The point PSF

One on-axis scatterer at the 40 mm focus is simulated with
[`pulse_echo_rf`][esdiva.reception.Reception.pulse_echo_rf], then beamformed into a
3-D volume with [`das_volume`][esdiva.beamforming.das_volume] (the focused transmit is a
converging-then-diverging wavefront, so the event's virtual source sits **at** the focus,
`z > 0`). Two orthogonal slices through the focus show the spot.

![Point PSF](assets/ex24_matrix_psf.png)

The main lobe is tight — about **1.5 mm** (≈ one wavelength, λ = 1.54 mm) laterally and in
elevation, and ≈ 1.5 mm axially (pulse-limited by the 2-cycle burst). Because each element
*diverges*, the matrix sees a wide solid angle, so even a 252-element array forms a
wavelength-scale focal spot.

!!! note "Grating lobes are expected here"
    The 5.48 mm pitch is **3.56 λ**, far above the λ/2 that suppresses grating lobes, and a
    diverging element does not taper them the way a large flat element would. The regular
    off-axis replicas in the PSF (the vertical columns in the xz slice, the ring in the xy
    slice) are those grating lobes — a real property of this coarse, large-pitch matrix,
    not a simulation artefact. A finer pitch or element directivity would push them down.
