---
icon: lucide/grid-3x3
---

# Example 24: Diverging-Multilens Matrix Point PSF

[Example 23](example23_optimize_radial_element.md) fitted a single rigid cap to the
diverging multilens element of Nardone et al. (*Nat. Commun.* 2025, s41467-025-64911-z).
The paper builds a large 3-D imaging probe by tiling **252** of these elements on an
**18×14** grid at **5.48 mm** pitch, driven at **1 MHz**, and shows the volumetric
point-spread function measured in water (their **Fig. 2b**). That PSF is *not* a focused
transmit: it is the ultrafast localization sequence — **16** central elements each fire a
single diverging wave in turn, and the echoes are compounded on receive. This example
reproduces it in eSDIva with [`CustomTransducer`][esdiva.transducers.CustomTransducer]
(252 copies of the example-23 free-aspheric cap), fires the 16-source sequence at one point
target, delay-and-sums the compounded echoes into a 3-D volume, and renders it the way the
paper does — grayscale, in dB, with a sigmoid opacity transfer function.

[Source on GitHub](https://github.com/EstebanRivera08/eSDIva/blob/main/examples/example24_multilens_matrix_psf.py)

---

## Assembling the matrix

The diverging cap is reused straight from example 23 (same fit), so a change to the lens
there flows through here. 252 curved caps × 16 transmits is heavy — keep `no_sub_diameter`
small, and preview the geometry with `sim.show()` before the simulation.

```python
from example23_optimize_radial_element import optimized_aspheric
from esdiva.transducers import CustomTransducer

element = optimized_aspheric(no_sub_diameter=10)   # one diverging cap, small patch count
# 18×14 centres on a 5.48 mm grid → positions_mm (252, 3)
tx = CustomTransducer([element] * 252, positions_mm, frequency_Hz=1e6)
rx = CustomTransducer([element] * 252, positions_mm, frequency_Hz=1e6)
```

TX and RX are separate copies of the same geometry. Both apertures get a 2-cycle 1 MHz
impulse response (bare drive) so the pulse-echo spectrum is band-limited.

![Setup preview](assets/ex24_setup.png)

---

## The 16-source diverging sequence

Each of the 16 most central elements fires one diverging wave, driven by a **one-hot**
apodization so only that element emits while every element receives. A single small element
radiates a spherical wave from its own centre, so its virtual source sits **on** the
aperture (`z = 0`). The sequence runs through
[`sequence_rf`][esdiva.reception.Reception.sequence_rf]; the 16 events are then compounded
coherently by [`das_volume`][esdiva.beamforming.das_volume], which recovers each event's
transmit time origin from its own geometry.

```python
central = np.argsort(np.hypot(positions_mm[:, 0], positions_mm[:, 1]))[:16]
tx_events = [
    {"delays": np.zeros(252), "apodization": one_hot(e),
     "virtual_source_mm": [positions_mm[e, 0], positions_mm[e, 1], 0.0]}
    for e in central
]
rf, coords = sim.sequence_rf(scat_mm, scat_amp, tx_events)   # (16, Erx, Nt)
vol, axes = das_volume(rf, coords, tx_events, tx, grid_mm, c=1540, fnum=0.5)
```

---

## The Fig. 2b-style volume render

The beamformed volume is envelope-detected along depth, log-compressed to dB, and
volume-rendered with PyVista using a **grayscale** colormap, a **sigmoid** opacity
transfer function (main lobe opaque, −40 dB background transparent), and the **`"smart"`**
volume mapper (ray-cast, so the spot renders smooth rather than blocky) — matching the
paper's Fig. 2b. A dark theme (black background, **all text white** — ruler, colorbar, box)
with the z axis pointing **down** (`camera.up = (0, 0, −1)`, the ultrasound depth
convention) puts the probe at the top and the target deeper below. The render box widens the
paper's PSF volume to **100 × 100 mm** laterally, the full **0–60 mm** depth, point
scatterer at **(0, 0, 40)**, with a wireframe outline and mm ruler so the physical extent is
readable. It is rendered on a coarse **1 mm** grid (the ray-cast mapper interpolates it
smooth) because this volume dominates the cost; the main lobe is measured separately on a
fine 0.1 mm grid.

![Point PSF volume render](assets/ex24_matrix_psf.png)

Three panels give the quantitative view: a fine xz slice, the xy grating-lobe star, and the
PSF **intensity profiles** (lateral and axial) with the **−6 dB** resolution marked.

![PSF slices](assets/ex24_psf_slices.png)

### How close is this to the paper?

The faithfully reproduced quantity is the **array-level signature**: the peak grating lobe
sits at about **−22.9 dB** here versus the paper's measured **−22.4 dB**, and the eight-arm
grating-lobe geometry matches. That level is fixed by the sparse 5.48 mm geometry and is
independent of depth.

The raw delay-and-sum main lobe here is about **2.5 mm** at −6 dB (≈ 1.3 mm axially,
pulse-limited by the 2-cycle burst), roughly twice the paper's quoted **1.06 mm**. That gap
is *not* a reconstruction knob this example gets wrong — sweeping the receive F-number, the
receive apodization (rect vs Hann), the element subdivision (and hence its directivity), the
transmit-source placement (central cluster vs spread across the aperture), and the target
depth all leave the main lobe near 2.5 mm. Its near-constancy with depth shows it is set by
the single-element diverging transmit and the element directivity, not by array-aperture
diffraction. The residual factor of two lives in the paper's unpublished *Beamforming
processing*, so this example reproduces the measurable array physics without claiming to
match the raw main-lobe width.

!!! note "Grating lobes are expected here"
    The 5.48 mm pitch is **3.56 λ**, far above the λ/2 that suppresses grating lobes, and a
    diverging element does not taper them the way a large flat element would. The regular
    off-axis replicas in the PSF (the star at ±11–13 mm in the xy slice) are those grating
    lobes — the 5.48 mm pitch places them at ≈ 16° (λ/pitch), i.e. ≈ 11.7 mm off-axis at
    40 mm depth. They are a real property of this coarse, large-pitch matrix, not a
    simulation artefact. A finer pitch or stronger element directivity would push them down.
