---
icon: lucide/layers
---

# Example 25: Moving and Combining Volumes on One Canvas

One 32-element 5 MHz linear array, focused at 30 mm, simulated **once** as a
monochromatic XZ plane. `Field` then moves and combines that result instead of
re-simulating, following the [recommended steps](../user-guide/fields.md):
canvas → wrap → pose → resample → combine.

[Source on GitHub](https://github.com/EstebanRivera08/eSDIva/blob/main/examples/example25_field_canvas.py)

![Single view, 3-view compound, moved vs re-simulated, two-way beam](assets/example25_field_canvas.png)

## A. Compounding three tilted views

The probe fires at −15°, 0° and +15° at **different times**, so the views do not
interfere: their **intensities** `|P|²` add.

```python
views = [Field(p**2, coords, scalars="Intensity").transform(rot_y(a)) for a in (-15, 0, 15)]
compound = sum(v.resample(canvas) for v in views)
```

## B. Moving a field equals re-simulating the moved probe

The moved field is compared with an `Emission` of the probe after
`tx.transform(rot_y(15))`, only where the moved field has data (its coverage mask).

```python
diff = beam.transform(rot_y(15)).resample(canvas) - Field(p_resim, c_resim)
```

Result: **2.1 % of peak beyond 5 mm, 3.0 % in the near field**. That residual is trilinear
interpolation on a 0.1 mm (≈ λ/3) grid, not physics. The source grid is simulated
**wider than the canvas** (x ±25 mm, z from 0.1 mm); with the source the same size as
the canvas, the tilted view leaves uncovered corners that read as 54 % "errors".

## C. Two-way beam

Transmit (F/4, 7.5 mm aperture) and receive (full 9.6 mm aperture), both focused at 30 mm,
act in sequence, so the pulse-echo amplitude response is their product:

```python
two_way = beam.resample(canvas) * Field(p_rx, canvas.coords)
```

−6 dB width at 30 mm: **TX 1.40 mm, RX 1.00 mm, two-way 0.80 mm** — the Gaussian-beam
prediction `1/w² = 1/w_tx² + 1/w_rx²` gives 0.81 mm.

## Run

```bash
uv run python examples/example25_field_canvas.py
```

About 3 s. Set `ESDIVA_SAVE_FIG=1` to write the figure instead of showing it.
