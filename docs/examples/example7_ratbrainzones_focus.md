# Example 7: Rat Brain Zone Focusing

Demonstrates focused ultrasound simulation targeting specific anatomical zones
of a rat brain using a BrainGlobe atlas and a linear array transducer.

## What you will learn

- Loading the rat atlas (`whs_sd_rat_39um`) and selecting motor/somatosensory regions
- Scaling the atlas by the lambda-bregma distance for a specific animal
- Combining atlas, transducer delays, and pressure field in one 3-D scene
- Positioning the brain relative to the transducer with affine transforms

## Prerequisites

This example requires the BrainGlobe atlas API.  The rat atlas data is
downloaded automatically on first run.

## Output

![Rat brain zones with focused pressure field](assets/rat_brain_zones.png)

## Run it

```bash
uv run examples/example7_ratbrainzones_focus.py
```

## Key code

```python
from pyfield.brain_atlas import BG_Atlas

brain_atlas = BG_Atlas("whs_sd_rat_39um", region_names=("root", "M1", "S1-hl"))

# Scale by animal-specific lambda-bregma distance
scale = np.eye(4)
scale[:3, :3] *= 8.0  # mm

# Transform and render
brain_atlas.transform(T_matrix=T_matrix, inplace=True)
```

[View full script on GitHub](https://github.com/EstebanRivera08/PyField/blob/main/examples/example7_ratbrainzones_focus.py)
