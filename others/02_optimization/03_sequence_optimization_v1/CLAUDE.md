# CLAUDE.md — Sequence Optimization v1

## Goal

Optimize virtual source (VS) positions and apodization profiles for diverging wave
compounding. Maximize uniform coverage across imaging region using summed pressure
fields from N virtual sources.

## Parameter Vector

Each VS has 4 learnable parameters stored as a single tensor `vs_i`:

```
[x_vs, z_vs, n, FD]
```

| Index | Name | Unit | Description | Constraints |
|-------|------|------|-------------|-------------|
| 0 | `x_vs` | mm | Lateral position of VS | unconstrained |
| 1 | `z_vs` | mm | Depth of VS (positive = behind array) | min: None |
| 2 | `n` | - | Super-Gaussian order (profile steepness) | min: 1.0 |
| 3 | `FD` | - | F-number ratio (F/D) | min: 0.1, max: 5.0 |

**Aperture derivation from FD:**
```
D = z_vs / FD          (aperture diameter in mm)
half_aperture = D / 2   (half-aperture in mm)
```

Constructor params `FD_init` (default 2.0) and `n_gauss_init` (default 4.0) set initial
values. `optimize_virtual_sources()` passes these through (`FD_init=1`, `n_gauss_init=4`).

## Apodization Profile

Super-Gaussian centered at `x_vs`:

```python
dx = |element_x - x_vs|          # distance from element to VS center
r  = dx / half_aperture           # normalized distance (0 at center, 1 at edge)
apod = exp(-0.5 * r^(2n))        # super-Gaussian
```

| n value | Profile shape |
|---------|---------------|
| 1 | Gaussian taper |
| 2-4 | Smooth compromise (init: n=4) |
| n -> inf | Hard rectangular window |

**Why super-Gaussian instead of sigmoid:**
- Sigmoid with high steepness (~33,000) saturates -> gradient vanishing for steepness param
- Sigmoid gradient only at aperture edge (2-3 elements contribute)
- Super-Gaussian gradient exists at **every element** via `d/dn = apod * -(r^(2n)) * ln(r^2)`
- `n` in range [1, 10] — same scale as x/z/FD params, no Adam scale mismatch

**Note:** At `r=1` (aperture edge), `apod = exp(-0.5) = 0.607` regardless of `n`.
For all elements to be ~1, need `D >> transducer_aperture` (i.e., small FD).
Example: z_vs=15mm, FD=0.5 -> D=30mm > 25mm aperture -> edge element r=0.83 -> apod=0.79 (n=2).
Optimizer can increase n for flatter profile if needed.

## Delay Mapping

VS position -> element delays (diverging wave):

```python
distances = sqrt((elem_x - x_vs)^2 + elem_y^2 + (elem_z - z_vs)^2)
delays = distances/c - min(distances/c)    # relative delays
```

## Loss Functions

All losses in `loss_functions.py`:

| Loss | What it does | Weight param |
|------|-------------|--------------|
| `compute_symmetry_loss` | MSE between left/right halves of pressure field | `symmetry_weight` |
| `compute_soft_coverage_loss` | Sigmoid-based fraction above dB threshold, output in [0,1] | `coverage_weight` |
| `compute_aperture_cost` | Mean active elements across VS (discourages wide apertures) | `aperture_weight` |
| `compute_mean_energy_loss` | Negative mean pressure (prevents collapse to zero) | `energy_weight` |
| `compute_resolution_loss` | Effective f-number penalty from VS geometry | `resolution_weight` |

`max_pr` is computed once from initial field and kept fixed during training to prevent
oscillatory gradients from SIR simulation normalization.

**Dynamic weight balancing:** `energy_weight` and `coverage_weight` are recomputed each
epoch relative to `aperture_weight * loss_aperture` to keep loss terms on similar scale.

## Architecture

```
VirtualSourceOptimizer
  ├── n TorchFieldFlexible instances (one per VS)
  │     ├── Parameter: vs_i = [x_vs, z_vs, n, FD]
  │     ├── Mapping: vs_i -> delays (element-level)
  │     └── Mapping: vs_i -> apodization (element-level, super-Gaussian)
  └── get_combined_field() -> sum of |P_i| across VS
```

`TorchFieldFlexible` handles:
- Parameter storage + gradient tracking (PyTorch module)
- Constraint enforcement via `clamp_()` after each optimizer step
- SIR forward simulation (differentiable through TorchField)
- Parameter mapping chain: learnable params -> derived params -> simulation

## Key Files

| File | Purpose |
|------|---------|
| `optimization_functions.py` | `VirtualSourceOptimizer` class + `optimize_virtual_sources()` |
| `loss_functions.py` | All loss functions (v1-v4, only v3/v4 active) |
| `plotting_functions.py` | Visualization of optimization results |
| `README.md` | Analysis of optimization challenges + recommendations |

## Known Issues / Design Decisions

1. **Gaussian filter on combined field**: `gaussian_filter_pytorch` smooths final
   pressure field with `sigma_points` (default 1). Helps coverage/energy metrics
   but adds spatial blur.

2. **Gradient clipping**: `clip_grad_norm_(params, max_norm=5.0)` — SIR simulation
   produces oscillatory gradients.

3. **Non-convex landscape**: Multiple equivalent solutions exist (VS symmetry).
   Consider multi-start or derivative-free methods (CMA-ES, Bayesian opt) for
   global exploration, then gradient fine-tune.

4. **Super-Gaussian r=0 guard**: `r.clamp(min=1e-6)` prevents `0^(2n)` producing
   NaN gradient at center element (0 * ln(0) = NaN in floating point).

## Typical Usage

```python
results = optimize_virtual_sources(
    transducer=tx,
    n_virtual_sources=3,
    field_points=field_points,
    num_epochs=200,
    lr=0.01,
    symmetry_weight=1.0,
    coverage_weight=0.5,
    aperture_weight=0.3,
    energy_weight=0.1,
    resolution_weight=1.0,
    x_init_mm=[0.0, -5.0, 5.0],
    z_init_mm=[10.0, 10.0, 10.0],
)
```

Output `results` dict contains: VS positions history, final pressure field,
loss histories, apodization, grid coordinates.
