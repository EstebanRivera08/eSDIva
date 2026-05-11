# Virtual Source Optimization — Analysis & Recommendations

## The Problem

Optimize positions `[x, z]` of N virtual sources (VS) for diverging wave compounding.
Goal: uniform coverage across imaging region using summed pressure fields.
Only 2 params per VS. Apodization derived from position via F/D=1 rule.

Each VS has 4 learnable params: `[x_vs, z_vs, n_steepness, D_aperture]`.
Total params: **12** (for 3 VS). Still small.

## What's Broken Right Now

### 1. SGD momentum=2 causes NaN

```python
torch.optim.SGD(params, lr=lr, momentum=2)  # line 503
```

**momentum > 1 = exponential blowup.** Velocity buffer grows every step.
PyTorch SGD: `v = momentum * v + grad`, then `param -= lr * v`.
With momentum=2, `v` doubles every step → NaN in ~20 iterations.
Valid range: 0.0–0.99. Typical: 0.9.

### 2. `pr / pr.max()` normalization kills gradients

```python
pr_norm = pr / (pr.max() + 1e-8)
```

`max()` is not smooth. When max-point jumps to a different spatial location
(which happens often as VS moves), gradient direction flips discontinuously.
This makes the loss surface jagged.

### 3. `torch.abs(vs[1])` in mappings

```python
focus_z_m = torch.abs(vs[1]) * 1e-3
```

Non-differentiable at z=0. For z < 0 (behind array, typical for diverging waves),
gradient of abs is -1, so sign is flipped. Works but creates a fold in the
landscape at z=0 that confuses optimizers.

### 4. `dB()` inside symmetry loss

```python
symmetry_loss = loss_MSE(dB(left_half), dB(right_half_flipped))
```

`dB = 20*log10(x)` → gradient ∝ 1/x. Near-zero pressure regions produce
huge gradients. These noisy outliers dominate the MSE, drowning the useful
signal from moderate-pressure regions.

### 5. Sigmoid apodization causes steepness gradient vanishing (FIXED)

**Problem:** Original apodization used sigmoid with steepness ~33,000 (= 10/pitch):

```python
apod = sigmoid(steepness * (half_aperture - dx))
```

At steepness=33,333 the transition zone is ~0.15mm (< 1 element). All elements
are in saturated zone (sigmoid ≈ 0 or 1). Gradient of sigmoid `σ(z)(1-σ(z)) ≈ 0`
everywhere → steepness parameter frozen during optimization.

Additionally, sigmoid gradient only exists at aperture **edge** — elements
inside/outside contribute nothing to steepness gradient. Scale mismatch:
steepness ~33,000 vs x_mm/z_mm ~10 breaks Adam's per-parameter scaling.

**Fix:** Replaced with **super-Gaussian** profile:

```python
r = dx / half_aperture        # normalized distance from VS center
apod = exp(-0.5 * r^(2n))     # n = super-Gaussian order
```

| Parameter | Behavior |
|-----------|----------|
| `n = 1`   | Gaussian taper |
| `n = 2-4` | Clinical compromise (smooth transition) |
| `n → ∞`   | Hard rect window |

Why this works:
- **Gradient w.r.t. n exists at every element** (not just edge):
  `d/dn = apod * -(r^(2n)) * ln(r²)` — all elements within aperture contribute
- **n ∈ [1, ~10]**: same scale as x_mm/z_mm/D_mm, no Adam scale mismatch
- **Naturally in [0, 1]**: exp() guarantees valid apodization, no clamping needed
- **Physically meaningful**: super-Gaussian apodization is used clinically

Constraint: `n ≥ 1` (Gaussian is the softest allowed profile).

## Why Gradient Descent Struggles Here

### Fundamental issues

1. **Physics simulator in the loop.** SIR computation involves discrete time
   sampling, JIT kernels, batch accumulation. Gradients are numerically noisy
   and oscillatory. Not smooth like a neural net.

2. **Non-convex, multi-modal landscape.** Multiple VS configurations produce
   similar coverage. Gradient descent finds the nearest local minimum, which
   depends entirely on initialization.

3. **Multi-objective loss.** Uniformity, coverage, aperture, energy pull in
   different directions. Loss surface is a Pareto front plateau. Gradient
   magnitude is small in trade-off regions → Adam stays near init because
   update ≈ 0.

4. **6 parameters.** Gradient descent shines with thousands+ of params.
   With 6, you can afford methods that explore the full landscape.

5. **Normalization by max creates non-smooth loss.** Every time the spatial
   location of peak pressure changes, loss function has a kink.

### Is it ill-posed?

**Partially.** The problem has multiple equivalent solutions (symmetry: VS at
+x and -x are interchangeable). But it's not ill-posed in the Hadamard sense
— solutions exist and are stable to perturbations. It's just **non-unique**
and **non-convex**.

## Recommendations

### Option A: Grid/Random Search (best for 6 params)

6 params = feasible brute force. Sample N configurations, evaluate loss, pick best.

```
x ∈ [-15, 15] mm, 10 points per VS
z ∈ [-20, -2] mm, 10 points per VS
```

For 3 VS with symmetry exploitation: ~1000 evaluations. Each forward pass
takes ~0.1-1s on GPU. Total: minutes, not hours. **Do this first** to
understand the landscape before any gradient method.

Variant: Latin Hypercube Sampling or Sobol sequences for better coverage.

### Option B: Bayesian Optimization (best bang for buck)

Use `optuna`, `botorch`, or `ax-platform`. Fits a Gaussian process surrogate
to the loss surface, then picks next point to evaluate based on expected
improvement. Designed for expensive black-box functions with few parameters.

- Handles non-convexity, multi-modality
- No gradients needed (treats simulator as black box)
- Typically finds good solutions in 50-200 evaluations
- Can handle multi-objective (Pareto front) natively with `botorch`

**This is the right tool for this problem.**

### Option C: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

Population-based, derivative-free optimizer. Gold standard for non-convex
optimization with 2-100 parameters. Library: `cma` (pip install cma).

- Adapts search distribution shape to the landscape
- Naturally handles multi-modality
- Population size ~10-20 for 6 params
- 100-500 generations typically sufficient

### Option D: Fix gradient descent (if you insist)

If you want gradient descent to work (e.g., as a proof-of-concept for
higher-dimensional problems later), fix these things:

1. ~~**Fix momentum**~~: Done. `momentum=0.9`, not 2
2. **Remove max-normalization**: Use `pr / pr.mean()` or raw pressure in loss
3. **Use gradient clipping**: Done. `clip_grad_norm_(params, max_norm=5.0)`
4. ~~**Fix steepness gradient**~~: Done. Super-Gaussian replaces sigmoid.
5. **Learning rate schedule**: Cosine annealing or reduce-on-plateau
6. **Multi-start**: Run 10-20 times from random initializations, keep best
7. **Simplify loss**: Start with just coverage + energy. Add terms one by one.
   Fewer competing objectives = cleaner gradient signal.
8. **Use L-BFGS**: Better than Adam/SGD for low-dimensional smooth-ish
   problems. Approximates Hessian. `torch.optim.LBFGS` — needs closure.

### Option E: Hybrid approach

1. Coarse search (grid/Bayesian/CMA-ES) to find promising basin
2. Fine-tune with gradient descent (Adam or L-BFGS) from best candidate

This gives global exploration + local precision.

## Recommended Path Forward

| Step | What | Why |
|------|------|-----|
| 1 | Grid search over VS positions | Map the landscape, understand what "good" looks like |
| 2 | Pick 1-2 loss terms that matter most | Simplify before optimizing |
| 3 | Bayesian opt or CMA-ES | Global optimization, no gradient needed |
| 4 | (Optional) Gradient fine-tune from best | Polish the solution |

## Is Gradient Descent Worth Pursuing Long-Term?

**Yes, but not for this problem size.** The value of gradient-based
optimization is for higher-dimensional problems (e.g., optimizing full delay
profiles across 128 elements, or joint optimization of multiple sequence
parameters). For those, you can't grid search.

For that future use case:
- Fix the normalization issues (use mean, not max)
- Use smoother loss functions (avoid dB domain in loss)
- Consider differentiable relaxations of discrete quantities
- L-BFGS or Adam with warm restarts (cosine annealing with restarts)
- The TorchField framework is solid for this — the gradient chain works,
  the problem is the loss surface, not the autodiff

## Summary

Problem: non-convex, multi-modal, 12 params (4 per VS). Gradient descent alone
is insufficient for global exploration but viable for local refinement.

**Fixed issues:**
- SGD momentum=2 → NaN (fixed: momentum=0.9)
- Sigmoid apodization → steepness gradient vanishing (fixed: super-Gaussian profile)
- Steepness ~33,000 scale mismatch with mm-scale params (fixed: n ∈ [1, ~10])

**Remaining:** non-convex landscape, multi-objective Pareto plateau.
Best approach: derivative-free global search (Bayesian opt or CMA-ES),
then gradient fine-tune from best candidate.
