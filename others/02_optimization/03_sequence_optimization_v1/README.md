# Virtual Source Optimization — Analysis & Recommendations

## The Problem

Optimize positions `[x, z]` of N virtual sources (VS) for diverging wave compounding.
Goal: uniform coverage across imaging region using summed pressure fields.
Only 2 params per VS. Apodization derived from position via F/D=1 rule.

Total params: **6** (for 3 VS). This is tiny.

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

1. **Fix momentum**: `momentum=0.9`, not 2
2. **Remove max-normalization**: Use `pr / pr.mean()` or raw pressure in loss
3. **Use gradient clipping**: Uncomment `clip_grad_norm_(params, max_norm=1.0)`
4. **Replace symmetry loss**: Use lateral CV (`compute_lateral_uniformity_loss`)
   instead of dB-domain MSE. Less noisy gradients.
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

Problem: non-convex, multi-modal, 6 params. Gradient descent = wrong tool.
NaN from SGD: momentum=2 bug. Adam stuck near init: flat Pareto plateau.
Solution: derivative-free global search (Bayesian opt or CMA-ES), then
optionally polish with gradients. Save gradient methods for when you scale
to higher-dimensional parameter spaces where brute force is impossible.
