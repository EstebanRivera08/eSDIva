# AGENT.md — `src/pyfield/future/`

Context file for working on the experimental differentiable TorchField stack.
This area is **under development** and NOT part of the public API (see
`CLAUDE.md`: "Anything labelled or using TorchField is under development and
will not be release soon, so keep independent and secret").

When you change any file in this directory, update the relevant section of
this document.

---

## What lives here

| File | Purpose |
|---|---|
| `TorchField.py` | First iteration. Monolithic `TorchField` class. Superseded. |
| `TorchField_v2.py` | Second iteration. Adds the JIT-scripted kernels `compute_patch_events_batch` and `accumulate_events_derivative`. Still has a fixed `(delays, apodization)` optimization surface. |
| `TorchField_flexible.py` | **Current active design.** Flexible parameter system built on `OptimizableParameter` + `ParameterMapping`. |
| `dopplerscan.py` | Unrelated Doppler scan prototype. |

New work should happen in `TorchField_flexible.py` and its helpers.

---

## `TorchFieldFlexible` — what it does and how

### High-level goal

Provide a differentiable (PyTorch) acoustic field simulator whose
**optimization surface is configurable at runtime**. Instead of hard-coding
"optimize delays and apodization", users declare:

1. **Optimizable parameters** — trainable `nn.Parameter`s with metadata
   (level, constraints, transform).
2. **Parameter mappings** — differentiable functions that compute derived
   parameters from other parameters (direct or computed).

The framework then resolves dependencies, runs the SIR kernel, and gives
back a torch tensor whose gradient flows back to all trainable parameters.

### Architecture

```
OptimizableParameter                 ParameterMapping
  ├── name, level                      ├── name, inputs, output
  ├── value : nn.Parameter             ├── function : callable
  ├── transform (optional sigmoid…)    └── compute(inputs, tx, device)
  └── get_value() applies transform

                        ▼
           TorchFieldFlexible (nn.Module)
           ├── _optimizable_params  : Dict[str, OptimizableParameter]
           ├── _parameter_mappings  : Dict[str, ParameterMapping]
           ├── _computed_cache      : Dict[str, Tensor]
           ├── get_parameter(name)  ──┐
           │                          │ resolves value by:
           │                          │   1. cache lookup
           │                          │   2. parameter_mappings first (NEW)
           │                          │   3. direct optimizable_params
           ├── add_optimizable_parameter(...)
           ├── add_parameter_mapping(...)
           ├── spatial_impulse_response(pts)
           │     uses get_parameter("delays"),
           │          get_parameter("apodization"),
           │          get_parameter("patch_centers")
           └── forward(field_info_mm, training=True/False)
```

### Parameters the SIR kernel needs

The `spatial_impulse_response` loop **hard-requires** these three names to
resolve (by direct param or by mapping):

| Name            | Shape             | Unit | Level    |
|-----------------|-------------------|------|----------|
| `delays`        | `[n_elements]`    | μs   | element  |
| `apodization`   | `[n_elements]`    | —    | element  |
| `patch_centers` | `[n_patches, 3]`  | μm   | patch    |

In addition, the kernel uses scalar constants `self.wx`, `self.wy` (patch
half-widths in μm) set at `__init__` from `tx.elem_width / tx.no_sub_x`.

> **Current limitation.** `wx`/`wy` are Python floats, NOT torch tensors,
> so you cannot currently optimize patch size or use non-uniform patches.
> This is Phase 2 work — see "Roadmap" below.

### Default parameter initialization

`_initialize_default_parameters()` registers (in order):

1. `delays`        — direct param, from `tx.delays * 1e6`, `requires_grad=False`
2. `apodization`   — direct param, from `tx.apodization`, `requires_grad=False`
3. `quad_vertices` — direct param, from `tx.sub_quad_verts * 1e6`,
   shape `[n_patches, 4, 3]`, `requires_grad=False`
4. `patch_centers` — **mapping** `quad_vertices → patch_centers` via
   `quad_vertices.mean(dim=1)`

The switch from "direct `patch_centers`" to "mapping derived from
`quad_vertices`" is what enables optimization of higher-level transducer
attributes (see below).

### Parameter resolution order — CRITICAL

`get_parameter(name)` resolves as follows:

1. Check `_computed_cache`
2. **Check `_parameter_mappings`** (mappings take precedence) — new
3. Check `_optimizable_params` (direct params)
4. Raise `ValueError`

This is the **opposite of what a naive reading would suggest**. The reason
is that mappings are explicit user additions and defaults should be
overridable without manual bookkeeping. Historically this was reversed
and caused silent gradient loss — see "Troubleshooting" below.

### Shape / unit conventions (internal)

- Spatial: **μm** (`self.space_m_to_unit = 1e6`)
- Time:    **μs** (`self.time_sec_to_unit = 1e6`)
- `self.c_unit`       = `c * 1e6 / 1e6 = c` (μm/μs numerically equal to m/s)
- `self.fs` is in Hz (scalar).

When you add a new parameter mapping that produces a geometry quantity,
**multiply by `space_m_to_unit`** before returning.

---

## Typical workflows

### A. Optimize per-element delays (simplest case)

```python
tf = TorchFieldFlexible(tx)
tf.add_optimizable_parameter(
    "delays",
    initial_value=tx.delays * 1e6,  # μs
    level="element",
    requires_grad=True,
    replace=True,
)
x, y, z, pr = tf(field_points, training=True)
loss = ...
loss.backward()
```

Because mappings take precedence over direct params, you could also do
this via a mapping from a "delay law" parameter.

### B. Virtual source optimization (indirect parameter → delays)

See `others/02_optimization/03_sequence_optimization_v1/optimize_virtual_sources.py`.
Pattern:

```python
tf.add_optimizable_parameter("vs", initial_value=[0, -10], level="global", requires_grad=True)

def vs_to_delays(**kwargs):
    vs = kwargs["vs"]                       # [2] tensor
    # Compute delays from vs using tx.element_centers (must be torch tensor
    # already on the right device — DO NOT reconstruct it with torch.tensor
    # from a tensor slice, that breaks the grad graph).
    ...
    return delays_us                        # [n_elements]

tf.add_parameter_mapping(
    name="vs_to_delays",
    function=vs_to_delays,
    inputs=["vs"],
    output="delays",
    level="element",
)
```

### C. Optimize element positions (requires `quad_vertices` chain)

```python
from pyfield.future.TorchField_flexible import build_rect_patch_vertices_torch

tf = TorchFieldFlexible(tx)
tf.add_optimizable_parameter(
    "element_centers",
    initial_value=tx.element_centers,  # m
    level="element",
    requires_grad=True,
)

def ec_to_verts(**kwargs):
    ec_m = kwargs["element_centers"]
    verts_m = build_rect_patch_vertices_torch(
        ec_m,
        elem_width=tx.elem_width,
        elem_height=tx.elem_height,
        no_sub_x=tx.no_sub_x,
        no_sub_y=tx.no_sub_y,
        elev_focus=getattr(tx, "elev_focus", 0.0),
    )
    return verts_m * 1e6  # μm

tf.add_parameter_mapping(
    name="element_centers_to_verts",
    function=ec_to_verts,
    inputs=["element_centers"],
    output="quad_vertices",
    level="patch",
)
```

The chain `element_centers → quad_vertices → patch_centers → SIR → pressure`
is fully differentiable.

---

## Troubleshooting

### Symptom: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` on `loss.backward()`

The gradient graph is broken somewhere. Common causes (ranked):

1. **Mapping silently shadowed.** Historically `get_parameter` checked
   direct params first, so a mapping `vs → delays` was ignored if the
   default `delays` direct param existed. **Fixed now** (mappings take
   precedence). If you still see this, verify your mapping's `output=` name
   matches what `spatial_impulse_response` queries (`delays`, `apodization`,
   `patch_centers`, `quad_vertices`).

2. **Mapping function re-wraps tensors with `torch.tensor(...)`.** Writing
   `torch.tensor([[vs[0], 0, vs[1]]])` on tensor elements creates a **new
   leaf tensor without grad** — the grad path back to `vs` is severed. Use
   `torch.stack`, component-wise arithmetic, or write values directly onto
   existing grad-carrying tensors.

3. **Using `training=False` output in a torch loss.** `forward(training=False)`
   wraps the SIR computation in `torch.no_grad()` **and** converts the result
   to numpy via `.detach().cpu().numpy()`. Use `training=True` for the
   forward pass you want to differentiate; use `training=False` only for
   display/evaluation. If you need a torch reference tensor, convert with
   `torch.as_tensor(np.asarray(pr_ref), dtype=torch.float32, device=dev)`
   after the call.

4. **In-place operations on a leaf zero tensor.** `H = torch.zeros(P, T)`
   followed by `H[i:j] = expr_with_grad` generally does propagate grad back
   through the RHS, but only if `H` itself is created on the same device
   without `requires_grad`. The current kernel handles this correctly; if
   you change it, verify with a tiny backward test.

5. **Constraint clamping during the forward pass.** `apply_constraints()`
   uses `clamp_` under `torch.no_grad()`, which is fine AFTER `optimizer.step()`
   but **must not be called inside the forward pass**. Only call it after
   `optimizer.step()`.

### Symptom: optimization runs but loss never decreases

- Check `get_optimizable_parameters()` returns a non-empty list and that
  the parameters you expect are in it.
- Verify gradient is non-zero after `loss.backward()`:
  `for p in tf.get_optimizable_parameters(): print(p.grad.norm())`
- If all grads are zero but the graph is intact, your loss likely doesn't
  depend on the parameters (e.g. you normalized `pr / pr.max()` which has
  zero gradient at its argmax).

### Symptom: shape mismatch in `repeat_interleave`

`delays` and `apodization` mappings must return shape `[n_elements]`, not
`[1, n_elements]`. `spatial_impulse_response` does
`delays_elem.repeat_interleave(no_sub_x * no_sub_y)` which is unary.

### Symptom: `__array_wrap__` deprecation warning

You mixed a numpy array and a torch tensor in arithmetic. Convert one side
with `torch.as_tensor(...)` before the operation.

---

## Roadmap (pending work)

- **Phase 2 — differentiable patch sizes.** Make `compute_patch_events_batch`
  accept per-patch `wx`, `wy` tensors. This requires changing the JIT
  signature in `TorchField_v2.py`. Only needed for optimizations that vary
  element/patch dimensions (currently impossible).
- **Phase 3 — differentiable patch orientation.** Currently the kernel
  assumes patches are axis-aligned in the plane z=const. Supporting rotated
  patches (convex arrays, tilted elements) requires using the full per-patch
  frame (`tangent_u`, `tangent_v`, `normal`) inside the kernel, not just
  `patch_centers`.
- **Phase 4 — torch `build_subdivisions` for non-linear transducers.**
  `build_rect_patch_vertices_torch` currently only supports the
  LinearArray / (mildly curved via `elev_focus`) case. Convex arrays and
  matrix transducers need their own torch builders.
- **Phase 5 — generic transducer attribute optimization.** Expose a helper
  that introspects `TransducerBase` subclasses and produces a mapping chain
  automatically.

---

## Testing checklist when modifying `TorchField_flexible.py`

1. Run the example in `others/02_optimization/03_sequence_optimization_v1/optimize_virtual_sources.py`
   for a few epochs. Loss should change and `vs_i` positions should drift.
2. Verify `tf(field_points, training=True)` returns a torch tensor with
   `requires_grad=True` (or at least `grad_fn is not None`).
3. Verify `tf(field_points, training=False)` returns a numpy array (same as
   before).
4. Run `uv run pytest tests/` if any tests reference the future stack.
5. Update this file if you added / removed / renamed a default parameter or
   changed the resolution order.
