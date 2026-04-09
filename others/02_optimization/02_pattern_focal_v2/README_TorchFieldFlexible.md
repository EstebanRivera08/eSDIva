## TorchField Flexible Framework - Complete Guide

A redesigned PyTorch-based differentiable acoustic simulator that enables gradient-based optimization of **arbitrary** transducer parameters through a flexible parameter mapping system.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Examples](#examples)
6. [Migration from Old TorchField](#migration-from-old-torchfield)
7. [API Reference](#api-reference)

---

## Overview

### What's New?

The flexible TorchField framework (`TorchField_flexible.py`) addresses limitations of the original implementation:

**Original TorchField:**
- ❌ Hardcoded delays and apodization as the only optimizable parameters
- ❌ No support for virtual parameters (e.g., virtual source positions)
- ❌ Couldn't optimize geometric parameters (element positions)
- ❌ Limited to element-level parameters

**TorchField Flexible:**
- ✅ Optimize **any** parameter: delays, apodization, positions, custom parameters
- ✅ Support for **virtual parameters** that compute derived values
- ✅ **Parameter mappings** for complex transformations
- ✅ Proper handling of global/element/patch-level parameters
- ✅ Composable transformations (e.g., virtual_source → delays → patches)

---

## Key Concepts

### 1. Optimizable Parameters

Wrapped parameters with metadata and constraints:

```python
class OptimizableParameter:
    name: str              # Parameter identifier
    value: nn.Parameter    # Actual PyTorch parameter
    level: str             # 'global', 'element', or 'patch'
    constraints: dict      # e.g., {'min': 0, 'max': 1}
    transform: callable    # Optional (e.g., sigmoid)
```

**Levels:**
- **global**: Single value or small vector (e.g., virtual source position [x, y, z])
- **element**: One value per transducer element (e.g., delays, apodization)
- **patch**: One value per subdivision patch (e.g., patch centers)

### 2. Parameter Mappings

Define how to compute derived parameters from optimizable ones:

```python
class ParameterMapping:
    function: callable     # Computation function
    inputs: List[str]      # Input parameter names
    output: str            # Output parameter name
    level: str             # Output level
```

**Example:** Virtual source → delays
```python
def vs_to_delays(virtual_source, tx, device):
    vs_mm = virtual_source.detach().cpu().numpy()
    delays = tx.compute_delays(focus_mm=vs_mm, apply=False)
    return torch.tensor(delays * 1e6, device=device)
```

### 3. Dependency Resolution

The framework automatically resolves parameter dependencies:

```
virtual_source (optimizable)
    ↓ (mapping)
delays (computed)
    ↓ (expand to patches)
patch_delays (used in simulation)
```

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────┐
│   Optimizable Parameters            │
│  (base parameters to optimize)      │
│                                     │
│  - virtual_source: [x, y, z]       │
│  - apod_width: float               │
│  - element_offsets: [n_elem, 3]   │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   Parameter Mappings                │
│  (compute derived parameters)       │
│                                     │
│  - vs → delays                     │
│  - apod_width → apodization        │
│  - offsets → patch_centers         │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   Final Transducer Parameters       │
│  (used in simulation)               │
│                                     │
│  - delays: [n_elements]            │
│  - apodization: [n_elements]       │
│  - patch_centers: [n_patches, 3]  │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   Acoustic Simulation               │
│                                     │
│  SIR → Pressure Field              │
└─────────────────────────────────────┘
```

---

## Quick Start

### Installation

The flexible framework is located in:
```
src/pyfield/psimulation/TorchField_flexible.py
```

### Basic Usage

```python
from pyfield.psimulation.TorchField_flexible import TorchFieldFlexible
from pyfield.transducers import LinearArrayTransducer

# Create transducer
tx = LinearArrayTransducer(n_elements=64, ...)

# Create TorchField
tf = TorchFieldFlexible(tx, use_gpu=True)

# Option 1: Optimize delays directly
tf._optimizable_params['delays'].value.requires_grad = True

# Option 2: Add virtual source and mapping
tf.add_optimizable_parameter(
    'virtual_source',
    initial_value=[0, 0, 30],  # mm
    level='global',
    requires_grad=True
)

def vs_to_delays(virtual_source, tx, device):
    vs_mm = virtual_source.detach().cpu().numpy()
    delays_s = tx.compute_delays(focus_mm=vs_mm, apply=False)
    return torch.tensor(delays_s * 1e6, device=device)

tf.add_parameter_mapping(
    name='vs_to_delays',
    function=vs_to_delays,
    inputs=['virtual_source'],
    output='delays',
    level='element'
)

# Optimize
optimizer = torch.optim.Adam(tf.get_optimizable_parameters(), lr=0.01)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    x, y, z, p = tf(field_points, training=True)

    # Compute loss
    loss = compute_loss(p, target)

    # Backward
    loss.backward()
    optimizer.step()
    tf.apply_constraints()
```

---

## Examples

### Example 1: Direct Delay Optimization

```python
# Make delays optimizable
tf._optimizable_params['delays'].value.requires_grad = True

# Optimize
optimizer = torch.optim.Adam(tf.get_optimizable_parameters(), lr=1e-3)

for epoch in range(50):
    optimizer.zero_grad()
    x, y, z, p = tf(field_points, training=True)
    loss = -p[target_x, target_y, target_z]  # Maximize at target
    loss.backward()
    optimizer.step()
```

**Use case:** Fine-tuning delays for precise focusing

---

### Example 2: Virtual Source Optimization

```python
# Add virtual source
tf.add_optimizable_parameter(
    'virtual_source',
    initial_value=[0, 0, 30],
    level='global',
    requires_grad=True
)

# Map virtual source → delays
tf.add_parameter_mapping(
    name='vs_to_delays',
    function=lambda virtual_source, tx, device: ...,
    inputs=['virtual_source'],
    output='delays',
    level='element'
)

# Optimize virtual source position
optimizer = torch.optim.Adam(tf.get_optimizable_parameters(), lr=0.5)
```

**Use case:** Optimizing plane wave steering angles, diverging wave sequences

**See:** `optimize_virtual_sources.py`

---

### Example 3: Custom Apodization Function

```python
# Add Gaussian apodization parameters
tf.add_optimizable_parameter('apod_center', initial_value=32.0, ...)
tf.add_optimizable_parameter('apod_width', initial_value=20.0, ...)

# Map parameters → apodization
def gaussian_apod(apod_center, apod_width, tx, device):
    elements = torch.arange(tx.n_elements, device=device)
    apod = torch.exp(-((elements - apod_center)**2) / (2*apod_width**2))
    return apod / apod.max()

tf.add_parameter_mapping(
    name='gaussian_apod',
    function=gaussian_apod,
    inputs=['apod_center', 'apod_width'],
    output='apodization',
    level='element'
)
```

**Use case:** Optimizing aperture shape for specific beam profiles

**See:** `examples_flexible_torchfield.py` - Example 3

---

### Example 4: Element Position Optimization

```python
# Add position offsets
tf.add_optimizable_parameter(
    'element_offsets',
    initial_value=np.zeros((n_elements, 3)),
    level='element',
    requires_grad=True,
    constraints={'min': -0.5, 'max': 0.5}  # ±0.5mm
)

# Map offsets → patch centers
def compute_patch_centers_with_offsets(element_offsets, tx, device):
    # Recompute patch centers with offsets
    original_centers = ...  # Extract from tx
    offsets_expanded = element_offsets.repeat_interleave(...)
    return (original_centers + offsets_expanded) * 1e6  # to μm

tf.add_parameter_mapping(
    name='offsets_to_centers',
    function=compute_patch_centers_with_offsets,
    inputs=['element_offsets'],
    output='patch_centers',
    level='patch'
)
```

**Use case:** Calibrating array defects, optimizing array geometry

**See:** `examples_flexible_torchfield.py` - Example 4

---

### Example 5: Pattern Matching with Binary Mask

Optimize delays and apodization to match a target pressure pattern.

**Features:**
- Sigmoid transform for apodization (keeps in [0,1])
- Pressure → pattern conversion
- Physics-based + pattern-matching loss

```python
# Add sigmoid transform
sigmoid_transform = lambda x: torch.sigmoid(10 * (x - 0.5))

tf.add_optimizable_parameter(
    'apodization',
    initial_value=np.ones(n_elements) * 0.5,
    transform=sigmoid_transform,
    requires_grad=True
)

# Loss function
def loss_energy(y_target_3D, pr):
    E_focus = y_target_3D * pr
    E_sides = (1 - y_target_3D) * pr
    return torch.log(E_sides.mean() + 1e-6) - torch.log(E_focus.mean() + 1e-6)

# Training loop
for epoch in range(num_epochs):
    x, y, z, pr = tf(field_points, training=True)
    pattern_2d = pattern_from_pr_3Dto2D(pr, max_pr)

    loss_phys = loss_energy(target_3d, pr)
    loss_comp = mse_loss(target_2d, pattern_2d)

    loss = loss_phys + alpha * loss_comp
    ...
```

**See:** `optimize_delays_apod_mask.py`

---

## Migration from Old TorchField

### Old Code

```python
from pyfield.psimulation import TorchField

# Create
tf = TorchField(tx, device=device)

# Parameters were hardcoded as nn.Parameter
tf.delays.requires_grad = True
tf.apodization.requires_grad = True

# Process functions were hardcoded
processed_apod = tf._process_apodization(tf.apodization)
```

### New Code

```python
from pyfield.psimulation.TorchField_flexible import TorchFieldFlexible

# Create
tf = TorchFieldFlexible(tx, use_gpu=True)

# Make parameters optimizable (more explicit)
tf._optimizable_params['delays'].value.requires_grad = True
tf._optimizable_params['apodization'].value.requires_grad = True

# Transforms are specified when adding parameter
tf.add_optimizable_parameter(
    'apodization',
    initial_value=apod,
    transform=lambda x: torch.sigmoid(10 * (x - 0.5)),  # Explicit
    requires_grad=True
)

# Get processed value
apod = tf.get_parameter('apodization')  # Already transformed
```

### Key Differences

| Aspect | Old TorchField | TorchField Flexible |
|--------|----------------|---------------------|
| Optimizable params | Hardcoded (delays, apod) | User-defined, arbitrary |
| Virtual params | Not supported | Full support via mappings |
| Transforms | Hardcoded methods | Specified per parameter |
| Parameter levels | Element-only | Global/Element/Patch |
| Geometric params | Not supported | Fully supported |
| Extensibility | Limited | Highly extensible |

---

## API Reference

### TorchFieldFlexible

#### Constructor

```python
TorchFieldFlexible(
    transducer,
    *,
    c: float = 1540.0,
    fs: float = 200e6,
    use_gpu: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True
)
```

#### Methods

**add_optimizable_parameter**
```python
tf.add_optimizable_parameter(
    name: str,
    initial_value: Union[float, List, np.ndarray],
    *,
    level: str = 'global',
    requires_grad: bool = True,
    constraints: Optional[Dict] = None,
    transform: Optional[Callable] = None,
    replace: bool = True
)
```

**add_parameter_mapping**
```python
tf.add_parameter_mapping(
    name: str,
    function: Callable,
    inputs: List[str],
    output: str,
    level: str,
    *,
    cache: bool = False,
    replace: bool = True
)
```

**get_parameter**
```python
tf.get_parameter(name: str) -> Tensor
```

**get_optimizable_parameters**
```python
tf.get_optimizable_parameters() -> List[nn.Parameter]
```

**apply_constraints**
```python
tf.apply_constraints()
```

**forward**
```python
tf(field_info_mm: Dict, *, batch_size: int = 2048,
   training: bool = False, normalize: bool = False)
-> Tuple[Tensor | ndarray, ...]
```

---

## File Structure

```
others/learning_focalization/
├── README_TorchFieldFlexible.md          # This file
├── examples_flexible_torchfield.py       # Comprehensive examples (5 scenarios)
├── optimize_delays_apod_mask.py          # Pattern matching optimization
├── optimize_virtual_sources.py           # Virtual source optimization
└── version2/
    └── TorchField.py                     # Old version (for reference)

src/pyfield/psimulation/
├── TorchField.py                         # Original implementation
├── TorchField_v2.py                      # Intermediate version
└── TorchField_flexible.py                # NEW: Flexible framework
```

---

## Best Practices

### 1. Start Simple

Begin with direct parameter optimization before adding complex mappings:

```python
# Step 1: Optimize delays directly
tf._optimizable_params['delays'].value.requires_grad = True

# Step 2: Add virtual source if needed
tf.add_optimizable_parameter('virtual_source', ...)
tf.add_parameter_mapping('vs_to_delays', ...)
```

### 2. Use Constraints

Always set reasonable bounds on parameters:

```python
tf.add_optimizable_parameter(
    'virtual_source',
    ...,
    constraints={'min': -50, 'max': 50}  # Physical limits
)
```

### 3. Clear Cache

Call `clear_cache()` is automatic in `forward()`, but if you manually get parameters:

```python
tf.clear_cache()  # Before new forward pass
param = tf.get_parameter('delays')
```

### 4. Apply Constraints

After each optimizer step:

```python
optimizer.step()
tf.apply_constraints()  # Clamp to valid ranges
```

### 5. Check Parameter Shapes

Verify shapes match expected levels:

```python
# Global: scalar or small vector
virtual_source: [3]  # [x, y, z]

# Element: one per element
delays: [n_elements]
apodization: [n_elements]

# Patch: one per subdivision
patch_centers: [n_elements * no_sub_x * no_sub_y, 3]
```

---

## Troubleshooting

### Issue: "Parameter 'X' not found"

**Cause:** Trying to get a parameter that hasn't been defined or mapped.

**Solution:** Check available parameters:
```python
print(list(tf._optimizable_params.keys()))
print([m.output for m in tf._parameter_mappings.values()])
```

### Issue: Gradients not flowing

**Cause:** Parameter not set as `requires_grad=True`.

**Solution:**
```python
tf._optimizable_params['param_name'].value.requires_grad = True
# or
tf.add_optimizable_parameter(..., requires_grad=True)
```

### Issue: Constraint violations

**Cause:** Forgot to call `apply_constraints()`.

**Solution:**
```python
optimizer.step()
tf.apply_constraints()  # Add this!
```

### Issue: Shape mismatch in mapping

**Cause:** Mapping function returns wrong shape for specified level.

**Solution:** Check level and expected shapes:
- `global`: any shape (but usually small)
- `element`: [n_elements] or [n_elements, ...]
- `patch`: [n_patches] or [n_patches, ...]

---

## Performance Tips

1. **Use batch_size wisely**: `batch_size=2048` is good default, increase for more GPU memory
2. **Cache expensive mappings**: Set `cache=True` for expensive computations
3. **Use GPU**: Always set `use_gpu=True` if available
4. **Minimize context switches**: Avoid `.cpu().numpy()` in mappings during training

---

## Citation

If you use this flexible framework in your research, please cite:

```
@software{pyfield_torchfield_flexible,
  title={TorchField Flexible: Differentiable Acoustic Field Simulator with Flexible Parameter Optimization},
  author={PyField Development Team},
  year={2025},
  url={https://github.com/EstebanRivera08/PyField}
}
```

---

## Contributing

Found a bug or want to add features? Please:

1. Check existing examples in `examples_flexible_torchfield.py`
2. Create a minimal reproducible example
3. Submit an issue or pull request

---

## Future Improvements

Potential extensions:
- [ ] Multi-frequency optimization
- [ ] Transient (time-domain) optimization
- [ ] Constraint regularization (smoothness penalties)
- [ ] Automatic differentiation of transducer geometry functions
- [ ] Support for non-uniform patch sizes

---

**Last Updated:** 2025-04-02
**Version:** 1.0
**Contact:** PyField Development Team
