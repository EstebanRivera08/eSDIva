---
icon: lucide/zap
---

# FST and SDI Methods

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

## Overview

Two algorithms evaluate the SIR at discrete time samples:

### FST method

Direct sample-by-sample computation. For each field point and each patch, the SIR is evaluated at every time sample within the arrival window. Accurate and simple, but scales linearly with both the number of time samples and patches.

**Best for**: small grids, reference validation, debugging.

### SDI — Sparse Delta Integration

Instead of evaluating the SIR at every sample, SDI identifies the sparse set of time instants where the integrand has a non-zero contribution and accumulates only those. Significantly faster for large dense field grids.

**Best for**: production runs with large field grids or high sampling rates.

### Auto selection

```python
sim(field_points, method="auto")
```

The `"auto"` method examines the problem size and selects whichever approach will be faster. Use `"auto"` unless you have a specific reason to force one method.
