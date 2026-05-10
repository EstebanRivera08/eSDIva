---
icon: lucide/cpu
---

# Transducer SIR

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

## Overview

The total transducer SIR is assembled from the individual patch SIRs by:

1. Computing `h_patch(t, r)` for every patch at every field point
2. Applying the per-element delay: time-shifting each patch response by `Δt_elem`
3. Applying the per-element apodization: scaling each patch response by `w_elem`
4. Summing all contributions:

$$
h_\text{tx}(\mathbf{r}, t) = \sum_{i} w_i \cdot h_{\text{patch},i}\!\left(\mathbf{r},\, t - \Delta t_i\right)
$$

For mono-element transducers the delay is zero and the apodization is uniform (all patches weighted equally).
