---
icon: lucide/waves
---

# Spatial Impulse Response

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

    For a brief conceptual overview, see [Getting Started](getting-started.md).

## Overview

The Spatial Impulse Response (SIR) `h(t, r)` characterises the acoustic response at field point **r** when the transducer is driven by a Dirac impulse. It encodes all geometric information about the transducer-to-point relationship.

The SIR is the foundation of the Tupholme–Stepanishen method: once `h(t, r)` is known for each field point, any pressure field can be derived by convolution with the desired excitation pulse.

The key result is:

$$
p(\mathbf{r}, t) = \rho \frac{\partial}{\partial t} \left[ v_n(t) * h(\mathbf{r}, t) \right]
$$

where $v_n(t)$ is the normal surface velocity (excitation) and $\rho$ is the medium density.
