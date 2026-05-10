---
icon: lucide/activity
---

# From SIR to Pressure

!!! warning "Coming soon"
    This section is under active development. Content will be added in a future release.

## Overview

Once the transducer SIR `h(t, r)` is known, pressure fields are derived in two ways:

### Monochromatic pressure

For a continuous-wave source at frequency $f_c$, the complex pressure amplitude is:

$$
p(\mathbf{r}) = j \rho f_c \cdot \mathcal{F}\{h(\mathbf{r}, t)\}\big|_{f_c}
$$

where $\mathcal{F}$ denotes the Fourier transform. PyField evaluates this at the centre frequency of the transducer.

### Transient pressure

For a pulsed excitation $v_n(t)$, pressure is obtained by convolution:

$$
p(\mathbf{r}, t) = \rho \frac{\partial v_n}{\partial t} * h(\mathbf{r}, t)
$$

PyField performs this convolution along the time axis at each field point, returning the full spatio-temporal pressure field `p(t, x, y, z)`.
