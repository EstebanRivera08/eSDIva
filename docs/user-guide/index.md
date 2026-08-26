---
icon: lucide/book-open
hide:
  - toc
---

# User Guide

Everything you need to simulate acoustic pressure fields and pulse-echo RF with
SonDI — from installation to transducer geometry, emission, and reception.

<div class="grid cards" markdown>

-   :lucide-rocket: **[Getting Started](getting-started.md)**

    ---

    Install SonDI, run your first simulation, and learn the key concepts: patch discretisation, coordinate system, and unit conventions.

-   :lucide-package: **[Installation](installation.md)**

    ---

    Requirements, GitHub install, development setup, optional dependencies, and verification.

-   :lucide-container: **[Transducers](transducers.md)**

    ---

    Mono-element and multi-element transducer types. Patch model, delays, apodization, and the coordinate convention for curved surfaces.

-   :lucide-audio-lines: **[Simulation](simulation.md)**

    ---

    **Emission** (monochromatic, transient, attenuation) and **Reception** (pulse-echo RF for PSF, phantoms, FMC, sequences). Field grid format, SDI method selection, medium properties.

-   :lucide-square-activity: **[Visualization](visualization.md)**

    ---

    2-D Matplotlib pressure planes, transient animations, and interactive 3-D PyVista scenes with composable helpers.

-   :lucide-wrench: **[Utilities](utilities.md)**

    ---

    Geometry functions, BrainGlobe brain atlas integration, and PyVista scene-building helpers.

</div>
