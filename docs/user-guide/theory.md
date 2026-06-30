---
icon: lucide/book-open
---

# Background Theory

Mathematical foundations of the Spatial Impulse Response (SIR) method as implemented in PyField.

<div class="grid cards" markdown>

-   :lucide-waves: **[Spatial Impulse Response](sir.md)**

    ---

    Definition of the SIR and its role in acoustic field computation. Relationship between transducer geometry and the impulse response.

-   :lucide-rectangle-horizontal: **[Rectangular Aperture SIR](rect-aperture.md)**

    ---

    Analytical SIR for a flat rectangular piston — the fundamental building block of the patch-based method.

-   :lucide-zap: **[FST and SDI Methods](methods.md)**

    ---

    Two algorithms for evaluating the SIR at discrete time samples: the FST direct approach and the Sparse Delta Integration (SDI) method.

-   :lucide-grid: **[Patch Subdivision](subdivision.md)**

    ---

    How curved and flat surfaces are decomposed into rectangular patches, including spherical-cap and Cartesian parameterisations.

-   :lucide-cpu: **[Transducer SIR](transducer-sir.md)**

    ---

    Assembling the total transducer response from individual patch SIRs with delays and apodization weighting.

-   :lucide-activity: **[From SIR to Pressure](pressure.md)**

    ---

    Deriving the monochromatic pressure field and convolving with excitation pulses for transient simulation.

</div>
