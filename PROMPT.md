Continue.

Before updating the documentation, we need to iterate on the following aspects of the curved‑surface filling functions:

1. **Border patches on circular apertures**  
   For curved circular transducers, we should allow a small tolerance (≈1% of the patch area) for patches that slightly exceed the circular boundary. This lets us keep the border patches instead of discarding them, since the overshoot is negligible and improves coverage.

2. **Allowing extreme curvatures (focus_mm → 0)**  
   The current filling algorithm does not accept configurations where  
   \[
   \text{focus\_mm} < \frac{\text{diameter\_mm}}{2},
   \]  
   but we need it to support the full range down to **focus_mm → 0**, which corresponds to a perfect hemisphere. The method should gracefully handle these high‑curvature cases.

3. **Alternative method for spherical surfaces**  
   If the current approach cannot robustly handle very small focal distances, then add a second method specifically designed for spherical geometries.  
   A suitable approach is to parameterize the surface in **spherical coordinates** and fill it by:
   - fixing a value of \((\phi, \rho)\) to define a ring,  
   - looping over \(\theta\) to place rectangular patches along that ring.  

   This would give us two complementary methods for filling curved surfaces with rectangular patches:
   - the existing curvature‑based method (improved with points 1–2),  
   - a spherical‑coordinate method optimized for hemispherical or near‑hemispherical apertures.

