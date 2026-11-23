"""Learn 01 — Transducers: an aperture is a set of small flat patches.

eSDIva tiles every transducer surface (flat, curved, multi-element) into small
rectangular patches and sums their contributions. On arrays, each element also gets
a time delay (to focus or steer the beam) and an apodization weight (to taper the
aperture and lower the side lobes).

You will see: a linear array (with its delay and apodization curves), a convex
array, a matrix array, a flat piston and a focused bowl, all colored by apodization.
"""

# %% Imports
from esdiva.transducers import (
    ConcaveCircularTransducer,
    ConvexArrayTransducer,
    FlatCircularTransducer,
    LinearArrayTransducer,
    MatrixArrayTransducer,
)

FC_HZ = 1.5e6  # Centre frequency.
FOCUS_MM = [0, 0, 50]  # Where the arrays focus (x, y, z) in mm; z is depth.

# %% Linear array: 64 elements in a row
# no_sub_x / no_sub_y = number of patches per element along x and y.
linear = LinearArrayTransducer(
    n_elements=64,
    element_width_mm=1.0,
    element_height_mm=12.0,
    kerf_mm=0.05,  # Gap between elements.
    no_sub_x=1,
    no_sub_y=10,
    frequency_Hz=FC_HZ,
)
# Delays: the outer elements fire first, so all wavelets reach the focus together.
linear.compute_delays(focus_mm=FOCUS_MM)
# FoverD = focal depth / active aperture width (the F-number): a larger F/D uses
# fewer elements, giving a longer and wider focal spot.
linear.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)
linear.plot_delays_apodization()
linear.show()

# %% Convex array: the same elements, laid on an arc
convex = ConvexArrayTransducer(
    n_elements=128,
    element_width_mm=0.5,
    element_height_mm=10.0,
    kerf_mm=0.1,
    radius_of_curvature_mm=60.0,
    no_sub_x=1,
    no_sub_y=10,
    frequency_Hz=FC_HZ,
)
convex.compute_delays(focus_mm=FOCUS_MM)
convex.compute_apodization(focus_mm=FOCUS_MM, FoverD=1.5)
convex.show()

# %% Matrix array: a 2-D grid of elements (focuses in 3-D)
matrix = MatrixArrayTransducer(
    n_elements_x=16,
    n_elements_y=16,
    element_width_mm=0.3,
    element_height_mm=0.3,
    kerf_x_mm=0.05,
    kerf_y_mm=0.05,
    no_sub_x=2,
    no_sub_y=2,
    frequency_Hz=FC_HZ,
)
matrix.compute_delays(focus_mm=FOCUS_MM)
matrix.compute_apodization(focus_mm=FOCUS_MM, FoverD=2.0)
matrix.show()

# %% Single elements: a flat disc and a spherical bowl (geometric focus)
piston = FlatCircularTransducer(
    diameter_mm=25.0, no_sub_diameter=30, frequency_Hz=FC_HZ
)
piston.show()

bowl = ConcaveCircularTransducer(
    diameter_mm=40.0, focus_mm=60.0, no_sub_diameter=30, frequency_Hz=FC_HZ
)
bowl.show()
