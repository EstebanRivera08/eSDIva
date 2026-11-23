"""Application — Any curved element: a parabolic bowl, and how fine to tile it.

`RadialProfileTransducer` builds an element from any axisymmetric surface height
z(r). Here a paraboloid, z = r²/(2R): near its centre it is curved like a sphere of
radius R, so it focuses R from its centre. The rim sits at z = 0 and the centre is
recessed by the sag (2.8 mm here), so the focus lands near z = R - sag ≈ 37 mm.
Like every eSDIva aperture it is tiled into small flat patches, and each must be small for the
SIR formula to hold at the field points: width w << sqrt(4·l·c/f) at distance l.
Too-coarse patches give a wrong field, worst close to the element.

You will see: the coarse and the fine patch mosaics, then the on-axis CW pressure
for both: they agree near the focus and part in the near field.
"""

# %% Imports
import matplotlib.pyplot as plt

from esdiva.emission import Emission
from esdiva.transducers import RadialProfileTransducer

FC_HZ = 2e6
R_MM = 40.0  # Radius of curvature at the centre of the paraboloid.

on_axis = {
    "x_extent": [0, 0],
    "y_extent": [0, 0],
    "z_extent": [2, 80],
    "dx": 0,
    "dy": 0,
    "dz": 0.2,
}

# %% Same surface, two subdivision densities
for n_sub in [6, 40]:  # Patches across the diameter.
    tx = RadialProfileTransducer(
        diameter_mm=30.0,
        surface_sag_mm=lambda r: r**2 / (2 * R_MM),  # Height z(r) in mm.
        no_sub_diameter=n_sub,
        frequency_Hz=FC_HZ,
    )
    tx.show()
    p, coords = Emission(tx, monochromatic=True)(on_axis)
    plt.plot(coords["z"], p.ravel() / 1e3, label=f"{n_sub} patches across")

plt.xlabel("Depth z (mm)")
plt.ylabel("|P| on axis (kPa)")
plt.title("Parabolic bowl, R = 40 mm")
plt.legend()
plt.show()
