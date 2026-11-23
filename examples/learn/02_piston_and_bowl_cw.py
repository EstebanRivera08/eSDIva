"""Learn 02 — Continuous-wave field of a flat piston and a focused bowl.

A monochromatic (continuous-wave, CW) simulation gives the steady pressure amplitude
|P(r)| at one frequency: a map with no time axis. A flat piston shows interference
fringes in its near field up to the Fresnel distance N = D²/(4λ), then a single
spreading lobe. A spherical bowl concentrates the energy near its centre of curvature.

You will see: one XZ pressure map per transducer, normalised to its own peak.
"""

# %% Imports
from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices
from esdiva.transducers import ConcaveCircularTransducer, FlatCircularTransducer

FC_HZ = 1e6
C = 1540.0  # Speed of sound (m/s).

# Field points: the XZ plane through the axis (y = 0), in mm.
XZ_PLANE = {
    "x_extent": [-20, 20],
    "y_extent": [0, 0],
    "z_extent": [1, 80],
    "dx": 0.5,
    "dy": 0,
    "dz": 0.5,
}

# %% Two transducers
piston = FlatCircularTransducer(
    diameter_mm=25.0, no_sub_diameter=60, frequency_Hz=FC_HZ
)
bowl = ConcaveCircularTransducer(
    diameter_mm=40.0, focus_mm=60.0, no_sub_diameter=30, frequency_Hz=FC_HZ
)

wavelength_mm = C / FC_HZ * 1e3
print(
    f"Piston near-field length N = D^2/(4*lambda) = {25.0**2 / (4 * wavelength_mm):.0f} mm"
)

# %% Simulate and plot each one
for name, tx in [("Flat piston", piston), ("Spherical bowl (R = 60 mm)", bowl)]:
    sim = Emission(tx, c=C, monochromatic=True)
    p, coords = sim(XZ_PLANE)  # p = |P| in Pa, shape (Nx, Ny, Nz).
    plot2D_pressure_slices(p / p.max(), coords=coords, db_scale=False, title=name)
