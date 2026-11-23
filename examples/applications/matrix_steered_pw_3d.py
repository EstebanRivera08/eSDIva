"""Application — A steered plane wave from a matrix array, animated in 3-D.

A 32 x 32 matrix array can steer in two directions, (θx, θy). Here it tilts a plane
wave 10° laterally. Pitch 0.3 mm vs λ = 0.51 mm at 3 MHz: grating lobes stay away
while the pitch d < λ / (1 + |sin θ|), i.e. 0.43 mm at 10°.

You will see: the wavefront crossing two orthogonal planes (XZ and YZ) in a 3-D
scene, above the array colored by its firing delays.
"""

# %% Imports
from esdiva.emission import Emission
from esdiva.plotting import add_transducer_mesh, plot3D_transient_slices
from esdiva.transducers import MatrixArrayTransducer
from esdiva.utilities import align_to_common_time, tone_burst

FC_HZ = 3e6
FS = 50e6

# %% Array and steering delays
tx = MatrixArrayTransducer(
    n_elements_x=32,
    n_elements_y=32,
    element_width_mm=0.275,
    element_height_mm=0.275,
    kerf_x_mm=0.025,
    kerf_y_mm=0.025,
    no_sub_x=1,
    no_sub_y=1,
    frequency_Hz=FC_HZ,
)
tx.compute_delays(angle_steering_deg=(10.0, 0.0))  # (θx, θy) in degrees.

# %% Transient field on two planes through the axis (cheaper than a full volume)
xz_plane = {
    "x_extent": [-8, 8],
    "y_extent": [0, 0],
    "z_extent": [5, 30],
    "dx": 0.05,
    "dy": 0,
    "dz": 0.05,
}
yz_plane = {
    "x_extent": [0, 0],
    "y_extent": [-8, 8],
    "z_extent": [5, 30],
    "dx": 0,
    "dy": 0.05,
    "dz": 0.05,
}
sim = Emission(tx, fs=FS, excitation=tone_burst(FC_HZ, n_cycles=2, fs=FS))
p_xz, coords_xz = sim(xz_plane)
p_yz, coords_yz = sim(yz_plane)

# Each plane has its own time axis: put both on one shared axis for the animation.
t, (p_xz, p_yz) = align_to_common_time([(p_xz, coords_xz), (p_yz, coords_yz)])

# %% Animate both planes above the array
planes = [
    {"plane": "xz", "data": p_xz.squeeze(), "translation": (0, 0, 0)},
    {"plane": "yz", "data": p_yz.squeeze(), "translation": (0, 0, 0)},
]
coords = {"x": coords_xz["x"], "y": coords_yz["y"], "z": coords_xz["z"]}
plotter = add_transducer_mesh(tx.get_mesh(), scalars="Delays")
plot3D_transient_slices(
    planes, coords=coords, plotter=plotter, time_array=t, db_scale=True
)
