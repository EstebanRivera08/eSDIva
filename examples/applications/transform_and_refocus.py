"""Application — Move a probe in space, then refocus it.

Real setups rarely have the probe at the origin pointing straight down. `aim`
places a transducer anywhere and tilts its beam axis toward a target; delays are
firing times, not geometry, so they must be recomputed after the move for the beam
to focus on the target in the new pose. Here a sparse spiral array (as in Learn 11)
sits 15 mm off to the side and focuses 60 mm away along its tilted axis.

You will see: the moved array colored by delays, then the YZ pressure plane in dB
with the focus on the tilted axis.
"""

# %% Imports
import numpy as np

from esdiva.emission import Emission
from esdiva.plotting import plot2D_pressure_slices
from esdiva.transducers import CustomTransducer, FlatCircularTransducer

FC_HZ = 1e6
POSITION_MM = np.array([0.0, -15.0, 0.0])
TILT_DEG = 20.0  # Tilt of the beam axis toward -y.

# %% 64-disc sparse spiral (built at the origin, facing +z)
k = np.arange(64)
r, angle = 25 * np.sqrt((k + 0.5) / 64), k * np.pi * (3 - np.sqrt(5))
disc = FlatCircularTransducer(diameter_mm=3.0, no_sub_diameter=6, frequency_Hz=FC_HZ)
tx = CustomTransducer(
    elements=[disc] * 64,
    positions_mm=np.column_stack([r * np.cos(angle), r * np.sin(angle), 0 * r]),
    frequency_Hz=FC_HZ,
)

# %% Move it, then refocus on a target 60 mm along the tilted axis
axis = np.array([0, -np.sin(np.deg2rad(TILT_DEG)), np.cos(np.deg2rad(TILT_DEG))])
target_mm = POSITION_MM + 60 * axis
tx.aim(position_mm=POSITION_MM, target_mm=target_mm)
tx.compute_delays(focus_mm=target_mm)  # Delays in the new pose.
tx.show(scalars="Delays")

# %% Field on the YZ plane through the target (create the simulator after moving)
yz_plane = {
    "x_extent": [0, 0],
    "y_extent": [-60, 15],
    "z_extent": [5, 90],
    "dx": 0,
    "dy": 0.4,
    "dz": 0.4,
}
p, coords = Emission(tx, monochromatic=True)(yz_plane)
print(f"Target at y = {target_mm[1]:.1f} mm, z = {target_mm[2]:.1f} mm")
plot2D_pressure_slices(p, coords=coords, db_scale=True, title="Tilted, refocused array")
