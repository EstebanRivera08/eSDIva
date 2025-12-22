# Import mat
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import io as sio


def mat_struct_to_dict(ms):
    # recursive conversion: mat_struct -> dict, object arrays -> lists
    if isinstance(ms, np.ndarray) and ms.dtype == object:
        return [mat_struct_to_dict(x) for x in ms.flat]
    if hasattr(ms, "_fieldnames"):
        d = {}
        for fn in ms._fieldnames:
            d[fn] = mat_struct_to_dict(getattr(ms, fn))
        return d
    if hasattr(ms, "__dict__"):
        return {
            k: mat_struct_to_dict(v)
            for k, v in vars(ms).items()
            if not k.startswith("_")
        }
    return ms


# ---------- find .mat file(s) ---------
BASE = Path(r"D:\alec")
filename = "data_test.mat"

file_path = BASE / filename

#  ---------- load fieldII data ---------
mat_struct = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)["vessel"]

# Explore the structure names
# explore_mat(mat_struct)

# IPython-native :
mat_matrix = mat_struct_to_dict(mat_struct)

## Prepare data for plotting or further processing
nx, ny, nz = mat_matrix.shape
dx, dy, dz = 1, 1, 1  # mm

# Create the 3D UniformGrid
mesh_vol = pv.ImageData(
    dimensions=(nx, ny, nz),
    spacing=(dx, dy, dz),
)

# ------------  Create volume mesh data -------------
scalars = "Values"  # Name of the scalar data
# Attach pressure data to the grid
mesh_vol.point_data[scalars] = mat_matrix.ravel(
    order="F"
)  # VERY important: Fortran order


# ------------- Plotting ----------------
# Set to True to save high-res figure, False to show on screen
save_fig = True


pv.set_plot_theme("dark")

pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.background = "black"
pv.global_theme.font.color = "white"
color = "white"


# Apply Gaussian smoothing to the volume data
smoothed_data = mesh_vol.gaussian_smooth(std_dev=0.5)


if save_fig:
    scale = 3.0  # Scaling factor for high-res screenshots
else:
    scale = 1.0  # No scaling for on-screen display

# keyword arguments for volume plotting
kwargs = {
    "scalars": scalars,
    "cmap": "plasma",
    "opacity": "linear",
    "mapper": "smart",
    "show_scalar_bar": True,
    "ambient": 1,
    "scalar_bar_args": {
        "title": "MB/mm^2/s",
        "title_font_size": int(20 * scale),
        "label_font_size": int(18 * scale),
        "vertical": True,
        "position_x": 0.8,
        "position_y": 0.1,
        "height": 0.3,
    },
    "clim": (0, 100),
}

# Create the plotter
window_size = (500 * scale, 700 * scale)
off_screen = False
if save_fig:
    off_screen = True

plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)

# Add the volume to the plotter
vol = plotter.add_volume(smoothed_data, **kwargs)
vol.prop.interpolation_type = "linear"

# Modify plotter additional features
plotter.add_axes(color=color)

plotter.show_grid(
    grid="back",
    color=color,
    font_size=12 * scale,
    location="outer",
    xtitle="Z (mm)",
    ytitle="Y (mm)",
    ztitle="X (mm)",
    n_xlabels=7,
    n_ylabels=3,
    n_zlabels=3,
    use_3d_text=False,
)

# If inversion in axis is needed, modify the camera up vector
# plotter.camera.up = (0, 0, -1)  # Set the camera up direction
plotter.camera_position = [
    (-127.77415810473877, 247.30146467713405, -160.74227274842573),
    (60.5, 17.5, 16.0),
    (-0.8353062518326543, -0.3755107142957661, 0.40156589633379713),
]

if save_fig:
    plotter.screenshot("vessels_3D_alec.png")
else:
    plotter.show()
