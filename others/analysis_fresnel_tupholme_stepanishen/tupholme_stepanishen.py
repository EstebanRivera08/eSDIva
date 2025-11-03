import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from pyfield.utilities.plotting_pyvista import add_markers

# Parameters
save_3D = False
save_2D = True
figure_name = "Far_fieldv2"
factor = 1
reduce = 10


# DEFINE FIELD POINT
field_point = (factor * 2, factor * 6, factor * 2)

# DEFINE THE SPHERE

sphere_radius = np.linalg.norm(field_point)

sphere = pv.Sphere(radius=sphere_radius, center=field_point)

# DEFINE THE PLANE
plane_direction = (0, 1, 0)
plane_center = (0, 0, 0)
i_size = 20
j_size = 20
plane = pv.Plane(
    center=plane_center, direction=plane_direction, i_size=i_size, j_size=j_size
)

# COMPUTE THE 2D CIRCLE
# Calculate intersection circle center and radius
xc, yc, zc = field_point
r = sphere_radius

if abs(yc) >= r:
    raise ValueError("The sphere does not intersect the plane y=0.")

r_circle = np.sqrt(r**2 - yc**2)
circle_center = (xc, 0, zc)

# Parametric circle in the plane y=0
theta = np.linspace(0, 2 * np.pi, 100)
circle_points = np.array(
    [
        (
            circle_center[0] + r_circle * np.cos(t),
            0,
            circle_center[2] + r_circle * np.sin(t),
        )
        for t in theta
    ]
)


circle_points = np.array(circle_points)
lines = np.hstack(
    [[len(circle_points) + 1], np.append(np.arange(len(circle_points)), 0)]
)
circle = pv.PolyData()
circle.points = circle_points
circle.lines = np.hstack([[len(circle_points)], np.arange(len(circle_points))])

# DEFINE A RECTANGULAR TRANSDUCER

patch_center = (0, 0, 0)
plane_direction = (0, 1, 0)

i_size = 2 * 1 / reduce
j_size = 3 * 1 / reduce
patch = pv.Plane(
    center=patch_center, direction=plane_direction, i_size=i_size, j_size=j_size
)

# ------------------- 3D PLOTTING ------------------------------
w_size = [620, 580]
if save_3D:
    plotter = pv.Plotter(window_size=w_size, off_screen=True)
else:
    plotter = pv.Plotter(window_size=w_size)
plotter.add_mesh(plane, color="lightgray", opacity=0.2)
plotter = add_markers([field_point], glyph_scale=0.1, color="blue", plotter=plotter)
plotter.add_mesh(sphere, color="blue", opacity=0.1)
plotter.add_mesh(
    patch,
    color="red",
    opacity=1,
    lighting=True,
    ambient=1,
)
plotter.add_mesh(circle, color="black", line_width=2, render_points_as_spheres=False)

plotter.add_axes(label_size=(0.1, 0.05))  # show XYZ axes

# View 1
plotter.camera_position = [
    (-8.073600463933749, 10.146895936034053, -10.769562816196027),
    (0.5673223382618482, 3.0935365604192753, 0.7363281950839548),
    (-0.2948150735154079, 0.6999573427839951, 0.6504950351162362),
]
# # View 2
# plotter.camera_position = [
#     (-8.46055403625795, 11.831518058965884, -3.072633927412624),
#     (0.5432969465455436, 2.940450124118541, 0.8360851625888374),
#     (-0.12260226003361233, 0.2927372991069939, 0.9483003530243954),
# ]

plotter.show(auto_close=False)
if save_3D:
    plotter.screenshot(
        figure_name + ".png",
        scale=3,
    )

print(plotter.camera_position)


# Now let's go to 2D projection

# Rectangle
y_rect = np.array([-i_size / 2, i_size / 2, i_size / 2, -i_size / 2, -i_size / 2])
x_rect = np.array([-j_size / 2, -j_size / 2, j_size / 2, j_size / 2, -j_size / 2])

# Circle
x_circle = circle_points[:, 2]
y_circle = circle_points[:, 0]

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot(x_circle, y_circle, "k-", label="Circle Plane")
# ax.fill(x_rect, y_rect, "r", alpha=0.9, label="Transducer")

# ax.set_title("2D Projection")
# ax.legend()
# ax.axis("equal")
# ax.set_xlim([np.floor(-i_size), np.ceil(i_size)])
# ax.set_ylim([np.floor(-j_size), np.ceil(j_size)])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.tick_params(axis="both", which="both", length=0)
# ax.invert_xaxis()  # <-- Add this line
# plt.show()
# if save_figures:
#     fig.savefig(figure_name + "_2D.png", dpi=300)


# Let's build the SIR response for each case
# Lets first compute the arcs at different radii

# Calculate intersection circle center and radius

yc = yc
print(yc, sphere_radius)
radii = np.linspace(yc, 1.15 * sphere_radius, 110)
# Find the closest to the is plot is 3D
ind_arc = np.argmin(np.abs(radii - sphere_radius))
print(ind_arc)
# radii[ind_arc] = sphere_radius


r_circle = np.sqrt(radii**2 - yc**2)
theta = np.linspace(np.pi / factor, 3 / 2 * np.pi / factor, 50)

circle_center = (xc, 0, zc)
# Parametric circle in the plane y=0
circles_in_plane = np.zeros((len(r_circle), len(theta), 2))
for i, r in enumerate(r_circle):
    circles_in_plane[i, :, 0] = xc + r * np.cos(theta)
    circles_in_plane[i, :, 1] = zc + r * np.sin(theta)


fig, ax = plt.subplots(figsize=(6, 6))
ax.fill(x_rect, y_rect, "r", alpha=0.9, label="Transducer")
for i, circle_coord in enumerate((r_circle)):
    x_circle = circles_in_plane[i, :, 0]
    y_circle = circles_in_plane[i, :, 1]
    if i == ind_arc:
        ax.plot(x_circle, y_circle, "k-")
    else:
        ax.plot(x_circle, y_circle, "k--", linewidth=1)
ax.set_title("2D Projection")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis="both", which="both", length=0)
ax.axis("equal")
ax.set_xlim([-0.99 * i_size, 1.01 * i_size])
ax.set_ylim([-0.99 * j_size, 1.01 * j_size])
ax.invert_xaxis()
plt.show()

if save_2D:
    fig.savefig(figure_name + "_arcs2D.png", dpi=300)
