import matplotlib.pyplot as plt
import numpy as np
import torch

from pyfield.psimulation import PyField
from pyfield.transducers import Domino, LinearArrayTransducer
from pyfield.utilities import plot_pressure_planes

# Create a PyField simulation
focus_mm = (0, -16)  # Focus at (0, -10) mm

tx = LinearArrayTransducer(
    n_elements=128,
    element_width_mm=0.198,  # mm
    element_height_mm=5,  # mm
    kerf_mm=0.002,  # mm
    no_sub_x=1,
    no_sub_y=10,
    elevation_focus_mm=35,  # mm
    frequency_Hz=6.4e6,  # MHz
)

# Field specification (imaging region)
field_points = {
    "x_extent": [-500 * 0.2, 500 * 0.2],  # mm, lateral extent
    "y_extent": [0, 0],  # mm, thin slice
    "z_extent": [0, 800 * 0.2],  # mm, depth range
    "dx": 2,
    "dy": 1.0,
    "dz": 2,
}


tx.compute_delays(focus_mm)
# tx.set_apodization(np.ones(tx.num_elements))
tx.compute_apodization(focus_mm)
tx.plot_delays_apodization()

# Create the PyField instance for transducer
tx_field = PyField(tx, fs=30e6)

# Compute the field
x, y, z, pr = tx_field(field_points)


# Gaussian smoothing
from scipy.ndimage import gaussian_filter

sigma_points = 1  # Convert to points
pr = gaussian_filter(pr, sigma=sigma_points)

# Plot the field
plot_pressure_planes(x, y, z, pr, db_scale=True)

# Test of the Virtual Source Optimization
from optimization_functions import VirtualSourceOptimizer, compute_soft_coverage_loss

transducer = tx
n_virtual_sources = 1
x_i, z_i = focus_mm[0], focus_mm[1]
x_init_mm = [x_i - 5, x_i, x_i + 5]  # Initial x position of the virtual source (mm)
z_init_mm = [z_i, z_i, z_i]  # Initial z position of the virtual source (mm)
x_init_mm = [x_i]
z_init_mm = [z_i]
use_gpu = True

torch.cuda.empty_cache()  # Clear GPU memory before optimization

# Create VS optimizer (with F/D=1 derived apodization)
vs_opt = VirtualSourceOptimizer(
    transducer,
    n_virtual_sources,
    field_points,
    use_gpu=use_gpu,
    x_init_mm=x_init_mm,
    z_init_mm=z_init_mm,
    fs=50e6,  # 100 MHz sampling for accurate gradients
)

batch_size = 1024
# Forward: compound field from all VS
x_tf, y_tf, z_tf, pr_tf = vs_opt.get_combined_field(
    batch_size=batch_size, training=True, sigma_points=2
)

#
if isinstance(pr_tf, torch.Tensor):
    pr_tf = pr_tf.detach().cpu().numpy()

    x_tf = x_tf.detach().cpu().numpy()

    y_tf = y_tf.detach().cpu().numpy()

    z_tf = z_tf.detach().cpu().numpy()

plot_pressure_planes(x_tf, y_tf, z_tf, pr_tf, db_scale=True)

# lis
