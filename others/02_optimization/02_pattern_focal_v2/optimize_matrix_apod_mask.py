"""
Optimize Delays and Apodization to Match Binary Mask Pattern

This script demonstrates how to optimize transducer delays and apodization
to achieve a target pressure pattern (binary mask) using TorchFieldFlexible.

"""

from pathlib import Path

import numpy as np
import torch
from optimize_v2 import optimize_delays_apod_for_pattern
from plotting import plot_results

from pyfield import PyField
from pyfield.transducers import Domino, Zeus_Matrix
from pyfield.utilities import plot_pressure_field, plot_pressure_planes

if __name__ == "__main__":
    print("\nExample 1: Linear Array - Focal Point Pattern")
    print("-" * 70)

    # ----------------- Load target pattern -----------------
    target_folder = r".\target_masks"
    target_filename = f"\matrix_customtarget2.npz"
    print(f"Loading target pattern {target_filename}")
    target_dic = np.load(target_folder + target_filename)
    target = target_dic["target"].T
    nx, ny = target.shape

    print(torch.__version__)

    use_cuda = True  # Set to False if you want to run on CPU
    if use_cuda:
        # Free space if any used in cuda
        torch.cuda.empty_cache()

    # transducer type and saving folder
    base_path = "./results/custommask/"
    txarray = "matrixarray"  # "linarray" or "matrixarray"
    c = 1540  # m/s
    z_focal = 5  # mm

    # Optimization settings
    Energy_loss_type = "linear"  # "linear" or "log"
    MSE_loss_type = "linear"  # "linear" or "log"
    n_delays = 100  # Skip delay optimization for this example
    n_apod = 200
    lr_delays = 1e-2
    lr_apod = 1e-2
    init_apod = 1
    alpha = None  # Weight for combining losses (if using combined los)
    optimizer_type = "Adam"  # "Adam" or "SGD"
    dx_lambdas = 1.3
    version = "col"

    # Select transducer
    if txarray == "linarray":
        tx = Domino()
        resultsfolder = "linear"

    elif txarray == "matrixarray":
        tx = Zeus_Matrix()
        resultsfolder = "matrix"

    # Field specification
    fc = tx.fc  # Hz
    if txarray == "linarray":
        aperture_mm = tx.n_elements * tx.pitch * 1e3  # mm
    elif txarray == "matrixarray":
        aperture_mm = tx.n_elem_x * tx.pitch_x * 1e3  # mm

    print(
        f"Transducer: {txarray}, Focal depth: {z_focal}mm, Aperture: {aperture_mm:.2f}mm"
    )

    lambda_mm = c / fc * 1e3  # mm
    if z_focal <= aperture_mm:
        # Reference is with FoverD=1
        FoverD = 1.0
    else:
        FoverD = z_focal / aperture_mm

    estimated_width_mm = lambda_mm * FoverD  # mm

    print(
        f"Transducer center frequency: {fc / 1e6:.2f} MHz, wavelength: {lambda_mm:.2f}mm"
    )
    print(f"Estimated focal spot width: {estimated_width_mm:.2f} mm")
    dx_mm = (
        round(dx_lambdas * estimated_width_mm * 1000) / 1000
    )  # mm, round to nearest 0.01mm
    min_dx_mm = 0.025
    if dx_mm <= min_dx_mm:
        dx_mm = min_dx_mm

    # Define field points around focal region

    field_points = {
        "x_extent": [-dx_mm * nx / 2, dx_mm * nx / 2],
        "y_extent": [-dx_mm * ny / 2, dx_mm * ny / 2],
        "z_extent": [z_focal, z_focal],
        "dx": dx_mm,
        "dy": dx_mm,
        "dz": 0.5,
    }

    # -------------- point like target --------------
    # Create target: single focal point
    # Dx = field_points["x_extent"][1] - field_points["x_extent"][0]
    # Dy = field_points["y_extent"][1] - field_points["y_extent"][0]
    # nx, ny = int(Dx / field_points["dx"]), int(Dy / field_points["dy"])
    # if nx % 2 == 0:
    #     print(nx, nx % 2)
    #     nx += 1
    # if ny % 2 == 0:
    #     ny += 1
    # # target = np.zeros((nx, ny))
    # target[nx // 2, ny // 2] = 1  # Center point

    # Run optimization
    def pH(lr):
        return f"pH{-np.log10(lr):.2f}"

    file_name = f"""optim_{txarray}_zfoc{z_focal}_loss1{Energy_loss_type}_loss2{MSE_loss_type}_ndel{n_delays}_napod{n_apod}_lrdel{pH(lr_delays)}_lrapod{pH(lr_apod)}_initdel0_initapod{init_apod}_dxlambdas{dx_lambdas}_{optimizer_type}{version}.npz"""

    save_path = str(Path(base_path) / resultsfolder / file_name)
    # print(save_path)
    results = optimize_delays_apod_for_pattern(
        tx,
        target,
        field_points,
        initial_delays=None,
        initial_apod=np.ones(tx.n_elements) * init_apod,
        num_epochs_delays=n_delays,
        num_epochs_apod=n_apod,
        loss1_type=Energy_loss_type,
        loss2_type=MSE_loss_type,
        loss_alpha=alpha,
        lr_delays=lr_delays,
        lr_apod=lr_apod,
        batch_size=2048,
        use_gpu=True,
        save_path=save_path,
        optimizer_type=optimizer_type,
    )
    print(
        "Max pressure at target:{:.4e}".format(results["pressure_final"].max().item())
    )
    # Plot
    plot_results(results, save_path=save_path.replace(".npz", "_summary.png"), tx=tx)

    ## Compute a xz slice with results
    plane_xz = {
        "x_extent": [-3, 3],
        "y_extent": [-3, 3],
        "z_extent": [-3 + z_focal, z_focal + 3],
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.1,
    }

    tx.set_delays(results["delays"])
    tx.set_apodization(results["apodization"])
    pf = PyField(tx)
    x, y, z, pr = pf(plane_xz)
    plot_pressure_field(
        x,
        y,
        z,
        pr,
        save_path=save_path.replace(".npz", "_plane.png"),
        camera_position="yz",
        camera_elevation=60,
        camera_azimuth=45,
    )
