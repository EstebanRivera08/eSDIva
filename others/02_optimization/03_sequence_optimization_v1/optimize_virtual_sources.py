"""
Optimize Virtual Source Positions for Diverging Wave Compounding

v2: Apodization derived from VS position via F/D=1 (clinical standard).
Only VS [x, z] positions are optimized (2 params per VS).
Loss: lateral uniformity + soft coverage + aperture cost + mean energy.

"""

import numpy as np
import torch
from optimization_functions import optimize_virtual_sources
from plotting_functions import plot_virtual_source_results

from pyfield.transducers import Domino, LinearArrayTransducer

print("torch version:", torch.__version__)


if __name__ == "__main__":
    print("\nOptimizing Virtual Sources for Diverging Wave Imaging")
    print("=" * 70)

    # --- Configuration ---
    n_vs = 1  # Number of virtual sources to optimize
    use_gpu = True
    data_folder = r"results/domino/"
    optimizer_type = "SGD"  # Adam works better with few params
    version = "_nonorm_init1"
    num_epochs = 500
    lr = 1e-1  # log-mean gives stronger gradient → can use higher LR
    x_init = [0, 2, 4]  # mm, initial x positions of virtual sources
    z_init = [-5, -11, -8]  # mm, initial z positions of virtual sources
    # x_init = np.random.uniform(-10, 10, size=n_vs)  # Random initial x positions
    # z_init = np.random.uniform(-15, 0, size=n_vs)  # Random initial z positions
    fs = 50e6  # Hz, sampling frequency for field computation
    # Create transducer

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

    # Test with different numbers of virtual sources
    print(f"\n{'=' * 70}")
    print(f"Testing with {n_vs} virtual sources")
    print("=" * 70)

    results = optimize_virtual_sources(
        tx,
        n_virtual_sources=n_vs,
        field_points=field_points,
        num_epochs=num_epochs,
        lr=lr,
        uniformity_weight=10,
        coverage_weight=10,
        aperture_weight=0,
        energy_weight=1,
        batch_size=2048,
        use_gpu=use_gpu,
        optimizer_type=optimizer_type,
        x_init_mm=x_init[:n_vs],
        z_init_mm=z_init[:n_vs],
        fs=fs,
    )

    # Save results
    output_file = f"vs{n_vs}_lr{lr}_nepoch{num_epochs}_{optimizer_type}{version}.npz"

    from pathlib import Path

    save_path = str(Path(data_folder) / output_file)

    plot_virtual_source_results(results, output_file=save_path.replace(".npz", ".png"))
    np.savez(save_path, **results)
    print(f"\nResults saved to: {output_file}")


# --- v1 configuration (commented out for reference) ---
# optimizer_type = "SGD"
# version = "_apod0.5"
# lr = 1e-1
# z_behind = -2
#
# results = optimize_virtual_sources(
#     tx,
#     n_virtual_sources=n_vs,
#     field_points=field_points,
#     num_epochs=num_epochs,
#     lr=lr,
#     sparsity_weight=0.1,
#     uniformity_weight=1.0,
#     coverage_weight=0.5,
#     energy_weight=1e-8,
#     batch_size=2048,
#     use_gpu=True,
#     optimizer_type=optimizer_type,
#     z_behind=z_behind,
#     x_spacing=x_spacing,
# )
