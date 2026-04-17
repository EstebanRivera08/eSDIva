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
    data_folder = r"results/domino/"
    optimizer_type = "SGD"  # Adam works better with few params
    version = "_v4_logcov"
    num_epochs = 300
    lr = 0.1  # log-mean gives stronger gradient → can use higher LR
    z_behind = -1  # mm behind the array (deeper = wider initial beam)
    x_spacing = 3  # mm spacing between virtual sources

    # Create transducer
    tx = Domino()

    # Field specification (imaging region)
    field_points = {
        "x_extent": [-10, 10],  # mm, lateral extent
        "y_extent": [0, 0],  # mm, thin slice
        "z_extent": [0, 15],  # mm, depth range
        "dx": 0.2,
        "dy": 1.0,
        "dz": 0.2,
    }

    # Test with different numbers of virtual sources
    for n_vs in [3]:
        print(f"\n{'=' * 70}")
        print(f"Testing with {n_vs} virtual sources")
        print("=" * 70)

        results = optimize_virtual_sources(
            tx,
            n_virtual_sources=n_vs,
            field_points=field_points,
            num_epochs=num_epochs,
            lr=lr,
            uniformity_weight=0,
            coverage_weight=1.0,
            aperture_weight=0,
            energy_weight=0,
            batch_size=2048,
            use_gpu=True,
            optimizer_type=optimizer_type,
            z_behind=z_behind,
            x_spacing=x_spacing,
        )

        # Save results
        output_file = (
            f"vs{n_vs}_lr{lr}_nepoch{num_epochs}_{optimizer_type}"
            f"_z{z_behind}_Dx_{x_spacing}{version}.npz"
        )

        from pathlib import Path

        save_path = str(Path(data_folder) / output_file)

        plot_virtual_source_results(
            results, output_file=save_path.replace(".npz", ".png")
        )
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
