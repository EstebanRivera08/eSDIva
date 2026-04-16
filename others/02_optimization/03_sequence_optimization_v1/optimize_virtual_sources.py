"""
Optimize Virtual Source Positions for Plane Wave Compounding

This script optimizes virtual source positions for a linear array to:
1. Maximize energy distribution uniformity in the imaging plane
2. Minimize the number of active elements (sparse aperture)
3. Find optimal steering angles/positions for plane wave imaging

The optimization searches for the best virtual source locations behind
(or in front of) the array that produce the desired field distribution
with minimal element activation.

"""

import numpy as np
import torch
from optimization_functions import optimize_virtual_sources
from plotting_functions import plot_virtual_source_results

from pyfield.transducers import Domino, LinearArrayTransducer

print("torch version:", torch.__version__)


if __name__ == "__main__":
    print("\nOptimizing Virtual Sources for Plane Wave Imaging")
    print("=" * 70)

    # Optimizer
    data_folder = r"results/domino/"
    optimizer_type = "Adam"  # Options: "SGD", "Adam"
    energies = "loss_sparse_uniform_coverage"  # Options: "uniform", "sparse", "coverage", "energy"
    version = "_v1"
    num_epochs = 300
    lr = 1e-1

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
            sparsity_weight=0.1,
            uniformity_weight=1.0,
            coverage_weight=0.5,
            energy_weight=1e-8,
            batch_size=2048,
            use_gpu=True,
            optimizer_type=optimizer_type,
        )

        # Save results
        output_file = f"vsource_{n_vs}vs_{lr}_nepoch{num_epochs}_optim{optimizer_type}{version}.npz"

        # Plot results
        from pathlib import Path

        save_path = str(Path(data_folder) / output_file)

        plot_virtual_source_results(
            results, output_file=save_path.replace(".npz", ".png")
        )
        np.savez(save_path, **results)
        print(f"\nResults saved to: {output_file}")
