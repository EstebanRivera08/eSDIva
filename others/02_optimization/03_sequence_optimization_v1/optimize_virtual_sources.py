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
    n_vs = 3  # Number of virtual sources to optimize
    use_gpu = True
    data_folder = r"results/twolearnings/"
    optimizer_type = "SGD"  # Adam works better with few params
    use_phase = False  # optimize for energy only (ignore phase)

    energy_weight = 1  # 1e-2
    resolution_weight = 1
    symmetry_weight = 10  # not used in this version
    aperture_weight = 1
    coverage_weight = 5
    comment2 = "init1_E_aper_1equiv_cov5_sym10_res1"
    # comment2 = ""

    avg_energy_value, avg_resolution_value = 500, 0.3  # for normalization in loss

    num_epochs = 500
    lr_pos = 1e-1
    lr_apod = 1e-2  # log-mean gives stronger gradient -> can use higher LR
    x_init = [-5, 0, 5]  # mm, initial x positions of virtual sources
    z_init = [-10, -10, -10]  # mm, initial z positions of virtual sources
    # x_init = [-10, 0, 10]  # mm, initial x positions of virtual sources
    # z_init = [-15, -15, -15]  # mm, initial z positions of virtual sources
    # x_init = np.random.uniform(-10, 10, size=n_vs)  # Random initial x positions
    # z_init = np.random.uniform(-15, 0, size=n_vs)  # Random initial z positions
    fs = 50e6  # Hz, sampling frequency for field computation

    # if resolution_weight == 0:
    #     ratio = "inf"
    # else:
    #     ratio = (
    #         round(
    #             1000
    #             * energy_weight
    #             * avg_energy_value
    #             / (resolution_weight * avg_resolution_value)
    #         )
    #         / 1000
    #     )
    #     ratio = f"{ratio:.3f}"
    #
    # ratio += f"_E{round(1000 * energy_weight) / 1000:.3f}"

    ratio = "fd2_n1"
    if use_phase:
        comment = "AmpAndPhase"
    else:
        comment = "AmplitudeOnly"

    version = f"_init1_{comment}_EnergyToRes_{ratio}{comment2}"
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
    # Save results
    output_file = f"vs{n_vs}_lrpos{lr_pos}_lrapod{lr_apod}_nepoch{num_epochs}_{optimizer_type}{version}.npz"

    # Test with different numbers of virtual sources
    print(f"\n{'=' * 70}")
    print(f"Testing with {n_vs} virtual sources")
    print(f"filename: {output_file}")
    print("=" * 70)

    results = optimize_virtual_sources(
        tx,
        n_virtual_sources=n_vs,
        field_points=field_points,
        num_epochs=num_epochs,
        lr=lr_pos,
        lr_apod=lr_apod,
        resolution_weight=resolution_weight,
        energy_weight=energy_weight,
        coverage_weight=coverage_weight,
        aperture_weight=aperture_weight,  # cost apertures
        symmetry_weight=symmetry_weight,  # symmetry
        target_fnumber=1.5,
        batch_size=2048,
        use_gpu=use_gpu,
        optimizer_type=optimizer_type,
        x_init_mm=x_init[:n_vs],
        z_init_mm=z_init[:n_vs],
        FD_init=2,
        n_gauss_init=1,
        fs=fs,
        use_phase=use_phase,
        coverage_threshold_db=-10,  # for soft coverage loss
    )

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
