"""
Optimize Delays and Apodization to Match Binary Mask Pattern

This script demonstrates how to optimize transducer delays and apodization
to achieve a target pressure pattern (binary mask) using TorchFieldFlexible.

Features:
- Sigmoid transformation for apodization (keeps values in [0, 1])
- Pressure-to-pattern conversion for comparison with binary target
- Combined physics-based and pattern-matching loss
- Works with both linear and matrix arrays

Usage:
    uv run others/learning_focalization/optimize_delays_apod_mask.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pyfield import PyField
from pyfield.psimulation.TorchField_flexible import TorchFieldFlexible
from pyfield.transducers import Domino, LinearArrayTransducer, Zeus_Matrix
from pyfield.utilities import plot_pressure_planes, to_dB

# ============================================================================
# Helper Functions (from original learning code)
# ============================================================================


def gaussian_1d(n, sigma=0.7, device="cpu", plot=False):
    """Create 1D Gaussian weights centered at middle."""
    x = torch.arange(n, device=device, dtype=torch.float32)
    center = (n - 1) / 2.0
    weights = torch.exp(-((x - center) ** 2) / (2 * (sigma * n) ** 2))
    weights = weights / weights.sum()

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(weights.cpu().numpy(), "o-")
        plt.xlabel("Z index")
        plt.ylabel("Weight")
        plt.title(f"Gaussian weights (sigma={sigma})")
        plt.grid(True)
        plt.show()

    return weights


def pattern_from_pr_3Dto2D(pr, max_pr_plane, z_weights=None):
    """
    Convert 3D pressure field to 2D pattern by weighted sum along z.

    Parameters
    ----------
    pr : Tensor [nx, ny, nz]
        Pressure field
    max_pr_plane : float
        Maximum pressure value for normalization
    z_weights : Tensor [nz], optional
        Weights for z-axis integration. If None, use uniform

    Returns
    -------
    Tensor [nx, ny]
        2D pattern normalized to [0, 1]
    """
    if z_weights is not None:
        # Reshape z_weights to broadcast: [1, 1, nz]
        z_w = z_weights.view(1, 1, -1)
        pattern = (pr * z_w).sum(dim=-1)
    else:
        pattern = pr.mean(dim=-1)

    # Normalize to [0, 1]
    pattern = pattern / max_pr_plane
    pattern = pattern.clamp(0, 1)

    return pattern


def stack_2D_to_3D(pattern_2d, nz, sigma=0.7):
    """
    Expand 2D pattern to 3D with Gaussian weighting along z.

    Parameters
    ----------
    pattern_2d : Tensor [nx, ny]
        2D pattern
    nz : int
        Number of z points
    sigma : float
        Sigma for Gaussian weighting

    Returns
    -------
    Tensor [nx, ny, nz]
        3D pattern
    """
    device = pattern_2d.device
    z_weights = gaussian_1d(nz, sigma=sigma, device=device)

    # Expand pattern_2d to [nx, ny, nz] and multiply by z weights
    pattern_3d = pattern_2d.unsqueeze(-1) * z_weights.view(1, 1, -1)

    return pattern_3d


# ============================================================================
# Loss Functions
# ============================================================================
def dB(x):
    """Convert linear value to decibels."""
    return 20 * torch.log10(x + 1e-20)


def loss_energy(y_target_3D, pr, loss_type="linear"):
    """
    Energy-based loss: maximize energy in target region, minimize elsewhere.

    Parameters
    ----------
    y_target_3D : Tensor [nx, ny, nz]
        Binary target mask (0 or 1)
    pr : Tensor [nx, ny, nz]
        Pressure field
    min_error : float
        Small constant for numerical stability

    Returns
    -------
    Tensor (scalar)
        Log loss value
    """

    if loss_type != "linear":
        E_focus = y_target_3D * pr**2
        E_sides = (1 - y_target_3D) * pr**2
        # Log loss: maximize focus, minimize sides
        log_loss = dB(E_sides.mean()) - dB(E_focus.mean())
    else:
        # Energy in focal region
        E_focus = y_target_3D * pr
        # Energy outside focal region
        E_sides = (1 - y_target_3D) * pr
        log_loss = (E_sides.mean() + 1e-6) / (E_focus.mean() + 1e-6)

    return log_loss


# ============================================================================
# Main Optimization Function
# ============================================================================


def optimize_delays_apod_for_pattern(
    transducer,
    target_mask,
    field_points,
    *,
    initial_delays=None,
    initial_apod=None,
    loss1_type="linear",
    loss2_type="linear",
    loss_alpha=None,
    num_epochs_delays=50,
    num_epochs_apod=50,
    lr_delays=1e-3,
    lr_apod=1e-2,
    batch_size=2048,
    sigma_z=0.7,
    use_gpu=True,
    save_path=None,
    optimizer_type = "Adam"
):
    """
    Optimize delays and apodization to match a target binary mask.

    Parameters
    ----------
    transducer : TransducerBase
        Transducer object
    target_mask : ndarray [nx, ny]
        Binary target pattern (0 or 1)
    field_points : dict
        Field specification
    num_epochs_delays : int
        Epochs for delay optimization
    num_epochs_apod : int
        Epochs for apodization optimization
    lr_delays : float
        Learning rate for delays
    lr_apod : float
        Learning rate for apodization
    batch_size : int
        Batch size for simulation
    sigma_z : float
        Sigma for Gaussian z-weighting
    use_gpu : bool
        Use GPU if available
    save_path : str, optional
        Path to save results

    Returns
    -------
    dict
        Results including optimized parameters, loss history, etc.
    """
    print("=" * 70)
    print("Optimizing Delays and Apodization for Pattern Matching")
    print("=" * 70)

    # Setup device
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert target to tensor
    target_2d = torch.tensor(target_mask, dtype=torch.float32, device=device)

    # Create TorchFieldFlexible
    tf = TorchFieldFlexible(transducer, use_gpu=use_gpu, verbose=True)

    # ========================================================================
    # Step 1: Compute reference (to get normalization constants)
    # ========================================================================
    print("\n[1/4] Computing reference field...")

    # Set focal point for initial delays/apodization
    focus_mm = [0, 0, field_points["z_extent"][0]]
    transducer.compute_delays(focus_mm=focus_mm)
    transducer.compute_apodization(focus_mm=focus_mm, FoverD=1.0)

    # Reinitialize TorchField with updated parameters
    tf = TorchFieldFlexible(transducer, use_gpu=use_gpu, verbose=False)

    with torch.no_grad():
        x, y, z, pr_ref = tf(field_points, training=False)
        pr_ref = torch.tensor(pr_ref, device=device)

    nz = pr_ref.shape[-1]
    z_weights = gaussian_1d(nz, sigma=sigma_z, device=device)

    # Compute max pressure for normalization
    max_pr_plane = ((pr_ref * z_weights.view(1, 1, -1)).sum(dim=-1)).max().item()
    print(f"Reference max pressure: {max_pr_plane:.4e}")

    # Create 3D target
    target_3d = stack_2D_to_3D(target_2d, nz, sigma=sigma_z)

    # ========================================================================
    # Step 2: Setup optimization
    # ========================================================================
    print("\n[2/4] Setting up optimization...")

    # Add sigmoid transform for apodization
    sigmoid_transform = lambda x: torch.sigmoid(10 * (x - 0.5))

    # Reinitialize with optimizable parameters
    # Start with zeros for delays, 0.5 for apodization (will be sigmoided)
    if initial_delays is None:
        initial_delays = np.zeros(transducer.n_elements)
    if initial_apod is None:
        initial_apod = np.ones(transducer.n_elements) * 0.5

    tf = TorchFieldFlexible(transducer, use_gpu=use_gpu, verbose=False)

    # Replace default parameters with optimizable ones
    tf.add_optimizable_parameter(
        "delays",
        initial_value=initial_delays * 1e6,  # to μs
        level="element",
        requires_grad=True,
        replace=True,
    )

    tf.add_optimizable_parameter(
        "apodization",
        initial_value=initial_apod,
        level="element",
        requires_grad=True,
        constraints={"min": 0.0, "max": 1.0},
        transform=sigmoid_transform,  # Apply sigmoid
        replace=True,
    )

    print(
        f"Optimizable parameters: {[p.name for p in tf._optimizable_params.values() if p.value.requires_grad]}"
    )

    # Loss function
    loss_mse = nn.MSELoss()

    # ========================================================================
    # Step 3: Optimize delays only
    # ========================================================================

    print(f"\n[3/4] Optimizing delays ({num_epochs_delays} epochs)...")

    # Freeze apodization
    tf._optimizable_params["apodization"].value.requires_grad = False
    tf._optimizable_params["delays"].value.requires_grad = True
    
    if optimizer_type == "Adam"
        optimizer = torch.optim.Adam(
            [
                {"params": tf._optimizable_params["delays"].value, "lr": lr_delays},
                {"params": tf._optimizable_params["apodization"].value, "lr": lr_apod},
            ]
        )
    elif optimizer_type == "SGD"
        optimizer = torch.optim.SGD(
            [
                {"params": tf._optimizable_params["delays"].value, "lr": lr_delays},
                {"params": tf._optimizable_params["apodization"].value, "lr": lr_apod},
            ]
        )

    loss_history_delays = []
    loss_energy_values = []
    loss_comp_values = []

    for epoch in range(num_epochs_delays):
        optimizer.zero_grad()

        # Forward pass
        x, y, z, pr = tf(field_points, training=True, batch_size=batch_size)

        # Convert to pattern
        pattern_2d = pattern_from_pr_3Dto2D(pr, max_pr_plane, z_weights)

        # Compute losses
        loss_phys = loss_energy(target_3d, pr, loss_type=loss1_type)
        if loss2_type == "linear":
            loss_comp = loss_mse(target_2d, pattern_2d)
        else:
            loss_comp = loss_mse(dB(target_2d), dB(pattern_2d))

        # Combined loss
        if epoch == 0:
            alpha0 = loss_phys.item() / (loss_comp.item() + 1e-6)

        if loss_alpha is None:
            alpha = loss_phys.item() / (loss_comp.item() + 1e-6)
        elif loss_alpha == "constant":
            alpha = alpha0
        else:
            alpha = loss_alpha
        loss = loss_phys + alpha * loss_comp

        # Backward
        loss.backward()
        optimizer.step()
        tf.apply_constraints()

        loss_history_delays.append(loss.item())
        loss_energy_values.append(loss_phys.item())
        loss_comp_values.append(loss_comp.item())

        if epoch % 10 == 0 or epoch == num_epochs_delays - 1:
            print(
                f"  Epoch {epoch:3d}: Loss={loss.item():.6f} "
                f"(phys={loss_phys.item():.6f}, comp={loss_comp.item():.6f})"
            )

    # ========================================================================
    # Step 4: Optimize delays + apodization
    # ========================================================================
    print(f"\n[4/4] Optimizing delays + apodization ({num_epochs_apod} epochs)...")

    # Unfreeze apodization
    tf._optimizable_params["apodization"].value.requires_grad = True
    tf._optimizable_params["delays"].value.requires_grad = True
 
    if optimizer_type == "Adam"
        optimizer = torch.optim.Adam(
            [
                {"params": tf._optimizable_params["delays"].value, "lr": lr_delays},
                {"params": tf._optimizable_params["apodization"].value, "lr": lr_apod},
            ]
        )
    elif optimizer_type == "SGD"
        optimizer = torch.optim.SGD(
            [
                {"params": tf._optimizable_params["delays"].value, "lr": lr_delays},
                {"params": tf._optimizable_params["apodization"].value, "lr": lr_apod},
            ]
        )

    loss_history_apod = []

    for epoch in range(num_epochs_apod):
        optimizer.zero_grad()

        x, y, z, pr = tf(field_points, training=True, batch_size=batch_size)
        pattern_2d = pattern_from_pr_3Dto2D(pr, max_pr_plane, z_weights)

        # Compute losses
        loss_phys = loss_energy(target_3d, pr, loss_type=loss1_type)
        if loss2_type == "linear":
            loss_comp = loss_mse(target_2d, pattern_2d)
        else:
            loss_comp = loss_mse(dB(target_2d), dB(pattern_2d))

        # Combined loss
        if epoch == 0:
            alpha0 = loss_phys.item() / (loss_comp.item() + 1e-6)

        if loss_alpha is None:
            alpha = loss_phys.item() / (loss_comp.item() + 1e-6)
        elif loss_alpha == "constant":
            alpha = alpha0
        else:
            alpha = loss_alpha

        loss = loss_phys + alpha * loss_comp

        loss.backward()
        optimizer.step()
        tf.apply_constraints()

        loss_history_apod.append(loss.item())
        loss_energy_values.append(loss_phys.item())
        loss_comp_values.append(loss_comp.item())

        if epoch % 10 == 0 or epoch == num_epochs_apod - 1:
            print(
                f"  Epoch {epoch:3d}: Loss={loss.item():.6f} "
                f"(phys={loss_phys.item():.6f}, comp={loss_comp.item():.6f})"
            )

    # ========================================================================
    # Final results
    # ========================================================================
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    # Get final parameters
    with torch.no_grad():
        delays_final = tf.get_parameter("delays").cpu().numpy() / 1e6  # to seconds
        apod_final = tf.get_parameter("apodization").cpu().numpy()

    # Compute final field
    with torch.no_grad():
        x, y, z, pr_final = tf(field_points, training=False)

    print(
        f"\nFinal delays range: [{delays_final.min() * 1e6:.2f}, {delays_final.max() * 1e6:.2f}] μs"
    )
    print(f"Final apodization range: [{apod_final.min():.3f}, {apod_final.max():.3f}]")

    # Save results
    results = {
        "delays": delays_final,
        "apodization": apod_final,
        "loss_history_delays": loss_history_delays,
        "loss_history_apod": loss_history_apod,
        "loss_energy_values": loss_energy_values,
        "loss_comp_values": loss_comp_values,
        "x": x,
        "y": y,
        "z": z,
        "pressure_final": pr_final,
        "target_mask": target_mask,
        "field_points": field_points,
    }

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **results)
        print(f"\nResults saved to: {save_path}")

    return results


# ============================================================================
# Visualization
# ============================================================================


def plot_results(results, save_path=None):
    """Plot optimization results."""
    fig = plt.figure(figsize=(16, 10))

    # Create gridspec with dedicated colorbar columns
    # Columns: [plot, cbar, plot, cbar, plot, cbar]
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05])

    # Loss history

    loss_energy = np.array(results["loss_energy_values"])
    loss_comp = np.array(results["loss_comp_values"])
    norm_loss_energy = (loss_energy - loss_energy.min()) / (
        loss_energy.max() - loss_energy.min() + 1e-6
    )
    norm_loss_comp = (loss_comp - loss_comp.min()) / (
        loss_comp.max() - loss_comp.min() + 1e-6
    )

    loss_history_delays = np.array(results["loss_history_delays"])
    loss_history_apod = np.array(results["loss_history_apod"])

    n_delays = len(loss_history_delays)
    n_apod = len(loss_history_apod)
    iter_delays = np.arange(n_delays)
    iter_apod = np.arange(n_delays, n_delays + n_apod)

    # Row 0: Loss history, Delays, Apodization (span 2 columns each for centering)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iter_delays, loss_history_delays, color="b", label="Delays only")
    ax1.plot(iter_apod, loss_history_apod, color="g", label="Delays + Apod")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(loc="lower left")
    ax1.grid(True)
    ax1.set_yscale("log")

    ax11 = ax1.twinx()
    ax11.plot(norm_loss_energy, "--", color="tab:orange", label="Energy Loss")
    ax11.plot(norm_loss_comp, "--", color="tab:red", label="Pattern Loss")
    ax11.set_ylabel("Normalized Loss (dashed)")
    ax11.set_yscale("log")
    ax11.legend(loc="upper right")
    # ax11.set_ylim(-0.05, 1.05)

    # Delays
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.plot(results["delays"] * 1e6, "or-")
    ax2.set_xlabel("Element Index")
    ax2.set_ylabel("Delay (μs)")
    ax2.set_title("Optimized Delays")
    ax2.grid(True)

    # Apodization
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax3.plot(results["apodization"], "ok-")
    ax3.set_xlabel("Element Index")
    ax3.set_ylabel("Apodization")
    ax3.set_title("Optimized Apodization")
    ax3.grid(True)

    # Row 1: Target pattern, Pressure fields with dedicated colorbar columns
    pr = results["pressure_final"]
    x, y, z = results["x"], results["y"], results["z"]
    extent_xz = [z.min(), z.max(), x.min(), x.max()]
    extent_xy = [y.min(), y.max(), x.min(), x.max()]

    # Target pattern
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(
        results["target_mask"],
        aspect="auto",
        cmap="gray",
        origin="lower",
        extent=extent_xy,
    )
    ax4.set_title("Target Pattern")
    ax4.set_xlabel("Y (mm)")
    ax4.set_ylabel("X (mm)")
    cax4 = fig.add_subplot(gs[1, 1])
    plt.colorbar(im4, cax=cax4)

    # Pressure field (dB)
    pr_xy = pr[:, :, len(z) // 2]
    pr_xy_norm = pr_xy / pr_xy.max()
    pr_db = to_dB(pr_xy_norm)

    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(
        pr_db,
        aspect="auto",
        origin="lower",
        extent=extent_xy,
        cmap="hot",
        vmin=-40,
        vmax=0,
    )
    ax5.set_xlabel("Y (mm)")
    ax5.set_ylabel("X (mm)")
    ax5.set_title("Pressure Field (dB)")
    cax5 = fig.add_subplot(gs[1, 3])
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar5.set_label("dB", rotation=270, labelpad=15)

    # Pressure field (normalized)
    ax6 = fig.add_subplot(gs[1, 4])
    im6 = ax6.imshow(
        pr_xy_norm,
        aspect="auto",
        origin="lower",
        extent=extent_xy,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    ax6.set_xlabel("Y (mm)")
    ax6.set_ylabel("X (mm)")
    ax6.set_title("Pressure Field (a.u.)")
    cax6 = fig.add_subplot(gs[1, 5])
    plt.colorbar(im6, cax=cax6)

    # Row 2: Comparison plot (spans all columns)
    target = results["target_mask"]
    target_flat = np.squeeze(target).T.flatten()
    pr_xy_flat = np.squeeze(pr_xy_norm).T.flatten()
    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(target_flat, "b-", alpha=0.5, label="Target")
    ax7.plot(pr_xy_flat, "r-", alpha=0.5, label="Achieved")
    ax7.set_xlim(0, len(target_flat))
    ax7.set_xlabel("Pixel Index")
    ax7.set_ylabel("Intensity")
    ax7.set_title("Target vs Achieved Pattern (flattened)")
    ax7.legend()
    ax7.grid(True)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
    plt.show()
    plt.close()


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("\nExample 1: Linear Array - Focal Point Pattern")
    print("-" * 70)

    print(torch.__version__)

    use_cuda = True  # Set to False if you want to run on CPU
    # transducer type and saving folder
    base_path = "./results/optimizer/"
    txarray = "linarray"  # "linarray" or "matrixarray"

    # Optimization settings
    Energy_loss_type = "linear"  # "linear" or "log"
    MSE_loss_type = "log"  # "linear" or "log"
    n_delays = 0  # Skip delay optimization for this example
    n_apod = 200
    lr_delays = 1e-3
    lr_apod = 1e-3
    alpha = None  # Weight for combining losses (if using combined loss)
    optimizer_type = "SGD"

    # Select transducer
    if txarray == "linarray":
        tx = Domino()
        resultsfolder = "linear"

    elif txarray == "matrixarray":
        tx = Zeus_Matrix()
        resultsfolder = "matrix"

    # Field specification
    c = 1540  # m/s
    z_focal = 8  # mm
    fc = tx.fc  # Hz
    aperture_mm = tx.n_elements * tx.pitch * 1e3  # mm
    lambda_mm = c / fc * 1e3  # mm
    estimated_width_mm = 1.4 * lambda_mm * z_focal / aperture_mm
    dx_lambdas = 1

    print(
        f"Transducer center frequency: {fc / 1e6:.2f} MHz, wavelength: {lambda_mm:.2f}mm"
    )
    print(f"Estimated focal spot width: {estimated_width_mm:.2f} mm")
    dx_mm = (
        round(dx_lambdas * estimated_width_mm * 100) / 100
    )  # mm, round to nearest 0.01mm

    # Define field points around focal region
    field_points = {
        "x_extent": [-3, 3],
        "y_extent": [-3, 3],
        "z_extent": [z_focal, z_focal],
        "dx": dx_mm,
        "dy": 0.20,
        "dz": 0.5,
    }

    # Create target: single focal point
    Dx = field_points["x_extent"][1] - field_points["x_extent"][0]
    Dy = field_points["y_extent"][1] - field_points["y_extent"][0]
    nx, ny = int(Dx / field_points["dx"]), int(Dy / field_points["dy"])
    if Dx % field_points["dx"] != 0:
        nx += 1
    if Dy % field_points["dy"] != 0:
        ny += 1
    target = np.zeros((nx, ny))
    target[nx // 2, ny // 2] = 1  # Center point

    # Run optimization
    def pH(lr):
        return f"pH{-np.log10(lr):.2f}"

    file_name =
    f"""optim_{txarray}_zfoc{z_focal}_loss1{Energy_loss_type}_loss2{MSE_loss_type}_ndel{n_delays}_napod{n_apod}_lrdel{pH(lr_delays)}_lrapod{pH(lr_apod)}_initdel0_initapod1_dxlambdas{dx_lambdas}{optimizer_type}.npz"""

    save_path = str(Path(base_path) / resultsfolder / file_name)
    # print(save_path)
    results = optimize_delays_apod_for_pattern(
        tx,
        target,
        field_points,
        initial_delays=None,
        initial_apod=np.ones(tx.n_elements),
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
        optimizer_type = "Adam"
    )
    print(
        "Max pressure at target:{:.4e}".format(results["pressure_final"].max().item())
    )
    # Plot
    plot_results(results, save_path=save_path.replace(".npz", "_summary.png"))

    ## Compute a xz slice with results
    plane_xz = {
        "x_extent": [-2, 2],
        "y_extent": [0, 0],
        "z_extent": [-3 + z_focal, z_focal + 3],
        "dx": dx_mm,
        "dy": 0.20,
        "dz": 0.075,
    }
    tx.set_delays(results["delays"])
    tx.set_apodization(results["apodization"])
    pf = PyField(tx)
    x, y, z, pr = pf(plane_xz)
    plot_pressure_planes(
        x,
        y,
        z,
        pr,
        save_path=save_path.replace(".npz", "_plane.png"),
        title="Optimized Pressure Field (XZ Plane)",
    )
