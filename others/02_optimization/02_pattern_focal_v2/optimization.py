import numpy as np
import torch
import torch.nn as nn

from pyfield.future.TorchField_flexible import TorchFieldFlexible

# Helper function to create Gaussian weights along z-axis for integration


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


# ===========================================================================
# Loss Functions
# ===========================================================================


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


# ===========================================================================
# Main Optimization Function
# ===========================================================================


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
    optimizer_type="Adam",
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
    sigmoid_transform = lambda x: torch.sigmoid(20 * (x - 0.5))

    # Delay are periodic, so we can use a sigmoid transform to keep them in a
    # reasonable range (e.g., ±2pi -> ±1/fc time)
    max_delay = 1 / transducer.fc * 1e6  # Max delay (us) ~ one period

    sigmoid_transform_delays = lambda x: max_delay * torch.sigmoid(10 / max_delay * x)

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
        transform=sigmoid_transform_delays,  # Apply sigmoid transform
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

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": tf._optimizable_params["delays"].value, "lr": lr_delays},
                {"params": tf._optimizable_params["apodization"].value, "lr": lr_apod},
            ]
        )
    elif optimizer_type == "SGD":
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

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": tf._optimizable_params["delays"].value, "lr": lr_delays},
                {"params": tf._optimizable_params["apodization"].value, "lr": lr_apod},
            ]
        )
    elif optimizer_type == "SGD":
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
