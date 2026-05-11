"""
Examples: Flexible TorchField Parameter Optimization

This script demonstrates various optimization scenarios using the flexible
TorchField framework:

1. Direct parameter optimization (delays, apodization)
2. Virtual source optimization (for diverging/focusing waves)
3. Custom apodization functions
4. Element position perturbations
5. Multi-parameter optimization

Each example is self-contained and can be run independently.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pyfield.psimulation.TorchField_flexible import TorchFieldFlexible
from pyfield.transducers import LinearArrayTransducer, Domino
from pyfield.plotting import plot2D_pressure_slices
from pyfield.utilities import to_dB

# ============================================================================
# Example 1: Direct Delay Optimization
# ============================================================================


def example1_direct_delay_optimization():
    """Optimize delays directly to achieve a target focal point."""
    print("\n" + "=" * 70)
    print("Example 1: Direct Delay Optimization")
    print("=" * 70)

    # Create transducer
    tx = LinearArrayTransducer(
        n_elements=64,
        element_width_mm=0.25,
        element_height_mm=12.0,
        kerf_mm=0.05,
        no_sub_x=2,
        no_sub_y=4,
        frequency_Hz=5e6,
    )

    # Initialize with uniform delays
    tx.compute_delays(focus_mm=[0, 0, 30])

    # Create TorchField
    tf = TorchFieldFlexible(tx, use_gpu=True, verbose=True)

    # Make delays optimizable
    tf._optimizable_params['delays'].value.requires_grad = True

    print(f"\n{tf}")

    # Define target field
    target_focus = [2, 0, 35]  # offset focus
    field_points = {
        "x_extent": [-3, 3],
        "y_extent": [-0.5, 0.5],
        "z_extent": [25, 40],
        "dx": 0.2,
        "dy": 1.0,
        "dz": 0.3,
    }

    # Optimizer
    optimizer = torch.optim.Adam(tf.get_optimizable_parameters(), lr=1e-2)

    # Training loop
    num_epochs = 50
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        x, y, z, p = tf(field_points, training=True, normalize=False)

        # Simple loss: maximize pressure at target point
        # Find index closest to target
        target_idx_x = torch.argmin((x - target_focus[0]) ** 2)
        target_idx_z = torch.argmin((z - target_focus[2]) ** 2)

        # Loss: negative pressure at target (maximize)
        loss = -p[target_idx_x, 0, target_idx_z]

        loss.backward()
        optimizer.step()
        tf.apply_constraints()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # Final result
    with torch.no_grad():
        x, y, z, p = tf(field_points, training=False, normalize=True)

    print(f"\nOptimization complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Initial loss: {losses[0]:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    # Plot pressure field
    p_db = to_dB(p)
    extent = [z.min(), z.max(), x.min(), x.max()]
    im = axes[1].imshow(
        p_db[:, 0, :],
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='hot',
        vmin=-40,
        vmax=0,
    )
    axes[1].plot(target_focus[2], target_focus[0], 'g*', markersize=15, label='Target')
    axes[1].set_xlabel("Z (mm)")
    axes[1].set_ylabel("X (mm)")
    axes[1].set_title("Optimized Pressure Field")
    axes[1].legend()
    plt.colorbar(im, ax=axes[1], label="Pressure (dB)")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Example 2: Virtual Source Optimization
# ============================================================================


def example2_virtual_source_optimization():
    """
    Optimize virtual source position instead of delays directly.

    This is more intuitive for diverging wave imaging where you want
    to optimize the virtual source location behind the array.
    """
    print("\n" + "=" * 70)
    print("Example 2: Virtual Source Optimization")
    print("=" * 70)

    # Create transducer
    tx = Domino()

    # Create TorchField
    tf = TorchFieldFlexible(tx, use_gpu=True, verbose=True)

    # Add virtual source as optimizable parameter
    initial_vs = [0, 0, 30]  # mm
    tf.add_optimizable_parameter(
        'virtual_source',
        initial_value=initial_vs,
        level='global',
        requires_grad=True,
        constraints={'min': -50, 'max': 50},  # reasonable bounds
    )

    # Add mapping: virtual_source → delays
    def compute_delays_from_virtual_source(virtual_source, tx, device):
        """Compute delays from virtual source position."""
        vs_mm = virtual_source.detach().cpu().numpy()
        delays_s = tx.compute_delays(focus_mm=vs_mm, apply=False)
        delays_us = torch.tensor(
            delays_s * 1e6, dtype=torch.float32, device=device
        )
        return delays_us

    tf.add_parameter_mapping(
        name='vs_to_delays',
        function=compute_delays_from_virtual_source,
        inputs=['virtual_source'],
        output='delays',
        level='element',
    )

    print(f"\n{tf}")

    # Now 'delays' will be computed from 'virtual_source' automatically
    # Test it
    field_points = {
        "x_extent": [-5, 5],
        "y_extent": [-0.5, 0.5],
        "z_extent": [20, 40],
        "dx": 0.3,
        "dy": 1.0,
        "dz": 0.5,
    }

    print("\nTesting virtual source mapping...")
    x, y, z, p = tf(field_points, training=False, normalize=True)
    print(f"Initial virtual source: {initial_vs}")
    print(f"Field computed successfully with shape: {p.shape}")

    # Now optimize virtual source position
    target_focus = [3, 0, 30]
    optimizer = torch.optim.Adam(tf.get_optimizable_parameters(), lr=0.5)

    num_epochs = 30
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        x, y, z, p = tf(field_points, training=True)

        # Loss: focus at target
        target_idx_x = torch.argmin((x - target_focus[0]) ** 2)
        target_idx_z = torch.argmin((z - target_focus[2]) ** 2)
        loss = -p[target_idx_x, 0, target_idx_z]

        loss.backward()
        optimizer.step()
        tf.apply_constraints()

        if epoch % 5 == 0:
            vs = tf.get_parameter('virtual_source').detach().cpu().numpy()
            print(f"Epoch {epoch}: VS = {vs}, Loss = {loss.item():.6f}")

    final_vs = tf.get_parameter('virtual_source').detach().cpu().numpy()
    print(f"\nOptimized virtual source: {final_vs}")
    print(f"Target focus: {target_focus}")


# ============================================================================
# Example 3: Custom Apodization Function
# ============================================================================


def example3_custom_apodization_function():
    """
    Optimize parameters of a custom apodization function.

    Instead of optimizing each apodization value independently,
    optimize parameters of a function (e.g., Gaussian center and width).
    """
    print("\n" + "=" * 70)
    print("Example 3: Custom Apodization Function")
    print("=" * 70)

    tx = LinearArrayTransducer(
        n_elements=64,
        element_width_mm=0.25,
        element_height_mm=12.0,
        kerf_mm=0.05,
        no_sub_x=2,
        no_sub_y=4,
        frequency_Hz=5e6,
    )

    tf = TorchFieldFlexible(tx, use_gpu=True, verbose=True)

    # Add Gaussian apodization parameters
    tf.add_optimizable_parameter(
        'apod_center',
        initial_value=32.0,  # center element
        level='global',
        requires_grad=True,
        constraints={'min': 0, 'max': 63},
    )

    tf.add_optimizable_parameter(
        'apod_width',
        initial_value=20.0,  # width in elements
        level='global',
        requires_grad=True,
        constraints={'min': 5, 'max': 50},
    )

    # Mapping: Gaussian parameters → apodization
    def gaussian_apodization(apod_center, apod_width, tx, device):
        """Compute Gaussian apodization from center and width."""
        n_elem = tx.n_elements
        elements = torch.arange(n_elem, dtype=torch.float32, device=device)

        # Gaussian function
        apod = torch.exp(-((elements - apod_center) ** 2) / (2 * apod_width ** 2))

        # Normalize to [0, 1]
        apod = apod / apod.max()

        return apod

    tf.add_parameter_mapping(
        name='gaussian_apod',
        function=gaussian_apodization,
        inputs=['apod_center', 'apod_width'],
        output='apodization',
        level='element',
    )

    print(f"\n{tf}")

    # Optimize to create a specific beam pattern
    field_points = {
        "x_extent": [-10, 10],
        "y_extent": [-0.5, 0.5],
        "z_extent": [25, 35],
        "dx": 0.2,
        "dy": 1.0,
        "dz": 0.5,
    }

    # Target: narrow beam at center
    optimizer = torch.optim.Adam(tf.get_optimizable_parameters(), lr=0.1)

    for epoch in range(20):
        optimizer.zero_grad()

        x, y, z, p = tf(field_points, training=True)

        # Loss: minimize beamwidth at focal depth
        # Get pressure at z=30mm
        z_idx = torch.argmin((z - 30) ** 2)
        lateral_profile = p[:, 0, z_idx]

        # Want narrow peak at center
        # Penalize energy away from center
        x_center_idx = len(x) // 2
        loss = -lateral_profile[x_center_idx] + 0.1 * lateral_profile.sum()

        loss.backward()
        optimizer.step()
        tf.apply_constraints()

        if epoch % 5 == 0:
            center = tf.get_parameter('apod_center').item()
            width = tf.get_parameter('apod_width').item()
            print(f"Epoch {epoch}: center={center:.1f}, width={width:.1f}, loss={loss.item():.6f}")

    # Visualize final apodization
    apod_final = tf.get_parameter('apodization').detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(apod_final, 'o-')
    axes[0].set_xlabel("Element Index")
    axes[0].set_ylabel("Apodization")
    axes[0].set_title("Optimized Gaussian Apodization")
    axes[0].grid(True)

    # Show field
    x, y, z, p = tf(field_points, training=False, normalize=True)
    p_db = to_dB(p)
    extent = [z.min(), z.max(), x.min(), x.max()]
    im = axes[1].imshow(
        p_db[:, 0, :],
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='hot',
        vmin=-40,
        vmax=0,
    )
    axes[1].set_xlabel("Z (mm)")
    axes[1].set_ylabel("X (mm)")
    axes[1].set_title("Pressure Field")
    plt.colorbar(im, ax=axes[1], label="Pressure (dB)")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Example 4: Element Position Perturbations
# ============================================================================


def example4_element_position_optimization():
    """
    Optimize small perturbations to element positions.

    This could be used to:
    - Compensate for manufacturing defects
    - Optimize array geometry
    - Study sensitivity to positioning errors
    """
    print("\n" + "=" * 70)
    print("Example 4: Element Position Perturbations")
    print("=" * 70)

    tx = LinearArrayTransducer(
        n_elements=32,  # Smaller for faster computation
        element_width_mm=0.3,
        element_height_mm=10.0,
        kerf_mm=0.05,
        no_sub_x=2,
        no_sub_y=4,
        frequency_Hz=5e6,
    )

    tf = TorchFieldFlexible(tx, use_gpu=True, verbose=True)

    # Add position offsets as optimizable (small perturbations)
    # Shape: [n_elements, 3] for (x, y, z) offsets
    initial_offsets = np.zeros((tx.n_elements, 3))

    tf.add_optimizable_parameter(
        'element_offsets',
        initial_value=initial_offsets,
        level='element',  # per-element
        requires_grad=True,
        constraints={'min': -0.5, 'max': 0.5},  # max ±0.5mm offset
    )

    # Mapping: offsets → patch centers
    def compute_patch_centers_with_offsets(element_offsets, tx, device):
        """Recompute patch centers with element position offsets."""
        # Get original patch centers
        original_centers = []
        for elem_idx in range(tx.n_elements):
            for sub_idx in range(tx.no_sub_x * tx.no_sub_y):
                patch_idx = elem_idx * (tx.no_sub_x * tx.no_sub_y) + sub_idx
                verts = tx.sub_quad_verts[patch_idx]
                original_centers.append(verts.mean(axis=0))

        original_centers = torch.tensor(
            np.array(original_centers),
            dtype=torch.float32,
            device=device
        )

        # Apply offsets (expand element offsets to patches)
        # element_offsets: [n_elements, 3]
        # Expand to [n_patches, 3]
        offsets_expanded = element_offsets.repeat_interleave(
            tx.no_sub_x * tx.no_sub_y, dim=0
        )

        # Add offsets (convert mm to μm)
        perturbed_centers = (original_centers + offsets_expanded) * 1e6  # to μm

        return perturbed_centers

    tf.add_parameter_mapping(
        name='offsets_to_centers',
        function=compute_patch_centers_with_offsets,
        inputs=['element_offsets'],
        output='patch_centers',
        level='patch',
    )

    print(f"\n{tf}")
    print("Note: Element positions can now be optimized!")
    print("This could be used to:")
    print("  - Correct for positioning errors")
    print("  - Optimize array geometry")
    print("  - Calibrate array defects")


# ============================================================================
# Example 5: Multi-Parameter Optimization (Virtual Source + Apodization)
# ============================================================================


def example5_multi_parameter_optimization():
    """
    Optimize multiple parameters simultaneously:
    - Virtual source position (controls delays)
    - Apodization function parameters

    This demonstrates the full power of the flexible framework.
    """
    print("\n" + "=" * 70)
    print("Example 5: Multi-Parameter Optimization")
    print("=" * 70)

    tx = LinearArrayTransducer(
        n_elements=64,
        element_width_mm=0.25,
        element_height_mm=12.0,
        kerf_mm=0.05,
        no_sub_x=2,
        no_sub_y=4,
        frequency_Hz=5e6,
    )

    tf = TorchFieldFlexible(tx, use_gpu=True, verbose=True)

    # Add virtual source
    tf.add_optimizable_parameter(
        'virtual_source',
        initial_value=[0, 0, 30],
        level='global',
        requires_grad=True,
    )

    # Add apodization parameters
    tf.add_optimizable_parameter(
        'apod_center',
        initial_value=32.0,
        level='global',
        requires_grad=True,
        constraints={'min': 0, 'max': 63},
    )

    tf.add_optimizable_parameter(
        'apod_width',
        initial_value=30.0,
        level='global',
        requires_grad=True,
        constraints={'min': 10, 'max': 50},
    )

    # Mappings
    def vs_to_delays(virtual_source, tx, device):
        vs_mm = virtual_source.detach().cpu().numpy()
        delays_s = tx.compute_delays(focus_mm=vs_mm, apply=False)
        return torch.tensor(delays_s * 1e6, dtype=torch.float32, device=device)

    def gaussian_apod(apod_center, apod_width, tx, device):
        elements = torch.arange(tx.n_elements, dtype=torch.float32, device=device)
        apod = torch.exp(-((elements - apod_center) ** 2) / (2 * apod_width ** 2))
        return apod / apod.max()

    tf.add_parameter_mapping(
        'vs_to_delays', vs_to_delays, ['virtual_source'], 'delays', 'element'
    )
    tf.add_parameter_mapping(
        'gaussian_apod', gaussian_apod, ['apod_center', 'apod_width'],
        'apodization', 'element'
    )

    print(f"\n{tf}")

    # Optimize for a specific target pattern
    # (Implementation similar to previous examples)

    print("\nAll parameters are now jointly optimizable!")
    print("This allows for complex optimization scenarios like:")
    print("  - Optimizing focus position AND beam shape")
    print("  - Multi-objective optimization")
    print("  - Adaptive beamforming")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("Flexible TorchField Examples")
    print("=" * 70)
    print("\nThese examples demonstrate different optimization scenarios")
    print("using the flexible parameter mapping framework.\n")

    # Run examples (comment out the ones you don't want to run)

    # example1_direct_delay_optimization()

    example2_virtual_source_optimization()

    # example3_custom_apodization_function()

    # example4_element_position_optimization()

    # example5_multi_parameter_optimization()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
