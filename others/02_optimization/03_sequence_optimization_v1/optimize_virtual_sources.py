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

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from pyfield.future.TorchField_flexible import TorchFieldFlexible
from pyfield.transducers import Domino, LinearArrayTransducer
from pyfield.utilities import to_dB

# ============================================================================
# Helper Functions
# ============================================================================
print("torch version:", torch.__version__)


def compute_element_usage_penalty(apodization, sparsity_weight=0.1):
    """
    Penalty for using too many elements.

    Encourages sparse solutions where only necessary elements are active.

    Parameters
    ----------
    apodization : Tensor [n_elements]
        Element apodization values
    sparsity_weight : float
        Weight for sparsity penalty

    Returns
    -------
    Tensor (scalar)
        Sparsity penalty
    """
    # L1 norm encourages sparsity
    return sparsity_weight * apodization.abs().mean()


def compute_uniformity_loss(pr_field, target_region_mask=None):
    """
    Measure field uniformity in the imaging region.

    Lower is better - want uniform energy distribution.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Pressure field
    target_region_mask : Tensor [nx, ny, nz], optional
        Binary mask for region of interest

    Returns
    -------
    Tensor (scalar)
        Uniformity loss (coefficient of variation)
    """
    if target_region_mask is not None:
        field_roi = pr_field * target_region_mask
    else:
        field_roi = pr_field

    # Coefficient of variation (CV = std / mean)
    # Lower CV = more uniform
    mean_val = field_roi.mean()
    std_val = field_roi.std()

    # Avoid division by zero
    cv = std_val / (mean_val + 1e-1)

    return cv


def compute_transmit_energy(apodization, pressure, pressure_ref=1):
    """
    Measure mean energy in the field.

    Higher is better - want to maximize energy delivery.

    Parameters
    ----------
    apodization : Tensor [n_elements]
        Element apodization values
    mean_pressure : Tensor (scalar)
        Mean pressure in the field

    Returns
    -------
    Tensor (scalar)
        Energy loss (negative of mean pressure)
    """
    # Energy is proportional to mean pressure and element usage
    mean_pressure = ((pressure + 1e-1) / (pressure_ref + 1e-1)).mean()
    energy = mean_pressure / apodization.mean()

    # Return negative (want to maximize energy)
    return -energy


def compute_coverage_loss(pr_field, threshold=0.3):
    """
    Measure fraction of imaging region with sufficient energy.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Normalized pressure field [0, 1]
    threshold : float
        Minimum acceptable pressure level

    Returns
    -------
    Tensor (scalar)
        Coverage loss (want to maximize, so return negative)
    """
    # Fraction of points above threshold
    coverage = (pr_field > threshold).to(torch.float32).mean()

    # Return negative (want to maximize coverage)
    return -coverage


# ============================================================================
# Virtual Source Optimizer
# ============================================================================


class VirtualSourceOptimizer:
    """
    Optimize virtual source positions for plane wave imaging.

    This class handles optimization of multiple virtual sources, finding
    the best positions that:
    - Maximize field uniformity in the imaging region
    - Minimize number of active elements
    - Provide good coverage

    Parameters
    ----------
    transducer : TransducerBase
        Transducer object
    n_virtual_sources : int
        Number of virtual sources to optimize
    field_points : dict
        Field specification
    use_gpu : bool
        Use GPU if available
    """

    def __init__(
        self,
        transducer,
        n_virtual_sources: int,
        field_points: dict,
        use_gpu: bool = True,
        c: float = 1540.0,
        z_behind: float = -3,
        x_spacing: float = 5,
    ):
        self.transducer = transducer
        self.n_vs = n_virtual_sources
        self.field_points = field_points
        self.use_gpu = use_gpu
        self.c = c  # Speed of sound in m/s
        self.z_behind = z_behind  # Default distance behind array for virtual sources
        self.x_spacing = x_spacing  # Spacing between virtual sources in mm

        # Device
        self.device = torch.device(
            "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Create TorchField for each virtual source
        self.torch_fields = []
        self.virtual_sources = []

        self._setup_virtual_sources()

    def _setup_virtual_sources(self):
        """Initialize virtual sources with default positions."""
        print(f"Setting up {self.n_vs} virtual sources...")

        # Initialize virtual sources at different positions
        # For plane waves: spread behind the array

        self.element_centers = torch.tensor(
            self.transducer.element_centers, dtype=torch.float32, device=self.device
        )  # [n_elements, 3]

        for i in range(self.n_vs):
            # Spread in x (lateral direction)
            x_pos = (i - self.n_vs // 2) * self.x_spacing  # 5mm spacing

            # Create TorchField
            tf = TorchFieldFlexible(
                self.transducer, use_gpu=self.use_gpu, verbose=False
            )

            # Add virtual source parameter
            vs_name = f"vs_{i}"
            tf.add_optimizable_parameter(
                vs_name,
                initial_value=[x_pos, self.z_behind],
                level="global",
                requires_grad=True,
            )

            # Add apodization parameter (per virtual source)
            apod_name = f"apod_{i}"
            tf.add_optimizable_parameter(
                apod_name,
                initial_value=self.transducer.compute_apodization(
                    focus_mm=[x_pos, 0, self.z_behind], inline=False
                ),
                level="element",
                requires_grad=True,
                transform=lambda x: torch.sigmoid(10 * (x - 0.5)),
            )

            # Mapping: virtual source → delays
            def make_vs_to_delays(vs_name):
                def vs_to_delays(**kwargs):
                    vs = kwargs[vs_name]
                    # Focus position in meters, preserving gradient flow by
                    # operating directly on vs elements (no torch.stack/tensor
                    # assembly, which is prone to breaking the graph).
                    focus_x_m = vs[0] * 1e-3
                    focus_z_m = torch.abs(vs[1]) * 1e-3

                    # Per-element distance components [n_elements]
                    ec = self.element_centers  # [n_elements, 3]
                    dx = ec[:, 0] - focus_x_m
                    dy = ec[:, 1]  # y = 0 for the virtual source
                    dz = ec[:, 2] - focus_z_m
                    distances = torch.sqrt(dx * dx + dy * dy + dz * dz)
                    delays_s = distances / self.c  # [n_elements]

                    # Normalize delays using a Python branch on a detached scalar
                    if vs[1].detach().item() <= 0:
                        # Diverging wave (virtual source behind array):
                        # earliest element fires first
                        delays = delays_s - delays_s.min()
                    else:
                        # Focusing: farthest element fires first
                        delays = delays_s.max() - delays_s

                    return delays * 1e6  # us, shape [n_elements]

                return vs_to_delays

            tf.add_parameter_mapping(
                name=f"vs_to_delays_{i}",
                function=make_vs_to_delays(vs_name),
                inputs=[vs_name],
                output="delays",
                level="element",
            )

            # Mapping: apod parameter → apodization
            def make_apod_mapping(apod_name):
                def get_apod(**kwargs):
                    return kwargs[apod_name]

                return get_apod

            tf.add_parameter_mapping(
                name=f"apod_mapping_{i}",
                function=make_apod_mapping(apod_name),
                inputs=[apod_name],
                output="apodization",
                level="element",
            )

            self.torch_fields.append(tf)
            self.virtual_sources.append(vs_name)

            print(f"  VS {i}: x={x_pos:.1f}, z={self.z_behind:.1f} mm")

    def get_combined_field(self, batch_size=2048, training=True):
        """
        Compute combined pressure field from all virtual sources.

        Returns
        -------
        Tensor [nx, ny, nz]
            Combined pressure field
        """
        # Compute field for each virtual source and sum
        pr_combined = None

        if training:
            for i, tf in enumerate(self.torch_fields):
                x, y, z, pr_i = tf(
                    self.field_points, training=training, batch_size=batch_size
                )

                if pr_combined is None:
                    pr_combined = pr_i
                else:
                    pr_combined = pr_combined + pr_i
        else:
            with torch.no_grad():
                for i, tf in enumerate(self.torch_fields):
                    x, y, z, pr_i = tf(
                        self.field_points, training=training, batch_size=batch_size
                    )

                    if pr_combined is None:
                        pr_combined = pr_i
                    else:
                        pr_combined = pr_combined + pr_i

        return x, y, z, pr_combined

    def get_total_apodization(self):
        """
        Get combined apodization from all virtual sources.

        Returns
        -------
        Tensor [n_elements]
            Total element usage
        """
        apod_total = None

        for tf in self.torch_fields:
            apod_i = tf.get_parameter("apodization")

            if apod_total is None:
                apod_total = apod_i
            else:
                apod_total = apod_total + apod_i

        return apod_total

    def get_optimizable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all optimizable parameters from all virtual sources."""
        params = []
        for tf in self.torch_fields:
            params.extend(tf.get_optimizable_parameters())
        return params

    def apply_constraints(self):
        """Apply constraints to all parameters."""
        for tf in self.torch_fields:
            tf.apply_constraints()


# ============================================================================
# Optimization Function
# ============================================================================


def optimize_virtual_sources(
    transducer,
    n_virtual_sources: int,
    field_points: dict,
    *,
    num_epochs: int = 100,
    lr: float = 0.01,
    sparsity_weight: float = 0.1,
    uniformity_weight: float = 1.0,
    coverage_weight: float = 0.5,
    energy_weight: float = 0.2,
    batch_size: int = 2048,
    use_gpu: bool = True,
    optimizer_type: str = "SGD",
):
    """
    Optimize virtual source positions and element apodization.

    Parameters
    ----------
    transducer : TransducerBase
        Transducer object
    n_virtual_sources : int
        Number of virtual sources
    field_points : dict
        Field specification
    num_epochs : int
        Training epochs
    lr : float
        Learning rate
    sparsity_weight : float
        Weight for element sparsity penalty
    uniformity_weight : float
        Weight for field uniformity
    coverage_weight : float
        Weight for field coverage
    batch_size : int
        Batch size
    use_gpu : bool
        Use GPU

    Returns
    -------
    dict
        Optimization results
    """
    print("=" * 70)
    print("Optimizing Virtual Source Positions")
    print("=" * 70)
    print(f"Number of virtual sources: {n_virtual_sources}")
    print(f"Transducer: {transducer.name}")
    print(f"Field: X={field_points['x_extent']}, Z={field_points['z_extent']}")
    print()

    # Create optimizer
    vs_opt = VirtualSourceOptimizer(
        transducer, n_virtual_sources, field_points, use_gpu=use_gpu
    )

    # Setup optimizer
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(vs_opt.get_optimizable_parameters(), lr=lr)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(vs_opt.get_optimizable_parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Loss history
    loss_history = []
    uniformity_history = []
    sparsity_history = []
    coverage_history = []
    energy_history = []

    # Training loop
    print("Computing reference field...")
    _, _, _, pr_ref = vs_opt.get_combined_field(batch_size=batch_size, training=False)
    # training=False returns a numpy array; convert to a detached torch tensor
    # so it can be used inside the differentiable loss expressions.
    pr_ref = torch.as_tensor(
        np.asarray(pr_ref), dtype=torch.float32, device=vs_opt.device
    )
    print(f"Training for {num_epochs} epochs...")
    print()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass - get combined field
        x, y, z, pr = vs_opt.get_combined_field(batch_size=batch_size, training=True)

        # Normalize
        pr_norm = pr / pr.max()

        # Get total apodization
        apod_total = vs_opt.get_total_apodization() / n_virtual_sources  # Normalize

        # Compute losses
        loss_energy = compute_transmit_energy(
            apod_total, pr, pr_ref
        )  # Maximize mean energy
        loss_uniformity = compute_uniformity_loss(pr_norm)
        loss_sparsity = compute_element_usage_penalty(apod_total, sparsity_weight)
        loss_coverage = compute_coverage_loss(pr_norm, threshold=0.3)

        # Combined loss
        loss = (
            uniformity_weight * loss_uniformity
            + sparsity_weight * loss_sparsity
            + coverage_weight * loss_coverage
            # + energy_weight * loss_energy
        )

        # Backward
        loss.backward()
        optimizer.step()
        vs_opt.apply_constraints()

        # Store history
        loss_history.append(loss.item())
        energy_history.append(-loss_energy.item())  # Negate for plotting
        uniformity_history.append(loss_uniformity.item())
        sparsity_history.append(loss_sparsity.item())
        coverage_history.append(-loss_coverage.item())  # Negate for plotting

        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch:3d}: "
                f"Loss={loss.item():.4f} "
                f"(Unif={loss_uniformity.item():.4f}, "
                f"Sparse={loss_sparsity.item():.4f}, "
                f"Cover={loss_coverage.item():.4f}, "
                f"Energy={loss_energy.item():.4f})"
            )

    print()
    print("=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    # Get final results
    with torch.no_grad():
        x, y, z, pr_final = vs_opt.get_combined_field(batch_size=batch_size)
        apod_final = vs_opt.get_total_apodization().cpu().numpy()

    # Get virtual source positions
    vs_positions = []
    for i, tf in enumerate(vs_opt.torch_fields):
        vs_pos = tf.get_parameter(f"vs_{i}").detach().cpu().numpy()
        vs_positions.append(vs_pos)
        print(f"VS {i}: position = [x={vs_pos[0]:6.2f}, z={vs_pos[1]:6.2f}] mm")

    print()
    print(f"Active elements (>0.1): {(apod_final > 0.1).sum()} / {len(apod_final)}")
    print(f"Element usage (mean apod): {apod_final.mean():.3f}")

    # Results
    results = {
        "virtual_source_positions": np.array(vs_positions),
        "apodization_total": apod_final,
        "loss_history": loss_history,
        "uniformity_history": uniformity_history,
        "sparsity_history": sparsity_history,
        "coverage_history": coverage_history,
        "energy_history": energy_history,
        "x": x.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "z": z.detach().cpu().numpy(),
        "pressure_final": pr_final.detach().cpu().numpy(),
        "n_virtual_sources": n_virtual_sources,
    }

    return results


# ============================================================================
# Visualization
# ============================================================================


def plot_virtual_source_results(results, output_file=None):
    """Plot virtual source optimization results."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3)

    # Loss history
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results["loss_history"], label="Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss")
    ax1.legend()
    ax1.grid(True)

    # Individual loss components
    def _norm_btwn_0_and_1(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(_norm_btwn_0_and_1(results["uniformity_history"]), label="Uniformity")
    ax2.plot(_norm_btwn_0_and_1(results["sparsity_history"]), label="Sparsity")
    ax2.plot(_norm_btwn_0_and_1(results["coverage_history"]), label="Coverage")
    ax2.plot(_norm_btwn_0_and_1(results["energy_history"]), label="Energy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Component")
    ax2.set_title("Loss Components")
    ax2.legend()
    ax2.grid(True)

    # Total apodization
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(results["apodization_total"], "o-")
    ax3.axhline(0.1, color="r", linestyle="--", label="Threshold")
    ax3.set_xlabel("Element Index")
    ax3.set_ylabel("Total Apodization")
    ax3.set_title("Element Usage (Combined)")
    ax3.legend()
    ax3.grid(True)

    # Virtual source positions
    ax4 = fig.add_subplot(gs[1, 0])
    vs_pos = results["virtual_source_positions"]
    ax4.scatter(vs_pos[:, 0], vs_pos[:, 1], s=100, c="red", marker="x")
    for i, pos in enumerate(vs_pos):
        ax4.annotate(
            f"VS{i}", (pos[0], pos[1]), xytext=(5, 5), textcoords="offset points"
        )
    ax4.axhline(0, color="k", linestyle="-", linewidth=2, label="Array")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Z (mm)")
    ax4.set_title("Virtual Source Positions")
    ax4.legend()
    ax4.grid(True)
    ax4.axis("equal")

    # Pressure field (XZ plane)
    ax5 = fig.add_subplot(gs[1, 1])
    pr = results["pressure_final"]
    x, y, z = results["x"], results["y"], results["z"]

    y_center = len(y) // 2
    pr_xz = pr[:, y_center, :]
    pr_db = to_dB(pr_xz)

    extent = [x.min(), x.max(), z.min(), z.max()]
    im5 = ax5.imshow(
        pr_db.T,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="hot",
        vmin=-40,
        vmax=0,
    )
    ax5.set_xlabel("Z (mm)")
    ax5.set_ylabel("X (mm)")
    ax5.set_title("Pressure Field (XZ plane, dB)")
    plt.colorbar(im5, ax=ax5, label="dB")

    # Normalized pressure field
    ax6 = fig.add_subplot(gs[1, 2])
    pr_norm = pr_xz / pr_xz.max()
    im6 = ax6.imshow(
        pr_norm.T,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="hot",
        vmin=0,
        vmax=1,
    )
    ax6.set_xlabel("Z (mm)")
    ax6.set_ylabel("X (mm)")
    ax6.set_title("Normalized Pressure Field")
    plt.colorbar(im6, ax=ax6)

    # Lateral profile at different depths
    ax7 = fig.add_subplot(gs[2, :])
    n_depths = 5
    z_indices = np.linspace(0, len(z) - 1, n_depths, dtype=int)

    for idx in z_indices:
        lateral_profile = pr_norm[:, idx]
        ax7.plot(x, lateral_profile, label=f"z={z[idx]:.1f}mm")

    ax7.set_xlabel("X (mm)")
    ax7.set_ylabel("Normalized Pressure")
    ax7.set_title("Lateral Profiles at Different Depths")
    ax7.legend()
    ax7.grid(True)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        print(f"Figure saved to: {output_file}")
    plt.show()


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("\nOptimizing Virtual Sources for Plane Wave Imaging")
    print("=" * 70)

    # Optimizer
    data_folder = r"results/domino/"
    optimizer_type = "SGD"  # Options: "SGD", "Adam"
    energies = "loss_sparse_uniform_coverage"  # Options: "uniform", "sparse", "coverage", "energy"
    num_epochs = 300
    lr = 1e-3

    # Create transducer
    tx = Domino()

    # Field specification (imaging region)
    field_points = {
        "x_extent": [-10, 10],  # mm, lateral extent
        "y_extent": [0, 0],  # mm, thin slice
        "z_extent": [0, 15],  # mm, depth range
        "dx": 0.25,
        "dy": 1.0,
        "dz": 0.25,
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
            energy_weight=0.01,
            batch_size=2048,
            use_gpu=True,
            optimizer_type=optimizer_type,
        )

        # Plot results
        plot_virtual_source_results(results)

        # Save results
        output_file = (
            f"vsource_{n_vs}vs_{lr}_nepoch{num_epochs}_optim{optimizer_type}.npz"
        )
        np.savez(output_file, **results)
        print(f"\nResults saved to: {output_file}")
