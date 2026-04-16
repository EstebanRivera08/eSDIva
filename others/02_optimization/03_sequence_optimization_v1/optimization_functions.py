from typing import List

import numpy as np
import torch
from loss_functions import (
    compute_coverage_loss,
    compute_element_usage_penalty,
    compute_transmit_energy,
    compute_uniformity_loss,
)

from pyfield.future.TorchField_flexible import TorchFieldFlexible

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
        x_spacing: float = 0,
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
                constraints={"min": 0.0, "max": 1.0},
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
    vs_positions_history = np.zeros((num_epochs, n_virtual_sources, 2))  #

    # Training loop
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
        loss_energy = compute_transmit_energy(apod_total, pr)  # Maximize mean energy
        loss_uniformity = compute_uniformity_loss(pr_norm)
        loss_sparsity = compute_element_usage_penalty(apod_total, sparsity_weight)
        loss_coverage = compute_coverage_loss(pr_norm, threshold=0.3)

        # Combined loss
        loss = (
            uniformity_weight * loss_uniformity
            + sparsity_weight * loss_sparsity
            + coverage_weight * loss_coverage
            + energy_weight * loss_energy
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
        # Store virtual source positions
        vs_positions = []
        for i in range(n_virtual_sources):
            vs_pos = (
                vs_opt.torch_fields[i].get_parameter(f"vs_{i}").detach().cpu().numpy()
            )
            vs_positions_history[epoch, i] = vs_pos

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
        "virtual_source_positions_history": vs_positions_history,
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
