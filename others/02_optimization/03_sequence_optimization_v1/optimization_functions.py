from typing import List

import numpy as np
import torch

# ===================================================================
# Helper functions for optimization
# ===================================================================
import torch.nn.functional as F
from loss_functions import (
    compute_aperture_cost,
    compute_lateral_uniformity_loss,
    compute_log_coverage_loss,
    compute_mean_energy_loss,
    compute_soft_coverage_loss,
)

# v1 imports (kept for reference)
# from loss_functions import (
#     compute_coverage_loss,
#     compute_element_usage_penalty,
#     compute_transmit_energy,
#     compute_uniformity_loss,
# )
from pyfield.future.TorchField_flexible import TorchFieldFlexible


def gaussian_kernel1d(sigma, truncate=4.0, device="cpu"):
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_filter_pytorch(pr, sigma_points):
    device = pr.device

    # Build 1D kernels
    k = gaussian_kernel1d(sigma_points, device=device)
    kx = k.view(1, 1, -1, 1)  # vertical
    ky = k.view(1, 1, 1, -1)  # horizontal

    # Ensure 4D tensor: [B, C, H, W]
    if pr.dim() == 2:
        pr = pr.unsqueeze(0).unsqueeze(0)
    elif pr.dim() == 3:
        # This means [x, y, z], but we have y = 1. We need [x, y, z] → [B, C, H, W]
        # with B=1, C=1, H=z, W=x. So we permute and unsqueeze.
        pr = pr.permute(1, 0, 2).unsqueeze(0)  # [1, 1, z, x]

    # Apply separable convolution
    pr = F.conv2d(pr, kx, padding=(kx.shape[2] // 2, 0))
    pr = F.conv2d(pr, ky, padding=(0, ky.shape[3] // 2))
    return pr.squeeze().unsqueeze(1)  # [x, y, z] → [x, 1, z]


# ============================================================================
# Virtual Source Optimizer
# ============================================================================


class VirtualSourceOptimizer:
    """
    Optimize virtual source positions for diverging wave imaging.

    Apodization is *derived* from VS position via F/D=1 rule (clinical
    standard), not freely optimized.  Only VS positions [x, z] are learnable.

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
    c : float
        Speed of sound in m/s
    x_init_mm : list / array / float
        Initial lateral position of VS (mm)
    z_init_mm : list / array / float
        Initial depth of VS (mm, negative = behind array)
    fs : float
        Sampling frequency for SIR simulation (Hz). Higher = more accurate
    """

    def __init__(
        self,
        transducer,
        n_virtual_sources: int,
        field_points: dict,
        use_gpu: bool = True,
        c: float = 1540.0,
        x_init_mm=None,
        z_init_mm=None,
        fs: float = 100e6,
    ):
        self.transducer = transducer
        self.n_vs = n_virtual_sources
        self.field_points = field_points
        self.use_gpu = use_gpu
        self.c = c  # Speed of sound in m/s
        self.fs = fs  # Sampling frequency for SIR simulation (Hz)

        if x_init_mm is None:
            x_init_mm = [0.0] * n_virtual_sources
        else:
            # chek if its list or ndarray with length n_virtual_sources
            if isinstance(x_init_mm, (list, np.ndarray)):
                if len(x_init_mm) != n_virtual_sources:
                    raise ValueError(
                        f"x_init_mm length must match n_virtual_sources ({n_virtual_sources})"
                    )
        if z_init_mm is None:
            z_init_mm = [-10.0] * n_virtual_sources
        else:
            if isinstance(z_init_mm, (list, np.ndarray)):
                if len(z_init_mm) != n_virtual_sources:
                    raise ValueError(
                        f"z_init_mm length must match n_virtual_sources ({n_virtual_sources})"
                    )

        self.x_init_mm = x_init_mm
        self.z_init_mm = z_init_mm

        # Device
        self.device = torch.device(
            "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Transducer geometry constants (metres, on device)
        self.element_centers = torch.tensor(
            self.transducer.element_centers, dtype=torch.float32, device=self.device
        )  # [n_elements, 3]
        self.pitch = self.transducer.pitch  # metres

        # Create TorchField for each virtual source
        self.torch_fields = []
        self.virtual_sources = []

        self._setup_virtual_sources()

    # ------------------------------------------------------------------
    # v2: derived apodization from F/D = 1
    # ------------------------------------------------------------------
    def _setup_virtual_sources(self):
        """Initialize virtual sources. Apodization derived from VS via F/D=1."""
        print(f"Setting up {self.n_vs} virtual sources (F/D=1 apodization)...")

        for i in range(self.n_vs):
            x_pos = self.x_init_mm[i]
            z_pos = self.z_init_mm[i]

            tf = TorchFieldFlexible(
                self.transducer, use_gpu=self.use_gpu, verbose=False, fs=self.fs
            )

            # --- Only learnable parameter: VS position [x_mm, z_mm] ---
            vs_name = f"vs_{i}"
            tf.add_optimizable_parameter(
                vs_name,
                initial_value=[x_pos, z_pos],
                level="global",
                requires_grad=True,
            )

            # --- Mapping: VS → delays (same as v1, works fine) ---
            def make_vs_to_delays(vs_name):
                def vs_to_delays(**kwargs):
                    vs = kwargs[vs_name]
                    focus_x_m = vs[0] * 1e-3
                    focus_z_m = torch.abs(vs[1]) * 1e-3

                    ec = self.element_centers
                    dx = ec[:, 0] - focus_x_m
                    dy = ec[:, 1]
                    dz = ec[:, 2] - focus_z_m
                    distances = torch.sqrt(dx * dx + dy * dy + dz * dz)
                    delays_s = distances / self.c

                    if vs[1].detach().item() <= 0:
                        delays = delays_s - delays_s.min()
                    else:
                        delays = delays_s.max() - delays_s

                    return delays * 1e6  # µs

                return vs_to_delays

            tf.add_parameter_mapping(
                name=f"vs_to_delays_{i}",
                function=make_vs_to_delays(vs_name),
                inputs=[vs_name],
                output="delays",
                level="element",
            )

            # --- Mapping: VS → apodization (F/D=1, differentiable) ---
            # Smooth rect window: sigmoid transition over ~1 pitch
            def make_vs_to_apodization(vs_name):
                def vs_to_apod(**kwargs):
                    vs = kwargs[vs_name]  # [x_mm, z_mm]
                    x_vs_m = vs[0] * 1e-3
                    z_vs_m = torch.abs(vs[1]) * 1e-3

                    # F/D = 1 → aperture diameter D = |z_vs|
                    half_D = z_vs_m / 2.0

                    # Lateral distance of each element to VS
                    dx = torch.abs(self.element_centers[:, 0] - x_vs_m)

                    # Sigmoid-smoothed rect: 1 inside aperture, 0 outside
                    # Steepness = 1/pitch → transition over ~1 element
                    steepness = 10 / self.pitch
                    apod = torch.sigmoid(steepness * (half_D - dx))

                    # Rectangular window with hard cutoff (non-differentiable, but
                    # simpler)

                    return apod  # [n_elements]

                return vs_to_apod

            tf.add_parameter_mapping(
                name=f"vs_to_apod_{i}",
                function=make_vs_to_apodization(vs_name),
                inputs=[vs_name],
                output="apodization",
                level="element",
            )

            self.torch_fields.append(tf)
            self.virtual_sources.append(vs_name)

            print(f"  VS {i}: x={x_pos:.1f}, z={z_pos:.1f} mm")

    # ------------------------------------------------------------------
    # v1: free apodization (commented out for reference)
    # ------------------------------------------------------------------
    # def _setup_virtual_sources_v1(self):
    #     """v1: Free apodization per VS. Too many DOF, optimizer struggles."""
    #     print(f"Setting up {self.n_vs} virtual sources (free apodization)...")
    #
    #     self.element_centers = torch.tensor(
    #         self.transducer.element_centers, dtype=torch.float32, device=self.device
    #     )
    #
    #     for i in range(self.n_vs):
    #         x_pos = (i - self.n_vs // 2) * self.x_spacing
    #
    #         tf = TorchFieldFlexible(
    #             self.transducer, use_gpu=self.use_gpu, verbose=False
    #         )
    #
    #         vs_name = f"vs_{i}"
    #         tf.add_optimizable_parameter(
    #             vs_name,
    #             initial_value=[x_pos, self.z_behind],
    #             level="global",
    #             requires_grad=True,
    #         )
    #
    #         # FREE apodization — 128 extra learnable params per VS
    #         apod_name = f"apod_{i}"
    #         initial_apod = np.ones(self.transducer.n_elements) * 0.5
    #         tf.add_optimizable_parameter(
    #             apod_name,
    #             initial_value=initial_apod,
    #             level="element",
    #             requires_grad=True,
    #             constraints={"min": 0.0, "max": 1.0},
    #         )
    #
    #         # VS → delays mapping (same as v2)
    #         def make_vs_to_delays(vs_name):
    #             def vs_to_delays(**kwargs):
    #                 vs = kwargs[vs_name]
    #                 focus_x_m = vs[0] * 1e-3
    #                 focus_z_m = torch.abs(vs[1]) * 1e-3
    #                 ec = self.element_centers
    #                 dx = ec[:, 0] - focus_x_m
    #                 dy = ec[:, 1]
    #                 dz = ec[:, 2] - focus_z_m
    #                 distances = torch.sqrt(dx * dx + dy * dy + dz * dz)
    #                 delays_s = distances / self.c
    #                 if vs[1].detach().item() <= 0:
    #                     delays = delays_s - delays_s.min()
    #                 else:
    #                     delays = delays_s.max() - delays_s
    #                 return delays * 1e6
    #             return vs_to_delays
    #
    #         tf.add_parameter_mapping(
    #             name=f"vs_to_delays_{i}",
    #             function=make_vs_to_delays(vs_name),
    #             inputs=[vs_name],
    #             output="delays",
    #             level="element",
    #         )
    #
    #         # FREE apod mapping — just passes through the learnable param
    #         def make_apod_mapping(apod_name):
    #             def get_apod(**kwargs):
    #                 return kwargs[apod_name]
    #             return get_apod
    #
    #         tf.add_parameter_mapping(
    #             name=f"apod_mapping_{i}",
    #             function=make_apod_mapping(apod_name),
    #             inputs=[apod_name],
    #             output="apodization",
    #             level="element",
    #         )
    #
    #         self.torch_fields.append(tf)
    #         self.virtual_sources.append(vs_name)
    #         print(f"  VS {i}: x={x_pos:.1f}, z={self.z_behind:.1f} mm")

    def get_combined_field(self, batch_size=2048, training=True):
        """
        Compute combined pressure field from all virtual sources.

        Returns
        -------
        Tensor [nx, ny, nz]
            Combined pressure field
        """
        pr_combined = None

        if training:
            for i, tf in enumerate(self.torch_fields):
                x, y, z, pr_i = tf(
                    self.field_points, training=training, batch_size=batch_size
                )

                pr_i = gaussian_filter_pytorch(pr_i, sigma_points=2)

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

                    pr_i = gaussian_filter_pytorch(pr_i, sigma_points=2)

                    if pr_combined is None:
                        pr_combined = pr_i
                    else:
                        pr_combined = pr_combined + pr_i

        return x, y, z, pr_combined

    def get_per_vs_apodization(self):
        """
        Get apodization for each VS (derived from F/D=1).

        Returns
        -------
        list of Tensor [n_elements]
            Per-VS apodization vectors
        """
        apod_list = []
        for tf in self.torch_fields:
            apod_i = tf.get_parameter("apodization")
            apod_list.append(apod_i)
        return apod_list

    def get_total_apodization(self):
        """
        Get combined apodization from all virtual sources.

        Returns
        -------
        Tensor [n_elements]
            Total element usage (sum across VS)
        """
        apod_list = self.get_per_vs_apodization()
        return sum(apod_list)

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
    uniformity_weight: float = 1.0,
    coverage_weight: float = 0.5,
    aperture_weight: float = 0.3,
    energy_weight: float = 0.1,
    sparsity_weight: float = 0.1,
    batch_size: int = 2048,
    use_gpu: bool = True,
    optimizer_type: str = "Adam",
    x_init_mm=None,
    z_init_mm=None,
):
    """
    Optimize virtual source positions for diverging wave imaging.

    v2: Apodization derived from VS position (F/D=1). Only VS [x,z]
    positions are optimized. Loss uses lateral uniformity + soft coverage
    + aperture cost + mean energy.

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
    uniformity_weight : float
        Weight for lateral uniformity loss
    coverage_weight : float
        Weight for soft coverage loss
    aperture_weight : float
        Weight for total active element cost
    energy_weight : float
        Weight for mean field energy (prevents collapse to zero)
    batch_size : int
        Batch size for TorchField forward pass
    use_gpu : bool
        Use GPU
    optimizer_type : str
        "SGD" or "Adam"
    x_init_mm : list / array / float
        Initial lateral position of VS (mm). If list/array, must match
        n_virtual_sources.
    z_init_mm : list / array / float
        Initial depth of VS (mm, negative = behind array). If list/array, must
        match n_virtual_sources.

    Returns
    -------
    dict
        Optimization results
    """
    print("=" * 70)
    print("Optimizing Virtual Source Positions (v2 — F/D=1 derived apod)")
    print("=" * 70)
    print(f"Virtual sources: {n_virtual_sources}")
    print(f"Transducer: {transducer.name} ({transducer.n_elements} elements)")
    print(f"Field: X={field_points['x_extent']}, Z={field_points['z_extent']}")
    print(
        f"Weights: unif={uniformity_weight}, cover={coverage_weight}, "
        f"aper={aperture_weight}, energy={energy_weight}"
    )
    print()

    # Create VS optimizer (with F/D=1 derived apodization)
    vs_opt = VirtualSourceOptimizer(
        transducer,
        n_virtual_sources,
        field_points,
        use_gpu=use_gpu,
        x_init_mm=x_init_mm,
        z_init_mm=z_init_mm,
    )

    # Setup torch optimizer
    params = vs_opt.get_optimizable_parameters()
    print(
        f"Optimizable parameters: {sum(p.numel() for p in params)} "
        f"({len(params)} tensors — only VS positions)"
    )

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # History tracking
    loss_history = []
    uniformity_history = []
    coverage_history = []
    aperture_history = []
    energy_history = []
    vs_positions_history = np.zeros((num_epochs, n_virtual_sources, 2))

    print(f"Training for {num_epochs} epochs...")
    print()

    # ================================================================
    # v3 Training loop — log-coverage (gradient everywhere)
    # ================================================================
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward: compound field from all VS
        x, y, z, pr = vs_opt.get_combined_field(batch_size=batch_size, training=True)

        # Normalize compound field
        pr_norm = pr / (pr.max() + 1e-8)

        # Per-VS apodization (derived from F/D=1)
        apod_list = vs_opt.get_per_vs_apodization()

        # --- v3 losses ---
        loss_uniform = compute_lateral_uniformity_loss(pr_norm)
        # loss_cover = compute_log_coverage_loss(pr_norm, eps=1e-3)
        loss_cover = compute_soft_coverage_loss(pr_norm, threshold=-10, steepness=1)
        loss_aperture = compute_aperture_cost(apod_list)
        loss_energy = compute_mean_energy_loss(pr_norm)

        # Combined loss
        loss = (
            uniformity_weight * loss_uniform
            + coverage_weight * loss_cover
            + aperture_weight * loss_aperture
            + energy_weight * loss_energy
        )

        # Backward + step
        loss.backward()

        # Gradient clipping — SIR simulation produces oscillatory gradients
        # torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)

        optimizer.step()
        vs_opt.apply_constraints()

        # --- History ---
        loss_history.append(loss.item())
        uniformity_history.append(loss_uniform.item())
        coverage_history.append(loss_cover.item())
        aperture_history.append(loss_aperture.item())
        energy_history.append(loss_energy.item())

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
                f"(Unif={loss_uniform.item():.4f}, "
                f"Cover={loss_cover.item():.4f}, "
                f"Aper={loss_aperture.item():.4f}, "
                f"Energy={loss_energy.item():.4f})"
            )

    # ================================================================
    # v2 Training loop (sigmoid coverage — commented out)
    # ================================================================
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     x, y, z, pr = vs_opt.get_combined_field(batch_size=batch_size, training=True)
    #     pr_norm = pr / (pr.max() + 1e-8)
    #     apod_list = vs_opt.get_per_vs_apodization()
    #     loss_uniform = compute_lateral_uniformity_loss(pr_norm)
    #     loss_cover = compute_soft_coverage_loss(pr_norm, threshold=-6, steepness=20.0)
    #     loss_aperture = compute_aperture_cost(apod_list)
    #     loss_energy = compute_mean_energy_loss(pr_norm)
    #     loss = (uniformity_weight * loss_uniform + coverage_weight * loss_cover
    #             + aperture_weight * loss_aperture + energy_weight * loss_energy)
    #     loss.backward()
    #     optimizer.step()
    #     vs_opt.apply_constraints()

    # ================================================================
    # v1 Training loop (commented out for reference)
    # ================================================================
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     x, y, z, pr = vs_opt.get_combined_field(batch_size=batch_size, training=True)
    #     pr_norm = pr / pr.max()
    #     apod_total = vs_opt.get_total_apodization() / n_virtual_sources
    #
    #     loss_energy = compute_transmit_energy(apod_total, pr)
    #     loss_uniformity = compute_uniformity_loss(pr_norm)
    #     loss_sparsity = compute_element_usage_penalty(apod_total, sparsity_weight)
    #     loss_coverage = compute_coverage_loss(pr_norm, threshold=0.3)
    #
    #     loss = (
    #         uniformity_weight * loss_uniformity
    #         + sparsity_weight * loss_sparsity
    #         + coverage_weight * loss_coverage
    #         # + energy_weight * loss_energy
    #     )
    #
    #     loss.backward()
    #     optimizer.step()
    #     vs_opt.apply_constraints()

    print()
    print("=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    # Final evaluation
    with torch.no_grad():
        x, y, z, pr_final = vs_opt.get_combined_field(batch_size=batch_size)
        apod_final = vs_opt.get_total_apodization().cpu().numpy()

    # Print final VS positions
    vs_positions = []
    for i, tf in enumerate(vs_opt.torch_fields):
        vs_pos = tf.get_parameter(f"vs_{i}").detach().cpu().numpy()
        vs_positions.append(vs_pos)
        print(f"VS {i}: position = [x={vs_pos[0]:6.2f}, z={vs_pos[1]:6.2f}] mm")

    print()
    print(f"Active elements (>0.1): {(apod_final > 0.1).sum()} / {len(apod_final)}")
    print(f"Element usage (mean apod): {apod_final.mean():.3f}")

    # Results dict (compatible with plotting_functions)
    results = {
        "virtual_source_positions": np.array(vs_positions),
        "virtual_source_positions_history": vs_positions_history,
        "apodization_total": apod_final / n_virtual_sources,  # normalize for plotting
        "loss_history": loss_history,
        "uniformity_history": uniformity_history,
        "sparsity_history": aperture_history,  # reuse key for plotting compat
        "coverage_history": coverage_history,
        "energy_history": energy_history,
        "x": x.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "z": z.detach().cpu().numpy(),
        "pressure_final": pr_final.detach().cpu().numpy(),
        "n_virtual_sources": n_virtual_sources,
    }
    return results
