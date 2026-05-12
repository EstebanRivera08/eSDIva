from typing import List

import numpy as np
import torch

# ===================================================================
# Helper functions for optimization
# ===================================================================
import torch.nn.functional as F
from loss_functions import (
    compute_aperture_cost,
    compute_mean_energy_loss,
    compute_resolution_loss,
    compute_soft_coverage_loss,
    compute_symmetry_loss,
)
from tqdm import tqdm

# v1 imports (kept for reference)
# from loss_functions import (
#     compute_coverage_loss,
#     compute_element_usage_penalty,
#     compute_transmit_energy,
#     compute_uniformity_loss,
# )
from pyfield.cache.TorchField_flexible import TorchFieldFlexible


def gaussian_kernel1d(sigma, truncate=4.0, device="cpu"):
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_filter_pytorch(pr, sigma_points):
    device = pr.device

    # Build 1D kernels
    # when pr is complex, we apply the filter to the magnitude (abs) and keep the phase
    # unchanged.

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
        FD_init=2.0,
        n_gauss_init=4.0,
        fs: float = 100e6,
        use_phase: bool = False,
    ):
        self.transducer = transducer
        self.n_vs = n_virtual_sources
        self.field_points = field_points
        self.use_gpu = use_gpu
        self.c = c  # Speed of sound in m/s
        self.fs = fs  # Sampling frequency for SIR simulation (Hz)
        self.use_phase = use_phase

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
        self.FD_init = FD_init

        # Device
        self.device = torch.device(
            "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Transducer geometry constants (metres, on device)
        self.element_centers = torch.tensor(
            self.transducer.element_centers, dtype=torch.float32, device=self.device
        )  # [n_elements, 3]
        self.pitch = self.transducer.pitch  # metres

        self.n_steepness_init = n_gauss_init
        # super-Gaussian order: n=1 → Gaussian, n→∞ → rect

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

        self.vs_pos_names = []
        self.vs_apod_names = []

        for i in range(self.n_vs):
            x_pos = self.x_init_mm[i]
            z_pos = abs(self.z_init_mm[i])

            FD = self.FD_init

            tf = TorchFieldFlexible(
                self.transducer, use_gpu=self.use_gpu, verbose=False, fs=self.fs
            )

            # --- Learnable params: split into position and apodization ---
            pos_name = f"vs_pos_{i}"
            apod_name = f"vs_apod_{i}"

            tf.add_optimizable_parameter(
                pos_name,
                initial_value=[x_pos, z_pos],
                level="global",
                requires_grad=True,
            )

            tf.add_optimizable_parameter(
                apod_name,
                initial_value=[self.n_steepness_init, FD],
                level="global",
                requires_grad=True,
                constraints={"min": [1.0, 0.1], "max": [None, 5.0]},
            )

            # --- Mapping: VS position → delays ---
            def make_vs_to_delays(pos_name):
                def vs_to_delays(**kwargs):
                    pos = kwargs[pos_name]  # [x_mm, z_mm]
                    focus_x_m = pos[0] * 1e-3
                    focus_z_m = torch.abs(pos[1]) * 1e-3

                    ec = self.element_centers
                    dx = ec[:, 0] - focus_x_m
                    dy = ec[:, 1]
                    dz = ec[:, 2] - focus_z_m
                    distances = torch.sqrt(dx * dx + dy * dy + dz * dz)
                    delays_s = distances / self.c

                    delays = delays_s - delays_s.min()

                    return delays * 1e6  # µs

                return vs_to_delays

            tf.add_parameter_mapping(
                name=f"vs_to_delays_{i}",
                function=make_vs_to_delays(pos_name),
                inputs=[pos_name],
                output="delays",
                level="element",
            )

            # --- Mapping: VS position + apod params → apodization ---
            # Super-Gaussian: n=1 → Gaussian, n→∞ → rect.
            def make_vs_to_apodization(pos_name, apod_name):
                def vs_to_apod(**kwargs):
                    pos = kwargs[pos_name]  # [x_mm, z_mm]
                    apod_p = kwargs[apod_name]  # [n, FD]
                    x_vs_m = pos[0] * 1e-3
                    z_vs_m = torch.abs(pos[1]) * 1e-3
                    n = apod_p[0]  # super-Gaussian order
                    FD = apod_p[1]  # F/D ratio
                    half_aperture_m = z_vs_m / FD / 2

                    # Lateral distance of each element to VS
                    dx = torch.abs(self.element_centers[:, 0] - x_vs_m)

                    # Normalized distance, clamped to avoid 0^n gradient NaN
                    r = (dx / (half_aperture_m + 1e-8)).clamp(min=1e-6)

                    # Super-Gaussian: apod = exp(-0.5 * r^(2n))
                    apod = torch.exp(-0.5 * r.pow(2 * n))

                    return apod  # [n_elements], naturally in [0, 1]

                return vs_to_apod

            tf.add_parameter_mapping(
                name=f"vs_to_apod_{i}",
                function=make_vs_to_apodization(pos_name, apod_name),
                inputs=[pos_name, apod_name],
                output="apodization",
                level="element",
            )

            self.torch_fields.append(tf)
            self.vs_pos_names.append(pos_name)
            self.vs_apod_names.append(apod_name)

            print(
                f"  VS {i}: x={x_pos:.1f}, z={z_pos:.1f} mm, "
                f"n={self.n_steepness_init:.1f}, FD={FD:.1f}"
            )

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

    def get_combined_field(self, batch_size=2048, training=True, sigma_points=1):
        """
        Compute combined pressure field from all virtual sources.

        When use_phase=True, also computes the Coherence Factor (CF):
            CF = mean(|Σ Pᵢ|²) / mean(Σ |Pᵢ|²)

        CF measures angular diversity via interference:
            - CF = N → all VS identical (no diversity, bad resolution)
            - CF → 1 → VS fully decorrelated (max diversity, good resolution)

        No phase unwrapping needed — CF operates on magnitudes only.

        Returns
        -------
        x, y, z : Tensor
            Grid coordinates
        pr_combined : Tensor [nx, ny, nz]
            Combined pressure magnitude field
        coherence_factor : Tensor (scalar) or None
            CF value when use_phase=True, None otherwise
        """
        pr_combined = None

        if training:
            for i, tf in enumerate(self.torch_fields):
                x, y, z, pr_i = tf(
                    self.field_points, training=training, batch_size=batch_size
                )

                if self.use_phase:
                    # Accumulate incoherent power: Σ |Pᵢ|²
                    # Coherent sum (complex)
                    if pr_combined is None:
                        pr_combined = pr_i
                    else:
                        pr_combined = pr_combined + pr_i
                else:
                    pr_i = pr_i.abs()
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
                    pr_i = torch.tensor(pr_i, device=self.device)

                    if self.use_phase:
                        if pr_combined is None:
                            pr_combined = pr_i
                        else:
                            pr_combined = pr_combined + pr_i
                    else:
                        pr_i = pr_i.abs()
                        if pr_combined is None:
                            pr_combined = pr_i
                        else:
                            pr_combined = pr_combined + pr_i

        # Final output: magnitude + smoothing for coverage/energy metrics
        pr_combined = gaussian_filter_pytorch(
            pr_combined.abs(), sigma_points=sigma_points
        )
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

    def get_param_groups(self, lr_pos: float, lr_apod: float) -> list:
        """Get parameter groups with separate learning rates.

        Returns list of dicts for torch optimizer param groups:
          [{"params": [pos tensors], "lr": lr_pos},
           {"params": [apod tensors], "lr": lr_apod}]
        """
        pos_params = []
        apod_params = []
        for i, tf in enumerate(self.torch_fields):
            pos_params.append(tf._optimizable_params[self.vs_pos_names[i]].value)
            apod_params.append(tf._optimizable_params[self.vs_apod_names[i]].value)
        return [
            {"params": pos_params, "lr": lr_pos},
            {"params": apod_params, "lr": lr_apod},
        ]

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
    lr_apod: float = 0.01,
    symmetry_weight: float = 1.0,
    coverage_weight: float = 0.5,
    aperture_weight: float = 0.3,
    energy_weight: float = 0.1,
    resolution_weight: float = 1.0,
    target_fnumber: float = 1.5,
    batch_size: int = 2048,
    use_gpu: bool = True,
    optimizer_type: str = "Adam",
    x_init_mm=None,
    z_init_mm=None,
    FD_init=1,
    n_gauss_init=4,
    fs: float = 100e6,
    use_phase: bool = False,
    coverage_threshold_db: float = -15.0,
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
    symmetry_weight : float
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
    fs : float
        Sampling frequency for SIR simulation (Hz). Higher = more accurate but slower.

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
        f"Weights: unif={symmetry_weight}, cover={coverage_weight}, "
        f"aper={aperture_weight}, energy={energy_weight}, "
        f"resol={resolution_weight} "
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
        FD_init=FD_init,
        n_gauss_init=n_gauss_init,
        fs=fs,
        use_phase=use_phase,
    )

    # Setup torch optimizer with separate lr for position and apodization params
    if lr_apod is None:
        lr_apod = lr  # same lr for both if not specified

    param_groups = vs_opt.get_param_groups(lr_pos=lr, lr_apod=lr_apod)
    all_params = [p for g in param_groups for p in g["params"]]
    print(
        f"Optimizable parameters: {sum(p.numel() for p in all_params)} "
        f"({len(all_params)} tensors, lr_pos={lr}, lr_apod={lr_apod})"
    )

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(param_groups)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(param_groups)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Resolution loss constants (geometry-only, computed once)
    lambda_m = vs_opt.c / transducer.fc  # wavelength [m]
    D_physical_m = transducer.n_elements * transducer.pitch  # aperture [m]
    z_field_range = field_points["z_extent"]  # [z_min_mm, z_max_mm]

    # History tracking
    loss_history = []
    uniformity_history = []
    coverage_history = []
    aperture_history = []
    energy_history = []
    resolution_history = []
    vs_positions_history = np.zeros((num_epochs, n_virtual_sources, 4))

    # Get the max_pr now and keep it fixed during training to prevent oscillatory
    # gradients from SIR simulation. We can compute it from the initial combined field.
    with torch.no_grad():
        x, y, z, pr_init = vs_opt.get_combined_field(
            batch_size=batch_size, training=False
        )
        max_pr = pr_init.max().item()
        mean_init_logpr = torch.log(pr_init + 1e-20).mean().item()

        print(f"Initial max pressure: {max_pr:.4f}")
    print(f"Training for {num_epochs} epochs...")
    print()

    # ================================================================
    # v3 Training loop — log-coverage (gradient everywhere)
    # ================================================================
    # Use tqdm for progress bar (optional)
    pbar = tqdm(range(num_epochs), desc="Optimizing VS", unit="epoch")
    for epoch in pbar:
        optimizer.zero_grad()

        # Forward: compound field from all VS
        x, y, z, pr = vs_opt.get_combined_field(batch_size=batch_size, training=True)

        # Per-VS apodization (derived from F/D=1)
        apod_list = vs_opt.get_per_vs_apodization()

        # --- v3 losses ---
        loss_symm = compute_symmetry_loss(pr, pr_max=max_pr)
        loss_cover = compute_soft_coverage_loss(
            pr,
            pr_max=max_pr,
            threshold=coverage_threshold_db,
        )
        loss_aperture = compute_aperture_cost(apod_list)
        loss_energy = compute_mean_energy_loss(pr)

        # --- v4 resolution losses ---
        # Geometric f-number penalty (cheap, no simulation)
        vs_pos_list = [
            vs_opt.torch_fields[i].get_parameter(vs_opt.vs_pos_names[i])
            for i in range(n_virtual_sources)
        ]
        loss_resolution = compute_resolution_loss(
            vs_pos_list,
            z_field_range,
            lambda_m,
            D_physical_m,
            target_fnumber=target_fnumber,
        )

        # Coherence factor loss (from pressure field, replaces angular diversity)
        # CF ∈ [1, N]. Minimize → maximize phase diversity → better resolution.
        # Normalized to [0, 1] by dividing by N.

        energy_weight = aperture_weight * loss_aperture.item() / loss_energy.item()
        # coverage_weight = aperture_weight * loss_aperture.item() / loss_cover.item()

        # Combined loss
        loss = (
            symmetry_weight * loss_symm
            + coverage_weight * loss_cover
            + energy_weight * loss_energy
            + resolution_weight * loss_resolution
            + aperture_weight * loss_aperture
        )

        # Backward + step
        loss.backward()

        # Gradient clipping — SIR simulation produces oscillatory gradients
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=10)

        optimizer.step()
        vs_opt.apply_constraints()

        # --- History ---
        loss_history.append(loss.item())
        uniformity_history.append(loss_symm.item())
        coverage_history.append(loss_cover.item())
        energy_history.append(loss_energy.item())
        resolution_history.append(loss_resolution.item())
        aperture_history.append(loss_aperture.item())

        for i in range(n_virtual_sources):
            pos = (
                vs_opt.torch_fields[i]
                .get_parameter(vs_opt.vs_pos_names[i])
                .detach()
                .cpu()
                .numpy()
            )
            apod_p = (
                vs_opt.torch_fields[i]
                .get_parameter(vs_opt.vs_apod_names[i])
                .detach()
                .cpu()
                .numpy()
            )
            vs_positions_history[epoch, i, :] = np.concatenate([pos, apod_p])

        # update tqdm description with current loss
        tqdm_desc = (
            f"Loss={loss.item():.3f} | "
            f"Energy={loss_energy.item():.3f} | "
            f"Resol={loss_resolution.item():.3f} | "
            f"Cover={loss_cover.item():.3f} | "
            f"Aper={loss_aperture.item():.3f} | "
            f"Symm={loss_symm.item():.3f}"
        )
        pbar.set_description(tqdm_desc)
        # # Print progress
        # if epoch % 10 == 0 or epoch == num_epochs - 1:
        #     print(
        #         f"Epoch {epoch:3d}: "
        #         f"Loss={loss.item():.4f} "
        #         f"(Unif={loss_uniform.item():.4f}, "
        #         f"Cover={loss_cover.item():.4f}, "
        #         f"Aper={loss_aperture.item():.4f}, "
        #         f"Energy={10 ** (-1 * loss_energy.item()):.4f})"
        #     )
        #
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
    #     loss = (symmetry_weight * loss_uniform + coverage_weight * loss_cover
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
    #         symmetry_weight * loss_uniformity
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
        pos = tf.get_parameter(vs_opt.vs_pos_names[i]).detach().cpu().numpy()
        apod_p = tf.get_parameter(vs_opt.vs_apod_names[i]).detach().cpu().numpy()
        vs_all = np.concatenate([pos, apod_p])
        vs_positions.append(vs_all)
        print(
            f"VS {i}: position = [x={pos[0]:6.2f}, z={pos[1]:6.2f}] mm, "
            f"n={apod_p[0]:.2f}, FD={apod_p[1]:.2f}"
        )

    print()
    print(f"Active elements (>0.1): {(apod_final > 0.1).sum()} / {len(apod_final)}")
    print(f"Element usage (mean apod): {apod_final.mean():.3f}")

    # Results dict (compatible with plotting_functions)
    results = {
        "virtual_source_positions": np.array(vs_positions),
        "virtual_source_positions_history": vs_positions_history,
        "apodization_total": apod_final / n_virtual_sources,  # normalize for plotting
        "apodization_per_vs": [apod.detach().cpu().numpy() for apod in apod_list],
        "loss_history": loss_history,
        "symmetry_history": uniformity_history,
        "aperture_history": aperture_history,
        "coverage_history": coverage_history,
        "energy_history": energy_history,
        "resolution_history": resolution_history,
        "loss_weights": {
            "symmetry": symmetry_weight,
            "coverage": coverage_weight,
            "aperture": aperture_weight,
            "energy": energy_weight,
            "resolution": resolution_weight,
        },
        "learning_rate": lr,
        "use_phase": use_phase,
        "coverage_threshold_db": coverage_threshold_db,
        "target_fnumber": target_fnumber,
        "x": x.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "z": z.detach().cpu().numpy(),
        "pressure_final": pr_final.detach().cpu().numpy(),
        "n_virtual_sources": n_virtual_sources,
    }
    return results
