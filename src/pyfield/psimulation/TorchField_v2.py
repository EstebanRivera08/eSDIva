"""
TorchField - PyTorch-based differentiable acoustic field simulator for optimization.

This module provides a differentiable version of PyField that enables gradient-based
optimization of transducer parameters (delays, apodization, etc.) using PyTorch.

Key differences from PyField:
- Parameters can be set as trainable for optimization
- Uses PyTorch tensors instead of NumPy arrays
- Supports GPU acceleration
- Provides gradient information for optimization

Example usage:
    >>> tx = LinearArrayTransducer(n_elements=64, ...)
    >>> tf = TorchField(tx, use_gpu=True)
    >>>
    >>> # Configure what to optimize
    >>> tf.configure_optimization(
    ...     optimize_delays=True,
    ...     optimize_apodization=True
    ... )
    >>>
    >>> # Run optimization loop
    >>> optimizer = torch.optim.Adam(tf.parameters(), lr=0.01)
    >>> for epoch in range(num_epochs):
    ...     optimizer.zero_grad()
    ...     x, y, z, p = tf(field_points, training=True)
    ...     loss = compute_loss(p, target)
    ...     loss.backward()
    ...     optimizer.step()
"""

import math
import warnings
from time import time as TIME
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from pyfield.utilities.helper_functions import create_3D_spatial_grid_from_points


# ============================================================================
# Core JIT-compiled functions (optimized for speed)
# ============================================================================


@torch.jit.script
def compute_patch_events_batch(
    wx: float,
    wy: float,
    diff: Tensor,
    dist: Tensor,
    delays: Tensor,
    apodization: Tensor,
    inv_c: float,
    inv_fs: float,
) -> Tensor:
    """
    Compute acoustic events (time points and amplitudes) for rectangular patches.

    Uses the trapezoid method from Tupholme-Stepanishen formulation.
    All inputs should be in consistent units (typically micrometers/microseconds).

    Parameters
    ----------
    wx, wy : float
        Patch dimensions
    diff : Tensor [B, M, 3]
        Distance vectors from patches to field points
    dist : Tensor [B, M]
        Euclidean distances
    delays : Tensor [B, M]
        Time delays per patch
    apodization : Tensor [B, M]
        Amplitude weights per patch
    inv_c : float
        Inverse speed of sound (time_unit/space_unit)
    inv_fs : float
        Inverse sampling frequency (time_unit)

    Returns
    -------
    Tensor [B, M, 5]
        Event times (t1, t2, t3, t4) and amplitude (hmax) for each patch
    """
    # Direction cosines
    xp = diff[..., 0] / dist
    yp = diff[..., 1] / dist

    # Projected patch dimensions
    xp_abs = torch.abs(xp) * wx * inv_c
    yp_abs = torch.abs(yp) * wy * inv_c

    # Trapezoid time intervals
    Dt1 = torch.min(xp_abs, yp_abs).clamp(min=inv_fs)
    Dt2 = torch.max(xp_abs, yp_abs).clamp(min=inv_fs)

    # Patch contribution amplitude
    area = (wx * wy) / (2 * math.pi * dist)

    # Event times
    t1 = (dist * inv_c) - 0.5 * (Dt1 + Dt2) + delays
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + (Dt1 + Dt2)

    # Maximum amplitude
    hmax = area * apodization / Dt2

    return torch.stack((t1, t2, t3, t4, hmax), dim=-1)


def accumulate_d2H_interpolation(
    d2H: Tensor,
    batch_idx: Tensor,
    t_event: Tensor,
    value: Tensor,
    sign: float,
    t0: float,
    inv_dt: float,
    T: int,
) -> None:
    """Linear interpolation accumulation for derivative of SIR."""
    f_idx = (t_event - t0) * inv_dt + 1
    idx_floor = torch.floor(f_idx).long().clamp(0, T - 1)
    w_floor = 1 - (f_idx - idx_floor)

    idx_ceil = (idx_floor + 1).clamp(0, T - 1)
    w_ceil = f_idx - idx_floor

    d2H.index_put_(
        (batch_idx, idx_ceil),
        sign * value * w_ceil,
        accumulate=True,
    )
    d2H.index_put_(
        (batch_idx, idx_floor),
        sign * value * w_floor,
        accumulate=True,
    )


def accumulate_events_derivative(
    events: Tensor,
    t0_us: float,
    dt_us: float,
    T: int,
) -> Tensor:
    """
    Accumulate patch events into spatial impulse response via integration.

    Parameters
    ----------
    events : Tensor [B, M, 5]
        Patch events (t1, t2, t3, t4, hmax)
    t0_us : float
        Start time
    dt_us : float
        Time step
    T : int
        Number of time samples

    Returns
    -------
    Tensor [B, T]
        Spatial impulse response H(t) for each field point
    """
    B, M, _ = events.shape
    device = events.device

    # Extract event components
    t1 = events[..., 0].reshape(-1)
    t2 = events[..., 1].reshape(-1)
    t3 = events[..., 2].reshape(-1)
    t4 = events[..., 3].reshape(-1)
    hmax = events[..., 4].reshape(-1)

    inv_dt = 1.0 / dt_us
    s1 = hmax / (t2 - t1)

    # Batch indices
    batch_idx = (
        torch.arange(B, device=device).unsqueeze(1).expand(B, M).reshape(-1)
    )

    # Accumulate second derivative
    d2H = torch.zeros(B, T, device=device)
    accumulate_d2H_interpolation(d2H, batch_idx, t1, s1, +1, t0_us, inv_dt, T)
    accumulate_d2H_interpolation(d2H, batch_idx, t2, s1, -1, t0_us, inv_dt, T)
    accumulate_d2H_interpolation(d2H, batch_idx, t3, s1, -1, t0_us, inv_dt, T)
    accumulate_d2H_interpolation(d2H, batch_idx, t4, s1, +1, t0_us, inv_dt, T)

    # Integrate twice to get H
    dH = torch.cumsum(d2H, dim=1)
    H = torch.cumsum(dH, dim=1) * dt_us

    return H


# ============================================================================
# Main TorchField class
# ============================================================================


class TorchField(nn.Module):
    """
    Differentiable acoustic field simulator for transducer optimization.

    This class wraps PyField's simulation logic in a PyTorch-compatible interface,
    enabling gradient-based optimization of transducer parameters.

    Parameters
    ----------
    transducer : TransducerBase
        Transducer object from pyfield.transducers
    c : float, optional
        Speed of sound in m/s. Default: 1540.0
    fs : float, optional
        Sampling frequency in Hz. Default: 200e6
    use_gpu : bool, optional
        Whether to use GPU if available. Default: True
    device : torch.device, optional
        Specific device to use. If None, auto-detect. Default: None
    verbose : bool, optional
        Print detailed information. Default: True

    Attributes
    ----------
    device : torch.device
        Compute device (CPU or CUDA)
    delays : nn.Parameter
        Element delays (trainable if configured)
    apodization : nn.Parameter
        Element apodization (trainable if configured)

    Examples
    --------
    Basic usage:
    >>> from pyfield.transducers import LinearArrayTransducer
    >>> tx = LinearArrayTransducer(n_elements=64, ...)
    >>> tf = TorchField(tx)
    >>> x, y, z, p = tf(field_points)

    Optimization:
    >>> tf.configure_optimization(optimize_delays=True)
    >>> optimizer = torch.optim.Adam(tf.parameters(), lr=0.01)
    >>> # ... training loop ...
    """

    def __init__(
        self,
        transducer,
        *,
        c: float = 1540.0,
        fs: float = 200e6,
        use_gpu: bool = True,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        super().__init__()

        # Device configuration
        if device is None:
            device = self._check_cuda_availability(use_gpu, verbose=verbose)
        self.device = device
        self.verbose = verbose

        # Store transducer reference
        self.tx = transducer

        # Physical parameters
        self.c = c  # m/s
        self.fs = fs  # Hz
        self.fc = transducer.fc  # Hz

        # Unit conversions (work in micrometers/microseconds for numerical stability)
        self.time_sec_to_unit = 1e6  # s -> μs
        self.space_m_to_unit = 1e6  # m -> μm
        self.c_unit = self.c * self.space_m_to_unit / self.time_sec_to_unit  # μm/μs

        # Extract transducer geometry
        self._extract_transducer_geometry()

        # Initialize parameters as tensors
        self._initialize_parameters()

        # Optimization configuration
        self._optimize_delays = False
        self._optimize_apodization = False
        self._constraints = {}

    def _extract_transducer_geometry(self):
        """Extract patch geometry from transducer."""
        # Get patch dimensions
        # For transducers with uniform subdivision
        self.no_sub_x = self.tx.no_sub_x
        self.no_sub_y = self.tx.no_sub_y
        self.n_elements = self.tx.n_elements

        # Patch dimensions (assume uniform for now)
        self.wx = (
            self.tx.elem_width / self.tx.no_sub_x * self.space_m_to_unit
        )  # μm
        self.wy = (
            self.tx.elem_height / self.tx.no_sub_y * self.space_m_to_unit
        )  # μm

        # Extract patch centers
        centers = []
        for elem_idx in range(self.n_elements):
            for sub_idx in range(self.no_sub_x * self.no_sub_y):
                patch_idx = elem_idx * (self.no_sub_x * self.no_sub_y) + sub_idx
                verts = self.tx.sub_quad_verts[patch_idx]
                centers.append(verts.mean(axis=0))

        self.centers = torch.tensor(
            np.array(centers) * self.space_m_to_unit,
            dtype=torch.float32,
            device=self.device,
        )  # μm

        self.n_patches = len(centers)

    def _initialize_parameters(self):
        """Initialize transducer parameters as PyTorch tensors."""
        # Get initial values from transducer
        apod_init = self.tx.apodization
        delays_init = self.tx.delays  # in seconds

        # Convert to tensors (not yet parameters - will be made trainable via configure_optimization)
        self.register_buffer(
            "_apodization_init",
            torch.tensor(apod_init, dtype=torch.float32, device=self.device),
        )
        self.register_buffer(
            "_delays_init",
            torch.tensor(
                delays_init * self.time_sec_to_unit,  # convert to μs
                dtype=torch.float32,
                device=self.device,
            ),
        )

        # Initially, just use the init values as non-trainable parameters
        self.delays = nn.Parameter(
            self._delays_init.clone(),
            requires_grad=False,
        )
        self.apodization = nn.Parameter(
            self._apodization_init.clone(),
            requires_grad=False,
        )

    def configure_optimization(
        self,
        *,
        optimize_delays: bool = False,
        optimize_apodization: bool = False,
        delay_constraints: Optional[Dict] = None,
        apodization_constraints: Optional[Dict] = None,
    ):
        """
        Configure which parameters should be optimized.

        Parameters
        ----------
        optimize_delays : bool
            Enable delay optimization
        optimize_apodization : bool
            Enable apodization optimization
        delay_constraints : dict, optional
            Constraints for delays, e.g.:
            {'min': -10e-6, 'max': 10e-6, 'regularization': 'smooth'}
        apodization_constraints : dict, optional
            Constraints for apodization, e.g.:
            {'min': 0.0, 'max': 1.0, 'regularization': 'l1'}

        Examples
        --------
        Optimize only delays:
        >>> tf.configure_optimization(optimize_delays=True)

        Optimize both with constraints:
        >>> tf.configure_optimization(
        ...     optimize_delays=True,
        ...     optimize_apodization=True,
        ...     delay_constraints={'min': -5e-6, 'max': 5e-6},
        ...     apodization_constraints={'min': 0.0, 'max': 1.0}
        ... )
        """
        self._optimize_delays = optimize_delays
        self._optimize_apodization = optimize_apodization

        # Set requires_grad based on configuration
        self.delays.requires_grad = optimize_delays
        self.apodization.requires_grad = optimize_apodization

        # Store constraints
        self._constraints['delays'] = delay_constraints or {}
        self._constraints['apodization'] = apodization_constraints or {}

        if self.verbose:
            print("\nOptimization configuration:")
            print(f"  Optimize delays: {optimize_delays}")
            print(f"  Optimize apodization: {optimize_apodization}")
            if optimize_delays and delay_constraints:
                print(f"  Delay constraints: {delay_constraints}")
            if optimize_apodization and apodization_constraints:
                print(f"  Apodization constraints: {apodization_constraints}")

    def apply_constraints(self):
        """
        Apply constraints to parameters (e.g., clamping to valid ranges).
        Call this after optimizer.step() to ensure parameters stay valid.
        """
        with torch.no_grad():
            # Apply delay constraints
            if self._optimize_delays and 'min' in self._constraints['delays']:
                self.delays.clamp_(
                    min=self._constraints['delays'].get('min', -float('inf')) * self.time_sec_to_unit,
                    max=self._constraints['delays'].get('max', float('inf')) * self.time_sec_to_unit,
                )

            # Apply apodization constraints
            if self._optimize_apodization:
                apod_min = self._constraints['apodization'].get('min', 0.0)
                apod_max = self._constraints['apodization'].get('max', 1.0)
                self.apodization.clamp_(min=apod_min, max=apod_max)

    def get_optimizable_parameters(self) -> List[nn.Parameter]:
        """
        Get list of parameters that are configured for optimization.

        Returns
        -------
        list of nn.Parameter
            Parameters with requires_grad=True
        """
        params = []
        if self._optimize_delays:
            params.append(self.delays)
        if self._optimize_apodization:
            params.append(self.apodization)
        return params

    def spatial_impulse_response(
        self,
        pts: Tensor,
        *,
        batch_size: int = 2048,
        delays: Optional[Tensor] = None,
        apodization: Optional[Tensor] = None,
    ) -> Tuple[float, Tensor]:
        """
        Compute spatial impulse response for field points.

        Parameters
        ----------
        pts : Tensor [P, 3]
            Field points in μm
        batch_size : int
            Batch size for processing
        delays : Tensor, optional
            Override delays (in μs). If None, use self.delays
        apodization : Tensor, optional
            Override apodization. If None, use self.apodization

        Returns
        -------
        t0 : float
            Start time in seconds
        H : Tensor [T, P]
            Spatial impulse response in SI units (m/s)
        """
        if delays is None:
            delays = self.delays
        if apodization is None:
            apodization = self.apodization

        P = pts.shape[0]

        # Compute temporal grid
        min_time_us, dt_us, T = self._compute_temporal_grid(pts)

        # Expand element-wise parameters to patch-wise
        expanded_delays = delays.repeat_interleave(self.no_sub_x * self.no_sub_y)
        expanded_apodization = apodization.repeat_interleave(
            self.no_sub_x * self.no_sub_y
        )

        # Initialize SIR tensor
        H = torch.zeros(P, T, device=self.device)

        # Process in batches
        desc = "Computing SIR" if self.verbose else None
        disable = not self.verbose

        for i in tqdm(range(0, P, batch_size), desc=desc, disable=disable, unit="batch"):
            j = min(i + batch_size, P)
            batch = pts[i:j]  # [B, 3]

            # Compute vectors and distances
            diff = batch.unsqueeze(1) - self.centers.unsqueeze(0)  # [B, M, 3]
            dist = diff.norm(dim=-1)  # [B, M]

            # Compute events
            events = compute_patch_events_batch(
                self.wx,
                self.wy,
                diff,
                dist,
                expanded_delays.unsqueeze(0).expand(j - i, -1),
                expanded_apodization.unsqueeze(0).expand(j - i, -1),
                inv_c=1 / self.c_unit,
                inv_fs=self.time_sec_to_unit / self.fs,
            )

            # Accumulate into SIR
            H[i:j] = accumulate_events_derivative(events, min_time_us, dt_us, T)

        # Convert back to SI units
        t0 = min_time_us / self.time_sec_to_unit  # seconds
        H = H.T * self.time_sec_to_unit / self.space_m_to_unit  # m/s

        return t0, H

    def compute_pressure_from_sir(
        self,
        h_sir: Tensor,
        grid_shape: Tuple[int, int, int],
        batch_size: int = 2048,
    ) -> Tensor:
        """
        Compute monochromatic pressure field from SIR via FFT.

        Parameters
        ----------
        h_sir : Tensor [T, P]
            Spatial impulse response
        grid_shape : tuple of int
            Shape (nx, ny, nz) of output grid
        batch_size : int
            Batch size for FFT processing

        Returns
        -------
        Tensor [nx, ny, nz]
            Pressure field amplitude at center frequency
        """
        n_time, n_points = h_sir.shape

        # Find frequency bin closest to fc
        freqs = torch.fft.fftfreq(n_time, d=1 / self.fs, device=self.device)
        idx = torch.argmin((freqs - self.fc) ** 2)

        # Process FFT in batches to save memory
        fft_results = []
        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            fft_batch = torch.fft.fft(h_sir[:, i:batch_end], dim=0)
            fft_results.append(fft_batch[idx].abs())

        # Concatenate and reshape
        fft_all = torch.cat(fft_results, dim=0)

        # Reshape: points are ordered as (z, x, y) from grid creation
        nx, ny, nz = grid_shape
        pr_field = fft_all.view(nz, nx, ny).permute(1, 2, 0)

        return pr_field

    def forward(
        self,
        field_info_mm: Dict,
        *,
        batch_size: int = 2048,
        training: bool = False,
        normalize: bool = False,
    ) -> Tuple[Union[Tensor, np.ndarray], ...]:
        """
        Compute acoustic pressure field.

        Parameters
        ----------
        field_info_mm : dict
            Field specification with keys:
            'x_extent', 'y_extent', 'z_extent', 'dx', 'dy', 'dz'
        batch_size : int
            Batch size for processing
        training : bool
            If True, keep gradients. If False, detach and convert to numpy
        normalize : bool
            If True, normalize pressure to max value

        Returns
        -------
        x, y, z : Tensor or ndarray
            Coordinate arrays
        p : Tensor or ndarray
            Pressure field
        """
        start_time = TIME()

        # Parse field points
        x, y, z, pts, P = self._check_points(field_info_mm)

        if training:
            # Compute with gradients
            if self.verbose:
                print(f"Computing field for {P} points with gradients...")

            t0, h = self.spatial_impulse_response(pts, batch_size=batch_size)
            pr = self.compute_pressure_from_sir(
                h, grid_shape=(len(x), len(y), len(z)), batch_size=batch_size
            )
        else:
            # Compute without gradients, return numpy
            if self.verbose:
                print(f"Computing field for {P} points (inference mode)...")

            with torch.no_grad():
                t0, h = self.spatial_impulse_response(pts, batch_size=batch_size)
                pr = self.compute_pressure_from_sir(
                    h, grid_shape=(len(x), len(y), len(z)), batch_size=batch_size
                )

                # Convert to numpy
                x = x.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                z = z.detach().cpu().numpy()
                pr = pr.detach().cpu().numpy()

        if normalize:
            pr = pr / pr.max()

        if self.verbose:
            elapsed = TIME() - start_time
            print(f"Field computed in {elapsed:.2f}s on {self.device}")

        return x, y, z, pr

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _check_cuda_availability(
        self, use_gpu: bool, verbose: bool = True
    ) -> torch.device:
        """Check CUDA availability and return appropriate device."""
        device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            device_cuda = torch.device("cuda:0")
            if verbose:
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
            return device_cuda if use_gpu else device_cpu
        else:
            if use_gpu and verbose:
                print("Warning: GPU requested but not available. Using CPU.")
            return device_cpu

    def _check_points(
        self, field_points_mm: Dict
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Parse field specification and convert to tensors."""
        x, y, z, pts = create_3D_spatial_grid_from_points(field_points_mm)

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        pts = torch.tensor(pts, dtype=torch.float32, device=self.device)

        # Convert to working units (μm)
        pts = pts * self.space_m_to_unit

        return x, y, z, pts, pts.shape[0]

    def _compute_temporal_grid(
        self, pts: Tensor, batch_size: int = 1024
    ) -> Tuple[float, float, int]:
        """Compute temporal grid parameters."""
        P = pts.shape[0]

        max_d = float("-inf")
        min_d = float("inf")

        # Find min/max distances to patches
        with torch.no_grad():
            for i in range(0, P, batch_size):
                batch_pts = pts[i : i + batch_size]
                dists = (batch_pts.unsqueeze(1) - self.centers.unsqueeze(0)).norm(
                    dim=-1
                )
                max_d = max(max_d, dists.max().item())
                min_d = min(min_d, dists.min().item())

        max_delay = self.delays.max().item()

        # Time range
        min_time_us = (min_d - 0.5 * (self.wx + self.wy)) / self.c_unit
        max_time_us = (max_d + 0.5 * (self.wx + self.wy)) / self.c_unit + max_delay

        dt_us = (1.0 / self.fs) * self.time_sec_to_unit
        T = int(math.ceil((max_time_us - min_time_us) / dt_us))

        return min_time_us, dt_us, T

    def __repr__(self) -> str:
        return (
            f"TorchField(\n"
            f"  transducer={self.tx.name},\n"
            f"  n_elements={self.n_elements},\n"
            f"  n_patches={self.n_patches},\n"
            f"  c={self.c} m/s,\n"
            f"  fc={self.fc/1e6:.1f} MHz,\n"
            f"  device={self.device},\n"
            f"  optimize_delays={self._optimize_delays},\n"
            f"  optimize_apodization={self._optimize_apodization}\n"
            f")"
        )
