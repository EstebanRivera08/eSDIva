"""
TorchField with Flexible Parameter Optimization Framework

This module provides a highly flexible parameter optimization system that allows:
- Optimizing any transducer parameter (delays, apodization, positions, etc.)
- Optimizing virtual parameters (e.g., virtual source positions)
- Using custom functions to compute derived parameters
- Proper handling of element-level vs patch-level parameters

Architecture:
    OptimizableParameter: Wraps a torch parameter with metadata
    ParameterMapping: Defines how to compute derived parameters
    TorchField: Orchestrates simulation with parameter transformations

Example - Virtual Source Optimization:
    >>> # Define virtual source position as optimizable
    >>> tf.add_optimizable_parameter(
    ...     'virtual_source',
    ...     initial_value=[0, 0, 30],  # [x, y, z] in mm
    ...     level='global'
    ... )
    >>>
    >>> # Map virtual source to delays
    >>> def compute_delays_from_vs(virtual_source, tx):
    ...     return tx.compute_delays(focus_mm=virtual_source, apply=False)
    >>>
    >>> tf.add_parameter_mapping(
    ...     name='virtual_source_to_delays',
    ...     function=compute_delays_from_vs,
    ...     inputs=['virtual_source'],
    ...     output='delays',
    ...     level='element'
    ... )

Example - Element Position Optimization:
    >>> # Optimize element positions directly
    >>> tf.add_optimizable_parameter(
    ...     'element_offsets',
    ...     initial_value=np.zeros((n_elements, 3)),
    ...     level='element'
    ... )
    >>>
    >>> # Map element positions to patch centers
    >>> def compute_patches_from_positions(element_offsets, tx):
    ...     # Recompute patch centers with offsets
    ...     ...
    >>>
    >>> tf.add_parameter_mapping(
    ...     name='positions_to_patches',
    ...     function=compute_patches_from_positions,
    ...     inputs=['element_offsets'],
    ...     output='patch_centers',
    ...     level='patch'
    ... )
"""

import math
import warnings
from collections import OrderedDict
from time import time as TIME
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from pyfield.utilities.helper_functions import create_3D_spatial_grid_from_points

# Import core functions from TorchField_v2
from .TorchField_v2 import (
    accumulate_events_derivative,
    compute_patch_events_batch,
)

# ============================================================================
# Parameter Management Classes
# ============================================================================


class OptimizableParameter:
    """
    Wrapper for an optimizable parameter with metadata.

    Parameters
    ----------
    name : str
        Parameter identifier
    value : array-like
        Initial value
    level : str
        Parameter level: 'global', 'element', or 'patch'
    shape : tuple, optional
        Expected shape. If None, infer from value
    requires_grad : bool
        Whether this parameter should be optimized
    constraints : dict, optional
        Constraints like {'min': 0, 'max': 1}
    transform : callable, optional
        Function applied during forward pass (e.g., sigmoid for [0,1] range)

    Attributes
    ----------
    value : nn.Parameter
        The actual PyTorch parameter
    """

    def __init__(
        self,
        name: str,
        value: Union[float, List, np.ndarray],
        *,
        level: str = "global",
        shape: Optional[Tuple] = None,
        requires_grad: bool = True,
        constraints: Optional[Dict] = None,
        transform: Optional[Callable] = None,
    ):
        self.name = name
        self.level = level
        self.constraints = constraints or {}
        self.transform = transform

        # Convert to tensor
        if isinstance(value, (int, float)):
            tensor_val = torch.tensor([value], dtype=torch.float32)
        else:
            tensor_val = torch.tensor(np.array(value), dtype=torch.float32)

        # Validate shape
        if shape is not None and tensor_val.shape != shape:
            raise ValueError(
                f"Parameter '{name}': expected shape {shape}, got {tensor_val.shape}"
            )

        self.value = nn.Parameter(tensor_val, requires_grad=requires_grad)

    def get_value(self) -> Tensor:
        """Get parameter value, applying transform if specified."""
        val = self.value
        if self.transform is not None:
            val = self.transform(val)
        return val

    def apply_constraints(self):
        """Apply constraints to parameter value."""
        if not self.constraints:
            return

        with torch.no_grad():
            if "min" in self.constraints or "max" in self.constraints:
                self.value.clamp_(
                    min=self.constraints.get("min", -float("inf")),
                    max=self.constraints.get("max", float("inf")),
                )

    def __repr__(self) -> str:
        return (
            f"OptimizableParameter(name='{self.name}', "
            f"level='{self.level}', "
            f"shape={tuple(self.value.shape)}, "
            f"requires_grad={self.value.requires_grad})"
        )


class ParameterMapping:
    """
    Defines how to compute a derived parameter from optimizable parameters.

    This enables complex parameter transformations like:
    - virtual_source → delays
    - element_positions → patch_centers
    - custom_apod_params → apodization

    Parameters
    ----------
    name : str
        Mapping identifier
    function : callable
        Function that computes output from inputs.
        Signature: fn(**input_params, tx=transducer, device=device) -> tensor
    inputs : list of str
        Names of input parameters (from OptimizableParameter or other mappings)
    output : str
        Name of output parameter
    level : str
        Output level: 'global', 'element', or 'patch'
    cache : bool
        Whether to cache the result (avoid recomputation)

    Examples
    --------
    Virtual source to delays:
    >>> def vs_to_delays(virtual_source, tx, device):
    ...     # virtual_source is [3] tensor (x, y, z)
    ...     vs_np = virtual_source.detach().cpu().numpy()
    ...     delays = tx.compute_delays(focus_mm=vs_np, apply=False)
    ...     return torch.tensor(delays, device=device)
    >>>
    >>> mapping = ParameterMapping(
    ...     name='vs_to_delays',
    ...     function=vs_to_delays,
    ...     inputs=['virtual_source'],
    ...     output='delays',
    ...     level='element'
    ... )
    """

    def __init__(
        self,
        name: str,
        function: Callable,
        inputs: List[str],
        output: str,
        level: str,
        *,
        cache: bool = False,
    ):
        self.name = name
        self.function = function
        self.inputs = inputs
        self.output = output
        self.level = level
        self.cache = cache
        self._cached_result = None
        self._cache_valid = False

    def compute(
        self,
        input_values: Dict[str, Tensor],
        tx: Any,
        device: torch.device,
    ) -> Tensor:
        """
        Compute output from inputs.

        Parameters
        ----------
        input_values : dict
            {param_name: value} for each input
        tx : TransducerBase
            Transducer object
        device : torch.device
            Compute device

        Returns
        -------
        Tensor
            Computed output value
        """
        if self.cache and self._cache_valid:
            return self._cached_result

        # Call function with inputs
        result = self.function(**input_values, tx=tx, device=device)

        if self.cache:
            self._cached_result = result
            self._cache_valid = True

        return result

    def invalidate_cache(self):
        """Invalidate cached result."""
        self._cache_valid = False

    def __repr__(self) -> str:
        return (
            f"ParameterMapping(name='{self.name}', "
            f"inputs={self.inputs}, "
            f"output='{self.output}', "
            f"level='{self.level}')"
        )


# ============================================================================
# Flexible TorchField
# ============================================================================


class TorchFieldFlexible(nn.Module):
    """
    Highly flexible differentiable acoustic field simulator.

    This version allows optimizing arbitrary parameters through a mapping system:
    - Define optimizable parameters (base parameters)
    - Define mappings that compute derived parameters
    - Automatic dependency resolution and computation

    The simulation flow:
        1. Optimizable parameters (e.g., virtual_source, custom params)
        2. Parameter mappings (e.g., virtual_source → delays)
        3. Final transducer parameters (delays, apodization, patch_centers)
        4. Acoustic simulation (SIR → pressure)

    Parameters
    ----------
    transducer : TransducerBase
        Base transducer object
    c : float
        Speed of sound in m/s
    fs : float
        Sampling frequency in Hz
    use_gpu : bool
        Use GPU if available
    device : torch.device, optional
        Specific device
    verbose : bool
        Print detailed info

    Examples
    --------
    See module docstring for detailed examples.
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

        # Device
        if device is None:
            device = self._check_cuda_availability(use_gpu, verbose=verbose)
        self.device = device
        self.verbose = verbose

        # Transducer
        self.tx = transducer

        # Physical parameters
        self.c = c
        self.fs = fs
        self.fc = transducer.fc

        # Unit conversions
        self.time_sec_to_unit = 1e6  # s → μs
        self.space_m_to_unit = 1e6  # m → μm
        self.c_unit = self.c * self.space_m_to_unit / self.time_sec_to_unit

        # Transducer dimensions
        self.no_sub_x = transducer.no_sub_x
        self.no_sub_y = transducer.no_sub_y
        self.n_elements = transducer.n_elements
        self.n_patches = self.n_elements * self.no_sub_x * self.no_sub_y

        # Get patch dimensions (assume uniform)
        self.wx = transducer.elem_width / transducer.no_sub_x * self.space_m_to_unit
        self.wy = transducer.elem_height / transducer.no_sub_y * self.space_m_to_unit

        # Parameter management
        self._optimizable_params: Dict[str, OptimizableParameter] = OrderedDict()
        self._parameter_mappings: Dict[str, ParameterMapping] = OrderedDict()

        # Cache for computed parameters
        self._computed_cache: Dict[str, Tensor] = {}

        # Initialize with default transducer parameters
        self._initialize_default_parameters(verbose=False)

    def _initialize_default_parameters(self, verbose: bool = True):
        """
        Initialize with transducer's current parameters as defaults.

        Users can replace these by adding their own optimizable parameters
        and mappings.
        """
        # Extract initial delays and apodization
        delays_init = self.tx.delays  # seconds
        apod_init = self.tx.apodization

        # Add as non-optimizable by default
        self.add_optimizable_parameter(
            "delays",
            initial_value=delays_init * self.time_sec_to_unit,  # to μs
            level="element",
            requires_grad=False,
            verbose=verbose,
        )

        self.add_optimizable_parameter(
            "apodization",
            initial_value=apod_init,
            level="element",
            requires_grad=False,
            constraints={"min": 0.0, "max": 1.0},
            verbose=verbose,
        )

        # Extract patch centers
        centers = []
        for elem_idx in range(self.n_elements):
            for sub_idx in range(self.no_sub_x * self.no_sub_y):
                patch_idx = elem_idx * (self.no_sub_x * self.no_sub_y) + sub_idx
                verts = self.tx.sub_quad_verts[patch_idx]
                centers.append(verts.mean(axis=0))

        self.add_optimizable_parameter(
            "patch_centers",
            initial_value=np.array(centers) * self.space_m_to_unit,  # to μm
            level="patch",
            requires_grad=False,
            verbose=verbose,
        )

    # ========================================================================
    # Parameter Management API
    # ========================================================================

    def add_optimizable_parameter(
        self,
        name: str,
        initial_value: Union[float, List, np.ndarray],
        *,
        level: str = "global",
        requires_grad: bool = True,
        constraints: Optional[Dict] = None,
        transform: Optional[Callable] = None,
        replace: bool = True,
        verbose: Optional[bool] = True,
    ):
        """
        Add an optimizable parameter.

        Parameters
        ----------
        name : str
            Parameter name (e.g., 'delays', 'virtual_source', 'custom_param')
        initial_value : float, list, or array
            Initial value
        level : str
            'global' (single value), 'element' (per element), or 'patch' (per patch)
        requires_grad : bool
            Whether to optimize this parameter
        constraints : dict, optional
            Constraints like {'min': 0, 'max': 1}
        transform : callable, optional
            Transform function (e.g., torch.sigmoid)
        replace : bool
            If True, replace existing parameter with same name

        Examples
        --------
        Optimize delays directly:
        >>> tf.add_optimizable_parameter(
        ...     'delays',
        ...     initial_value=tx.delays,
        ...     level='element',
        ...     requires_grad=True
        ... )

        Add virtual source:
        >>> tf.add_optimizable_parameter(
        ...     'virtual_source',
        ...     initial_value=[0, 0, 30],  # mm
        ...     level='global',
        ...     requires_grad=True
        ... )

        Add custom apodization parameters:
        >>> tf.add_optimizable_parameter(
        ...     'apod_center',
        ...     initial_value=32,  # center element
        ...     level='global',
        ...     requires_grad=True,
        ...     constraints={'min': 0, 'max': 63}
        ... )
        """
        if name in self._optimizable_params and not replace:
            raise ValueError(
                f"Parameter '{name}' already exists. Use replace=True to overwrite."
            )

        param = OptimizableParameter(
            name=name,
            value=initial_value,
            level=level,
            requires_grad=requires_grad,
            constraints=constraints,
            transform=transform,
        )

        self._optimizable_params[name] = param

        # Register as PyTorch parameter
        self.register_parameter(f"_param_{name}", param.value)

        if verbose:
            print(f"Added optimizable parameter: {param}")

    def add_parameter_mapping(
        self,
        name: str,
        function: Callable,
        inputs: List[str],
        output: str,
        level: str,
        *,
        cache: bool = False,
        replace: bool = True,
    ):
        """
        Add a mapping that computes a parameter from other parameters.

        Parameters
        ----------
        name : str
            Mapping name
        function : callable
            Function with signature: fn(**inputs, tx=transducer, device=device) -> tensor
        inputs : list of str
            Input parameter names
        output : str
            Output parameter name
        level : str
            Output level: 'global', 'element', or 'patch'
        cache : bool
            Cache result (avoid recomputation)
        replace : bool
            Replace existing mapping

        Examples
        --------
        Virtual source to delays:
        >>> def vs_to_delays(virtual_source, tx, device):
        ...     vs_np = virtual_source.detach().cpu().numpy()
        ...     delays = tx.compute_delays(focus_mm=vs_np, apply=False)
        ...     return torch.tensor(delays, dtype=torch.float32, device=device) * 1e6
        >>>
        >>> tf.add_parameter_mapping(
        ...     name='vs_to_delays',
        ...     function=vs_to_delays,
        ...     inputs=['virtual_source'],
        ...     output='delays',
        ...     level='element'
        ... )

        Custom Gaussian apodization:
        >>> def gaussian_apod(apod_center, apod_width, tx, device):
        ...     elements = torch.arange(tx.n_elements, device=device, dtype=torch.float32)
        ...     apod = torch.exp(-((elements - apod_center)**2) / (2 * apod_width**2))
        ...     return apod
        >>>
        >>> tf.add_parameter_mapping(
        ...     name='gaussian_apod',
        ...     function=gaussian_apod,
        ...     inputs=['apod_center', 'apod_width'],
        ...     output='apodization',
        ...     level='element'
        ... )
        """
        if name in self._parameter_mappings and not replace:
            raise ValueError(
                f"Mapping '{name}' already exists. Use replace=True to overwrite."
            )

        mapping = ParameterMapping(
            name=name,
            function=function,
            inputs=inputs,
            output=output,
            level=level,
            cache=cache,
        )

        self._parameter_mappings[name] = mapping

        if self.verbose:
            print(f"Added parameter mapping: {mapping}")

    def get_parameter(self, name: str) -> Tensor:
        """
        Get a parameter value, computing it if necessary.

        This handles dependency resolution:
        1. Check if it's a direct optimizable parameter
        2. Check if there's a mapping that computes it
        3. Check cache

        Parameters
        ----------
        name : str
            Parameter name

        Returns
        -------
        Tensor
            Parameter value
        """
        # Check cache first
        if name in self._computed_cache:
            return self._computed_cache[name]

        # Check if it's a direct optimizable parameter
        if name in self._optimizable_params:
            value = self._optimizable_params[name].get_value().to(self.device)
            self._computed_cache[name] = value
            return value

        # Check if there's a mapping that computes it
        for mapping in self._parameter_mappings.values():
            if mapping.output == name:
                # Get input values (recursively)
                input_values = {inp: self.get_parameter(inp) for inp in mapping.inputs}

                # Compute
                value = mapping.compute(input_values, self.tx, self.device)
                self._computed_cache[name] = value
                return value

        raise ValueError(
            f"Parameter '{name}' not found. "
            f"Available: {list(self._optimizable_params.keys())}, "
            f"Computed: {[m.output for m in self._parameter_mappings.values()]}"
        )

    def clear_cache(self):
        """Clear computed parameter cache. Call before forward pass."""
        self._computed_cache.clear()
        for mapping in self._parameter_mappings.values():
            mapping.invalidate_cache()

    def apply_constraints(self):
        """Apply constraints to all optimizable parameters."""
        for param in self._optimizable_params.values():
            param.apply_constraints()

    def get_optimizable_parameters(self) -> List[nn.Parameter]:
        """Get list of parameters with requires_grad=True."""
        return [
            param.value
            for param in self._optimizable_params.values()
            if param.value.requires_grad
        ]

    # ========================================================================
    # Simulation Methods
    # ========================================================================

    def spatial_impulse_response(
        self,
        pts: Tensor,
        *,
        batch_size: int = 2048,
    ) -> Tuple[float, Tensor]:
        """
        Compute SIR using current parameters.

        Parameters
        ----------
        pts : Tensor [P, 3]
            Field points in μm
        batch_size : int
            Batch size

        Returns
        -------
        t0 : float
            Start time (seconds)
        H : Tensor [T, P]
            SIR in SI units (m/s)
        """
        P = pts.shape[0]

        # Get required parameters (computed via mappings if needed)
        delays_elem = self.get_parameter("delays")  # [n_elements] in μs
        apod_elem = self.get_parameter("apodization")  # [n_elements]
        patch_centers = self.get_parameter("patch_centers")  # [n_patches, 3] in μm

        # Expand element-level to patch-level
        delays_patch = delays_elem.repeat_interleave(self.no_sub_x * self.no_sub_y)
        apod_patch = apod_elem.repeat_interleave(self.no_sub_x * self.no_sub_y)

        # Compute temporal grid
        min_time_us, dt_us, T = self._compute_temporal_grid(
            pts, patch_centers, delays_patch
        )

        # Initialize SIR
        H = torch.zeros(P, T, device=self.device)

        # Batch processing
        desc = "Computing SIR" if self.verbose else None
        for i in tqdm(
            range(0, P, batch_size), desc=desc, disable=not self.verbose, unit="batch"
        ):
            j = min(i + batch_size, P)
            batch = pts[i:j]

            diff = batch.unsqueeze(1) - patch_centers.unsqueeze(0)
            dist = diff.norm(dim=-1)

            events = compute_patch_events_batch(
                self.wx,
                self.wy,
                diff,
                dist,
                delays_patch.unsqueeze(0).expand(j - i, -1),
                apod_patch.unsqueeze(0).expand(j - i, -1),
                inv_c=1 / self.c_unit,
                inv_fs=self.time_sec_to_unit / self.fs,
            )

            H[i:j] = accumulate_events_derivative(events, min_time_us, dt_us, T)

        # Convert to SI
        t0 = min_time_us / self.time_sec_to_unit
        H = H.T * self.time_sec_to_unit / self.space_m_to_unit

        return t0, H

    def compute_pressure_from_sir(
        self,
        h_sir: Tensor,
        grid_shape: Tuple[int, int, int],
        batch_size: int = 2048,
    ) -> Tensor:
        """Compute pressure from SIR via FFT."""
        n_time, n_points = h_sir.shape

        freqs = torch.fft.fftfreq(n_time, d=1 / self.fs, device=self.device)
        idx = torch.argmin((freqs - self.fc) ** 2)

        fft_results = []
        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            fft_batch = torch.fft.fft(h_sir[:, i:batch_end], dim=0)
            fft_results.append(fft_batch[idx].abs())

        fft_all = torch.cat(fft_results, dim=0)
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
        Compute pressure field.

        Before each forward pass, clears parameter cache to ensure
        fresh computation with updated optimizable parameters.
        """
        start_time = TIME()

        # Clear cache to recompute with updated parameters
        self.clear_cache()

        # Parse field
        x, y, z, pts, P = self._check_points(field_info_mm)

        if self.verbose:
            mode = "training" if training else "inference"
            print(f"Computing field ({mode}) for {P} points...")

        if training:
            t0, h = self.spatial_impulse_response(pts, batch_size=batch_size)
            pr = self.compute_pressure_from_sir(h, (len(x), len(y), len(z)), batch_size)
        else:
            with torch.no_grad():
                t0, h = self.spatial_impulse_response(pts, batch_size=batch_size)
                pr = self.compute_pressure_from_sir(
                    h, (len(x), len(y), len(z)), batch_size
                )
                x = x.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                z = z.detach().cpu().numpy()
                pr = pr.detach().cpu().numpy()

        if normalize:
            if isinstance(pr, Tensor):
                pr = pr / pr.max()
            else:
                pr = pr / pr.max()

        if self.verbose:
            print(f"Field computed in {TIME() - start_time:.2f}s")

        return x, y, z, pr

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _check_cuda_availability(self, use_gpu: bool, verbose: bool) -> torch.device:
        """Check CUDA and return device."""
        if torch.cuda.is_available() and use_gpu:
            if verbose:
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda:0")
        else:
            if use_gpu and verbose:
                print("Warning: GPU requested but not available. Using CPU.")
            return torch.device("cpu")

    def _check_points(self, field_info_mm: Dict):
        """Parse field specification."""
        x, y, z, pts = create_3D_spatial_grid_from_points(field_info_mm)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        pts = torch.tensor(pts, dtype=torch.float32, device=self.device)

        pts = pts * self.space_m_to_unit

        return x, y, z, pts, pts.shape[0]

    def _compute_temporal_grid(
        self, pts: Tensor, patch_centers: Tensor, delays: Tensor, batch_size: int = 1024
    ) -> Tuple[float, float, int]:
        """Compute temporal grid."""
        P = pts.shape[0]
        max_d = float("-inf")
        min_d = float("inf")

        with torch.no_grad():
            for i in range(0, P, batch_size):
                batch_pts = pts[i : i + batch_size]
                dists = (batch_pts.unsqueeze(1) - patch_centers.unsqueeze(0)).norm(
                    dim=-1
                )
                max_d = max(max_d, dists.max().item())
                min_d = min(min_d, dists.min().item())

        max_delay = delays.max().item()
        min_time_us = (min_d - 0.5 * (self.wx + self.wy)) / self.c_unit
        max_time_us = (max_d + 0.5 * (self.wx + self.wy)) / self.c_unit + max_delay
        dt_us = (1.0 / self.fs) * self.time_sec_to_unit
        T = int(math.ceil((max_time_us - min_time_us) / dt_us))

        return min_time_us, dt_us, T

    def __repr__(self) -> str:
        opt_params = [
            p.name for p in self._optimizable_params.values() if p.value.requires_grad
        ]
        return (
            f"TorchFieldFlexible(\n"
            f"  transducer={self.tx.name},\n"
            f"  n_elements={self.n_elements},\n"
            f"  optimizable_params={opt_params},\n"
            f"  mappings={list(self._parameter_mappings.keys())},\n"
            f"  device={self.device}\n"
            f")"
        )
