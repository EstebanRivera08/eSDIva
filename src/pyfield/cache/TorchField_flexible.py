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
    >>> def compute_quad_vert_from_positions(element_offsets, tx):
    ...     # Recompute patch centers with offsets
    ...     ...
    >>> tf.add_parameter_mapping(
    ...     name='positions_to_quad_vertices',
    ...     function=compute_quad_vert_from_positions,
    ...     inputs=['element_offsets'],
    ...     output='quad_vertices',
    ...     level='patch'
    ...     )
     ... )
"""

import math
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
# Differentiable geometry builders
# ============================================================================


def build_rect_patch_vertices_torch(
    element_centers: Tensor,
    *,
    elem_width: float,
    elem_height: float,
    no_sub_x: int,
    no_sub_y: int,
    elev_focus: float = 0.0,
) -> Tensor:
    """
    Torch-differentiable equivalent of ``LinearArrayTransducer._build_subdivisions``.

    Produces the full ``quad_vertices`` tensor (shape ``[n_patches, 4, 3]``)
    used by :class:`TorchFieldFlexible`. The gradient graph is preserved
    through ``element_centers`` so element positions can be optimized end-to-end.

    The patch layout is the one used by ``LinearArrayTransducer``:

    * Each element is subdivided into ``no_sub_x × no_sub_y`` rectangular
      sub-patches laid out in the local (x, y) plane of the element.
    * Sub-patch ordering matches the numpy builder:
      outer loop over ``i`` (x-sub-index), inner loop over ``j`` (y-sub-index).
    * When ``elev_focus > 0``, an elevation cylindrical lens is applied so
      that sub-patch vertices are lifted onto an arc of radius ``elev_focus``
      in the yz-plane (flat lens at ``elev_focus == 0``).

    Parameters
    ----------
    element_centers : Tensor [n_elements, 3]
        Element centre positions in metres. May require grad.
    elem_width : float
        Element width (x-extent) in metres.
    elem_height : float
        Element height (y-extent) in metres.
    no_sub_x, no_sub_y : int
        Number of sub-patches per element in x and y.
    elev_focus : float, optional
        Cylindrical elevation focal length in metres. ``0.0`` = flat lens.

    Returns
    -------
    Tensor [n_patches, 4, 3]
        Quad vertices in metres. Carries grad if ``element_centers`` does.
        ``n_patches = n_elements * no_sub_x * no_sub_y``.
        The four corners per patch are ordered (low-x low-y), (high-x low-y),
        (high-x high-y), (low-x high-y) — same as the numpy builder.
    """
    device = element_centers.device
    dtype = element_centers.dtype

    # Local patch grid edges (metres), relative to the element centre
    xs = torch.linspace(
        -elem_width / 2, elem_width / 2, no_sub_x + 1, device=device, dtype=dtype
    )
    ys = torch.linspace(
        -elem_height / 2, elem_height / 2, no_sub_y + 1, device=device, dtype=dtype
    )

    # Build the (no_sub_x, no_sub_y, 4) arrays of corner x- and y-coordinates
    # in the element-local frame. Order: (lo-x, lo-y), (hi-x, lo-y),
    # (hi-x, hi-y), (lo-x, hi-y) — matches LinearArrayTransducer._build_subdivisions.
    x_lo = xs[:-1]  # [no_sub_x]
    x_hi = xs[1:]  # [no_sub_x]
    y_lo = ys[:-1]  # [no_sub_y]
    y_hi = ys[1:]  # [no_sub_y]

    # corner_x: [no_sub_x, 4]
    corner_x = torch.stack([x_lo, x_hi, x_hi, x_lo], dim=1)
    # corner_y: [no_sub_y, 4]
    corner_y = torch.stack([y_lo, y_lo, y_hi, y_hi], dim=1)

    # Broadcast to [no_sub_x, no_sub_y, 4]
    local_x = corner_x.unsqueeze(1).expand(no_sub_x, no_sub_y, 4)
    local_y = corner_y.unsqueeze(0).expand(no_sub_x, no_sub_y, 4)
    local_z = torch.zeros_like(local_x)

    # Apply cylindrical elevation lens (before combining with element centres)
    if elev_focus > 0:
        # z_offset(y) = R - sqrt(R² - y²), clamped to avoid sqrt of negative
        R = elev_focus
        local_z = R - torch.sqrt(torch.clamp(R * R - local_y * local_y, min=0.0))

    # Stack into local_corners: [no_sub_x, no_sub_y, 4, 3]
    local_corners = torch.stack([local_x, local_y, local_z], dim=-1)

    # Flatten sub-patch grid: [no_sub_x * no_sub_y, 4, 3]
    local_corners_flat = local_corners.reshape(no_sub_x * no_sub_y, 4, 3)

    # Broadcast-add element centres. element_centers: [n_elements, 1, 1, 3]
    # local_corners_flat:                             [1,         M, 4, 3] where M = no_sub_x*no_sub_y
    # result:                                         [n_elements, M, 4, 3]
    n_elements = element_centers.shape[0]
    ec = element_centers.view(n_elements, 1, 1, 3)
    lc = local_corners_flat.unsqueeze(0)  # [1, M, 4, 3]

    if elev_focus > 0:
        # In the numpy builder, the z-offset from the cylindrical lens
        # REPLACES any z component of the element centre (corners[:, 2] += elev_focus - ...).
        # For a flat linear array element_centers[:, 2] is 0, so the two
        # formulations agree; we nevertheless mirror the numpy behaviour by
        # only adding the (x, y) components of the element centre.
        xy = ec[..., :2]
        z_local = lc[..., 2:3].expand(n_elements, -1, 4, 1)
        xy_full = lc[..., :2] + xy
        verts = torch.cat([xy_full, z_local], dim=-1)
    else:
        verts = lc + ec

    # Flatten to [n_patches, 4, 3]
    return verts.reshape(n_elements * no_sub_x * no_sub_y, 4, 3)


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

    # [TODO] : apply_contraints should be a function to be applied to values (is more general)

    def apply_constraints(self):
        """Apply constraints to parameter value."""
        if not self.constraints:
            return

        with torch.no_grad():
            # --- MIN ---
            min_raw = self.constraints.get("min", None)
            if min_raw is None:
                min_val = None
            elif isinstance(min_raw, (int, float)):
                min_val = float(min_raw)
            else:
                # list/array/tensor-like → replace None with -inf
                cleaned = [(-float("inf") if v is None else v) for v in min_raw]
                min_val = torch.as_tensor(cleaned, device=self.value.device)

            # --- MAX ---
            max_raw = self.constraints.get("max", None)
            if max_raw is None:
                max_val = None
            elif isinstance(max_raw, (int, float)):
                max_val = float(max_raw)
            else:
                # list/array/tensor-like → replace None with -inf
                cleaned = [(float("inf") if v is None else v) for v in max_raw]
                max_val = torch.as_tensor(cleaned, device=self.value.device)

            # Optional: shape validation for per‑entry constraints
            if isinstance(min_val, torch.Tensor) and min_val.shape != self.value.shape:
                raise ValueError(
                    f"min constraint shape mismatch: {min_val.shape} vs {self.value.shape}"
                )

            if isinstance(max_val, torch.Tensor) and max_val.shape != self.value.shape:
                raise ValueError(
                    f"max constraint shape mismatch: {max_val.shape} vs {self.value.shape}"
                )
                # Apply clamp
            self.value.clamp_(min=min_val, max=max_val)

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
    - element_positions → quad_vertices -> patch_centers
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
        print(
            f"Initialized with fs={self.fs / 1e6} MHz, c={self.c} m/s, device={self.device}"
        )

    def _initialize_default_parameters(self, verbose: bool = True):
        """
        Initialize with transducer's current parameters as defaults.

        The default parameter graph is:

        ::

            delays         (direct)  [n_elements]        μs
            apodization    (direct)  [n_elements]        —
            quad_vertices  (direct)  [n_patches, 4, 3]   μm
            patch_centers  (mapping) [n_patches, 3]      μm   = quad_vertices.mean(dim=1)

        Users can override any of these by:

        - Replacing a direct parameter with ``add_optimizable_parameter(name, ..., replace=True)``
        - Adding a mapping with the same ``output`` name; mappings take
          precedence over direct parameters in ``get_parameter``.

        Using ``quad_vertices`` as the geometric foundation (instead of
        ``patch_centers`` directly) lets higher-level transducer attributes —
        element positions, element sizes, etc. — be optimized through a
        differentiable chain ``attribute → quad_vertices → patch_centers → SIR``.
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

        # Stack per-patch quad vertices: list of (4,3) arrays → (n_patches, 4, 3)
        quad_verts_m = np.stack(
            [np.asarray(v, dtype=np.float64) for v in self.tx.sub_quad_verts],
            axis=0,
        )
        self.add_optimizable_parameter(
            "quad_vertices",
            initial_value=quad_verts_m * self.space_m_to_unit,  # to μm
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

        Resolution order (mappings take precedence over direct parameters so
        that user-added mappings override auto-registered defaults without
        requiring manual bookkeeping):

        1. Check ``_computed_cache``
        2. Check ``_parameter_mappings`` — if a mapping produces ``name``,
           call it (recursing into its inputs)
        3. Check ``_optimizable_params`` — return the direct parameter
        4. Raise ``ValueError``

        Parameters
        ----------
        name : str
            Parameter name

        Returns
        -------
        Tensor
            Parameter value
        """
        # 1. Cache
        if name in self._computed_cache:
            return self._computed_cache[name]

        # 2. Mappings first — user-added mappings override defaults
        for mapping in self._parameter_mappings.values():
            if mapping.output == name:
                input_values = {inp: self.get_parameter(inp) for inp in mapping.inputs}
                value = mapping.compute(input_values, self.tx, self.device)
                self._computed_cache[name] = value
                return value

        # 3. Direct optimizable parameters
        if name in self._optimizable_params:
            value = self._optimizable_params[name].get_value().to(self.device)
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
        quad_vertices = self.get_parameter("quad_vertices")  # [n_patches, 4, 3] in μm

        patch_centers = quad_vertices.mean(dim=1)  # [n_patches, 3] in μm

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
            fft_results.append(fft_batch[idx])

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

        if self.verbose or not training:
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

        T = int(max((max_time_us - min_time_us) / dt_us, 1) + 1)

        # Memory estimate for h_sir (P × T × 4 bytes float32).
        h_sir_gb = P * T * 4 / 1e9
        if h_sir_gb >= 2.0:
            print(
                f"\nWARNING: grid Nx×Ny×Nz = {P:,} points and T={T}— "
                f"estimated h_sir ~= {h_sir_gb:.1f} GB "
                "This will likely cause a memory error.\n"
                "  -> Reduce dx/dy/dz or, shrink the extent, or compute a 2-D plane. "
            )
        elif h_sir_gb >= 0.5:
            print(
                f"INFO: grid Nx×Ny×Nz = {P:,} points and T = {T} — "
                f"estimated h_sir ~= {h_sir_gb * 1e3:.0f} MB "
                "Consider a coarser grid if memory is limited.\n"
            )

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
