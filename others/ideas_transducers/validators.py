"""
Validation utilities for transducer geometries and parameters.

This module provides reusable validation functions and decorators for ensuring
transducer parameters are physically meaningful and computationally valid.
"""

import numpy as np
from typing import Tuple, Union, Optional


def validate_positive(value: float, name: str, strict: bool = False) -> float:
    """
    Validate that a value is positive (or non-negative).

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        The name of the parameter (for error messages).
    strict : bool, optional
        If True, require value > 0. If False (default), allow value >= 0.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if strict:
        if value <= 0:
            raise ValueError(f"{name} must be positive (> 0), got {value}")
    else:
        if value < 0:
            raise ValueError(f"{name} must be non-negative (>= 0), got {value}")
    return value


def validate_range(
    value: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    inclusive: bool = True,
) -> float:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        The name of the parameter (for error messages).
    min_val : float, optional
        Minimum allowed value.
    max_val : float, optional
        Maximum allowed value.
    inclusive : bool, optional
        If True, bounds are inclusive. Default is True.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If validation fails.
    """
    cmp_op = "<=" if inclusive else "<"
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        elif not inclusive and value <= min_val:
            raise ValueError(f"{name} must be > {min_val}, got {value}")

    if max_val is not None:
        if inclusive and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        elif not inclusive and value >= max_val:
            raise ValueError(f"{name} must be < {max_val}, got {value}")

    return value


def validate_integer(
    value: Union[int, float], name: str, min_val: int = 1
) -> int:
    """
    Validate that a value is a positive integer.

    Parameters
    ----------
    value : int or float
        The value to validate.
    name : str
        The name of the parameter (for error messages).
    min_val : int, optional
        Minimum allowed value (default: 1).

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    ValueError
        If validation fails.
    TypeError
        If value is not an integer type.
    """
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < min_val:
        raise ValueError(
            f"{name} must be >= {min_val}, got {value}"
        )
    return int(value)


def validate_kerf(kerf_mm: float, element_size_mm: float, name: str = "kerf") -> float:
    """
    Validate kerf (gap between elements).

    Parameters
    ----------
    kerf_mm : float
        Kerf in millimeters.
    element_size_mm : float
        Element size in millimeters (for proportionality check).
    name : str, optional
        Parameter name for error messages. Default: "kerf".

    Returns
    -------
    float
        The validated kerf value.

    Raises
    ------
    ValueError
        If kerf is negative or unreasonably large.
    """
    if kerf_mm < 0:
        raise ValueError(f"{name} must be non-negative, got {kerf_mm}")
    
    # Warn if kerf is very large compared to element size
    if kerf_mm > element_size_mm:
        import warnings
        warnings.warn(
            f"{name} ({kerf_mm} mm) exceeds element size ({element_size_mm} mm). "
            "This may result in significant gaps between elements.",
            UserWarning,
        )
    return kerf_mm


def validate_subdivisions(no_sub_x: int, no_sub_y: int) -> Tuple[int, int]:
    """
    Validate subdivision counts.

    Parameters
    ----------
    no_sub_x : int
        Number of subdivisions in x-direction.
    no_sub_y : int
        Number of subdivisions in y-direction.

    Returns
    -------
    tuple of int
        (no_sub_x, no_sub_y) after validation.

    Raises
    ------
    ValueError
        If subdivisions are invalid.
    TypeError
        If subdivisions are not integers.
    """
    no_sub_x = validate_integer(no_sub_x, "no_sub_x", min_val=1)
    no_sub_y = validate_integer(no_sub_y, "no_sub_y", min_val=1)
    return no_sub_x, no_sub_y


def validate_focus_coordinates(
    focus_mm: Union[list, tuple, np.ndarray],
    allow_2d: bool = True,
) -> np.ndarray:
    """
    Validate and normalize focus coordinates.

    Parameters
    ----------
    focus_mm : sequence of float
        Focus coordinates in millimeters. Can be 2D [x, z] or 3D [x, y, z].
    allow_2d : bool, optional
        If True, allow 2D coordinates [x, z] (y=0 is assumed). Default is True.

    Returns
    -------
    ndarray, shape (3,)
        Focus coordinates as [x, y, z] in meters (converted from mm).

    Raises
    ------
    ValueError
        If focus coordinates are invalid.
    """
    focus = np.atleast_1d(focus_mm)
    
    if focus.ndim != 1:
        raise ValueError(
            f"Focus must be 1D array or sequence, got shape {focus.shape}"
        )

    if allow_2d and focus.size == 2:
        # Convert [x, z] to [x, 0, z]
        focus_3d = np.array([focus[0], 0.0, focus[1]])
    elif focus.size == 3:
        focus_3d = focus.copy()
    else:
        raise ValueError(
            f"Focus must have 2 [x, z] or 3 [x, y, z] elements, got {focus.size}"
        )

    return focus_3d * 1e-3  # Convert mm to meters


def validate_f_over_d(f_over_d: Optional[float], focus_depth_mm: float) -> float:
    """
    Validate F/D ratio.

    Parameters
    ----------
    f_over_d : float or None
        F/D ratio (focal length / diameter).
    focus_depth_mm : float
        Focal depth in millimeters (for sanity checks).

    Returns
    -------
    float
        The validated F/D ratio, or 1.0 if None.

    Raises
    ------
    ValueError
        If F/D ratio is invalid.
    """
    if f_over_d is None:
        return 1.0

    if f_over_d <= 0:
        raise ValueError(f"F/D ratio must be positive, got {f_over_d}")

    if f_over_d < 0.1:
        import warnings
        warnings.warn(
            f"F/D ratio ({f_over_d}) is very small. This may result in "
            "a very tight aperture.",
            UserWarning,
        )

    return float(f_over_d)


def validate_apodization_weights(
    weights: np.ndarray,
    n_elements: int,
    name: str = "apodization",
) -> np.ndarray:
    """
    Validate apodization weight array.

    Parameters
    ----------
    weights : ndarray
        Apodization weights.
    n_elements : int
        Expected number of elements.
    name : str, optional
        Parameter name for error messages.

    Returns
    -------
    ndarray
        The validated and normalized weights.

    Raises
    ------
    ValueError
        If weights are invalid.
    """
    weights = np.atleast_1d(weights).astype(float)

    if weights.size != n_elements:
        raise ValueError(
            f"{name} size ({weights.size}) must match n_elements ({n_elements})"
        )

    if np.any(weights < 0):
        raise ValueError(f"{name} must be non-negative everywhere")

    if np.all(weights == 0):
        import warnings
        warnings.warn(
            f"{name} is all zeros. No energy will be transmitted.",
            UserWarning,
        )

    return weights


def validate_delays(
    delays: np.ndarray,
    n_elements: int,
    name: str = "delays",
) -> np.ndarray:
    """
    Validate delay array.

    Parameters
    ----------
    delays : ndarray
        Delay values in seconds.
    n_elements : int
        Expected number of elements.
    name : str, optional
        Parameter name for error messages.

    Returns
    -------
    ndarray
        The validated delays.

    Raises
    ------
    ValueError
        If delays are invalid.
    """
    delays = np.atleast_1d(delays).astype(float)

    if delays.size != n_elements:
        raise ValueError(
            f"{name} size ({delays.size}) must match n_elements ({n_elements})"
        )

    if np.any(np.isnan(delays)) or np.any(np.isinf(delays)):
        raise ValueError(f"{name} contains NaN or inf values")

    return delays


def validate_speed_of_sound(c_mps: Optional[float] = None) -> float:
    """
    Validate speed of sound.

    Parameters
    ----------
    c_mps : float or None
        Speed of sound in m/s. If None, returns default.

    Returns
    -------
    float
        Speed of sound in m/s.

    Raises
    ------
    ValueError
        If speed of sound is invalid.
    """
    if c_mps is None:
        return 1540.0  # Default for soft tissue

    if c_mps <= 0:
        raise ValueError(f"Speed of sound must be positive, got {c_mps}")

    if c_mps < 800 or c_mps > 3000:
        import warnings
        warnings.warn(
            f"Speed of sound ({c_mps} m/s) is outside typical range [800, 3000]. "
            "Are you sure about this value?",
            UserWarning,
        )

    return float(c_mps)
