import torch

from pyfield.utilities import to_dB

# ============================================================================
# v1 Loss Functions (original — free apodization, global CV, hard threshold)
# ============================================================================

# def compute_element_usage_penalty(apodization, sparsity_weight=0.1):
#     """L1 norm on apodization to encourage sparsity."""
#     return sparsity_weight * apodization.abs().mean()
#
#
# def compute_uniformity_loss(pr_field, target_region_mask=None):
#     """Global CV = std/mean. Problem: dominated by depth decay, not lateral spread."""
#     if target_region_mask is not None:
#         field_roi = pr_field * target_region_mask
#     else:
#         field_roi = pr_field
#     mean_val = field_roi.mean()
#     std_val = field_roi.std()
#     cv = std_val / (mean_val + 1e-1)  # epsilon too large
#     return cv
#
#
# def compute_transmit_energy(apodization, pressure, pressure_ref=1):
#     """Negative of mean_pressure / mean_apod. Was commented out in the loop."""
#     mean_pressure = ((pressure**2 + 1e-1) / (pressure_ref**2 + 1e-1)).mean()
#     energy = mean_pressure / apodization.mean()
#     return -energy
#
#
# def compute_coverage_loss(pr_field, threshold=0.3):
#     """Hard threshold → zero gradient everywhere. Optimizer blind to this term."""
#     coverage = (pr_field > threshold).to(torch.float32).mean()
#     return -coverage


# ============================================================================
# v2 Loss Functions (derived apodization, lateral CV, soft coverage)
# ============================================================================
def dB(x):
    """Convert linear to dB scale."""
    x_norm = x / x.max()
    return 20 * torch.log10(x_norm + 1e-6)


# ============================================================================
# v3 Loss — log-mean coverage (replaces sigmoid coverage)
# ============================================================================


def compute_log_coverage_loss(pr_field, eps=1e-3):
    """
    Log-mean (geometric mean) coverage loss.

    Maximizes the geometric mean of the pressure field.  Equivalent to
    minimising -mean(log(pr)).

    Why this works when sigmoid fails:
      - sigmoid gradient at dark regions (pr≈0) is **zero** → optimizer blind
      - log gradient at dark regions is **-1/(pr+eps)** → very strong
      - dark regions actively "pull" the beam toward them

    Think of it as: every field point votes "I want more pressure" with a
    voice proportional to 1/pressure.  Quiet points shout the loudest.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Pressure field (raw or normalised, both work)
    eps : float
        Floor preventing log(0).  Smaller = more aggressive pull from dark
        regions, but can cause gradient explosion.  1e-3 is a good start.

    Returns
    -------
    Tensor (scalar)
        Negative log-mean (minimise this → maximise geometric mean)
    """
    return -torch.log(pr_field + eps).mean()


def compute_lateral_uniformity_loss(pr_field):
    """
    Lateral uniformity per depth slice, then averaged.

    Why better: depth decay is physics (can't fix). What VS positions control
    is lateral spread. CV per z-slice isolates that.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Pressure field (not necessarily normalized)

    Returns
    -------
    Tensor (scalar)
        Mean CV across depth slices (lower = more uniform laterally)
    """
    # Take the y=0 slice → [nx, nz]
    pr_2d = pr_field[:, pr_field.shape[1] // 2, :]

    # Per depth-column statistics
    col_mean = pr_2d.mean(dim=0)  # [nz]
    col_std = pr_2d.std(dim=0)  # [nz]

    # CV per depth, only where mean is meaningful (avoid near-zero depth slices)
    cv_per_depth = col_std / (col_mean + 1e-6)

    return cv_per_depth.mean()


def compute_soft_coverage_loss(pr_field, threshold=-6, steepness=20.0):
    """
    Differentiable coverage using sigmoid instead of hard threshold.

    Why better: hard threshold ``(pr > t)`` has zero gradient → optimizer
    blind.  Sigmoid provides smooth gradient everywhere.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Normalized pressure field [0, 1]
    threshold : float
        Target level (0.5 ≈ -6 dB)
    steepness : float
        Sigmoid sharpness (higher = closer to hard threshold but noisier grad)

    Returns
    -------
    Tensor (scalar)
        Coverage loss (1 - soft_fraction_above_threshold), minimize to 0.
    """
    pr_dB = dB(pr_field)

    soft_above = torch.sigmoid(steepness * (pr_dB - threshold))

    soft_coverage = soft_above.sum() / soft_above.numel()  # fraction
    return 1.0 - soft_coverage


def compute_aperture_cost(apod_list):
    """
    Total active-element cost across all virtual source firings.

    With F/D=1 derived apodization, deeper VS → wider aperture → more
    active elements → higher cost.  This creates healthy tension with
    uniformity/coverage (which want wide beams = deep VS).

    Parameters
    ----------
    apod_list : list of Tensor [n_elements]
        Per-VS apodization vectors (derived from VS position)

    Returns
    -------
    Tensor (scalar)
        Normalized total active elements [0, 1]
    """
    total = sum(apod.sum() for apod in apod_list)
    n_elements = apod_list[0].shape[0]
    n_vs = len(apod_list)
    return total / (n_elements * n_vs)


def compute_mean_energy_loss(pr_field):
    """
    Negative mean pressure — want to maximize field energy.

    Prevents the optimizer from trivially achieving "uniformity" by
    making everything equally low.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Pressure field

    Returns
    -------
    Tensor (scalar)
        Negative mean (minimize → maximize energy)
    """
    return -pr_field.mean()
