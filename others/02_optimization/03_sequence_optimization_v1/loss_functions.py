import torch

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

loss_MSE = torch.nn.MSELoss()


def dB(x, max_val=None, min_val=1e-20):
    """Convert linear to dB scale."""
    if max_val is None:
        max_val = x.max()

    x_norm = x / max_val
    return 20 * torch.log10(x_norm + min_val)


# ============================================================================
# v3 Loss — log-mean coverage (replaces sigmoid coverage)
# ============================================================================
#
#
# def compute_log_coverage_loss(pr_field, eps=1e-3):
#     """
#     Log-mean (geometric mean) coverage loss.
#
#     Maximizes the geometric mean of the pressure field.  Equivalent to
#     minimising -mean(log(pr)).
#
#     Why this works when sigmoid fails:
#       - sigmoid gradient at dark regions (pr≈0) is **zero** → optimizer blind
#       - log gradient at dark regions is **-1/(pr+eps)** → very strong
#       - dark regions actively "pull" the beam toward them
#
#     Think of it as: every field point votes "I want more pressure" with a
#     voice proportional to 1/pressure.  Quiet points shout the loudest.
#
#     Parameters
#     ----------
#     pr_field : Tensor [nx, ny, nz]
#         Pressure field (raw or normalised, both work)
#     eps : float
#         Floor preventing log(0).  Smaller = more aggressive pull from dark
#         regions, but can cause gradient explosion.  1e-3 is a good start.
#
#     Returns
#     -------
#     Tensor (scalar)
#         Negative log-mean (minimise this → maximise geometric mean)
#     """
#     return -torch.log(pr_field + eps).mean()
#
#
# def compute_lateral_uniformity_loss(pr_field):
#     """
#     Lateral uniformity per depth slice, then averaged.
#
#     Why better: depth decay is physics (can't fix). What VS positions control
#     is lateral spread. CV per z-slice isolates that.
#
#     Parameters
#     ----------
#     pr_field : Tensor [nx, ny, nz]
#         Pressure field (not necessarily normalized)
#
#     Returns
#     -------
#     Tensor (scalar)
#         Mean CV across depth slices (lower = more uniform laterally)
#     """
#     # Take the y=0 slice → [nx, nz]
#     pr_2d = pr_field[:, pr_field.shape[1] // 2, :]
#
#     # Per depth-column statistics
#     col_mean = pr_2d.mean(dim=0)  # [nz]
#     col_std = pr_2d.std(dim=0)  # [nz]
#
#     # CV per depth, only where mean is meaningful (avoid near-zero depth slices)
#     cv_per_depth = col_std / (col_mean + 1e-6)
#
#     return cv_per_depth.mean()
#


def compute_symmetry_loss(pr_field, *, pr_max=None, db_scale=False):
    """
    Symmetry loss: mean absolute difference between left and right halves.

    Why: encourages symmetric beams, which are often desirable in imaging.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Pressure field (not necessarily normalized)

    Returns
    -------
    Tensor (scalar)
        Mean absolute difference between left and right halves (lower = more symmetric)
    """
    # Take the y=0 slice → [nx, nz]
    pr_2d = pr_field[:, pr_field.shape[1] // 2, :]

    # Split into left and right halves
    if pr_max is None:
        pr_max = pr_2d.max()

    mid_x = pr_2d.shape[0] // 2
    left_half = pr_2d[:mid_x, :] / pr_max  # [nx//2, nz]
    right_half = pr_2d[mid_x + 1 :, :] / pr_max  # [nx//2, nz]

    # Flip right half for symmetry comparison
    right_half_flipped = torch.flip(right_half, dims=[0])  # [nx//2, nz]

    # Compute mean absolute difference
    if db_scale:
        left_half = dB(left_half)
        right_half_flipped = dB(right_half_flipped)

    symmetry_loss = loss_MSE(left_half, right_half_flipped)

    return symmetry_loss


def compute_soft_coverage_loss(pr_field, *, pr_max=None, threshold=-6, steepness=5.0):
    """
    Differentiable coverage using sigmoid.

    Each field point contributes sigmoid(steepness * (pr_dB - threshold)) ∈ [0, 1].
    Mean over field gives fraction above threshold. Bounded by construction.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Pressure field
    pr_max : float or Tensor
        Reference max pressure for dB conversion
    threshold : float
        Coverage threshold in dB (e.g., -6, -15)
    steepness : float
        Sigmoid sharpness. 5-10 = good compromise (gradient within ±2-4 dB of threshold)

    Returns
    -------
    Tensor (scalar)
        Coverage loss: 1 - mean(sigmoid), in [0, 1]. Minimize → maximize coverage.
    """
    pr_dB = dB(pr_field, max_val=pr_max)

    soft_above = torch.sigmoid(steepness * (pr_dB - threshold))
    soft_coverage = soft_above.mean()  # fraction in [0, 1]

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
    n_elements = apod_list[0].shape[0] if apod_list else 1
    total = (
        sum(apod.sum() for apod in apod_list) / len(apod_list) / n_elements
    )  # Average
    return total


def compute_mean_energy_loss(pr_field, pr_max=1, apod_list=1):
    """
    Negative mean pressure — want to maximize field energy.

    Prevents the optimizer from trivially achieving "uniformity" by
    making everything equally low.

    Parameters
    ----------
    pr_field : Tensor [nx, ny, nz]
        Pressure field
    apod_list : list of Tensor [n_elements]
        Per-VS apodization vectors (not necessarily normalized)

    Returns
    -------
    Tensor (scalar)
        - meanprssure/mean(apod), minimize to increase energy per active element.
    """
    if isinstance(apod_list, list):
        total_apod = sum(apod.sum() for apod in apod_list)
    else:
        total_apod = apod_list

    log_mean_pr = torch.log(pr_field + 1e-20).mean()  # Avoid log(0)
    return 1 - log_mean_pr / pr_max  # Avoid division by zero


# ============================================================================
# v4 Resolution Losses
# - compute_resolution_loss: geometric f-number (cheap, no simulation)
# - Coherence Factor (CF): computed in VirtualSourceOptimizer.get_combined_field
#   replaces compute_angular_diversity_loss (kept below for reference)
# ============================================================================


def compute_resolution_loss(
    vs_positions, z_field_range, lambda_m, D_physical_m, target_fnumber=1.5
):
    """
    Effective f-number penalty based on virtual aperture geometry.

    Penalizes configurations where the effective synthetic f-number exceeds
    a target (poor lateral resolution). Operates on VS geometry only — no
    forward simulation needed (cheap).

    Physics:
        D_eff(z) = D_physical + D_virtual * z / |z_vs_mean|
        F# = z / D_eff
        FWHM_lateral ~ 1.4 * lambda * F#

    Parameters
    ----------
    vs_positions : list of Tensor [2]
        Virtual source positions [x_mm, z_mm] for each VS.
    z_field_range : tuple (z_min_mm, z_max_mm)
        Depth range of imaging region (mm).
    lambda_m : float
        Wavelength in metres.
    D_physical_m : float
        Physical aperture width in metres.
    target_fnumber : float
        Target f-number. Lower = better resolution. F#=1 is clinical standard.

    Returns
    -------
    Tensor (scalar)
        Mean excess f-number (minimize this → better resolution).
    """
    vs_pos = torch.stack(vs_positions)  # [N, 2]
    x_vs = vs_pos[:, 0] * 1e-3  # metres
    z_vs = torch.abs(vs_pos[:, 1]) * 1e-3  # metres (positive distance behind)

    # Virtual aperture span
    D_virtual = x_vs.max() - x_vs.min()

    # Mean VS depth behind array
    z_vs_mean = z_vs.mean()

    # Sample field depths
    z_min = z_field_range[0] * 1e-3
    z_max = z_field_range[1] * 1e-3
    z_points = torch.linspace(z_min, z_max, steps=20, device=vs_pos.device)

    # Effective aperture at each depth: D_eff(z) = D_phys + D_virtual * z / |z_vs|
    D_eff = D_physical_m + D_virtual * z_points / (z_vs_mean + 1e-6)

    # Effective f-number
    f_number_eff = z_points / (D_eff + 1e-6)

    # Penalize only when F# exceeds target (softplus = smooth ReLU)
    excess = torch.nn.functional.softplus(f_number_eff - target_fnumber)

    return excess.mean()


def compute_angular_diversity_loss(vs_positions, z_field_center_mm):
    """
    Angular diversity (repulsion) loss.

    Penalizes small pairwise angular separations between virtual sources
    as seen from the field center. Prevents VS collapse to identical positions.

    Strong gradient when angles approach zero (log penalty).

    Parameters
    ----------
    vs_positions : list of Tensor [2]
        Virtual source positions [x_mm, z_mm] for each VS.
    z_field_center_mm : float
        Representative depth of field center (mm).

    Returns
    -------
    Tensor (scalar)
        Negative log of mean pairwise angular separation.
        Minimize → maximize angular diversity.
    """
    vs_pos = torch.stack(vs_positions)  # [N, 2]
    N = vs_pos.shape[0]

    if N < 2:
        return torch.tensor(0.0, device=vs_pos.device)

    x_vs = vs_pos[:, 0]  # mm
    z_vs = vs_pos[:, 1]  # mm (negative = behind array)

    # Angle from field center to each VS
    # Field center at (0, z_field_center), VS at (x_vs, z_vs)
    angles = torch.atan2(x_vs, z_field_center_mm - z_vs)  # [N] radians

    # Pairwise angular differences (upper triangle only)
    angle_diffs = (angles.unsqueeze(0) - angles.unsqueeze(1)).abs()  # [N, N]
    mask = torch.triu(torch.ones(N, N, device=vs_pos.device), diagonal=1).bool()
    pairwise = angle_diffs[mask]  # [N*(N-1)/2]

    # Negative log: strong gradient when angles collapse
    mean_separation = pairwise.mean()
    return -torch.log(mean_separation + 1e-6)
