import torch


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

    # Return CV (want to minimize)
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
    mean_pressure = ((pressure**2 + 1e-1) / (pressure_ref**2 + 1e-1)).mean()
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
