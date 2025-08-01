import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# z_weights = z_weights / z_weights.max()  # Normalize weights to sum to 1
def gaussian_1d(
    size: torch.Tensor,
    sigma: float,
    *,
    plot: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute a 1D Gaussian function.

    Parameters:
    - x: torch.Tensor, the input tensor.
    - sigma: float, the standard deviation of the Gaussian.

    Returns:
    - gaussian: torch.Tensor, the computed Gaussian values.

    """
    if size > 1:
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize to sum to 1
    else:
        kernel_1d = torch.tensor(1)

    if plot:
        plt.plot(kernel_1d.cpu().numpy(), label=f"Gaussian 1D (σ={sigma})")
        plt.xlabel("Index")
        plt.ylabel("Weight")
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()
    return kernel_1d.to(device)  # Normalize to sum to 1


def pattern_from_pr_3Dto2D(pressure, max, *, sigma=0.5):
    # Apply the weights to the pressure tensor along the z-axis
    """
    Create a focalization mask from a pressure field.
    Parameters:
    - pressure: torch.Tensor, the pressure field tensor (shape: [x_len, y_len, z_len]).
    - max: float, the maximum value for normalization.
    - size: int, the size of the Gaussian kernel (default: 5).
    - sigma: float, the standard deviation of the Gaussian (default: 0.5).
    Returns:
    - focal_mask: torch.Tensor, the focalization mask (shape: [x_len, y_len]).
    """
    size = pressure.shape[-1]  # Use z dimension as size
    # Create a Gaussian kernel for the z-axis
    z_weights = gaussian_1d(size, sigma, device=pressure.device)
    z_weights = z_weights.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, z_len]
    pressure_disk = (pressure * z_weights).sum(dim=-1)  # Weighted sum along the z-axis
    pressure_disk = pressure_disk / max  # Normalize

    # Use a differentiable thresholding operation
    focal_mask = torch.sigmoid(10 * (pressure_disk - 0.5))

    return focal_mask.squeeze()  # Remove z dimension


def stack_2D_to_3D(matrix_2D: torch.Tensor, nz: int, *, sigma=0.5) -> torch.Tensor:
    """
    Stack a 2D matrix along a new dimension to create a 3D matrix.

    Parameters:
    - matrix_2D: torch.Tensor, the input 2D matrix (shape: [H, W]).
    - depth: int, the size of the new dimension.

    Returns:
    - matrix_3D: torch.Tensor, the resulting 3D matrix (shape: [H, W, depth]).
    """
    # Add a new dimension at the end
    matrix_3D = matrix_2D.unsqueeze(-1)  # Shape: [H, W, 1]

    # Repeat the matrix along the new dimension
    matrix_3D = matrix_3D.repeat(1, 1, nz)  # Shape: [H, W, depth]

    # Create a Gaussian kernel for the z-axis
    if sigma != 0:
        z_weights = gaussian_1d(nz, sigma, device=matrix_2D.device)
        # Normalize the weights to its max, so that at the center of the z-axis, the weights are 1
        # and at the edges, the weights are 0
        z_weights = (
            z_weights.unsqueeze(0).unsqueeze(0) / z_weights.max()
        )  # Shape: [1, 1, depth]
    else:
        z_weights = torch.zeros(nz, device=matrix_2D.device)
        z_weights[nz // 2] = 1.0  # Set the center weight to 1
        z_weights = z_weights.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, depth]

    # Apply the Gaussian weights to the 3D matrix
    matrix_3D = matrix_3D * z_weights

    return matrix_3D.squeeze(0).squeeze(0)


def pattern_from_pr_3Dto3D(pressure, max):
    # Apply the weights to the pressure tensor along the z-axis
    """
    Create a focalization mask from a pressure field.
    Parameters:
    - pressure: torch.Tensor, the pressure field tensor (shape: [B, C, H, W, z_len]).
    - max: float, the maximum value for normalization.
    - size: int, the size of the Gaussian kernel (default: 5).
    - sigma: float, the standard deviation of the Gaussian (default: 0.5).
    Returns:
    - focal_mask: torch.Tensor, the focalization mask (shape: [B, C, H, W]).
    """
    pressure_disk = pressure / max  # Normalize

    # Use a differentiable thresholding operation
    focal_mask = torch.sigmoid(10 * (pressure_disk - 0.5))

    return focal_mask


def gaussian_filter_1d(
    tensor: torch.Tensor, *, kernel_size: int = 3, sigma: float = 0.5
) -> torch.Tensor:
    """
    Apply a Gaussian filter to a 1D tensor.
    Parameters:
    - tensor: torch.Tensor, the input 1D tensor (shape: [N]).
    - kernel_size: int, the size of the Gaussian kernel (must be odd).
    - sigma: float, the standard deviation of the Gaussian.
    Returns:
    - filtered_tensor: torch.Tensor, the filtered 1D tensor.
    """
    # Ensure tensor is a 1D tensor
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be a 1D tensor (N,)")
    if kernel_size // 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd

    # Ensure the input tensor has the correct shape for conv1d (N, C, L)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # Create the Gaussian kernel
    kernel = gaussian_1d(kernel_size, sigma, plot=False, device=tensor.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # Apply the Gaussian filter using conv1d
    filtered_tensor = F.conv1d(tensor, kernel, padding=kernel_size // 2)
    # Remove the batch and channel dimensions
    return filtered_tensor.squeeze(0).squeeze(0)  # Flatten the output tensor


def gaussian_2d(
    size: torch.Tensor,
    sigma: float,
    *,
    plot: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute a 2D Gaussian function.

    Parameters:
    - size: torch.Tensor, the size of the Gaussian kernel (must be odd).
    - sigma: float, the standard deviation of the Gaussian.

    Returns:
    - gaussian: torch.Tensor, the computed 2D Gaussian values.
    """
    kernel_1d = gaussian_1d(size, sigma)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)  # Outer product to create 2D Gaussian
    kernel_2d = kernel_2d / kernel_2d.sum()  # Normalize to sum to 1
    if plot:
        plt.imshow(kernel_2d.cpu().numpy(), cmap="hot", interpolation=None)
        plt.title(f"Gaussian 2D (σ={sigma})")
        plt.colorbar()
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()
        plt.show()
        plt.close()
    return kernel_2d.to(device)  # Normalize to sum to 1


def gaussian_filter_2d(
    tensor: torch.Tensor, *, kernel_size: int = 3, sigma: float = 0.5
) -> torch.Tensor:
    """
    Apply a Gaussian filter to a 2D tensor.

    Parameters:
    - tensor: torch.Tensor, the input 2D tensor (shape: [H, W]).
    - kernel_size: int, the size of the Gaussian kernel (must be odd).
    - sigma: float, the standard deviation of the Gaussian.

    Returns:
    - filtered_tensor: torch.Tensor, the filtered 2D tensor.
    """
    # Ensure tensor is a 2D tensor
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be a 2D tensor (H, W)")
    if kernel_size // 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd

    # Ensure the input tensor has the correct shape for conv2d (N, C, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Create the Gaussian kernel
    kernel = gaussian_2d(kernel_size, sigma, plot=False, device=tensor.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Apply the Gaussian filter using conv2d
    filtered_tensor = F.conv2d(tensor, kernel, padding=kernel_size // 2)

    # Remove the batch and channel dimensions
    return filtered_tensor.squeeze(0).squeeze(0)  # Flatten the output tensor
