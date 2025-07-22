import torch
import torch.nn.functional as F


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.

    Parameters:
    - size: int, the size of the kernel (must be odd).
    - sigma: float, the standard deviation of the Gaussian.

    Returns:
    - kernel: torch.Tensor, the 2D Gaussian kernel.
    """
    # Create a 1D Gaussian kernel
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()  # Normalize

    # Create a 2D Gaussian kernel by outer product
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()  # Normalize again

    return kernel_2d


def apply_gaussian_filter(
    tensor: torch.Tensor, kernel_size: int, sigma: float
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
    # Ensure the input tensor has the correct shape for conv2d (N, C, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Move the kernel to the same device as the input tensor
    kernel = kernel.to(tensor.device)

    # Apply the Gaussian filter using conv2d
    filtered_tensor = F.conv2d(tensor, kernel, padding=kernel_size // 2)

    # Remove the batch and channel dimensions
    return filtered_tensor.squeeze(0).squeeze(0)
