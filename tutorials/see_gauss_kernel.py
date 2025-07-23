import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from helper_function import gaussian_kernel

# Generate a Gaussian kernel
kernel = gaussian_kernel(3, 0.5)
print(kernel.sum())


# Visualize the kernel
plt.imshow(kernel.detach().cpu().numpy(), cmap="gray")
plt.title("Gaussian Kernel")
plt.colorbar()
plt.show()
