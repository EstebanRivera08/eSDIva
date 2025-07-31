import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# from helper_function import gaussian_kernel

# # Generate a Gaussian kernel
# kernel = gaussian_kernel(3, 0.5)
# print(kernel.sum())


# # Visualize the kernel
# plt.imshow(kernel.detach().cpu().numpy(), cmap="gray")
# plt.title("Gaussian Kernel")
# plt.colorbar()
# plt.show()


target_folder = r".\target_masks"
target_filename = r"/linear_lambda2.npz"


target_dic = np.load(target_folder + target_filename)
target_matrix = target_dic["target"]
wavelength = target_dic["wavelength"]
x_length_mm = target_dic["x_length_mm"]
y_length_mm = target_dic["y_length_mm"]
dx = target_dic["dx"]
dy = target_dic["dy"]

extent = [-x_length_mm / 2, x_length_mm / 2, -y_length_mm / 2, y_length_mm / 2]
plt.imshow(target_matrix.T, cmap="gray", interpolation=None, extent=extent)
plt.title("Target Mask")
plt.colorbar()
plt.xlabel("X-axis (mm)")
plt.ylabel("Y-axis (mm)")
plt.show()
