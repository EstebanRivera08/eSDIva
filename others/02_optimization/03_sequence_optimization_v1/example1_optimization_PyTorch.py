"""
Simple PyTorch Optimization Examples
=====================================
This script demonstrates how optimization works in PyTorch with clear, step-by-step
visualization of:
1. 1D function optimization: f(x) = x^2 (parabola - simple case)
2. 2D function optimization: f(x,y) = 10*sin(x+y)/(x+y) (complex landscape)

The script shows:
- How to set up a simple optimization loop in PyTorch
- How gradients are computed and used to update parameters
- How to visualize the optimization trajectory
- How general for loops work in Python with step-by-step iteration
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# =============================================================================
# PART 1: 1D OPTIMIZATION - Finding minimum of f(x) = x^2
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: 1D OPTIMIZATION - f(x) = x^2")
print("=" * 80)


# Create the function x^2
def function_1d(x):
    return x**2


# Step 1: Initialize parameter
x = torch.tensor([3.0], requires_grad=True)
# We turn True the flag requires_grad to do optimization
print(f"Initial x: {x.item():.4f}")
print("Minimizing f(x) = x^2, which has its minimum at x = 0")

# Step 2: Set up optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD([x], lr=learning_rate)
# The optimizer is SGD (Stochastic Gradient Descent) and we pass the parameter x to
# optimize, along with the learning rate which controls how big the steps are in the
# optimization process

# Step 3: Optimization loop with visualization
num_epochs = 20
x_history = [x.item()]  # Store history for visualization
loss_history = []
gradient_history = []

for epoch in tqdm(range(num_epochs), desc="Optimizing 1D function", unit="epoch"):
    # 1) Zero gradients from previous iteration
    optimizer.zero_grad()

    # 2) Forward pass: compute loss
    loss = function_1d(x)  # we want to minimize f(x) = x^2, so we compute the loss as
    # the value of the function at x

    # 3) Backward pass: compute gradients
    loss.backward()

    # 4) Update parameters using optimizer
    optimizer.step()

    # Store history
    x_history.append(x.item())
    loss_history.append(loss.item())

    # Print progress every 5 epochs
    gradient = x.grad.item() if x.grad is not None else 0
    gradient_history.append(gradient)


print("\n Optimization complete!")
print(f"  Final x: {x.item():.4f}")
print(f"  Final loss (final f(x)): {loss_history[-1]:.6f}")
print(f"  Started at x = {x_history[0]:.4f}, ended at x = {x_history[-1]:.4f}")

# Visualization for 1D

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Function and optimization path
ax = axes[0]
x_plot = np.linspace(-3.5, 3.5, 200)
y_plot = function_1d(torch.tensor(x_plot)).detach().numpy()
x_history = np.array(x_history)
y_history = np.array(loss_history)
loss_history = np.array(loss_history)

ax.plot(x_plot, y_plot, "k-", linewidth=2, label="f(x) = x²", zorder=0)
ax.plot(
    x_history[:-1],
    loss_history,
    "ro-",
    linewidth=1.5,
    markersize=5,
    label="Optimization steps",
)
ax.plot(x_history[0], loss_history[0], "g*", markersize=20, label="Start", zorder=3)
ax.plot(
    x_history[-2],
    loss_history[-1],
    "r*",
    markersize=20,
    label="End (minimum)",
)

# Plot gradient arrows using the gradient history
for i in range(len(x_history) - 1):
    x_i = x_history[i]
    y_i = loss_history[i]
    x_next = x_i + (-gradient_history[i] * learning_rate)  # Next x based on gradient
    y_next = function_1d(torch.tensor(x_next)).item()  # Next loss based on new x
    dx = x_next - x_i
    dy = y_next - y_i
    ax.arrow(
        x_i,
        y_i,
        dx,
        dy,
        length_includes_head=True,
        head_width=0.2,
        head_length=0.3,
        fc="blue",
        ec="blue",
        alpha=0.8,
        zorder=10,
    )
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)
ax.set_title("1D Optimization: f(x) = x²", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xlim([-3.5, 3.5])

# Plot 2: Loss over epochs
ax = axes[1]
ax.plot(range(len(loss_history)), loss_history, "b-o", linewidth=2, markersize=6)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss f(x)", fontsize=12)
ax.set_title("Loss Decrease Over Time", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

plt.tight_layout()
plt.show()


# =============================================================================
# PART 2: 2D OPTIMIZATION - Finding minimum of f(x,y) = 10*sin(x+y)/(x+y)
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: 2D OPTIMIZATION - f(x,y) = 10*sin(x+y)/(x+y)")
print("=" * 80)


def function_2d(x, y):
    # Avoid exact zero - use small epsilon for safety
    result = torch.sin(x) * torch.cos(y) * torch.exp(-0.1 * (x**2 + y**2))
    return result


# Step 1: Initialize parameters
x_2d = torch.tensor([1.0], requires_grad=True)
y_2d = torch.tensor([3.0], requires_grad=True)
print(f"  Initial: x = {x_2d.item():.4f}, y = {y_2d.item():.4f}")

# Step 2: Set up optimizer
learning_rate_2d = 0.1
optimizer_2d = torch.optim.Adam([x_2d, y_2d], lr=learning_rate_2d)
print(f"  Optimizer: Adam with learning rate = {learning_rate_2d}")
# Adam is an adaptive optimizer that adjusts learning rates for each parameter based on
# estimates of first and second moments of the gradients, which can lead to faster
# convergence in complex landscapes compared to simple SGD.

# Step 3: Optimization loop
print("-" * 80)

num_epochs_2d = 200
x_2d_history = [x_2d.item()]
y_2d_history = [y_2d.item()]
loss_2d_history = []
gradient_2d_history = []

print("\nRunning 2D optimization:\n")

for epoch in tqdm(range(num_epochs_2d), desc="Optimizing 2D function", unit="epoch"):
    # 1) Zero gradients
    optimizer_2d.zero_grad()

    # 2) Forward pass: compute loss (we want to maximize f(x,y), so we minimize -f(x,y))
    loss_2d = -1 * function_2d(x_2d, y_2d)

    # 3) Backward pass: compute gradients
    loss_2d.backward()

    # 4) Update parameters
    optimizer_2d.step()

    x_2d_history.append(x_2d.item())
    y_2d_history.append(y_2d.item())
    loss_2d_history.append(loss_2d.item())
    gradient_2d_history.append((x_2d.grad.item(), y_2d.grad.item()))

print("\n Optimization complete!")
print(f"  Final: x = {x_2d.item():.4f}, y = {y_2d.item():.4f}")
print(f"  Final loss: {loss_2d_history[-1]:.6f}")

# Visualization for 2D

fig = plt.figure(figsize=(16, 5))

# Create 2D grid for plotting
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Compute function values on grid
Z_grid = function_2d(torch.tensor(X_grid), torch.tensor(Y_grid)).detach().numpy()

# Plot 1: 3D surface
# take the list and turn them into arrays for plotting
x_2d_history = np.array(x_2d_history)
y_2d_history = np.array(y_2d_history)
loss_2d_history = np.array(loss_2d_history)

ax1 = fig.add_subplot(131, projection="3d")
surf = ax1.plot_surface(X_grid, Y_grid, Z_grid, cmap="viridis", alpha=0.8)
ax1.plot(
    x_2d_history[:-1],
    y_2d_history[:-1],
    -1 * loss_2d_history,
    "ro-",
    linewidth=2,
    markersize=5,
)
ax1.plot(
    [x_2d_history[0]],
    [y_2d_history[0]],
    [-1 * loss_2d_history[0]],
    "g*",
    markersize=15,
    label="Start",
)
ax1.plot(
    [x_2d_history[-2]],
    [y_2d_history[-2]],
    [-1 * loss_2d_history[-1]],
    "r*",
    markersize=15,
    label="End",
)
ax1.set_xlabel("x", fontsize=10)
ax1.set_ylabel("y", fontsize=10)
ax1.set_zlabel("f(x,y)", fontsize=10)
ax1.set_title("3D Surface: f(x,y)", fontsize=12, fontweight="bold")
ax1.legend()
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Plot 2: Contour plot with trajectory
ax2 = fig.add_subplot(132)
contour = ax2.contour(X_grid, Y_grid, Z_grid, levels=15, cmap="viridis")
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(x_2d_history, y_2d_history, "ro-", linewidth=1.5, markersize=4, label="Path")
ax2.plot(x_2d_history[0], y_2d_history[0], "g*", markersize=20, label="Start")
ax2.plot(x_2d_history[:-2], y_2d_history[:-2], "r*", markersize=20, label="End")
ax2.set_xlabel("x", fontsize=10)
ax2.set_ylabel("y", fontsize=10)
ax2.set_title("Contour Plot with Optimization Path", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Loss over epochs
ax3 = fig.add_subplot(133)
ax3.plot(range(len(loss_2d_history)), loss_2d_history, "b-o", linewidth=2, markersize=4)
ax3.set_xlabel("Epoch", fontsize=10)
ax3.set_ylabel("Loss f(x,y)", fontsize=10)
ax3.set_title("Loss Decrease Over Time", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# PART 3: COMPARISON AND EXPLANATION OF LOOP MECHANICS
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: EXPLANATION OF OPTIMIZATION LOOP MECHANICS")
print("=" * 80)

explanation = """
OPTIMIZATION LOOP ANATOMY:
--------------------------
Step-by-step what happens at each iteration:

1. optimizer.zero_grad()
   - Resets gradients from previous iteration
   - Without this, gradients would accumulate

2. loss = function(parameters)
   - Compute loss/function value
   - This creates a computation graph

3. loss.backward()
   - Computes gradients using backpropagation
   - For f(x) = x², gradient df/dx = 2x
   - Stores gradients in parameter.grad

4. optimizer.step()
   - Updates parameters using gradients
   - For SGD: x = x - learning_rate * gradient
   - For Adam: more complex adaptive update

KEY CONCEPTS:
-------------
• The learning rate controls step size
  - Too small: slow convergence
  - Too large: may overshoot or diverge

• Gradients point in direction of increase
  - We move opposite (negative) to decrease loss

• Different optimizers use gradients differently
  - SGD: simple gradient descent
  - Adam: adaptive per-parameter learning rates
  - Momentum: considers previous steps

PYTORCH FEATURES USED:
---------------------
• requires_grad=True: enables gradient tracking
• .backward(): automatic differentiation
• optimizer.step(): parameter update
• .detach(): remove from computation graph
"""

print(explanation)

# Create a side-by-side comparison visualization

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1D results
ax = axes[0, 0]
x_plot = np.linspace(-3.5, 3.5, 200)
y_plot = function_1d(torch.tensor(x_plot)).detach().numpy()
ax.plot(x_plot, y_plot, "b-", linewidth=2)
ax.plot(x_history[:-1], loss_history, "ro-", linewidth=1.5, markersize=5)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("1D: f(x) = x²", fontweight="bold")
ax.grid(True, alpha=0.3)

# 1D - trajectory in x space
ax = axes[0, 1]
ax.plot(range(len(x_history)), x_history, "b-o", linewidth=2, markersize=5)
ax.axhline(y=0, color="g", linestyle="--", label="Minimum")
ax.set_xlabel("Epoch")
ax.set_ylabel("x value")
ax.set_title("1D: Parameter Over Time", fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend()

# 2D - contour with path
ax = axes[1, 0]
contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=15, cmap="viridis")
ax.plot(x_2d_history, y_2d_history, "ro-", linewidth=1.5, markersize=4)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D: Optimization Path on Contours", fontweight="bold")
fig.colorbar(contour, ax=ax)

# 2D - trajectory in parameter space
ax = axes[1, 1]
ax.plot(
    range(len(x_2d_history)), x_2d_history, "r-o", linewidth=2, label="x", markersize=4
)
ax.plot(
    range(len(y_2d_history)), y_2d_history, "b-s", linewidth=2, label="y", markersize=4
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Parameter value")
ax.set_title("2D: Parameters Over Time", fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
