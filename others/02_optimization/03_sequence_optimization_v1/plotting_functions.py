import matplotlib.pyplot as plt
import numpy as np

from pyfield.utilities import to_dB


def plot_virtual_source_results(results, output_file=None):
    """Plot virtual source optimization results."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3)

    # Loss history
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results["loss_history"], "r", label="Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss")
    ax1.legend()
    ax1.grid(True)

    # Individual loss components
    def _norm_btwn_0_and_1(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(_norm_btwn_0_and_1(results["uniformity_history"]), label="Uniformity")
    ax2.plot(_norm_btwn_0_and_1(results["sparsity_history"]), label="Sparsity")
    ax2.plot(_norm_btwn_0_and_1(results["coverage_history"]), label="Coverage")
    ax2.plot(_norm_btwn_0_and_1(results["energy_history"]), label="Energy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Component")
    ax2.set_title("Loss Components")
    ax2.legend()
    ax2.grid(True)

    # Total apodization
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(results["apodization_total"], "ok-")
    ax3.axhline(0.1, color="r", linestyle="--", label="Threshold")
    ax3.set_xlabel("Element Index")
    ax3.set_ylabel("Total Apodization")
    ax3.set_title("Element Usage (Combined)")
    ax3.legend()
    ax3.grid(True)

    # Virtual source positions
    ax4 = fig.add_subplot(gs[1, 0])
    vs_pos = results["virtual_source_positions"]
    vs_history = results["virtual_source_positions_history"]

    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    if vs_history.shape[1] > len(colors):
        # add random colors if more virtual sources than predefined colors
        np.random.seed(0)
        colors = colors + [
            np.random.rand(
                3,
            )
            for _ in range(vs_history.shape[1] - len(colors))
        ]

    for vs_idx in range(vs_history.shape[1]):
        x_values = vs_history[:, vs_idx, 0].squeeze()
        z_values = vs_history[:, vs_idx, 1].squeeze()
        ax4.plot(
            x_values,
            z_values,
            "-",
            label=f"x [vs_idx={vs_idx}]",
            color=colors[vs_idx],
        )
    ax4.scatter(vs_pos[:, 0], vs_pos[:, 1], s=100, c="black", marker="o")
    for i, pos in enumerate(vs_pos):
        ax4.annotate(
            f"VS{i}", (pos[0], pos[1]), xytext=(5, 5), textcoords="offset points"
        )
    ax4.axhline(0, color="k", linestyle="-", linewidth=2, label="Array")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Z (mm)")
    ax4.set_title("Virtual Source Positions")
    ax4.legend()
    ax4.grid(True)
    ax4.axis("equal")

    # Pressure field (XZ plane)
    ax5 = fig.add_subplot(gs[1, 1])
    pr = results["pressure_final"]
    x, y, z = results["x"], results["y"], results["z"]

    y_center = len(y) // 2
    pr_xz = pr[:, y_center, :]
    pr_db = to_dB(pr_xz)

    extent = [x.min(), x.max(), z.min(), z.max()]
    im5 = ax5.imshow(
        pr_db.T,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="hot",
        vmin=-40,
        vmax=0,
    )
    ax5.set_xlabel("Z (mm)")
    ax5.set_ylabel("X (mm)")
    ax5.set_title("Pressure Field (XZ plane, dB)")
    plt.colorbar(im5, ax=ax5, label="dB")

    # Normalized pressure field
    ax6 = fig.add_subplot(gs[1, 2])
    pr_norm = pr_xz / pr_xz.max()
    im6 = ax6.imshow(
        pr_norm.T,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="hot",
        vmin=0,
        vmax=1,
    )
    ax6.set_xlabel("Z (mm)")
    ax6.set_ylabel("X (mm)")
    ax6.set_title("Normalized Pressure Field")
    plt.colorbar(im6, ax=ax6)

    # Lateral profile at different depths
    ax7 = fig.add_subplot(gs[2, :])

    for vs_idx in range(vs_history.shape[1]):
        x_values = vs_history[:, vs_idx, 0].squeeze()
        z_values = vs_history[:, vs_idx, 1].squeeze()
        ax7.plot(
            x_values - x_values[0],
            "-",
            label=f"x [vs_idx={vs_idx}]",
            color=colors[vs_idx],
        )
        ax7.plot(
            z_values - z_values[0],
            "--",
            label=f"z [vs_idx={vs_idx}]",
            color=colors[vs_idx],
        )

    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("Change in Position (mm)")
    ax7.set_title("Change in Virtual Source Position Over Epochs")
    ax7.legend()
    ax7.grid(True)

    plt.tight_layout()
    if output_file is not None:
        # create path
        from pathlib import Path

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        print(f"Figure saved to: {output_file}")
    plt.show()

    plt.figure(figsize=(6, 5))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pr_cover = sigmoid(1 * (pr_db - (-10)))  # steepness=20, threshold=-6 dB
    im6 = plt.imshow(
        pr_cover.T,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="hot",
    )
    plt.show()
