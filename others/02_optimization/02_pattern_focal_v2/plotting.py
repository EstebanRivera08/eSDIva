import matplotlib.pyplot as plt
import numpy as np

from pyfield.utilities import to_dB


def plot_results(results, save_path=None, tx=None):
    """Plot optimization results v2."""

    print("\nPlotting results...")
    fig = plt.figure(figsize=(14, 10))

    # Create gridspec with dedicated colorbar columns
    # gs = fig.add_gridspec(3, 6, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05])
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1])

    # loss history

    # Columns: [plot, cbar, plot, cbar, plot, cbar]
    loss_energy = np.array(results["loss_energy_values"])
    loss_comp = np.array(results["loss_comp_values"])
    norm_loss_energy = (loss_energy - loss_energy.min()) / (
        loss_energy.max() - loss_energy.min() + 1e-6
    )
    norm_loss_comp = (loss_comp - loss_comp.min()) / (
        loss_comp.max() - loss_comp.min() + 1e-6
    )

    loss_history_delays = np.array(results["loss_history_delays"])
    loss_history_apod = np.array(results["loss_history_apod"])

    n_delays = len(loss_history_delays)
    n_apod = len(loss_history_apod)
    iter_delays = np.arange(n_delays)
    iter_apod = np.arange(n_delays, n_delays + n_apod)

    # Row 0: Loss history, Delays, Apodization (span 2 columns each for centering)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iter_delays, loss_history_delays, color="b", label="Delays only")
    ax1.plot(iter_apod, loss_history_apod, color="g", label="Delays + Apod")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(loc="lower left")
    ax1.grid(True)
    ax1.set_yscale("log")

    ax11 = ax1.twinx()
    ax11.plot(norm_loss_energy, "--", color="tab:orange", label="Energy Loss")
    ax11.plot(norm_loss_comp, "--", color="tab:red", label="Pattern Loss")
    ax11.set_ylabel("Normalized Loss (dashed)")
    ax11.set_yscale("log")
    ax11.legend(loc="upper right")
    # ax11.set_ylim(-0.05, 1.05)

    # Delays
    ax2 = fig.add_subplot(gs[0, 1])
    delays = results["delays"]  # convert to microseconds
    if tx is not None:
        ax2 = tx.plot_delays(delays, ax=ax2)
        if hasattr(ax2, "_im"):
            cb = plt.colorbar(ax2._im, ax=ax2)
    else:
        ax2.plot(delays * 1e6, "or-")

    ax2.set_xlabel("Element Index")
    ax2.set_ylabel("Delay (μs)")
    ax2.set_title("Optimized Delays")
    ax2.grid(True)

    # Apodization
    ax3 = fig.add_subplot(gs[0, 2])
    apod_final = results["apodization"]
    if tx is not None:
        ax3 = tx.plot_apodization(apod_final, ax=ax3)
        if hasattr(ax3, "_im"):
            cb = plt.colorbar(ax3._im, ax=ax3)
    else:
        ax3.plot(apod_final, "ok-")
    ax3.set_xlabel("Element Index")
    ax3.set_ylabel("Apodization")
    ax3.set_title("Optimized Apodization")
    ax3.grid(True)

    # Row 1: Target pattern, Pressure fields with dedicated colorbar columns
    pr = results["pressure_final"]
    x, y, z = results["x"], results["y"], results["z"]
    extent_xz = [z.min(), z.max(), x.min(), x.max()]
    extent_xy = [x.min(), x.max(), y.min(), y.max()]

    # Target pattern
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(
        results["target_mask"].T,
        aspect="auto",
        cmap="gray",
        origin="lower",
        extent=extent_xy,
    )
    ax4.set_title("Target Pattern")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    cb = plt.colorbar(im4, ax=ax4)
    # cax4 = fig.add_subplot(gs[1, 1])
    # plt.colorbar(im4, cax=cax4)

    # Pressure field (dB)
    pr_xy = pr[:, :, len(z) // 2]
    pr_xy_norm = pr_xy / pr_xy.max()
    pr_db = to_dB(pr_xy_norm)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(
        pr_db.T,
        aspect="auto",
        origin="lower",
        extent=extent_xy,
        cmap="hot",
        vmin=-40,
        vmax=0,
    )
    ax5.set_xlabel("X (mm)")
    ax5.set_ylabel("Y (mm)")
    ax5.set_title("Pressure Field (dB)")
    # cax5 = fig.add_subplot(gs[1, 3])
    # cbar5 = plt.colorbar(im5, cax=cax5)
    cbar5 = plt.colorbar(im5, ax=ax5)
    cbar5.set_label("dB", rotation=270, labelpad=15)

    # Pressure field (normalized)
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(
        pr_xy_norm.T,
        aspect="auto",
        origin="lower",
        extent=extent_xy,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    ax6.set_xlabel("X (mm)")
    ax6.set_ylabel("Y (mm)")
    ax6.set_title("Pressure Field (a.u.)")
    plt.colorbar(im6, ax=ax6, label="a.u.")
    # cax6 = fig.add_subplot(gs[1, 5])
    # plt.colorbar(im6, cax=cax6)

    # Row 2: Comparison plot (spans all columns)
    target = results["target_mask"]
    target_flat = np.squeeze(target).T.flatten()
    pr_xy_flat = np.squeeze(pr_xy_norm).T.flatten()
    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(target_flat, "b-", alpha=0.5, label="Target")
    ax7.plot(pr_xy_flat, "r-", alpha=0.5, label="Achieved")
    ax7.set_xlim(0, len(target_flat))
    ax7.set_xlabel("Pixel Index")
    ax7.set_ylabel("Intensity")
    ax7.set_title("Target vs Achieved Pattern (flattened)")
    ax7.legend()
    ax7.grid(True)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
    plt.show()
    plt.close()
