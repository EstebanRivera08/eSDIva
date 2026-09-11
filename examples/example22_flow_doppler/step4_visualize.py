"""
Step 4 — draw the comparison.

Reads only the stored products from step 3; it never re-beamforms, so the
figures can be reworked without repeating the acquisition.

One row per sequence: the B-mode it reconstructed, its colour Doppler on a
velocity scale shared by all rows, and its power Doppler. The compounded
sequence should match the conventional one having cost 8x fewer
emissions; the long-ensemble sequence, spending the conventional budget on
slow-time samples instead of on scanning lines, should be visibly cleaner.

The second figure carries the trade itself — cost against estimate scatter —
and the measured radial profile, which shows how far the point-spread function
flattens the parabola.

Run with:
    uv run examples/example22_flow_doppler/step4_visualize.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from step1_define_flow_phantom import (
    FIG_FOLDER,
    IQ_FILE,
    METRICS_FILE,
    SAVE_FIG,
    SPEEDUP,
    THETA_DEG,
    V_PEAK,
)

print("\n--- Example 22 - Step 4: figures ---\n")
if not IQ_FILE.exists():
    raise SystemExit(
        f"No results at {IQ_FILE} — run step3_doppler_processing.py first."
    )

d = np.load(IQ_FILE)
m = json.loads(METRICS_FILE.read_text())
arms = ("A", "B", "C")
truth_peak = m["truth_m_s"]
truth_mean = m["arms"]["A"]["v_roi_true_m_s"]
v_nyq = m["v_nyquist_m_s"]
# Display range is set by the FLOW, not by the Nyquist limit: spanning +/-v_nyq
# would render the whole vessel one flat tint and hide the parabolic profile.
VLIM = 1.3 * truth_peak * 100

# Sequences are named for what they are, with what each one cost.
TITLES = {
    "A": "Conventional focused",
    "B": "Compounded ultrafast",
    "C": "Compounded ultrafast\nlong ensemble",
}
# "ensemble" is the number of SLOW-TIME SAMPLES the estimator sees, which is
# the quantity that must match for the comparison to be fair. What one sample
# costs differs: a conventional sample is one focused emission on that line, a
# compounded sample is a whole frame of N_ANGLES emissions.
COST = {
    arm: f"{m['arms'][arm]['n_emissions']} emissions  ·  "
    f"{m['arms'][arm]['ensemble']} slow-time samples"
    for arm in arms
}
PALETTE = {"A": "#3b6ea5", "B": "#c4622d", "C": "#4a8c5c"}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)

# ============================================================================
# FIGURE 1 — the three sequences, side by side
# ============================================================================
fig = plt.figure(figsize=(13.5, 12.0))
gs = fig.add_gridspec(
    3,
    4,
    width_ratios=[1, 1, 1, 0.05],
    hspace=0.12,
    wspace=0.10,
    left=0.13,
    right=0.90,
    top=0.875,
    bottom=0.05,
)

for r, arm in enumerate(arms):
    info = m["arms"][arm]
    v_map, power, mask = d[f"{arm}_v_map"], d[f"{arm}_power"], d[f"{arm}_mask"]
    x_mm, z_mm = d[f"{arm}_x_mm"], d[f"{arm}_z_mm"]
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    env = np.abs(d[f"{arm}_iq"]).mean(axis=0)
    bmode = 20 * np.log10(env / env.max() + 1e-12)
    # Power Doppler is shown only where there IS flow signal. With receiver
    # noise present, an unmasked map is mostly noise floor and every row
    # saturates into a single colour.
    power_db = np.where(mask, 10 * np.log10(power / power.max() + 1e-12), np.nan)

    axes = [fig.add_subplot(gs[r, c]) for c in range(3)]
    for ax in axes:
        ax.imshow(
            bmode.T,
            extent=extent,
            cmap="gray",
            vmin=-45,
            vmax=0,
            aspect="equal",
            interpolation="nearest",
        )
        if r == 2:
            ax.set_xlabel("x (mm)")
        else:
            ax.set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[0].set_ylabel("z (mm)")

    im_v = axes[1].imshow(
        np.where(mask, v_map, np.nan).T * 100,
        extent=extent,
        cmap="bwr",
        aspect="equal",
        vmin=-VLIM,
        vmax=VLIM,
        interpolation="nearest",
    )
    im_p = axes[2].imshow(
        power_db.T,
        extent=extent,
        cmap="hot",
        aspect="equal",
        vmin=-20,
        vmax=0,
        interpolation="nearest",
    )

    if r == 0:
        for ax, name in zip(axes, ("B-mode", "Colour Doppler", "Power Doppler")):
            ax.set_title(name, pad=9, fontweight="semibold")

    # Row heading to the left of the B-mode, like a table stub.
    axes[0].text(
        -0.34,
        0.5,
        TITLES[arm],
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11.5,
        fontweight="semibold",
        color=PALETTE[arm],
        linespacing=1.4,
    )
    axes[0].text(
        -0.21,
        0.5,
        COST[arm],
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    axes[1].text(
        0.035,
        0.05,
        f"vessel mean {info['v_roi_mean_m_s'] * 100:.1f}   "
        f"scatter {info['v_std_m_s'] * 100:.1f} cm/s",
        transform=axes[1].transAxes,
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.75", "pad": 2.5},
    )

cb = fig.colorbar(im_v, cax=fig.add_subplot(gs[0:2, 3]))
cb.set_label(f"axial velocity (cm/s)     Nyquist ±{v_nyq * 100:.0f}", fontsize=9)
fig.colorbar(im_p, cax=fig.add_subplot(gs[2, 3])).set_label(
    "flow power (dB)", fontsize=9
)

fig.suptitle(
    "Compounded ultrafast Doppler matches conventional imaging "
    f"for {SPEEDUP:.0f}x fewer emissions",
    fontsize=13.5,
    fontweight="semibold",
    y=0.955,
)
fig.text(
    0.515,
    0.905,
    f"Vessel at {THETA_DEG:.0f}° to the beam, {V_PEAK * 100:.0f} cm/s peak flow "
    f"(true axial peak {truth_peak * 100:.1f}, vessel mean "
    f"{truth_mean * 100:.1f} cm/s).\n"
    "Identical phantom, probe, Doppler PRF, receiver noise and estimator "
    "throughout — only the transmit differs.",
    ha="center",
    fontsize=9.5,
    color="0.3",
    linespacing=1.5,
)

if SAVE_FIG:
    plt.savefig(FIG_FOLDER / "ex22_bercoff_arms.png", dpi=145, bbox_inches="tight")
    print(f"Saved ex22_bercoff_arms.png to {FIG_FOLDER}")
else:
    plt.show()

# ============================================================================
# FIGURE 2 — the trade, and what the PSF does to the profile
# ============================================================================
fig2, ax = plt.subplots(1, 3, figsize=(15, 4.5))

# -- spectra ---------------------------------------------------------------
for arm in arms:
    ax[0].plot(
        d[f"{arm}_v_axis"] * 100,
        d[f"{arm}_spectrum"],
        lw=1.7,
        color=PALETTE[arm],
        label=TITLES[arm].replace("\n", ", "),
    )
ax[0].axvline(truth_peak * 100, color="0.25", ls="--", lw=1.4)
ax[0].annotate(
    f"true peak {truth_peak * 100:.1f}",
    (truth_peak * 100, 1.0),
    xytext=(5, -2),
    textcoords="offset points",
    fontsize=8,
    color="0.25",
    va="top",
)
ax[0].axvspan(-v_nyq * 100, 0, color="0.94", zorder=0)
ax[0].set(
    title="Velocity spectrum at the sample gate",
    xlabel="axial velocity (cm/s)",
    ylabel="power (normalised)",
    xlim=(-v_nyq * 100, v_nyq * 100),
    ylim=(0, 1.08),
)
ax[0].legend(fontsize=8, frameon=False, loc="upper left")
ax[0].grid(alpha=0.25)

# -- the trade -------------------------------------------------------------
# Each label is placed away from its own point and from the connecting arrow,
# so nothing collides with the axis or with the annotation between them.
LABEL_POS = {
    "A": {"xytext": (0, 16), "ha": "center"},
    "B": {"xytext": (0, 16), "ha": "center"},
    "C": {"xytext": (0, -30), "ha": "center"},
}
for arm in arms:
    info = m["arms"][arm]
    ax[1].scatter(
        info["n_emissions"],
        info["v_std_m_s"] * 100,
        s=140,
        color=PALETTE[arm],
        zorder=3,
        edgecolor="white",
        linewidth=1.6,
    )
    ax[1].annotate(
        TITLES[arm],
        (info["n_emissions"], info["v_std_m_s"] * 100),
        textcoords="offset points",
        fontsize=8,
        color="0.25",
        linespacing=1.3,
        **LABEL_POS[arm],
    )
# The paper's two options, drawn as the two moves available from a conventional
# acquisition: spend less (A -> B), or spend the same budget on ensemble length
# (B -> C). Arrow ends and captions are positioned FROM THE DATA, so a retuned
# sequence cannot strand a caption beside the wrong point.
XY = {a: (m["arms"][a]["n_emissions"], m["arms"][a]["v_std_m_s"] * 100) for a in arms}


def _trade(src, dst, text, dy, rad):
    ax[1].annotate(
        "",
        xy=XY[dst],
        xytext=XY[src],
        arrowprops={
            "arrowstyle": "->",
            "color": "0.5",
            "lw": 1.3,
            "connectionstyle": f"arc3,rad={rad}",
        },
    )
    ax[1].text(
        (XY[src][0] * XY[dst][0]) ** 0.5,  # midpoint on the log cost axis
        0.5 * (XY[src][1] + XY[dst][1]) + dy,
        text,
        fontsize=8.5,
        color="0.35",
        ha="center",
        va="center",
        linespacing=1.4,
    )


_trade("A", "B", f"{m['speedup']:.0f}x fewer emissions,\nsame answer", 1.6, -0.3)
_trade(
    "B",
    "C",
    f"same budget, {m['arms']['B']['v_std_m_s'] / m['arms']['C']['v_std_m_s']:.1f}x"
    "\ntighter estimate",
    -1.7,
    0.3,
)
ax[1].set(
    title="The trade: cost against estimate scatter",
    xlabel="emissions (acquisition cost)",
    ylabel="velocity scatter in vessel (cm/s)",
    xscale="log",
    xlim=(40, 1800),
    # Derived, never hardcoded: a fixed ceiling silently clipped the
    # short-ensemble point off the panel once its scatter exceeded it.
    ylim=(0, 1.3 * max(v[1] for v in XY.values())),
)
ax[1].grid(alpha=0.25, which="both")

# -- what the PSF does -----------------------------------------------------
# The resolution cell is a good fraction of the 3 mm lumen, so every voxel
# averages its neighbourhood and the parabola comes back flattened.
prof = d["C_profile_measured"] * 100
prof_true = d["C_profile_true"] * 100
r_mid = np.linspace(0, 1, len(prof) + 1)
r_mid = 0.5 * (r_mid[:-1] + r_mid[1:])
ax[2].plot(r_mid, prof_true, "k--", lw=1.6, label="true (Poiseuille)")
ax[2].plot(r_mid, prof, "o-", color=PALETTE["C"], lw=1.9, ms=5.5, label="measured")
ax[2].fill_between(r_mid, prof_true, prof, color=PALETTE["C"], alpha=0.15)
ax[2].set(
    title="What the PSF does: the profile flattens",
    xlabel="radius r/R   (0 = vessel axis, 1 = wall)",
    ylabel="axial velocity (cm/s)",
    xlim=(0, 1),
    ylim=(0, None),
)
ax[2].annotate(
    f"axis reads {m['arms']['C']['psf_core_ratio']:.2f}× true",
    (r_mid[0], prof[0]),
    xytext=(4, -20),
    textcoords="offset points",
    fontsize=8,
    color="0.25",
)
ax[2].annotate(
    f"wall reads {m['arms']['C']['psf_wall_ratio']:.2f}× true",
    (r_mid[-1], prof[-1]),
    xytext=(-96, 12),
    textcoords="offset points",
    fontsize=8,
    color="0.25",
)
ax[2].legend(fontsize=8, frameon=False)
ax[2].grid(alpha=0.25)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig(FIG_FOLDER / "ex22_bercoff_tradeoff.png", dpi=145, bbox_inches="tight")
    print(f"Saved ex22_bercoff_tradeoff.png to {FIG_FOLDER}")
else:
    plt.show()

print(
    f"\nTrue vessel mean {truth_mean * 100:.2f} cm/s (axis peak {truth_peak * 100:.1f})"
)
for arm in arms:
    info = m["arms"][arm]
    label = TITLES[arm].replace("\n", ", ")
    print(
        f"  {label:<40}{info['n_emissions']:>5} em   "
        f"mean {info['v_roi_mean_m_s'] * 100:5.2f}   "
        f"scatter {info['v_std_m_s'] * 100:5.2f} cm/s"
    )
print(f"\nCompounded ultrafast used {m['speedup']:.1f}x fewer emissions.")
