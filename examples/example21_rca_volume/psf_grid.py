"""Compounding ladder on 12 isolated scatterers — is the compound working?

Twelve unit point targets on a lattice spanning the scenario's volume (two
lateral x two elevation x three depth positions), simulated once through the
full diverging-wave sequence, then beamformed once per ring subset (centre
source only, centre + first ring, ..., all). With no speckle in the frame the
images show the pure point-spread function at every position — how it varies
across the volume (off-axis blur, depth) and exactly what each level of
coherent compounding buys (mainlobe, sidelobe skirt, off-axis clean-up).

Run with (pick the scenario in step 1 or via the SCENARIO env var):
    uv run examples/example21_rca_volume/psf_grid.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from step1_define_phantom_TX_RX import (
    C,
    DOWNSAMPLING,
    FIG_DIR,
    FS,
    SC,
    SCENARIO,
    dw_events,
    excitation,
    pulse_center_lag_s,
)
from config import SAVE_FIG

from pyfield.beamforming import das_volume
from pyfield.reception import ReceptionSDI
from pyfield.utilities import to_dB

VOL = SC["volume"]
x0, x1 = VOL["x_extent"]
z0, z1 = VOL["z_extent"]
XS = [0.5 * x0, 0.5 * x1]
YS = [0.0, SC["slice_y"]]
ZS = [z0 + f * (z1 - z0) for f in (0.2, 0.5, 0.8)]
POINTS = np.array([[x, y, z] for y in YS for x in XS for z in ZS])
DB_RANGE = 40

# Ring subsets from the virtual-source layout itself: sources share a radius
# per ring, so the cumulative count per unique radius gives the ladder
# (e.g. centre / +ring-8 / +ring-16).
radii = np.round(np.linalg.norm(SC["vs_mm"][:, :2], axis=1), 6)
counts = [np.sum(radii <= r) for r in np.unique(radii)]
SUBSETS = [(f"{n} VS", list(range(n))) for n in counts]

print(
    f"\n--- 12-point compounding ladder, '{SCENARIO}' ({len(POINTS)} scatterers) ---\n"
)

tx, rx = SC["make_probe"](SC["fc"]), SC["make_probe"](SC["fc"])
sim = ReceptionSDI(
    tx,
    rx,
    c=C,
    fs=FS,
    excitation=excitation(SC["fc"]),
    method="spectral",
    verbose=False,
)
events = dw_events(tx, SC["vs_mm"])
rf, coords = sim.sequence_rf(
    POINTS,
    np.ones(len(POINTS)),
    [{"delays": ev["delays"], "apodization": ev["apodization"]} for ev in events],
    downsampling=DOWNSAMPLING,
)
print(f"RF {rf.shape} — beamforming the ladder…")

t0 = np.asarray(coords["t0_per_event"], dtype=np.float64)


def bf_plane(idx, y0):
    """Beamform a thin y-slab from the event subset; return envelope + axes."""
    idx = np.asarray(idx)
    grid = dict(SC["grid"])
    grid["y_extent"] = [y0 - 0.2, y0 + 0.21]
    vol, axes = das_volume(
        rf[idx],
        {"dt": coords["dt"], "t0_per_event": t0[idx]},
        [events[i] for i in idx],
        rx,
        grid,
        c=C,
        fnum=0.5,
        rx_apodization="rect",
        t_offset_s=pulse_center_lag_s(SC["fc"]),
    )
    env = np.abs(hilbert(vol, axis=2))
    return env[:, env.shape[1] // 2, :], axes


def fwhm(prof, ax_mm):
    above = np.flatnonzero(prof >= prof.max() / 2)
    return ax_mm[above[-1]] - ax_mm[above[0]] if above.size else np.nan


fig, axs = plt.subplots(len(SUBSETS), len(YS), figsize=(11, 4.7 * len(SUBSETS)))
axs = np.atleast_2d(axs)
for i, (label, idx) in enumerate(SUBSETS):
    print(f"\n{label}: per-point lateral / axial FWHM (mm)")
    for j, y0 in enumerate(YS):
        img, axes = bf_plane(idx, y0)
        x, z = axes["x_mm"], axes["z_mm"]
        axs[i, j].imshow(
            to_dB(img).T,
            origin="upper",
            cmap="gray",
            vmin=-DB_RANGE,
            vmax=0,
            extent=[x[0], x[-1], z[-1], z[0]],
            aspect="equal",
        )
        axs[i, j].set(
            title=f"{label} · y = {y0:+.0f} mm", xlabel="x (mm)", ylabel="z (mm)"
        )
        for xp in XS:
            for zp in ZS:
                # ±3 mm window around each point: lateral and axial cuts.
                iw = np.abs(x - xp) < 3.0
                kw = np.abs(z - zp) < 3.0
                sub = img[np.ix_(iw, kw)]
                pi, pk = np.unravel_index(np.argmax(sub), sub.shape)
                lat = fwhm(sub[:, pk], x[iw])
                axl = fwhm(sub[pi, :], z[kw])
                print(f"  ({xp:+5.1f}, {y0:+2.0f}, {zp:4.1f}): {lat:5.2f} / {axl:4.2f}")
fig.suptitle("12 isolated points — PSF vs position and vs compounding", y=0.995)
fig.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(str(FIG_DIR / f"{SCENARIO}_psf_ladder.png"), dpi=150, bbox_inches="tight")
print(f"\nsaved {FIG_DIR / f'{SCENARIO}_psf_ladder.png'}")
if not SAVE_FIG:
    plt.show()
