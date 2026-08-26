"""
Example 19: Dual-Probe Pulse-Echo — transform() TX/RX + Reception show()

Pitch-catch configuration used in flow imaging and vector Doppler research:
a mono-element circular piston on the left transmits, a linear array on the
right receives — both rigidly moved with `transform()` and tilted so their
beam axes cross at a shallow common target.

  1. TX circular piston translated left, tilted toward the middle
  2. RX linear array translated right, tilted toward the middle
  3. `sim.show()` — 3-D preview of both apertures + scatterers
  4. Pulse-echo RF on the tilted RX aperture

Run with:
    uv run examples/example19_dualprobe_reception_show.py
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_FOLDER, SAVE_FIG, SCALE

import sondi.transducers as transducers
from sondi.reception import Reception

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_MM = np.array([0.0, 0.0, 7.0])  # beams cross here (shallow target)
PROBE_OFFSET_MM = 8.0  # lateral distance of each probe from the mid-plane
C = 1540.0
FS = 200e6
PULSE_CYCLES = 2

print("\n--- Example 19: Dual-Probe Pulse-Echo (transform + show) ---\n")

if SAVE_FIG:
    FIG_FOLDER.mkdir(exist_ok=True)


def aim_at_target(offset_x_mm: float) -> np.ndarray:
    """4x4 transform: translate probe to `offset_x_mm` laterally, tilt about y
    so the beam axis (+z at the canonical pose) points at TARGET_MM."""
    th = np.arctan2(TARGET_MM[0] - offset_x_mm, TARGET_MM[2])
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [np.cos(th), 0, np.sin(th)],
            [0, 1, 0],
            [-np.sin(th), 0, np.cos(th)],
        ]
    )
    T[:3, 3] = [offset_x_mm, 0.0, 0.0]
    return T


# ============================================================================
# STEP 1: TX PROBE — MONO-ELEMENT PISTON, LEFT, TILTED TOWARD THE MIDDLE
# ============================================================================
tx = transducers.FlatCircularTransducer(
    diameter_mm=4.0, no_sub_diameter=25, frequency_Hz=12.5e6
)
tx.transform(aim_at_target(-PROBE_OFFSET_MM))

# ============================================================================
# STEP 2: RX PROBE — LINEAR ARRAY, RIGHT, FLIPPED + TILTED TOWARD THE MIDDLE
# ============================================================================
rx = transducers.Domino()

# Hanning receive apodization over the full aperture, computed in the
# canonical frame BEFORE the move (apodization is per-element state, the
# rigid transform does not change it). Focus at the target distance.
target_dist_mm = float(np.linalg.norm(TARGET_MM - [PROBE_OFFSET_MM, 0, 0]))
aperture_mm = rx.n_elements * 0.11  # Domino pitch
rx.compute_apodization(
    focus_mm=[0, 0, target_dist_mm],
    FoverD=target_dist_mm / aperture_mm,
    apodization_type="hanning",
)

# Rotate the array 90° about z (x→y) so its element axis lies along y instead
# of x, then aim at the target. Right-multiplying applies the rotation first,
# in the canonical frame.
rot_z90 = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
rx.transform(aim_at_target(+PROBE_OFFSET_MM) @ rot_z90)

tilt_deg = np.rad2deg(np.arctan2(PROBE_OFFSET_MM, TARGET_MM[2]))
print(f"TX piston at x=-{PROBE_OFFSET_MM} mm, RX array at x=+{PROBE_OFFSET_MM} mm")
print(f"Both tilted {tilt_deg:.1f}° toward the target at {TARGET_MM} mm")

# ============================================================================
# STEP 3: SCATTERERS + 3-D SETUP PREVIEW
# ============================================================================
# 10 scatterers clustered around the beam crossing.
N_scat = 30
rng = np.random.default_rng(19)
scatterer_pos = (TARGET_MM + rng.uniform(-2, 2, size=(N_scat, 3)) * [1, 0, 1]).astype(
    np.float32
)
scatterer_amp = rng.uniform(1, 2.0, size=N_scat).astype(np.float32)

fc = tx.fc
t_pulse = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
excitation = (np.sin(2 * np.pi * fc * t_pulse) * np.hanning(len(t_pulse))).astype(
    np.float32
)
# NOTE: no piezo impulse_response is set, so the elements are ideally
# broadband (echo = drive convolved with the aperture SIRs only). Fine for an
# API demo; for realistic imaging set tx/rx `.impulse_response` as in example20.

# The simulator snapshots both apertures at construction, AFTER the moves.
sim = Reception(tx, rx, c=C, fs=FS, excitation=excitation)


if SAVE_FIG:
    scale = SCALE  # higher-res screenshot: scales window + all fonts together
    save_path = str(FIG_FOLDER / "ex19_dualprobe_setup.png")
else:
    scale = 1.0
    save_path = None
# 3-D preview: TX piston (blue), tilted RX array (salmon), scatterers faded
# by amplitude.
# Bare scene: keep the scalar colouring on the apertures/scatterers but drop
# every colour bar and the legend (show_scalar_bar=False on each mesh).
plotter = sim.show(
    scatterer_pos,
    scatterer_amp,
    RX_color="Apodization",
    legend=False,
    TX_kwargs={"show_scalar_bar": False},
    RX_kwargs={"show_scalar_bar": False},
    show_scalar_bar=False,
    scale=scale,  # higher-res screenshot: scales window + all fonts together
    off_screen=SAVE_FIG,  # headless render so plotter.screenshot() works below
    return_plotter=True,
)

plotter.remove_bounds_axes()  # drop the X/Y/Z grid added by show()
plotter.camera_position = [
    (-22.752762440240886, 13.822424446767888, -5.364591941710554),
    (0.2984648214293899, 0.36829171869645116, 3.5419471180135615),
    (0.4089767709732527, 0.10026744868751436, -0.9070195364698537),
]
if SAVE_FIG:
    plotter.screenshot(save_path, transparent_background=True)
else:
    plotter.show()
# print(plotter.camera_position)

# ============================================================================
# STEP 4: PULSE-ECHO RF ON THE TILTED RX
# ============================================================================
rf, coords = sim(scatterer_pos, scatterer_amp)
t = coords["t0"] + np.arange(rf.shape[1]) * coords["dt"]
print(f"RF shape: {rf.shape}  (E_rx, Nt)")

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(
    rf.T / (np.abs(rf).max() + 1e-30),
    aspect="auto",
    extent=[0, rf.shape[0] - 1, t[-1] * 1e6, t[0] * 1e6],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)
plt.colorbar(im, ax=ax, label="RF (norm.)")
ax.set_xlabel("RX element")
ax.set_ylabel("Time (µs)")
ax.set_title(f"Pitch-catch RF — piston TX / array RX, ±{tilt_deg:.0f}° tilt")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(str(FIG_FOLDER / "ex19_dualprobe_rf.png"), dpi=150)

plt.show()


del plotter
print("\nDone.")
