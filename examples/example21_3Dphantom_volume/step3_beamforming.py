"""
Step 3 — beamform each transmit event to IQ, compound coherently, measure.

The literature-canonical volumetric pipeline, nothing exotic:

1. Per event, 3-D delay-and-sum with the GENERAL beamformer (`das_volume`):
   the transmit wavefront geometry (here a diverging wave; the same call
   handles plane-wave, focused and synthetic-aperture events) is recovered
   from the event's own delays + virtual source, the echo returns over the
   direct path, and every voxel reads its aligned sample from every channel.
   Receive aperture: FULL width, rect — the ~1λ element faces already
   apodize the aperture through their own directivity (edge elements view a
   mid-depth voxel at ~30° and weigh x0.6), so adding Hann on top
   double-apodizes and doubles the PSF (measured on an isolated point:
   0.98 mm at f#1 Hann vs 0.52 mm at full-aperture rect on the ZeUS).
2. IQ per event: the beamformed RF volume is made analytic along the axial
   axis (Hilbert transform) — the complex IQ whose magnitude is the envelope.
3. Coherent compounding: the per-event IQ volumes are summed complex —
   in-phase echoes (true reflectors) add linearly, the tilt-dependent
   transmit fringes average down as 1/N.
4. TGC: one depth-only gain curve g(z) = 1 / smoothed lateral MEDIAN
   envelope, wire columns masked out (a wire spans the full elevation and
   would bias the median at its depths; the tubes cover well under half the
   width, which the median ignores by construction — the digital analogue of
   setting the TGC sliders on tissue). A slow function of depth alone cannot
   create lateral contrast; it only levels the diffraction/transmit-overlap
   ramp so one display window fits the volume.
5. Envelope = |IQ|, 30 dB log compression, grayscale.

Outputs: the compounded complex IQ volume + voxel axes saved to
``out/<scenario>/IQ/`` (the visualization scripts read this — beamforming
never has to be re-run to look at the data), contrast/SNR/wire-PSF metrics
(printed + ``out/<scenario>/metrics.json``), B-mode triptych and
phantom-truth vs image slices in ``figures/<scenario>/``.
Re-run freely: beamforming never touches the RF checkpoints.

Run with (pick the scenario in step 1 or via the SCENARIO env var):
    uv run examples/example21_3Dphantom_volume/step3_beamforming.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from step1_define_phantom_TX_RX import (
    C,
    FIG_DIR,
    IQ_DIR,
    OUT_DIR,
    RF_DIR,
    SC,
    SCENARIO,
    dw_events,
    phantom_map,
)
from config import SAVE_FIG

from esdiva.beamforming import das_volume
from esdiva.io import RFDataset
from esdiva.utilities import to_dB

FNUM = 0.5  # full receive aperture (see the module docstring, point 1)
RX_APOD = "rect"
DB_RANGE = 30  # after TGC the speckle fills 30 dB of greyscale
GRID = SC["grid"]
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n--- Example 21 · Step 3: beamforming, scenario '{SCENARIO}' ---\n")

# ============================================================================
# LOAD CHECKPOINTED RF + TIMING RECAP
# ============================================================================
ds = RFDataset(RF_DIR)
ds.summary()
rf, coords = ds.load_all()

contents = json.loads((RF_DIR / "contents.json").read_text())
sim_total = sum(ev.get("duration_s") or 0.0 for ev in contents["events"].values())
print(f"\nSimulation wall time (from contents file): {sim_total / 3600:.2f} h")

# ============================================================================
# 1–3. PER-EVENT DAS → IQ → COHERENT COMPOUND
# ============================================================================
probe = SC["make_probe"](SC["fc"])
events = dw_events(probe, SC["vs_mm"])  # delays + virtual_source_mm per event
t0_ev = np.asarray(coords["t0_per_event"], dtype=np.float64)

t_bf = time.perf_counter()
iq = None
for e, event in enumerate(events):
    # das_volume takes the pulse-centre lag from coords["pulse_center_lag_s"]
    # (stored by the acquisition) and applies it as t_offset_s automatically.
    vol_e, axes = das_volume(
        rf[e : e + 1],
        {
            "dt": coords["dt"],
            "t0_per_event": t0_ev[e : e + 1],
            "pulse_center_lag_s": coords["pulse_center_lag_s"],
        },
        [event],
        probe,
        GRID,
        c=C,
        fnum=FNUM,
        rx_apodization=RX_APOD,
    )
    iq_e = hilbert(vol_e, axis=2)  # analytic along the axial axis
    iq = iq_e if iq is None else iq + iq_e
t_bf = time.perf_counter() - t_bf
x_mm, y_mm, z_mm = axes["x_mm"], axes["y_mm"], axes["z_mm"]
print(
    f"Beamformed {iq.shape} voxels x {len(events)} events x "
    f"{rf.shape[1]} channels in {t_bf:.1f} s"
)

# The compounded complex IQ volume is the beamforming deliverable: the
# visualization scripts load it from here instead of re-beamforming.
IQ_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    IQ_DIR / "iq_volume.npz",
    iq=iq.astype(np.complex64),
    x_mm=x_mm,
    y_mm=y_mm,
    z_mm=z_mm,
)
print(f"Saved compounded IQ volume to {IQ_DIR / 'iq_volume.npz'}")

# ============================================================================
# 4. TGC — depth-only gain from the lateral MEDIAN envelope. The wire columns
#    are masked out (a wire spans the full elevation, so it biases the median
#    at its depths); the tubes need no mask — they cover well under half the
#    lateral width, which the median ignores by construction.
# ============================================================================
# Lateral PSF λ·z/D at the tier depth: every exclusion margin below scales
# with it, so the same masks stay valid from the 0.3 mm ZeUS PSF to the
# ~0.8 mm Vermon PSF (fixed margins tuned on one probe contaminate on the
# other — a bright wire's mainlobe is PSF-wide, not 0.4 mm wide).
PSF_MM = C / SC["fc"] * 1e3 * SC["tier_z"] / SC["aperture_mm"]
MARGIN = max(0.5, 1.5 * PSF_MM)
no_wire_cols = np.abs(x_mm - SC["wire_x"]) > MARGIN
prof = np.median(np.abs(iq[no_wire_cols]), axis=(0, 1))
prof = gaussian_filter1d(prof, sigma=2.0 / GRID["dz"])  # 2 mm in samples
tgc = prof.max() / prof.clip(min=prof.max() * 1e-3)
env = np.abs(iq) * tgc[None, None, :]
env_db = to_dB(env)

# ============================================================================
# METRICS — contrast vs clean speckle, speckle SNR, wire PSF widths
# ============================================================================
(CY_X, CY_Z), CY_R, _ = SC["tubes"][0]  # anechoic cyst tube
(LE_X, LE_Z), LE_R, _ = SC["tubes"][1]  # x4 lesion tube
(C2_X, C2_Z), C2_R, _ = SC["tubes"][2]  # x4 tube under the cyst
(SP_X, SP_Y, SP_Z), SP_R, _ = SC["spheres"][0]  # elevation sphere
X, Y, Z = np.meshgrid(x_mm, y_mm, z_mm, indexing="ij")
r2c = (X - CY_X) ** 2 + (Z - CY_Z) ** 2
r2l = (X - LE_X) ** 2 + (Z - LE_Z) ** 2
r2c2 = (X - C2_X) ** 2 + (Z - C2_Z) ** 2
r2s = (X - SP_X) ** 2 + (Y - SP_Y) ** 2 + (Z - SP_Z) ** 2
# Target cores: 70 % radius avoids the partial-volume rim.
cyst = r2c < (0.7 * CY_R) ** 2
les = r2l < (0.7 * LE_R) ** 2
les2 = r2c2 < (0.7 * C2_R) ** 2
sph = r2s < (0.7 * SP_R) ** 2
# Speckle reference: every voxel ≥ MARGIN clear of every target. After the
# depth-only TGC the mean envelope is flat in depth by construction, so
# speckle from any depth is a fair background (the same-depth bands of the
# campaign reports were needed only before TGC; at the crowded tier depth a
# wide-PSF probe would leave no clean same-depth voxels at all).
bg = r2c > (CY_R + MARGIN) ** 2
bg &= (r2l > (LE_R + MARGIN) ** 2) & (r2c2 > (C2_R + MARGIN) ** 2)
bg &= r2s > (SP_R + MARGIN) ** 2
bg &= np.abs(X - SC["wire_x"]) > MARGIN
# Stay off the grid rim, where the dynamic receive aperture is clipped.
bg &= (Z > z_mm[0] + 0.5) & (Z < z_mm[-1] - 0.5)

metrics = {
    "cyst_dB": 20 * np.log10(env[cyst].mean() / env[bg].mean()),
    "lesion_dB": 20 * np.log10(env[les].mean() / env[bg].mean()),
    "lesion2_dB": 20 * np.log10(env[les2].mean() / env[bg].mean()),
    "sphere_dB": 20 * np.log10(env[sph].mean() / env[bg].mean()),
    "speckle_SNR": env[bg].mean() / env[bg].std(),
}


def fwhm_mm(profile, axis_mm):
    """Full width at half maximum of a 1-D envelope profile, in mm.

    Measured outward from the peak (first drop below half on each side), so
    a neighbouring bright target inside the window cannot widen the reading.
    """
    ip = int(np.argmax(profile))
    half = profile[ip] / 2.0
    lo = ip
    while lo > 0 and profile[lo - 1] >= half:
        lo -= 1
    hi = ip
    while hi < profile.size - 1 and profile[hi + 1] >= half:
        hi += 1
    return axis_mm[hi] - axis_mm[lo]


iy0 = np.argmin(np.abs(y_mm))
ixw = np.argmin(np.abs(x_mm - SC["wire_x"]))
# Measurement window: wide enough to reach the half-maximum of the widest
# PSF, clamped so the neighbouring lesion column (its edge is 1.7 mm from
# the wire) can never out-peak the wire inside the window.
wire_win = min(max(0.8, 2.5 * PSF_MM), 1.5)
win = np.abs(x_mm - SC["wire_x"]) < wire_win
for zw in SC["wire_z"]:
    izw = np.argmin(np.abs(z_mm - zw))
    metrics[f"wire_z{zw:g}_lat_FWHM_mm"] = fwhm_mm(env[win, iy0, izw], x_mm[win])
    winz = np.abs(z_mm - zw) < 0.8
    metrics[f"wire_z{zw:g}_ax_FWHM_mm"] = fwhm_mm(env[ixw, iy0, winz], z_mm[winz])
# The axial wire (along z at y=0) reads the lateral PSF continuously with
# depth; sample it midway between the lateral wires, clear of their crossings.
for zm in (np.asarray(SC["wire_z"][:-1]) + np.asarray(SC["wire_z"][1:])) / 2.0:
    izm = np.argmin(np.abs(z_mm - zm))
    metrics[f"zwire_z{zm:g}_lat_FWHM_mm"] = fwhm_mm(env[win, iy0, izm], x_mm[win])

print(f"\n'{SCENARIO}' volume, same-depth metrics (plain DAS + TGC):")
print(
    f"  cyst    {metrics['cyst_dB']:+6.1f} dB   (anechoic r={CY_R} — want strongly negative)"
)
print(f"  lesion  {metrics['lesion_dB']:+6.1f} dB   (x4 r={LE_R} — want ≈ +12)")
print(f"  lesion2 {metrics['lesion2_dB']:+6.1f} dB   (x4 r={C2_R}, under the cyst)")
print(f"  sphere  {metrics['sphere_dB']:+6.1f} dB   (x4 r={SP_R}, elevation-offset)")
print(f"  speckle SNR {metrics['speckle_SNR']:.2f}  (Rayleigh 1.91)")
for zw in SC["wire_z"]:
    print(
        f"  wire z={zw:g}: lateral FWHM {metrics[f'wire_z{zw:g}_lat_FWHM_mm']:.2f} mm, "
        f"axial {metrics[f'wire_z{zw:g}_ax_FWHM_mm']:.2f} mm"
    )
for zm in (np.asarray(SC["wire_z"][:-1]) + np.asarray(SC["wire_z"][1:])) / 2.0:
    print(
        f"  axial wire @ z={zm:g}: lateral FWHM "
        f"{metrics[f'zwire_z{zm:g}_lat_FWHM_mm']:.2f} mm"
    )

(OUT_DIR / "metrics.json").write_text(
    json.dumps(
        {
            "sim_total_s": sim_total,
            "beamform_s": t_bf,
            "n_voxels": int(np.prod(iq.shape)),
            "metrics": {k: float(v) for k, v in metrics.items()},
        },
        indent=2,
    )
)
print(f"Saved {OUT_DIR / 'metrics.json'}")

# ============================================================================
# FIGURE 1 — B-mode triptych: tubes plane, sphere plane, C-plane
# ============================================================================
iys = np.argmin(np.abs(y_mm - SP_Y))
iz0 = np.argmin(np.abs(z_mm - SC["tier_z"]))
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
im_kw = {"cmap": "gray", "vmin": -DB_RANGE, "vmax": 0, "aspect": "equal"}
axs[0].imshow(
    env_db[:, iy0, :].T,
    origin="upper",
    **im_kw,
    extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
)
axs[0].set(title="y = 0 (cyst + lesion tubes)", xlabel="x (mm)", ylabel="z (mm)")
axs[1].imshow(
    env_db[:, iys, :].T,
    origin="upper",
    **im_kw,
    extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
)
axs[1].set(
    title=f"y = {SP_Y:+.0f} (sphere above cyst)", xlabel="x (mm)", ylabel="z (mm)"
)
m = axs[2].imshow(
    env_db[:, :, iz0].T,
    origin="lower",
    **im_kw,
    extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
)
axs[2].set(title=f"C-plane z = {SC['tier_z']:g} mm", xlabel="x (mm)", ylabel="y (mm)")
fig.colorbar(m, ax=axs, label="dB", shrink=0.8)
plt.savefig(str(FIG_DIR / f"ex21_{SCENARIO}_bmode.png"), dpi=150, bbox_inches="tight")
if not SAVE_FIG:
    plt.show()
plt.close()

# ============================================================================
# FIGURE 2 — phantom truth vs image: the same three planes
# ============================================================================
truth = phantom_map(
    {k: GRID[k] for k in ("x_extent", "y_extent", "z_extent")},
    (x_mm.size, y_mm.size, z_mm.size),
    SC["tubes"],
    SC["spheres"],
)


def _mark_wires(ax, plane):
    """Overlay the PSF wires in red — the lateral wires run along y at
    x=wire_x, so they appear as points in an x–z panel and as a vertical
    line in the x–y C-plane; the axial wire (along z at y=0) is the dotted
    vertical line in the y=0 panel only."""
    if plane.startswith("xz"):
        for zw in SC["wire_z"]:
            ax.plot(SC["wire_x"], zw, "r+", ms=12, mew=2)
        if plane == "xz0":  # the axial wire lives in the y=0 plane
            ax.axvline(SC["wire_x"], color="r", lw=0.6, ls=":")
    else:  # xy C-plane
        ax.axvline(SC["wire_x"], color="r", lw=0.6, ls=":")
        ax.plot(SC["wire_x"], 0, "r+", ms=12, mew=2)


ext_xz = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
ext_xy = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
cuts = [
    ("y = 0", lambda v: v[:, iy0, :].T, ext_xz, "x (mm)", "z (mm)", "upper", "xz0"),
    (
        f"y = {SP_Y:+.0f} mm",
        lambda v: v[:, iys, :].T,
        ext_xz,
        "x (mm)",
        "z (mm)",
        "upper",
        "xz",
    ),
    (
        f"z = {SC['tier_z']:g} mm",
        lambda v: v[:, :, iz0].T,
        ext_xy,
        "x (mm)",
        "y (mm)",
        "lower",
        "xy",
    ),
]
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for j, (title, cut, ext, xl, yl, orig, plane) in enumerate(cuts):
    axs[0, j].imshow(
        cut(truth), origin=orig, cmap="gray", vmin=0, vmax=2, extent=ext, aspect="equal"
    )
    axs[0, j].set(title=f"phantom · {title}", xlabel=xl, ylabel=yl)
    m = axs[1, j].imshow(
        cut(env_db),
        origin=orig,
        cmap="gray",
        vmin=-DB_RANGE,
        vmax=0,
        extent=ext,
        aspect="equal",
    )
    axs[1, j].set(title=f"image · {title}", xlabel=xl, ylabel=yl)
    _mark_wires(axs[0, j], plane)
# One shared dB colorbar (the image row); the phantom row is a 0–2 truth map,
# so a single dB bar to the right is enough to read the whole figure.
fig.colorbar(m, ax=axs.ravel().tolist(), location="right", shrink=0.6, label="dB")
plt.savefig(
    str(FIG_DIR / f"ex21_{SCENARIO}_slices_compare.png"), dpi=150, bbox_inches="tight"
)
if not SAVE_FIG:
    plt.show()
plt.close()

print(f"\nFigures saved to {FIG_DIR}")
print("For 3-D renders (setup scene, volume, MPR) run visualize_beamformed_volume.py.")
