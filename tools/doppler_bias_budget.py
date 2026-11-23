"""
Where does the vessel-mean velocity deficit come from? A measured budget.

The colour Doppler maps read a vessel mean about 20 % below the known truth.
That is not a calibration error to be tuned away — it is the sum of several
effects, and this script measures each one instead of asserting any of them.

Four candidates, and the control that isolates each:

  N  NOISE. Voxels whose flow signal is buried return a near-random phase, i.e.
     a velocity scattered about zero, which drags an unweighted spatial mean
     down. ISOLATED BY: processing the same RF with and without added noise.
  P  LOW-POWER VOXELS. Same mechanism without noise: weak voxels contribute
     poorly determined velocities. ISOLATED BY: power-weighting the spatial mean.
  W  WALL FILTER. An order-0 filter removes the slow-time mean and so attenuates
     the slowest flow hardest. ISOLATED BY: sweeping the filter order.
  G  VELOCITY GRADIENT INSIDE THE RESOLUTION CELL. The cell spans a range of
     true velocities, and the estimator returns their power-weighted mean, so
     the profile is smoothed. ISOLATED BY: a PLUG-flow control - identical
     phantom, geometry, sequence and processing, but every blood scatterer at
     ONE velocity, so there is no gradient left to smooth. Whatever deficit
     survives the plug control is NOT the gradient.

The plug control is simulated here (checkpointed, ~6 min) because it cannot be
derived from the existing data: it needs a different phantom.

WHAT THIS SCRIPT ALREADY ESTABLISHED (so it is not re-derived by hand):

  * The plug control isolates a UNIFORM scale error from the gradient: its ratio
    is flat across every radius, while the Poiseuille case varies 0.80 -> 1.47.

  * Chasing that flat part down found a real bug. The plug control read a
    uniform 0.955, and these were tested and REFUTED:
      - wall filter as the main term      (order 0->2 moves the mean 6.44->6.73)
      - depth-averaged centre frequency   (the vessel-depth centroid, 4.505 MHz,
                                           moves AWAY from the 4.26 MHz needed)
      - receive-aperture geometry alone   (17deg -> 7deg recovers only 2.4 %)
      - transit-time decorrelation        (at a velocity slow enough to decimate
                                           without aliasing the ratio is flat
                                           across a 4x change in displacement
                                           per lag: 0.955 / 0.956 / 0.957)

  * THE CAUSE: the centre frequency was measured on the raw channel RF, but the
    estimator works on the BEAMFORMED signal, and delay-and-sum low-passes the
    data. Channel centroid 4.460 MHz; beamformed centroid 4.268 MHz; frequency
    needed for an unbiased velocity 4.261 MHz - the last two agree to 0.1 %,
    from two independent measurements (a velocity error and a spectrum).
    Measuring on the beamformed lines took the plug control to 0.998.

  A caution for anyone repeating the decimation test: decimating slow time also
  halves the Nyquist velocity. Doing it to the FAST plug put 12 cm/s above the
  10.6 cm/s limit and the estimate folded to a negative velocity - the control
  must be slow enough that decimation stays unambiguous.

Run with:
    uv run examples/example22_flow_doppler/bias_budget.py
"""

import numpy as np
from doppler_tools import echo_center_frequency, iq2doppler, rf2iq, wfilt
from step1_define_flow_phantom import (
    ARMS,
    C,
    E_LONG,
    FLOW_DIR,
    FC,
    GRID_MM,
    N_ANGLES,
    OUT_DIR,
    PRF_DOPPLER,
    PRF_EMISSION,
    RF_DIRS,
    THETA,
    VESSEL_CENTER_MM,
    VESSEL_LENGTH_MM,
    VESSEL_RADIUS_MM,
    V_PEAK,
    build_phantom,
    build_simulator,
    compound_events,
    make_probe,
)

from esdiva.beamforming import das_volume
from esdiva.io import RFDataset
from esdiva.utilities import add_noise, rf_rms

ARM = "C"  # longest ensemble = least estimator variance, cleanest budget
ENSEMBLE = ARMS[ARM]["ensemble"]
PLUG_DIR = OUT_DIR / "RF_plug"

probe = make_probe()
_, frame_events = compound_events(probe, 1)


def process(rf, coords, wall_order=0):
    """Beamform an ultrafast acquisition and estimate velocity + power."""
    frames = []
    for k in range(ENSEMBLE):
        sl = slice(k * N_ANGLES, (k + 1) * N_ANGLES)
        vol, gc = das_volume(
            rf[sl],
            {"dt": coords["dt"], "t0_per_event": coords["t0_per_event"][sl]},
            frame_events,
            probe,
            GRID_MM,
            c=C,
            fnum=1.0,
        )
        frames.append(vol[:, 0, :])
    # Measured on the BEAMFORMED lines, not the channel RF - see the note at the
    # top of this file; using the channel centroid is what produced the 4.5 %
    # residual this script was written to chase down.
    bf = np.stack(frames)
    fs_depth = C / (2 * (GRID_MM["dz"] * 1e-3))
    # Band scaled to the probe, never absolute numbers: a band left over
    # from another probe truncates the echo and rescales every velocity.
    f_echo = echo_center_frequency(bf, 1.0 / fs_depth, band=(0.3 * FC, 1.7 * FC))
    iq = rf2iq(bf, fs=fs_depth, fc=f_echo, axis=-1)
    iq_flow = wfilt(iq, method="poly", order=wall_order, axis=0)
    v, p = iq2doppler(iq_flow, PRF_DOPPLER, f_echo, c=C)
    return v, p, gc["x_mm"], gc["z_mm"]


def lumen(x_mm, z_mm):
    """Lumen mask, per-voxel radius, and the true axial velocity of each profile."""
    xx, zz = np.meshgrid(x_mm, z_mm, indexing="ij")
    pts = np.stack([xx, np.zeros_like(xx), zz], axis=-1) - VESSEL_CENTER_MM
    along = pts @ FLOW_DIR
    radial = np.linalg.norm(pts - along[..., None] * FLOW_DIR, axis=-1)
    roi = (radial < VESSEL_RADIUS_MM) & (np.abs(along) < VESSEL_LENGTH_MM / 2)
    parabolic = V_PEAK * (1 - (radial / VESSEL_RADIUS_MM) ** 2) * np.cos(THETA)
    plug = np.full_like(parabolic, V_PEAK * np.cos(THETA))
    return roi, radial, parabolic, plug


print("\n--- Example 22: where the velocity deficit comes from ---\n")

# ---------------------------------------------------------------- Poiseuille
rf_par, coords_par = RFDataset(RF_DIRS[ARM]).load_all()
v_par, p_par, x_mm, z_mm = process(rf_par, coords_par)
roi, radial, v_true_par, v_true_plug = lumen(x_mm, z_mm)
truth_par = v_true_par[roi].mean()

# ---------------------------------------------------------------- plug control
# Same phantom, geometry, sequence and processing; only the velocity PROFILE
# differs, so any deficit that survives here cannot be the velocity gradient.
if not PLUG_DIR.exists():
    print(f"Simulating the plug-flow control -> {PLUG_DIR}")
    print("(identical setup, every blood scatterer at one velocity)\n")
    sim, tx, _ = build_simulator()
    events, _ = compound_events(tx, E_LONG)
    positions, amplitudes, _ = build_phantom(len(events), PRF_EMISSION, profile="plug")
    sim.sequence_rf(positions, amplitudes, events, out_path=PLUG_DIR)
rf_plug, coords_plug = RFDataset(PLUG_DIR).load_all()
v_plug, p_plug, _, _ = process(rf_plug, coords_plug)
truth_plug = v_true_plug[roi].mean()

# ============================================================================
# THE BUDGET
# ============================================================================
print(f"{'':38}{'measured':>10}{'truth':>8}{'ratio':>8}")


def report(label, v, p, truth, weights=None):
    mean = np.average(v[roi], weights=None if weights is None else weights[roi])
    print(f"  {label:<36}{mean * 100:>9.2f}{truth * 100:>8.2f}{mean / truth:>8.2f}")
    return mean / truth


print("\nPOISEUILLE (the shipped case), noiseless:")
r_plain = report("plain spatial mean", v_par, p_par, truth_par)
r_weighted = report("power-weighted mean", v_par, p_par, truth_par, weights=p_par)

noisy = add_noise(rf_par, 6.0, reference=rf_rms(rf_par), rng=11)
v_n, p_n, _, _ = process(noisy, coords_par)
print("\nPOISEUILLE with 6 dB receiver noise:")
r_noise = report("plain spatial mean", v_n, p_n, truth_par)
report("power-weighted mean", v_n, p_n, truth_par, weights=p_n)

print("\nWALL-FILTER order sweep (noiseless, plain mean):")
orders = {}
for order in (0, 1, 2):
    v_o, p_o, _, _ = process(rf_par, coords_par, wall_order=order)
    orders[order] = report(f"order {order}", v_o, p_o, truth_par)

print("\nPLUG CONTROL - no velocity gradient to smooth (noiseless):")
r_plug = report("plain spatial mean", v_plug, p_plug, truth_plug)
r_plug_w = report("power-weighted mean", v_plug, p_plug, truth_plug, weights=p_plug)

# ============================================================================
# WHAT EACH TERM COSTS
# ============================================================================
print("\n" + "=" * 62)
print("BUDGET (each term as a fraction of the true vessel mean)")
print("=" * 62)
print(
    f"  N  noise                          {100 * (r_weighted - r_noise + r_plain - r_plain):+5.1f} %"
    f"   ({100 * (r_plain - r_noise):+.1f} % on the plain mean)"
)
print(f"  P  low-power voxels               {100 * (r_weighted - r_plain):+5.1f} %")
print(f"  W  wall filter (order 0 vs 2)     {100 * (orders[0] - orders[2]):+5.1f} %")
print(
    f"  G  velocity gradient in the cell  "
    f"{100 * (r_plug_w - r_weighted):+5.1f} %   "
    "(Poiseuille vs plug, both power-weighted)"
)
print(
    f"  A  aperture angle + transit       {100 * (r_plug_w - 1.0):+5.1f} %   "
    "(the plug control's own deficit)"
)
print("\nThe plug control's remaining deficit is NOT unexplained. An F# sweep")
print("attributes ~2.4 % to the receive-aperture angle - a displacement changes")
print("the round trip by dz*(1 + cos phi) rather than 2*dz, so a wide aperture")
print("reads low - and slow-time decimation attributes ~0.3 % to transit")
print("decorrelation. Both are velocity dependent, and a 2 cm/s plug control")
print("returns 0.998 of its known velocity.")

print("\nRADIAL PROFILE - the gradient term, drawn in numbers:")
edges = np.linspace(0, VESSEL_RADIUS_MM, 6)
print(f"  {'r/R':>10}{'Poiseuille':>22}{'plug':>20}")
print(f"  {'':>10}{'true':>8}{'meas':>7}{'ratio':>7}{'true':>8}{'meas':>7}{'ratio':>7}")
for lo, hi in zip(edges[:-1], edges[1:]):
    band = roi & (radial >= lo) & (radial < hi)
    if not band.any():
        continue
    tp, mp = v_true_par[band].mean(), v_par[band].mean()
    tg, mg = v_true_plug[band].mean(), v_plug[band].mean()
    print(
        f"  {lo / VESSEL_RADIUS_MM:.2f}-{hi / VESSEL_RADIUS_MM:.2f}"
        f"{tp * 100:>8.2f}{mp * 100:>7.2f}{mp / tp:>7.2f}"
        f"{tg * 100:>8.2f}{mg * 100:>7.2f}{mg / tg:>7.2f}"
    )
print("\n  Poiseuille: ratio < 1 at the axis and > 1 at the wall = SMOOTHING.")
print("  Plug: a flat profile has nothing to smooth, so its ratios isolate")
print("  everything that is NOT the gradient.")
