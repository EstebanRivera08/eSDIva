"""
Step 3 — beamform all three arms and estimate velocity with ONE estimator.

The comparison is only meaningful if nothing downstream of the transmit differs,
so every arm goes through the identical chain: beamform -> `rf2iq` (analytic
signal, then down-mix to baseband IQ) -> `wfilt` (clutter) -> `iq2doppler`
(lag-one autocorrelation). The only thing that changes is how the transmit spent its
emissions. Results are stored for step 4 to draw.

HOW EACH ARM IS BEAMFORMED
--------------------------
Conventional (A): each focused transmit illuminates ONE line, so its echoes are
beamformed onto that line's voxel column only — reconstructing a whole image
from one focused transmit would just be inventing data the transmit never
insonified. The 8 emissions of a line's packet become that column's slow time.

Ultrafast (B, C): every emission insonifies the whole box, so the `N_ANGLES`
angles of one frame are summed COHERENTLY by `das_volume` (that is what
synthesises a transmit focus everywhere), and each compound frame is one
slow-time sample. Compounding never crosses frames: that would average away the
very phase that carries the flow.

THE CENTRE FREQUENCY MUST BE MEASURED
-------------------------------------
Every Doppler estimator converts phase to velocity with `v = c·PRF/(4π·f)·Δφ`,
and `f` is the centre frequency of the RECEIVED echo, not the number printed on
the probe. The two-way chain (drive x TX impulse response x RX impulse response,
plus the aperture's own response) reshapes the spectrum, so the echo here centres
near 2.45 MHz on a 3 MHz probe; assuming the nominal value biases every velocity
by -18 %. Estimating the frequency instead of assuming it is the correction
Loupas introduced over the classic Kasai estimator; the version here is a global
simplification of that idea, measured from the data.

It must be measured on the signal the ESTIMATOR processes. Delay-and-sum
low-passes the data, so the beamformed signal is centred below the raw channel
RF. On the earlier 5 MHz build of this example that gap was 4.27 against
4.46 MHz, and using the channel figure biased every velocity low by 4.5 %;
measuring on the beamformed lines removed it, taking a plug-flow control from
0.955 to 0.998 of its known velocity.

Run with:
    uv run examples/example22_flow_doppler/step3_doppler_processing.py
"""

import json

import numpy as np
from doppler_tools import (
    doppler_spectrum,
    echo_center_frequency,
    iq2doppler,
    max_velocity_envelope,
    rf2iq,
    wfilt,
)
from step1_define_flow_phantom import (
    ARMS,
    C,
    FLOW_DIR,
    FC,
    THETA,
    VESSEL_CENTER_MM,
    VESSEL_LENGTH_MM,
    VESSEL_RADIUS_MM,
    V_PEAK,
    CLUTTER_ROI_MM,
    DX_MM,
    E_LONG,
    E_SHORT,
    GATE_MM,
    GRID_MM,
    IQ_FILE,
    METRICS_FILE,
    N_ANGLES,
    PRF_DOPPLER,
    RF_DIRS,
    SPEEDUP,
    V_AXIAL_TRUE,
    compound_events,
    focused_events,
    line_positions_mm,
    make_probe,
)

from esdiva.beamforming import das_volume
from esdiva.io import RFDataset
from esdiva.utilities import add_noise, rf_rms

print("\n--- Example 22 - Step 3: Doppler processing (3 arms) ---\n")

probe = make_probe()
_, frame_events = compound_events(probe, 1)
# The SAME event list the acquisition fired: das_volume recovers each transmit's
# time origin from its own per-element delays, so a stand-in would desynchronise
# the reconstruction from the data.
conventional_events = focused_events(probe, 1)  # one per line, un-repeated


def _radial_profile(field, roi, radial_mm, n_bands=6):
    """Mean of `field` in equal radial bands across the lumen, axis outwards."""
    edges = np.linspace(0.0, VESSEL_RADIUS_MM, n_bands + 1)
    return np.array(
        [
            field[roi & (radial_mm >= lo) & (radial_mm < hi)].mean()
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )


def beamform_conventional(rf, coords, ensemble):
    """Arm A: each focused transmit -> its own line, packets -> slow time."""
    x_lines = line_positions_mm()
    columns = []
    for i, x_line in enumerate(x_lines):
        line_grid = dict(GRID_MM, x_extent=[float(x_line), float(x_line) + DX_MM])
        frames = []
        for e in range(ensemble):
            n = i * ensemble + e
            vol, gc = das_volume(
                rf[n : n + 1],
                {"dt": coords["dt"], "t0_per_event": coords["t0_per_event"][n : n + 1]},
                [conventional_events[i]],
                probe,
                line_grid,
                c=C,
                fnum=1.0,
            )
            frames.append(vol[0, 0, :])
        columns.append(np.stack(frames))  # (ensemble, Nz)
        if i == 0:
            z_mm = gc["z_mm"]
    stacked = np.stack(columns, axis=1)  # (ensemble, N_LINES, Nz)
    return stacked, x_lines, z_mm


def beamform_compound(rf, coords, ensemble):
    """Arms B and C: angles summed within a frame, frames kept separate."""
    frames = []
    for k in range(ensemble):
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
    return np.stack(frames), gc["x_mm"], gc["z_mm"]


def vessel_geometry(x_mm, z_mm):
    """Lumen mask and the TRUE axial velocity at every voxel of the image plane.

    Every arm is scored over this same set of voxels. Scoring each arm over its
    own power-thresholded mask would compare different pixel populations — an
    arm that detects less flow would be judged on only its brightest voxels and
    could look artificially clean.

    The truth is evaluated voxel by voxel from the Poiseuille profile that built
    the phantom, NOT from a closed-form average. The image plane cuts the vessel
    along a DIAMETER, so its voxels sample the radius uniformly and their mean
    velocity is (2/3)·v_peak — not the (1/2)·v_peak that averaging over the
    circular cross-section would give. Computing it numerically from the actual
    voxel set avoids having to get that distinction right twice.
    """
    xx, zz = np.meshgrid(x_mm, z_mm, indexing="ij")
    pts = np.stack([xx, np.zeros_like(xx), zz], axis=-1) - VESSEL_CENTER_MM
    along = pts @ FLOW_DIR
    radial = np.linalg.norm(pts - along[..., None] * FLOW_DIR, axis=-1)
    roi = (radial < VESSEL_RADIUS_MM) & (np.abs(along) < VESSEL_LENGTH_MM / 2)
    v_true = V_PEAK * (1.0 - (radial / VESSEL_RADIUS_MM) ** 2) * np.cos(THETA)
    return roi, v_true, radial


# ============================================================================
# RECEIVER NOISE — required for the sensitivity claim to mean anything
# ============================================================================
# eSDIva's RF is noiseless BY DESIGN, and without noise a longer ensemble has
# nothing to average down: arm C's whole point would be invisible. Real Doppler
# sensitivity is a contest against the receiver's thermal noise, so one is added
# here — white, Gaussian, per channel and per emission.
#
# The noise level is a property of the RECEIVER, not of the sequence, so a single
# sigma is derived once and applied identically to all three arms. Anything else
# would hand one sequence a quieter amplifier than another.
NOISE_SNR_DB = 6.0  # single-emission channel SNR, referenced to the tissue echo
noise_reference = None
noise_rng = np.random.default_rng(11)

results, metrics = {}, {}
for arm in ("A", "B", "C"):
    rf_dir = RF_DIRS[arm]
    if not rf_dir.exists():
        raise SystemExit(f"No RF at {rf_dir} — run step2_acquire_RF.py first.")
    rf, coords = RFDataset(rf_dir).load_all()
    ensemble = ARMS[arm]["ensemble"]

    if noise_reference is None:  # fixed once, from the first arm, reused for all
        noise_reference = rf_rms(rf)
        print(
            f"Receiver noise: {NOISE_SNR_DB:.0f} dB channel SNR, referenced to "
            f"RMS {noise_reference:.3e} - the same absolute floor for every arm"
        )
    rf = add_noise(rf, NOISE_SNR_DB, reference=noise_reference, rng=noise_rng)

    if arm == "A":
        beamformed, x_mm, z_mm = beamform_conventional(rf, coords, ensemble)
    else:
        beamformed, x_mm, z_mm = beamform_compound(rf, coords, ensemble)

    # Identical processing from here on, so any difference is the transmit's.
    # The centre frequency is measured on the BEAMFORMED lines, not the channel
    # RF: delay-and-sum low-passes the data (sample interpolation + coherent
    # aperture summation), so the signal the estimator sees is centred below the
    # raw echo. Measured on the earlier 5 MHz build: 4.27 beamformed vs 4.46 MHz
    # on the channels; the channel figure biased every velocity low by 4.5 %, and
    # a plug-flow control moved from 0.955 to 0.998 of truth once corrected.
    fs_depth = C / (2.0 * (GRID_MM["dz"] * 1e-3))
    # Band scaled to the probe, never absolute numbers: a band left over
    # from another probe truncates the echo and rescales every velocity.
    f_echo = echo_center_frequency(
        beamformed, 1.0 / fs_depth, band=(0.3 * FC, 1.7 * FC)
    )
    iq = rf2iq(beamformed, fs=fs_depth, fc=f_echo, axis=-1)
    # Order 0 subtracts the ensemble mean, removing exactly what is perfectly
    # still - all this phantom contains. Real tissue motion would need a higher
    # order or method="eig".
    iq_flow = wfilt(iq, method="poly", order=0, axis=0)
    v_map, power = iq2doppler(iq_flow, PRF_DOPPLER, f_echo, c=C)

    gx = int(np.argmin(np.abs(x_mm - GATE_MM[0])))
    gz = int(np.argmin(np.abs(z_mm - GATE_MM[1])))
    cx = int(np.argmin(np.abs(x_mm - CLUTTER_ROI_MM[0])))
    cz = int(np.argmin(np.abs(z_mm - CLUTTER_ROI_MM[1])))
    roi = (slice(max(cx - 4, 0), cx + 5), slice(max(cz - 8, 0), cz + 9))
    p_tissue = 10 * np.log10(
        (np.abs(iq_flow[:, roi[0], roi[1]]) ** 2).mean()
        / (np.abs(iq[:, roi[0], roi[1]]) ** 2).mean()
    )

    # Average the spectrum over a few neighbouring voxels of the sample volume.
    # A single voxel of an 8-sample ensemble is dominated by estimator variance;
    # spatial averaging inside the gate is what a real system does too.
    gate_block = iq_flow[:, gx - 1 : gx + 2, gz - 3 : gz + 4].reshape(ensemble, -1)
    spectra = [
        doppler_spectrum(gate_block[:, j], PRF_DOPPLER, f_echo, c=C)
        for j in range(gate_block.shape[1])
    ]
    v_axis = spectra[0][0]
    spectrum = np.mean([sp for _, sp in spectra], axis=0)
    spectrum /= spectrum.max()
    v_edge = max_velocity_envelope(v_axis, spectrum, threshold=0.5)
    v_gate = float(v_map[gx, gz])

    mask = power > 0.05 * power.max()
    # Score every arm over the SAME geometric lumen, and use its MEAN rather than
    # one voxel: a single voxel of an 8-sample ensemble is dominated by estimator
    # variance, so comparing single voxels between arms would test noise.
    roi, v_true_map, radial_mm = vessel_geometry(x_mm, z_mm)
    v_roi_true = float(v_true_map[roi].mean())
    v_roi_mean = float(np.mean(v_map[roi]))
    # Spread over that same fixed ROI is the sensitivity measure: with identical
    # true flow, a noisier estimator scatters the map more.
    v_std = float(np.std(v_map[roi]))
    # Sensitivity in absolute terms: how much of the lumen carries flow power
    # well above this arm's own tissue-region floor.
    floor = np.median(power[~roi])
    detected = float(np.mean(power[roi] > 10.0 * floor))

    results[arm] = {
        "v_map": v_map,
        "power": power,
        "mask": mask,
        "v_axis": v_axis,
        "spectrum": spectrum,
        "iq": iq,
        "iq_flow": iq_flow,
        "x_mm": x_mm,
        "z_mm": z_mm,
        "gate": (gx, gz),
        # Radial profile: measured against the parabola that built the phantom.
        # Step 4 draws these to show how far the PSF flattens the profile.
        "profile_measured": _radial_profile(v_map, roi, radial_mm),
        "profile_true": _radial_profile(v_true_map, roi, radial_mm),
    }
    metrics[arm] = {
        "label": ARMS[arm]["label"],
        "n_emissions": int(ARMS[arm]["n_emissions"]),
        "ensemble": int(ensemble),
        "f_echo_Hz": float(f_echo),
        "v_gate_m_s": v_gate,
        "v_edge_m_s": float(v_edge),
        "v_roi_mean_m_s": v_roi_mean,
        "v_roi_true_m_s": v_roi_true,
        "v_std_m_s": v_std,
        "detected_frac": detected,
        "psf_flattening": "core reads slow, wall reads fast — see the profile",
        "psf_core_ratio": float(
            v_map[roi & (radial_mm < 0.4 * VESSEL_RADIUS_MM)].mean()
            / v_true_map[roi & (radial_mm < 0.4 * VESSEL_RADIUS_MM)].mean()
        ),
        "psf_wall_ratio": float(
            v_map[roi & (radial_mm > 0.8 * VESSEL_RADIUS_MM)].mean()
            / v_true_map[roi & (radial_mm > 0.8 * VESSEL_RADIUS_MM)].mean()
        ),
        "flow_voxels": int(mask.sum()),
        "wall_filter_tissue_dB": float(p_tissue),
    }
    print(
        f"ARM {arm} ({ARMS[arm]['label']}): {ARMS[arm]['n_emissions']} emissions, "
        f"ensemble {ensemble}"
    )
    print(f"   beamformed {beamformed.shape}, f_echo {f_echo / 1e6:.2f} MHz")
    print(
        f"   gate velocity {v_gate * 100:+6.2f} cm/s   "
        f"envelope {v_edge * 100:+6.2f} cm/s"
    )
    print(
        f"   vessel ROI mean {v_roi_mean * 100:+6.2f} cm/s   "
        f"spread {v_std * 100:5.2f}   detected {100 * detected:4.0f}%"
    )

# ============================================================================
# THE PAPER'S CLAIMS, CHECKED
# ============================================================================
truth = V_AXIAL_TRUE
print("\n" + "=" * 68)
print(f"Truth: peak axial velocity {truth * 100:.1f} cm/s")
print("=" * 68)
V_ROI_TRUE = metrics["A"]["v_roi_true_m_s"]  # identical for every arm
print(f"Vessel-mean truth over the image plane: {V_ROI_TRUE * 100:.2f} cm/s")
print("=" * 68)
print(f"{'arm':<26}{'emis':>7}{'ROI mean':>10}{'spread':>9}{'detect':>8}")
for arm in ("A", "B", "C"):
    m = metrics[arm]
    print(
        f"{m['label']:<26}{m['n_emissions']:>7}"
        f"{m['v_roi_mean_m_s'] * 100:>10.2f}{m['v_std_m_s'] * 100:>9.2f}"
        f"{100 * m['detected_frac']:>7.0f}%"
    )

d_ab = abs(metrics["B"]["v_roi_mean_m_s"] - metrics["A"]["v_roi_mean_m_s"])
print("\nCLAIM 1 - same performance, fewer emissions:")
print(f"  ultrafast uses {SPEEDUP:.1f}x fewer emissions than conventional")
print(
    f"  vessel-mean velocities agree to {d_ab * 100:.2f} cm/s "
    f"({100 * d_ab / abs(V_ROI_TRUE):.1f}% of the true vessel mean)"
)

# The resolution cell is a sizeable fraction of the 3 mm lumen, so every voxel
# averages its neighbourhood: the measured profile comes out FLATTER than the
# true parabola - too slow on the axis, too fast at the wall - which pulls the
# vessel MEAN below the truth. Measured, not assumed: a uniform scale error
# would give the same ratio at both radii, and it does not.
core_r, wall_r = metrics["C"]["psf_core_ratio"], metrics["C"]["psf_wall_ratio"]
print("\nPSF SMOOTHING - the profile is flattened, not scaled:")
print(f"  vessel core (r<0.4R) reads {core_r:.2f}x the true velocity there")
print(f"  vessel wall (r>0.8R) reads {wall_r:.2f}x")

print("\nCLAIM 2 - same budget buys sensitivity instead:")
print(
    f"  arm C spends arm A's {metrics['A']['n_emissions']} emissions on a "
    f"{E_LONG // E_SHORT}x longer ensemble"
)
print(
    f"  velocity spread {metrics['B']['v_std_m_s'] * 100:.2f} -> "
    f"{metrics['C']['v_std_m_s'] * 100:.2f} cm/s, "
    f"detection {100 * metrics['B']['detected_frac']:.0f}% -> "
    f"{100 * metrics['C']['detected_frac']:.0f}%"
)

# Both compounded arms must recover the same flow as the conventional reference,
# and every arm must land near the known truth. A sign error, a wrong centre
# frequency or a mis-scaled PRF in any arm breaks these.
for arm in ("A", "B", "C"):
    m = metrics[arm]
    assert m["v_roi_mean_m_s"] > 0, (
        f"arm {arm}: flow recedes, velocity must be positive"
    )
    assert 0.6 * V_ROI_TRUE < m["v_roi_mean_m_s"] < 1.3 * V_ROI_TRUE, (
        f"arm {arm} vessel-mean velocity {m['v_roi_mean_m_s'] * 100:.1f} cm/s "
        f"should track the true vessel mean {V_ROI_TRUE * 100:.1f} cm/s"
    )
    # The maximum-velocity envelope is only asserted where the ensemble can
    # support it. Eight slow-time samples cannot resolve a spectral edge against
    # noise - a real pulsed-wave system builds its spectrogram from hundreds of
    # samples - so for arms A and B the envelope is reported, not tested.
    if m["ensemble"] >= 32:
        assert 0.6 * truth < m["v_edge_m_s"] < 1.5 * truth, (
            f"arm {arm} envelope {m['v_edge_m_s'] * 100:.1f} cm/s should track "
            f"the known peak {truth * 100:.1f} cm/s"
        )
assert d_ab < 0.3 * abs(V_ROI_TRUE), (
    "ultrafast and conventional must agree: that is the paper's claim"
)
# Claim 2: the same emission budget, spent on ensemble instead of on scanning
# lines, must produce a measurably tighter velocity estimate.
assert metrics["C"]["v_std_m_s"] < 0.8 * metrics["B"]["v_std_m_s"], (
    f"arm C's longer ensemble should tighten the estimate: "
    f"{metrics['C']['v_std_m_s'] * 100:.2f} vs {metrics['B']['v_std_m_s'] * 100:.2f} cm/s"
)
print("\nOK - all three arms recover the known flow; the claims hold.")

IQ_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    IQ_FILE,
    **{
        f"{arm}_{k}": v
        for arm, r in results.items()
        for k, v in r.items()
        if k not in ("gate",)
    },
    **{f"{arm}_gate": np.array(r["gate"]) for arm, r in results.items()},
)
METRICS_FILE.write_text(
    json.dumps(
        {
            "arms": metrics,
            "truth_m_s": float(truth),
            "speedup": float(SPEEDUP),
            "v_nyquist_m_s": float(C * PRF_DOPPLER / (4.0 * metrics["B"]["f_echo_Hz"])),
        },
        indent=2,
    )
)
print(f"\nStored {IQ_FILE.name} and {METRICS_FILE.name}")
print("Next: step4_visualize.py")
