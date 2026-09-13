"""
Step 1 — define the Bercoff compound-Doppler experiment.

This example reproduces the central comparison of ultrafast compound Doppler
imaging (Bercoff et al., IEEE TUFFC 58(1):134-147, 2011): a colour-flow image
built from a few TILTED PLANE WAVES, coherently summed, delivers the same
Doppler performance as conventional line-by-line focused Doppler while spending
far fewer emissions - 8x fewer in the geometry set up here.

Everything steps 2-4 need is declared HERE, once. Nothing is simulated in this
file; run it directly and it previews the setup in 3-D.

THE CLAIM BEING TESTED
----------------------
Conventional colour Doppler scans the box one focused line at a time, and each
line needs its own ensemble ("packet") of E emissions to measure a velocity. The
cost is therefore `N_lines x E`. Ultrafast compounding insonifies the WHOLE box
with every emission, so one ensemble of E compound frames images everything at
once, costing `N_angles x E`. The saving is the ratio `N_lines / N_angles`,
reported as 10-15x for a typical colour box of ~100 lines with ~9 angles.

Three acquisitions on ONE phantom make the comparison, all sharing the same
Doppler PRF, ensemble estimator and velocity scale:

    A  conventional   N_LINES x E_SHORT   focused transmits (the reference)
    B  ultrafast      N_ANGLES x E_SHORT  compounded plane waves (the speed claim)
    C  sensitivity    N_ANGLES x E_LONG   same emission budget as A, spent on a
                                          ~10x longer ensemble instead

A and B test "same performance, far fewer emissions". C tests the paper's second
option: keep the acquisition time and buy sensitivity instead of speed.

THE DESIGN RULE, AND HOW IT CONSTRAINS THIS SEQUENCE
----------------------------------------------------
The number of angles you can afford is set by

    N_angles = PRF_max / PRF_doppler

where `PRF_max = c / (2·z_max)` is the depth limit — you cannot fire again until
the last echo is back — and `PRF_doppler` is the slow-time rate you need for the
velocity scale. That rule is not decoration here: at `z_max = 21 mm`,
`PRF_max = 34.2 kHz`, so nine angles cap the Doppler PRF at 3.7 kHz, and with it
the unambiguous velocity. Compounding is never free; it is always paid for in
velocity range.

THE PROBE, AND WHY EVERY DIMENSION IS TIED TO THE WAVELENGTH
------------------------------------------------------------
A 3 MHz linear array: 64 elements on a half-wavelength pitch, a 14 mm elevation
aperture, and an acoustic lens focused on the vessel.

Elevation is the direction the image cannot show, and it is where a Doppler study
quietly goes wrong: a voxel collects blood from above and below the imaging plane
too, where the vessel is narrower and the flow slower, so the measured profile
comes back flattened. The lens puts the elevation slice at 0.59 mm against a 3 mm
lumen, so what the colour map reports is the flow, not the beam.

`elevation_focus_mm` is the lens RADIUS OF CURVATURE, not the focal depth: the rim
is the z = 0 datum, so the line focus lands one sagitta shallower. At 3 MHz that
sagitta is 1.46 mm - nearly three wavelengths - so the difference is not a detail.
`step1` asks the probe (`elevation_focus_depth_mm`) instead of assuming.

SIZING: WHAT ACTUALLY COSTS, AND WHAT IS FREE
----------------------------------------------
The bill is (scatterers x emissions), and lowering the frequency does not reduce
it: line spacing, resolution cell and speckle density all scale with the
wavelength together, so the required scatterer count is the same at 3 MHz as at
12 MHz. What costs is the DIMENSIONLESS design - how many lines, how many
scatterers per resolution cell, and how many wavelengths across the vessel is.

So the counts here are set from the MEASURED resolution cell (`CELL_MM3`), not
from a total picked by eye: 5 blood scatterers per cell and 1.5 for tissue. Get
that wrong and the phantom is under-sampled, at which point the flow map turns
moth-eaten and every texture statistic is an artefact of the random draw - the
count alone never tells you, because it only means something next to the cell.

The box is 80 lines rather than a clinical ~100 because both terms of the bill
grow with it: a wider box needs more lines AND a longer vessel to cross it. The
reported speed-up is N_lines/N_angles by construction, so this box shows 8.0x
where a 100-line one shows 10-15x. Same mechanism, slightly shorter lever.

WHAT DOPPLER ACTUALLY MEASURES
------------------------------
Not a frequency shift inside one echo — the PHASE an echo gains from one
emission to the next, because the blood moved `v/PRF` in between. A scatterer
receding by `dz` lengthens the round trip by `2·dz`, turning the echo phase by
`4·pi·f·dz/c`. Stacking those increments over an ensemble is every Doppler
estimator there is.

`sequence_rf` reproduces exactly that: one scatterer cloud PER EMISSION,
`(N_events, N_scat, 3)`, so each transmit sees the medium where it truly is.
eSDIva has no slow-time clock, so the trajectory is ours to write — here
`pos = pos0 + v·(m/PRF)` in `blood_at()`. Scatterers are frozen WITHIN each
emission, so the RF carries the inter-emission phase a Doppler estimator reads,
not an intra-pulse frequency shift; that is the same modelling boundary Field II
works within.

Only the AXIAL velocity component is visible, which is why the vessel is tilted
`THETA_DEG` from the beam and the truth to recover is `V_PEAK·cos(THETA)`.

Run with:
    uv run examples/example22_flow_doppler/step1_define_flow_phantom.py
(previews the phantom; normally imported by steps 2-4).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import FIG_FOLDER, SAVE_FIG  # noqa: E402

from esdiva.reception import Reception  # noqa: E402
from esdiva.transducers import LinearArrayTransducer  # noqa: E402

# ============================================================================
# MEDIUM, PROBE AND PULSE
# ============================================================================
C = 1540.0  # m/s
FS = 30e6  # Hz — RF sampling, 10 per carrier cycle
FC = 3e6  # Hz — probe centre frequency (NOMINAL; measured in step 3)
PULSE_CYCLES = 2
LAMBDA_MM = C / FC * 1e3  # 0.513 mm

# A 3 MHz linear array: 64 elements on a half-wavelength pitch, which is the
# density that samples the aperture without grating lobes at any steering angle.
# EVERY dimension below is tied to the wavelength, because the resolution cell
# scales with it and the cell is what this study is really about.
N_ELEMENTS = 64
ELEMENT_WIDTH_MM = 0.220
KERF_MM = 0.020
PITCH_MM = ELEMENT_WIDTH_MM + KERF_MM  # 0.240 mm = 0.47 lambda
APERTURE_MM = N_ELEMENTS * PITCH_MM  # 15.36 mm

# Elevation: a 14 mm aperture with an acoustic lens focused ON THE VESSEL. This
# is the axis the image cannot show, and an unfocused aperture averages in
# out-of-plane blood that is slower, flattening the measured profile.
# ELEVATION_FOCUS_MM is the lens RADIUS OF CURVATURE, and the lens rim is the
# z = 0 datum, so the line focus (the arc's centre of curvature) sits one sagitta
# shallower: radius = sqrt(z_focus^2 + (H/2)^2) puts it exactly on the vessel.
# The sagitta is 1.46 mm here, nearly three wavelengths - far too large to
# hand-wave, which is why the probe is asked (`elevation_focus_depth_mm`).
ELEMENT_HEIGHT_MM = 14.0
ELEVATION_FOCUS_MM = 17.464  # -> line focus at 16.000 mm
NO_SUB_Y = 6  # patches across the curved lens; 6 matches 10 to 0.04 mm

# ============================================================================
# THE COLOUR BOX AND THE TWO SEQUENCES
# ============================================================================
BOX_X_MM = 9.225  # half-width of the imaged colour box
BOX_Z_MM = (10.5, 21.5)
Z_MAX_MM = 22.5  # deepest scatterer — this sets PRF_max

# Conventional: one focused line per lateral position, spaced about lambda/2,
# which is the density a scanner actually uses. 96 lines is the ~100-line colour
# box the paper compares against - and at 3 MHz that box is physically 24.6 mm
# wide, because line spacing is set by the wavelength. Lower frequency does not
# make the comparison cheaper: the same LINE COUNT needs a bigger scene, and the
# scatterer count grows with it.
# 72 lines rather than the ~100 of a clinical colour box. The cost of this
# comparison is (scatterers x emissions) and BOTH grow with the box - a wider box
# needs more lines AND a longer vessel to cross it - so the box is the single
# most expensive choice here. The speed-up the paper reports is N_lines/N_angles
# by construction: 8.0x at this width, 10-15x for a clinical one.
N_LINES = 72
DX_MM = 2 * BOX_X_MM / N_LINES  # 0.256 mm, about lambda/2
Z_FOCUS_MM = 16.0  # transmit focus, at the vessel depth

# Ultrafast: 9 tilted plane waves, the number reported to match conventional
# resolution and sensitivity. At 0.47 lambda pitch the grating-lobe condition
# sin(theta) < lambda/pitch - 1 is satisfied at EVERY angle, so the +/-5 degree
# span here is chosen to match the paper rather than forced by the array.
N_ANGLES = 9
# Span set by the APERTURE, not by habit: coherent compounding sharpens the
# synthesised transmit focus only until the tilt span matches the aperture's own
# half-angle (25.6 deg at the vessel here), then it saturates. Measured on a
# point target: +/-5 deg gives a 1.90 mm lateral PSF, +/-10 -> 1.70, +/-15 ->
# 1.50, +/-20 -> 1.20, +/-26 -> 1.20. The half-wavelength pitch is what allows
# it - the grating-lobe condition sin(theta) < lambda/pitch - 1 never binds.
ANGLES_DEG = np.linspace(-20.0, 20.0, N_ANGLES)

E_SHORT = 8  # ensemble length for arms A and B (a conventional packet)
# Arm C trades A's emission budget for ensemble length. 4x rather than the
# paper's ~15x: enough to show the variance drop, and it keeps the whole
# three-arm comparison inside a coffee break on a laptop.
E_LONG = 32

# Slow-time rate, identical in every arm so all three share one velocity scale.
# 3.7 kHz x 9 angles = 33.3 kHz of emissions, just inside the 34.2 kHz the depth
# allows. The rule bites exactly as the paper describes it.
PRF_DOPPLER = 3700.0
PRF_MAX = C / (2.0 * Z_MAX_MM * 1e-3)  # depth limit on the emission rate
PRF_EMISSION = PRF_DOPPLER * N_ANGLES  # what arm B/C actually fire at

N_EMISSIONS_A = N_LINES * E_SHORT
N_EMISSIONS_B = N_ANGLES * E_SHORT
N_EMISSIONS_C = N_ANGLES * E_LONG
SPEEDUP = N_EMISSIONS_A / N_EMISSIONS_B

# ============================================================================
# THE VESSEL
# ============================================================================
THETA_DEG = 60.0  # tilt away from the beam axis
# Chosen against the Nyquist limit below, not for roundness: the measured
# spectral envelope overshoots the true axis peak by ~15 % (PSF broadening),
# so the peak needs real headroom under v_nyquist, not just to fit under it.
V_PEAK = 0.45  # m/s on the vessel axis -> 22.5 cm/s axial, ~44 % of Nyquist
VESSEL_CENTER_MM = np.array([0.0, 0.0, 16.0])
# A 4 mm lumen against the MEASURED resolution cell below. The lateral axis is
# the weak one - coherent compounding of 9 plane waves resolves 1.25 mm where a
# single focused transmit resolves 0.70 - so the lumen is sized against that.
VESSEL_RADIUS_MM = 2.0
VESSEL_LENGTH_MM = 24.0

# MEASURED resolution cell, not the lambda*F# estimate. The analytic formula
# under-predicted it by 1.5x here, and a scatterer count is meaningless except
# relative to the cell, so the cell is measured on a point target instead:
# -6 dB widths of 1.25 mm lateral (9-angle compound), 0.61 mm elevation and
# 0.32 mm axial. The elevation figure matches lambda*z/H to 3 %, which is what a
# FOCUSED elevation aperture should do - the same formula was refuted for an
# unfocused one.
CELL_LAT_MM = 1.25  # 9-angle compound; a single focused transmit gives 0.70
CELL_ELEV_MM = 0.61  # matches lambda*z/H to 3 % - a FOCUSED elevation aperture
CELL_AX_MM = 0.32
CELL_MM3 = CELL_LAT_MM * CELL_ELEV_MM * CELL_AX_MM

# Speckle is the interference of many scatterers per resolution cell, so counts
# are set from CELL_MM3: 5 per cell for blood (the textbook target, and the one
# that decides whether the flow map is continuous or moth-eaten) and 1.5 for
# tissue, which is the honest compromise with runtime.
N_TISSUE = 6000  # 1.2 per resolution cell
N_BLOOD = 6180  # 5.0 per resolution cell
# Blood scatters far more weakly than tissue — red cells are much smaller than a
# wavelength, which is why a vessel reads as a DARK lumen on B-mode and why flow
# has to be dug out from under the tissue echo. Field II vessel studies use 40-60
# dB; -14 dB is used here so the lumen is visibly dark while the flow still
# survives the SHORT 8-emission ensemble of arms A and B.
BLOOD_ECHOGENICITY = 0.2
# The slab must be thicker than the vessel in y, or the lumen would be clipped
# out of plane and the elevation average would run over blood that is not there.
BOX_MM = {"x": (-10.2, 10.2), "y": (-2.3, 2.3), "z": (9.5, Z_MAX_MM)}
SEED = 2026
# Tissue-only patch for measuring wall-filter suppression. The vessel crosses
# the whole depth of the box, so the patch is offset LATERALLY instead: at
# z = 11 mm the lumen sits near x = -8.7 mm, well clear of here.
CLUTTER_ROI_MM = (8.0, 11.0)

# ============================================================================
# BEAMFORMING GRID
# ============================================================================
# ONE elevation slice: the imaging plane is what a linear array resolves, and
# averaging several y-slices only smears the echo axially.
GRID_MM = {
    "x_extent": [-BOX_X_MM, BOX_X_MM],
    "y_extent": [-0.05, 0.05],
    "z_extent": [BOX_Z_MM[0], BOX_Z_MM[1]],
    "dx": DX_MM,  # one voxel column per conventional line
    "dy": 0.10,
    # The beamformed line is what the Doppler chain demodulates, so its depth
    # step has to sample the CARRIER, not just the envelope: 0.10 mm gives
    # fs_depth = c/(2*dz) = 7.7 MHz, about 3 samples per cycle at 2.5 MHz.
    "dz": 0.10,
}
GATE_MM = (0.0, 16.0)  # (x, z) of the spectral-Doppler sample volume

# ---- derived ---------------------------------------------------------------
THETA = np.deg2rad(THETA_DEG)
FLOW_DIR = np.array([np.sin(THETA), 0.0, np.cos(THETA)])
V_AXIAL_TRUE = V_PEAK * np.cos(THETA)  # the number Doppler must recover
V_NYQUIST = C * PRF_DOPPLER / (4.0 * FC)  # nominal-frequency estimate

OUT_DIR = Path(__file__).parent / "out"
RF_DIRS = {arm: OUT_DIR / f"RF_{arm}" for arm in ("A", "B", "C")}
IQ_FILE = OUT_DIR / "doppler_results.npz"
METRICS_FILE = OUT_DIR / "metrics.json"

ARMS = {
    "A": {
        "label": "conventional focused",
        "n_emissions": N_EMISSIONS_A,
        "ensemble": E_SHORT,
    },
    "B": {
        "label": "ultrafast compound",
        "n_emissions": N_EMISSIONS_B,
        "ensemble": E_SHORT,
    },
    "C": {
        "label": "ultrafast, long ensemble",
        "n_emissions": N_EMISSIONS_C,
        "ensemble": E_LONG,
    },
}


def make_probe():
    """One linear array. TX and RX must be SEPARATE instances.

    Reception applies delays and apodization PER RECEIVE CHANNEL, so passing one
    object as both would also stamp the transmit law onto the receive traces and
    silently corrupt the RF.
    """
    return LinearArrayTransducer(
        n_elements=N_ELEMENTS,
        element_width_mm=ELEMENT_WIDTH_MM,
        element_height_mm=ELEMENT_HEIGHT_MM,
        kerf_mm=KERF_MM,
        no_sub_x=1,
        no_sub_y=NO_SUB_Y,
        elevation_focus_mm=ELEVATION_FOCUS_MM,
        frequency_Hz=FC,
    )


def excitation():
    """The electrical drive: a 2-cycle Hanning-windowed burst at `FC`."""
    t = np.arange(0, PULSE_CYCLES / FC, 1.0 / FS)
    return (np.sin(2 * np.pi * FC * t) * np.hanning(len(t))).astype(np.float32)


def build_simulator():
    """Probe pair + `Reception`, with the piezo impulse responses SET.

    A physical probe band-passes the signal twice — drive x TX impulse response
    x RX impulse response. Omitting them models ideally broadband elements, and
    the aperture's own low-frequency diffraction tails then dominate the received
    spectrum. For a Doppler study that is not cosmetic: the velocity scale is
    inversely proportional to the echo's centre frequency (step 3), so a wrong
    spectrum is a wrong velocity.
    """
    tx, rx = make_probe(), make_probe()
    exc = excitation()
    tx.impulse_response = exc.copy()
    rx.impulse_response = exc.copy()
    sim = Reception(tx, rx, c=C, fs=FS, excitation=exc, verbose=False)
    return sim, tx, rx


def line_positions_mm():
    """Lateral centre of each conventional scan line.

    Deliberately the SAME lateral positions as the beamforming grid's voxel
    columns, so a conventional line and an ultrafast voxel column can be
    compared directly without resampling one onto the other.
    """
    return np.arange(-BOX_X_MM, BOX_X_MM, DX_MM)[:N_LINES]


def focused_events(probe, ensemble=E_SHORT):
    """Arm A: one focused transmit per line, each repeated `ensemble` times.

    A conventional colour-Doppler scanner dwells on a line for a whole PACKET of
    emissions before stepping to the next, because the velocity at that line is
    estimated from the slow-time phase within its own packet. The event order
    here is therefore line-major: all of line 0's ensemble, then line 1's.

    `virtual_source_mm` carries the transmit focus for `das_volume`, which needs
    the wavefront geometry; the simulator itself reads only delays/apodization.
    """
    events = []
    for x_line in line_positions_mm():
        focus = [float(x_line), 0.0, Z_FOCUS_MM]
        probe.compute_delays(focus_mm=focus)
        delays = np.asarray(probe.delays, np.float32).copy()
        for _ in range(ensemble):
            events.append(
                {
                    "delays": delays,
                    "apodization": np.ones(N_ELEMENTS, np.float32),
                    "virtual_source_mm": focus,
                }
            )
    probe.compute_delays(angle_steering_deg=0.0)  # leave the probe unsteered
    return events


def compound_events(probe, ensemble=E_SHORT):
    """Arms B and C: `N_ANGLES` tilted plane waves, re-fired every frame.

    The SAME angle list repeats for every compound frame. Keeping the transmit
    pattern identical frame to frame is what makes the slow-time phase a pure
    motion effect — a transmit that changed between frames would move the echoes
    by itself, and no estimator could separate that from flow.
    """
    frame = []
    for angle in ANGLES_DEG:
        probe.compute_delays(angle_steering_deg=float(angle))
        frame.append(
            {
                "delays": np.asarray(probe.delays, np.float32).copy(),
                "apodization": np.ones(N_ELEMENTS, np.float32),
                "angles_deg": float(angle),
            }
        )
    probe.compute_delays(angle_steering_deg=0.0)
    return frame * ensemble, frame


def build_phantom(n_emissions, prf_emission, profile="poiseuille"):
    """Stationary tissue speckle + a tilted vessel, one cloud per emission.

    Parameters
    ----------
    n_emissions : int
        Length of the sequence this phantom will feed.
    prf_emission : float
        Rate at which those emissions are fired (Hz). Blood advances `v/PRF`
        between consecutive emissions, so the sequences differ here: the
        compounded ones fire at `N_ANGLES x PRF_DOPPLER`, the conventional one
        sends a single beam per slow-time sample at `PRF_DOPPLER`.
    profile : {'poiseuille', 'plug'}, default 'poiseuille'
        Velocity profile across the lumen. ``'poiseuille'`` is the physical
        parabola; ``'plug'`` puts every scatterer at `V_PEAK` and exists as the
        control in `bias_budget.py`, where removing the velocity gradient is
        what separates PSF smoothing from an outright scale error.

    Returns
    -------
    positions_mm : (n_emissions, N_scat, 3) numpy.ndarray
        The moving phantom `sequence_rf` consumes. Tissue rows are identical in
        every frame; only the blood rows advance.
    amplitudes : (N_scat,) numpy.ndarray
        Fixed for the whole sequence.
    truth : dict
        Bookkeeping for steps 3 and 4.
    """
    rng = np.random.default_rng(SEED)

    # Vessel cross-section basis: e1 in the imaging plane, e2 along elevation.
    e1 = np.array([np.cos(THETA), 0.0, -np.sin(THETA)])
    e2 = np.array([0.0, 1.0, 0.0])
    s0 = rng.uniform(-VESSEL_LENGTH_MM / 2, VESSEL_LENGTH_MM / 2, N_BLOOD)
    # sqrt() of a uniform draw fills the disc evenly; radius alone would crowd
    # the axis and bias the velocity distribution toward the fast centre.
    radius = VESSEL_RADIUS_MM * np.sqrt(rng.uniform(0, 1, N_BLOOD))
    phi = rng.uniform(0, 2 * np.pi, N_BLOOD)
    offset = radius[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    # Poiseuille: laminar flow in a tube is parabolic — fastest on the axis, zero
    # at the wall. This is what gives the Doppler spectrum its width.
    if profile == "poiseuille":
        v_blood = V_PEAK * (1.0 - (radius / VESSEL_RADIUS_MM) ** 2)
    elif profile == "plug":
        # Every scatterer at one velocity. Not physical for a tube, but it is
        # the control that removes the velocity GRADIENT while changing nothing
        # else, which is what isolates gradient smoothing from a scale error.
        v_blood = np.full(N_BLOOD, V_PEAK)
    else:
        raise ValueError(f"unknown profile {profile!r}; use 'poiseuille' or 'plug'.")
    amp_blood = (BLOOD_ECHOGENICITY * rng.standard_normal(N_BLOOD)).astype(np.float32)

    # Tissue everywhere except inside the lumen — blood and tissue must not
    # overlap, or stationary scatterers would sit in the flow region and the wall
    # filter would have nothing to remove there.
    tissue = np.column_stack(
        [
            rng.uniform(*BOX_MM["x"], N_TISSUE),
            rng.uniform(*BOX_MM["y"], N_TISSUE),
            rng.uniform(*BOX_MM["z"], N_TISSUE),
        ]
    )
    d = tissue - VESSEL_CENTER_MM
    along = d @ FLOW_DIR
    in_lumen = (
        np.linalg.norm(d - along[:, None] * FLOW_DIR, axis=1) < VESSEL_RADIUS_MM
    ) & (np.abs(along) < VESSEL_LENGTH_MM / 2)
    tissue = tissue[~in_lumen]
    amp_tissue = rng.standard_normal(len(tissue)).astype(np.float32)

    def blood_at(m):
        """Blood positions (mm) at emission `m` — the whole motion model.

        Scatterers that run off the end of the tube are re-injected at the other
        end, which keeps the vessel uniformly full for the entire acquisition
        instead of slowly draining it.
        """
        s = s0 + v_blood * 1e3 * m / prf_emission
        s = (s + VESSEL_LENGTH_MM / 2) % VESSEL_LENGTH_MM - VESSEL_LENGTH_MM / 2
        return VESSEL_CENTER_MM + s[:, None] * FLOW_DIR + offset

    positions = np.stack(
        [np.vstack([tissue, blood_at(m)]) for m in range(n_emissions)]
    ).astype(np.float32)
    amplitudes = np.concatenate([amp_tissue, amp_blood])
    truth = {
        "n_tissue": len(tissue),
        "n_blood": N_BLOOD,
        "v_blood": v_blood,
        "radius_mm": radius,
    }
    return positions, amplitudes, truth


def describe():
    """Print the numbers that decide whether the comparison is well posed."""
    print(
        f"Probe      : {N_ELEMENTS} elements, pitch {PITCH_MM:.3f} mm "
        f"({PITCH_MM / LAMBDA_MM:.2f} lambda) at {FC / 1e6:.0f} MHz"
    )
    probe = make_probe()
    elev_slice = LAMBDA_MM * probe.elevation_focus_depth_mm / ELEMENT_HEIGHT_MM
    print(
        f"Elevation  : {ELEMENT_HEIGHT_MM:.1f} mm aperture, lens radius "
        f"{ELEVATION_FOCUS_MM:.1f} mm -> line focus at "
        f"{probe.elevation_focus_depth_mm:.2f} mm"
    )
    print(
        f"  -> slice ~lambda*z/H = {elev_slice:.2f} mm against a "
        f"{2 * VESSEL_RADIUS_MM:.1f} mm lumen "
        f"({elev_slice / (2 * VESSEL_RADIUS_MM):.2f} of it)"
    )
    # MEASURED on a point target, not lambda*F#: the analytic figure reads 0.53 mm
    # laterally where the compounded PSF is really 1.25 mm, and a resolution cell
    # quoted from a formula is how a phantom ends up under-sampled.
    d = 2 * VESSEL_RADIUS_MM
    print(
        f"Cell       : {CELL_LAT_MM:.2f} x {CELL_ELEV_MM:.2f} x {CELL_AX_MM:.2f} mm "
        f"measured = {CELL_MM3:.3f} mm3  (F# {Z_FOCUS_MM / APERTURE_MM:.2f})"
    )
    print(
        f"  -> as a fraction of the {d:.1f} mm lumen: "
        f"{CELL_LAT_MM / d:.2f} lateral, {CELL_ELEV_MM / d:.2f} elevation, "
        f"{CELL_AX_MM / d:.2f} axial"
    )
    print(f"Colour box : +/-{BOX_X_MM:.1f} mm x {BOX_Z_MM[0]:.0f}-{BOX_Z_MM[1]:.0f} mm")
    print()
    print("Sequences (all share one Doppler PRF, so one velocity scale):")
    print(
        f"  A conventional : {N_LINES} focused lines x {E_SHORT} "
        f"= {N_EMISSIONS_A:5d} emissions"
    )
    print(
        f"  B ultrafast    : {N_ANGLES} angles x {E_SHORT} "
        f"= {N_EMISSIONS_B:5d} emissions   -> {SPEEDUP:.1f}x fewer"
    )
    print(
        f"  C sensitivity  : {N_ANGLES} angles x {E_LONG} "
        f"= {N_EMISSIONS_C:5d} emissions   (A's budget, {E_LONG // E_SHORT}x "
        "the ensemble)"
    )
    print()
    # The design rule of the method: angles are bought with the PRF the depth
    # allows, and paid for out of the velocity scale.
    print(
        f"Design rule: N_angles = PRF_max/PRF_doppler = {PRF_MAX / 1e3:.1f} kHz / "
        f"{PRF_DOPPLER / 1e3:.1f} kHz = {PRF_MAX / PRF_DOPPLER:.1f}"
    )
    if N_ANGLES > PRF_MAX / PRF_DOPPLER:
        print(f"  !! {N_ANGLES} angles exceed the depth budget - lower PRF_DOPPLER")
    else:
        print(
            f"  -> {N_ANGLES} angles fit (emission PRF {PRF_EMISSION / 1e3:.1f} kHz "
            f"<= PRF_max {PRF_MAX / 1e3:.1f} kHz at {Z_MAX_MM:.0f} mm)"
        )
    print()
    print(f"Flow       : {V_PEAK * 100:.0f} cm/s peak, {THETA_DEG:.0f} deg to the beam")
    print(
        f"  -> axial component {V_AXIAL_TRUE * 100:.1f} cm/s  <- the truth to recover"
    )
    print(f"  -> v_nyquist = c*PRF_doppler/(4*fc) = {V_NYQUIST * 100:.1f} cm/s")
    if abs(V_AXIAL_TRUE) >= V_NYQUIST:
        print("  !! ALIASED: raise PRF_DOPPLER, use fewer angles, or slow the flow")
    else:
        print(f"  -> headroom x{V_NYQUIST / abs(V_AXIAL_TRUE):.2f}")


if __name__ == "__main__":
    print("\n--- Example 22 - Step 1: Bercoff compound-Doppler experiment ---\n")
    describe()
    positions, amplitudes, truth = build_phantom(4, PRF_EMISSION)
    print(f"\nPhantom    : {truth['n_tissue']} tissue + {truth['n_blood']} blood")
    print("  NOTE: below fully developed speckle density (5-10 per resolution")
    print("  cell); kept small so the whole comparison runs in minutes.")

    sim, tx, _ = build_simulator()
    if SAVE_FIG:
        FIG_FOLDER.mkdir(exist_ok=True)
    # A moving cloud is previewed at its FIRST emission.
    sim.show(
        positions,
        amplitudes,
        TX_color="blue",
        legend=False,
        save_path=str(FIG_FOLDER / "ex22_flow_setup.png") if SAVE_FIG else None,
    )
    print("\nPreview done. Next: step2_acquire_RF.py")
