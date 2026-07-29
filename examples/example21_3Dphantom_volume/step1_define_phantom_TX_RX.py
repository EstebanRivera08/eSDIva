"""
Step 1 — define the scenario: probe (TX = RX), phantom, and transmit sequence.

Everything the acquisition (step 2) and the beamforming (step 3) need is
declared here ONCE, per scenario. Pick the scenario below (or set the
``SCENARIO`` environment variable) and run the steps in order:

    SCENARIO = "zeus5"    ZeUS 55x55, 3025 ch, driven at 5 MHz (pitch 0.97λ:
                          no grating lobes; ~0.3 mm PSF). The flagship run.
    SCENARIO = "zeus10"   Same ZeUS probe at its 10 MHz centre frequency:
                          pitch 2λ → the echo field is spatially ALIASED and
                          the image is clutter-limited. Kept as the
                          cautionary counterpart (expensive: ~8 h).
    SCENARIO = "vermon"   Vermon-type 32x32, 1024 ch, 3 MHz — the research
                          probe of the 4-D ultrafast literature, on the
                          IDENTICAL phantom (~0.8 mm PSF; minutes to run —
                          the fast end-to-end pipeline test).

All three scenarios image the SAME phantom (identical volume, targets and
seed — the "contrast ladder"): a vertical column with a x4 hyperechoic
sphere (elevation-offset: absent at y=0, present in the +y slice — the proof
the volume is genuinely 3-D), an anechoic cyst at the volume mid-depth (the
compound transmit-focus peak) and a x4 tube below it; a second x4 tube
column on the other side; and dim PSF wires (amplitude 4 ≈ +10 dB over
speckle) on the clear lane between the columns — three lateral (along y) at
fixed depths plus a column of point beads (along z at y=0) crossing them, so
the lateral PSF is read at a ladder of depths. Only what the physics
dictates changes per scenario: the scatterer COUNT (the resolution cell
(λz/D)²·pulse/2 shrinks with frequency and aperture) and the virtual-source
layout (coverage + steep-wavefront rules).

The transmit basis is diverging waves (DW): each event delays the elements
as if the wavefront came from a virtual point source BEHIND the array, so
one shot insonifies the whole volume; compounding a few tens of tilted
sources synthesizes transmit focus everywhere. The virtual-source layouts
obey the coverage rule (a DW only insonifies the cone from the source
through the aperture edges) and each probe's steep-wavefront tolerance.

THE IMPULSE RESPONSE — essential, not optional
----------------------------------------------
The pulse-echo chain of a physical probe is ``e ⊛ h_e ⊛ h_r``: the electric
drive ``e`` is band-passed TWICE through the piezo impulse responses of the
transmitting and receiving elements. The probe builders below therefore set
``impulse_response`` on every transducer (a 2-cycle burst at ``fc``, Field II
practice), and the Reception class folds ``exc ⊛ ir_tx ⊛ ir_rx`` in the
frequency domain. Omitting the impulse responses models ideal BROADBAND
elements: the low-frequency tails of the aperture (diffraction) impulse
responses then dominate the received spectrum — measured on a point target
the spectral centroid fell 3.0 → 1.86 MHz, widening the lateral PSF by 60 %
and raising the near-in sidelobe skirt from −22 to −10 dB; that skirt was
the dominant clutter source in every early campaign.

One caveat: the RFDataset checkpoint fingerprint covers the excitation but
NOT the impulse responses — if you change the pulse model, delete the RF
folder yourself (the resume cannot detect the change).

Run with:
    uv run examples/example21_3Dphantom_volume/step1_define_phantom_TX_RX.py
(prints the selected scenario; normally imported by steps 2 and 3).
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import FIG_FOLDER

from pyfield.transducers import MatrixArrayTransducer

# Default scenario: vermon — the fastest acquisition, so the whole pipeline
# (steps 1-3 + visualizers) regenerates the documentation figures in minutes.
SCENARIO = os.environ.get("SCENARIO", "zeus10")

# --- Medium / sampling (shared by all scenarios) -------------------------------
C = 1540.0  # speed of sound (m/s)
FS = 100e6  # simulation sampling rate (Hz); RF is stored decimated
PULSE_CYCLES = 2  # cycles of the Hann-windowed drive burst
BEAD_DZ_MM = 1.0  # axial-bead spacing (mm), FIXED for every scenario so the
# beads sit at the SAME depths on all probes (comparable across frequency);
# 1 mm ≥ twice the coarsest axial resolution (Vermon PULSE_CYCLES·λ/2 = 0.51 mm)
# → beads always resolved, never λ-tied.
DOWNSAMPLING = 2  # store RF at FS/2 = 50 MHz (Nyquist-safe for all drives)
SEED = 2026  # one seed → every script rebuilds the identical phantom

# Data is grouped per scenario so probes never collide:
# out/<scenario>/RF  — the checkpointed acquisition (step 2, RFDataset)
# out/<scenario>/IQ  — the beamformed complex IQ volume + axes (step 3);
#                      visualization scripts read this, never re-beamform
# Figures go to the shared examples asset folder (docs/examples/assets), named
# ex21_<scenario>_*.png so every image traces back to this example.
OUT_DIR = Path(__file__).parent / "out" / SCENARIO
RF_DIR = OUT_DIR / "RF"
IQ_DIR = OUT_DIR / "IQ"
FIG_DIR = FIG_FOLDER


# --- Probes ---------------------------------------------------------------------
def make_zeus(fc: float = 5e6):
    """Build the full 55x55 ZeUS matrix (3025 elements individually wired).

    0.29 mm square elements, 0.30 mm pitch, 16.49 mm aperture; one patch per
    element (a 0.29 mm face is far inside the far-field patch limit
    ``w << sqrt(4·l·c/f)`` ≈ 7 mm at the shallowest target). Every element
    transmits and receives — the regime where the spectral pulse-echo kernel
    is fastest (one batched call builds all 3025 receive spectra) and where
    Field II needs ~20x longer per event.

    The piezo ``impulse_response`` (2-cycle burst at ``fc``) is set on the
    probe — essential: without it the elements are ideally broadband and the
    aperture impulse-response tails dominate the received spectrum (PSF +60 %,
    sidelobe skirt −22 → −10 dB).

    Returns
    -------
    MatrixArrayTransducer
        55x55 elements, 0.29 mm square, 0.30 mm pitch.
    """
    probe = MatrixArrayTransducer(
        n_elements_x=55,
        n_elements_y=55,
        element_width_mm=0.29,
        element_height_mm=0.29,
        kerf_x_mm=0.01,
        kerf_y_mm=0.01,
        no_sub_x=1,
        no_sub_y=1,
        frequency_Hz=fc,
    )
    probe.impulse_response = excitation(fc)
    return probe


def make_vermon(fc: float = 3e6):
    """Build the Vermon-type 32x32 matrix (1024 elements, 0.30 mm pitch).

    The published 4-D ultrafast research probe (Provost et al. 2014, Phys.
    Med. Biol. 59): at 3 MHz (λ = 0.513 mm) the 0.30 mm pitch is 0.58λ — the
    echo field is properly sampled — and the small 9.6 mm aperture gives a
    ~0.8 mm PSF at the phantom's 15 mm mid-depth. The physical probe has
    three dead rows; modelled contiguous (the gaps mainly raise far
    sidelobes slightly).

    The piezo ``impulse_response`` (2-cycle burst at ``fc``) is set on the
    probe — essential, see `make_zeus`.

    Returns
    -------
    MatrixArrayTransducer
        32x32 elements, 0.275 mm square, 0.30 mm pitch (0.58λ at 3 MHz).
    """
    probe = MatrixArrayTransducer(
        n_elements_x=32,
        n_elements_y=32,
        element_width_mm=0.275,
        element_height_mm=0.275,
        kerf_x_mm=0.025,
        kerf_y_mm=0.025,
        no_sub_x=1,
        no_sub_y=1,
        frequency_Hz=fc,
    )
    probe.impulse_response = excitation(fc)
    return probe


# --- Excitation -----------------------------------------------------------------
def excitation(fc: float):
    """Hann-windowed 2-cycle drive burst at ``fc`` (float32, sampled at FS)."""
    t = np.arange(0, PULSE_CYCLES / fc, 1.0 / FS)
    return (np.sin(2 * np.pi * fc * t) * np.hanning(t.size)).astype(np.float32)


# --- Phantom: the shared contrast-ladder design -----------------------------------
def resolution_cell_mm3(z_mm: float, fc: float, aperture_mm: float) -> float:
    """Approximate resolution-cell volume at depth ``z_mm``.

    Each lateral width is the one-way diffraction limit λ·z/D of the
    aperture; axially the cell is half the pulse length (two-way travel).
    Sets the scatterer density: fully developed Rayleigh speckle needs
    ≥ ~5–10 scatterers per cell.
    """
    lam = C / fc * 1e3
    lateral = lam * z_mm / aperture_mm  # per axis (x and y)
    axial = PULSE_CYCLES * lam / 2.0
    return lateral * lateral * axial


def tube_map(box_mm, shape, tubes):
    """3-D echogenicity map: background 1 with cylindrical tubes along y.

    Each tube's axis runs along y (elevation), so membership depends only on
    the in-plane distance to the axis: a point belongs if
    ``(x−cx)² + (z−cz)² < r²`` at any y — a vessel crossing the imaging
    plane, same (x, z) cross-section at every elevation.

    Parameters
    ----------
    box_mm : dict
        ``x_extent``/``y_extent``/``z_extent`` of the phantom (mm).
    shape : tuple[int, int, int]
        Map resolution ``(Nx, Ny, Nz)``.
    tubes : list[tuple]
        Each ``((cx, cz), radius_mm, gain)`` — gain 0 = anechoic,
        >1 = hyperechoic.

    Returns
    -------
    (Nx, Ny, Nz) numpy.ndarray
        Relative scattering strength for `make_phantom`.
    """
    axes = [np.linspace(*box_mm[f"{ax}_extent"], n) for ax, n in zip("xyz", shape)]
    X, _, Z = np.meshgrid(*axes, indexing="ij")
    emap = np.ones(shape)
    for (cx, cz), r, gain in tubes:
        emap[(X - cx) ** 2 + (Z - cz) ** 2 < r**2] = gain
    return emap


def phantom_map(box_mm, shape, tubes, spheres):
    """Echogenicity map of the contrast-ladder phantom: tubes + spheres.

    Tubes first (y-invariant cylinders), then the spheres painted on top —
    by design they never overlap.

    Returns
    -------
    (Nx, Ny, Nz) numpy.ndarray
        Relative scattering strength for `make_phantom`.
    """
    emap = tube_map(box_mm, shape, tubes)
    axes = [np.linspace(*box_mm[f"{ax}_extent"], n) for ax, n in zip("xyz", shape)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    for (cx, cy, cz), r, gain in spheres:
        emap[(X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 < r**2] = gain
    return emap


def build_phantom(sc: dict):
    """Scatterer cloud of a scenario (positions mm, amplitudes). Seeded.

    Random sub-wavelength scatterers at ``sc["per_cell"]`` per resolution
    cell (≥ ~5–10 gives fully developed Rayleigh speckle), amplitudes
    ``N(0,1)`` x echogenicity map, plus dim PSF wires — dense scatterer
    lines (spacing < λ/2 → acoustically continuous) whose image collapses to
    the PSF, so their width reads the resolution: three along y (elevation)
    at fixed depths, plus a column of point beads along z (axial) crossing
    them at y=0 (beads, not a dense line — see below). Wire
    amplitude 4 ≈ +10 dB over speckle: bright enough to read, faint enough
    that the sidelobe skirt stays below the speckle (amplitude 10 once washed
    whole depth planes).
    """
    from pyfield.utilities import make_phantom

    vol = sc["volume"]
    vol_mm3 = np.prod([np.diff(vol[f"{ax}_extent"])[0] for ax in "xyz"])
    z_mid = float(np.mean(vol["z_extent"]))
    n_scat = int(
        sc["per_cell"]
        * vol_mm3
        / resolution_cell_mm3(z_mid, sc["fc"], sc["aperture_mm"])
    )
    emap = phantom_map(vol, (96, 96, 64), sc["tubes"], sc["spheres"])
    pos, amp = make_phantom(vol, n_scat, echogenicity_map=emap, seed=SEED)
    ys = np.arange(*vol["y_extent"], sc["wire_dy"])
    wires = np.array([[sc["wire_x"], y, z] for z in sc["wire_z"] for y in ys])
    pos = np.vstack([pos, wires])
    amp = np.concatenate([amp, np.full(len(wires), sc["wire_amp"])])
    return pos, amp


# --- Transmit sequence ------------------------------------------------------------
def dw_events(transducer, virtual_sources_mm):
    """Diverging-wave TX events, one per virtual source.

    Element ``e`` fires at its extra travel time from the virtual source,
    referenced to the closest element, so the emitted field is a spherical
    wavefront diverging from the source position::

        delays_e = (|r_e − r_vs| − min_e|r_e − r_vs|) / c

    Each event also carries its ``virtual_source_mm`` so the general
    beamformer (`das_volume`) can recover the transmit wavefront geometry
    directly from the event — strip that key before ``sequence_rf`` (step 2
    does), which only consumes ``delays``/``apodization``.

    Parameters
    ----------
    transducer : MatrixArrayTransducer
        The transmitting matrix (all elements active, unit apodization).
    virtual_sources_mm : (N, 3) array-like
        Virtual-source positions in mm (z < 0 = behind the array).

    Returns
    -------
    list[dict]
        Events with ``delays``, unit ``apodization`` and ``virtual_source_mm``.
    """
    centers = transducer.element_centers  # (E, 3), metres
    events = []
    for vs in np.atleast_2d(np.asarray(virtual_sources_mm, dtype=float)):
        d = np.linalg.norm(centers - vs * 1e-3, axis=1) / C
        events.append(
            {
                "delays": (d - d.min()).astype(np.float32),
                "apodization": np.ones(centers.shape[0], np.float32),
                "virtual_source_mm": np.asarray(vs, dtype=float),
            }
        )
    return events


def _rings(z_mm, radii_deg_pairs):
    """Virtual sources at ``z_mm``: centre + rings ``(radius_mm, n, az0_deg)``.

    Successive rings are rotated (``az0``) so they interleave in azimuth —
    the transmit tilt disc is sampled uniformly at several radii, which is
    what sharpens the synthesized transmit focus and widens its depth of
    field when the events are compounded.
    """
    vs = [[0.0, 0.0, z_mm]]
    for r, n, az0 in radii_deg_pairs:
        for a in az0 + np.arange(n) * (360.0 / n):
            vs.append([r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a)), z_mm])
    return np.array(vs)


# The FIRST transmit event is the on-axis diverging wave; the next 8 form a
# ring at the SAME steering angle on every probe, so ``vs_mm[:9]`` is a
# comparable 9-event acquisition across scenarios — the base ring is matched by
# transmit TILT, not radius, because the coverage rule sets a different virtual-
# source depth ``z_vs`` for each aperture. The tilt is fixed by the ZeUS-10 MHz
# reference (its 9 sources are r = 4 mm at z_vs = −40 mm → atan(4/40) = 5.7°);
# each probe's base radius is then |z_vs|·tan θ. Extra rings (indices ≥ 9) add
# the wider tilt diversity each aperture can afford for the full-quality
# compound, so ``vs_mm`` (full) images best while ``vs_mm[:9]`` stays matched.
BASE_TILT_DEG = float(np.degrees(np.arctan(4.0 / 40.0)))  # 5.71°, ZeUS-10 reference


def _base_ring(z_mm, extra=()):
    """Centre + a shared 8-source base ring at ``BASE_TILT_DEG`` (indices 1–8),
    then any ``extra`` rings appended — see the comment above for why."""
    r0 = abs(z_mm) * np.tan(np.deg2rad(BASE_TILT_DEG))
    return _rings(z_mm, [(r0, 8, 0.0), *extra])


# --- Scenarios --------------------------------------------------------------------
# The contrast-ladder layout, per probe scale. All coordinates in mm.
# zeus*: cyst r=2 at the z=15 mid-depth, x4 tube r=1 below it (z=18.5) and a
# second x4 tube column at +3.2; x4 sphere r=0.8 above the cyst,
# elevation-offset to y=+2; wires at z = 11/15/19 on the x=0.5 lane.
_ZEUS_LAYOUT = {
    "volume": {
        "x_extent": [-5.5, 5.5],
        "y_extent": [-3.5, 3.5],
        "z_extent": [10.0, 20.0],
    },
    "tubes": [
        ((-2.8, 15.0), 2.0, 0.0),  # anechoic cyst tube (z-span 13–17)
        ((3.2, 15.0), 1.0, 4.0),  # x4 hyperechoic tube
        ((-2.8, 18.5), 1.0, 4.0),  # x4 tube UNDER the cyst (contrast ladder)
    ],
    "spheres": [((-2.8, 2.0, 11.5), 0.8, 4.0)],
    "wire_x": 0.5,
    "wire_z": np.array([11.0, 15.0, 19.0]),
    "wire_amp": 4.0,
    "tier_z": 15.0,
    "slice_y": 2.0,  # elevation slice through the sphere
    "grid": {
        "x_extent": [-5.2, 5.2],
        "y_extent": [-3.2, 3.2],
        "z_extent": [10.3, 19.7],
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.05,
    },
    "aperture_mm": 55 * 0.30 - 0.01,  # 16.49
}

SCENARIOS = {
    # The flagship: 5 MHz on the ZeUS makes the 0.30 mm pitch 0.97λ (grating
    # lobes leave the field) and the ~1λ element face accepts wide-angle
    # echoes. 21 CLOSE virtual sources (z=−20: up to 16.7° tilt) — affordable
    # only because the relaxed pitch removed the steep-wavefront penalty.
    "zeus5": {
        **_ZEUS_LAYOUT,
        "fc": 5e6,
        "make_probe": make_zeus,
        "per_cell": 10,
        "checkpoint_chunks": 2,
        "wire_dy": 0.1,  # < λ/2 = 0.154 mm → acoustically continuous wire
        # Base 8-ring at 5.7° (r = 20·tan5.7° = 2.0 mm), then two wider rings
        # out to r = 6 mm (16.7° tilt): 21 sources total, vs_mm[:9] comparable.
        "vs_mm": _base_ring(-20.0, [(4.0, 8, 0.0), (6.0, 4, 22.5)]),
    },
    # The cautionary counterpart: at 10 MHz the same pitch is 2λ — the echo
    # field is spatially aliased and every voxel collects faint coherent
    # copies of speckle from elsewhere; the cyst fills with clutter no
    # software can reject. 9 FAR sources (z=−40, r=4): at 2λ pitch every
    # degree of local wavefront steepness feeds the aliasing, so tilt
    # diversity must stay small. Expensive (~1.3 M scatterers, ~8 h).
    "zeus10": {
        **_ZEUS_LAYOUT,
        "fc": 10e6,
        "make_probe": make_zeus,
        "per_cell": 5,
        "checkpoint_chunks": 4,
        "wire_dy": 0.05,  # < λ/2 = 0.077 mm at 10 MHz
        # The reference layout: 9 sources = centre + the base 8-ring (r = 4 mm
        # at z_vs = −40 → 5.7°), which fixes BASE_TILT_DEG for every scenario.
        "vs_mm": _base_ring(-40.0),
    },
    # The real-probe scenario: the IDENTICAL phantom (same volume, same
    # targets) imaged by the Vermon at 3 MHz — only the scatterer count
    # (resolution cell ∝ λ³/aperture²) and the virtual sources change. At
    # z=15 mm its 9.6 mm aperture gives a ~0.8 mm PSF, so the cyst r=2 is
    # still ~2.5 PSF (resolvable) and the sphere r=0.8 ≈ 1 PSF (a bright
    # blob that still proves elevational sectioning).
    #
    # Virtual sources — derived from the COVERAGE rule, not copied from the
    # ZeUS: a diverging wave from (r, −z_s) only insonifies the cone from
    # the source through the aperture edges, so covering the far corner
    # (x=5.5, z=20) with a 4.8 mm half-aperture needs
    # 4.8 + (4.8 − r)·20/z_s ≥ 5.5 → r ≤ 4.45 mm at z_s = 10. The rings
    # r = 2.2/4.4 sit at that limit → max tilt at the tier ≈ 10°, the most
    # transmit diversity this small aperture can buy over this FOV (the
    # ZeUS sources at z=−20, r ≤ 6 would BOTH miss the deep corners
    # (reaching only x = 3.6) AND tilt less — same design rule, not same
    # coordinates, is what makes the probes comparable). Cheap: ~23k
    # scatterers → this scenario is the fast end-to-end pipeline test.
    "vermon": {
        **_ZEUS_LAYOUT,
        "fc": 3e6,
        "make_probe": make_vermon,
        "per_cell": 10,
        "checkpoint_chunks": 1,
        "wire_dy": 0.15,  # < λ/2 = 0.257 mm at 3 MHz
        "aperture_mm": 32 * 0.30 - 0.025,  # 9.575 (overrides the ZeUS value)
        # Base 8-ring at 5.7° (r = 10·tan5.7° = 1.0 mm), then two wider rings
        # out to r = 4.4 mm (23.7° tilt, at the coverage limit): 25 sources
        # total, vs_mm[:9] comparable with the other probes.
        "vs_mm": _base_ring(-10.0, [(2.5, 8, 0.0), (4.4, 8, 11.25)]),
    },
}

if SCENARIO not in SCENARIOS:
    raise ValueError(f"unknown scenario {SCENARIO!r}; pick one of {list(SCENARIOS)}")
SC = SCENARIOS[SCENARIO]

if __name__ == "__main__":
    print(f"Scenario '{SCENARIO}':")
    print(f"  probe      {SC['make_probe'].__name__}, fc = {SC['fc'] / 1e6:.0f} MHz")
    print(
        f"  sequence   {len(SC['vs_mm'])} diverging waves, z_vs = {SC['vs_mm'][0, 2]:.0f} mm"
    )
    pos, amp = build_phantom(SC)
    print(f"  phantom    {pos.shape[0]:,d} scatterers ({SC['per_cell']}/cell)")
    print(f"  RF data    out/{SCENARIO}/RF   ·   IQ volume   out/{SCENARIO}/IQ")
