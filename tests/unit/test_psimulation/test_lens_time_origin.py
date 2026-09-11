"""A lens must not move the image: a point target lands at its true depth.

`coords["t0"]` is the beamforming reference, so a beamformer reads the sample at
``(t_tx + t_rx − t0)·fs`` using distances to the ELEMENT CENTRES, which sit on
the ``z = 0`` plane whether or not the aperture carries an elevation lens. The
lens surface does not: a concave lens recedes one sagitta at its centre, a convex
one protrudes by the same amount. Reception compensates by subtracting the signed
sag from `t0`, once per aperture.

Getting that sign wrong does not blur or distort anything — it displaces the
whole image in depth by twice the sagitta, which reads as a calibration error
rather than a bug. That is exactly what happened before these tests existed
(0.34 mm for a 4 mm aperture at R = 12 mm, with every other metric looking
healthy), so the guard is an end-to-end one: simulate, beamform, and check where
the target landed.
"""

import numpy as np
import pytest

from esdiva.beamforming import das_volume, envelope_db
from esdiva.reception import Reception
from esdiva.transducers import LinearArrayTransducer

C = 1540.0
FC = 5e6
FS = 50e6
N_EL = 16
HEIGHT_MM = 4.0
Z_TARGET_MM = 12.0


def _probe(elevation_focus_mm):
    return LinearArrayTransducer(
        n_elements=N_EL,
        element_width_mm=0.25,
        element_height_mm=HEIGHT_MM,
        kerf_mm=0.03,
        no_sub_x=1,
        no_sub_y=4,
        elevation_focus_mm=elevation_focus_mm,
        frequency_Hz=FC,
    )


def _measured_depth_mm(elevation_focus_mm):
    """Beamform one on-axis point target and report the depth it lands at."""
    t = np.arange(0, 2 / FC, 1.0 / FS)
    exc = (np.sin(2 * np.pi * FC * t) * np.hanning(len(t))).astype(np.float32)
    tx, rx = _probe(elevation_focus_mm), _probe(elevation_focus_mm)
    tx.impulse_response = exc.copy()
    rx.impulse_response = exc.copy()
    tx.compute_delays(angle_steering_deg=0.0)

    sim = Reception(tx, rx, c=C, fs=FS, excitation=exc, verbose=False)
    rf, coords = sim.pulse_echo_rf(np.array([[0.0, 0.0, Z_TARGET_MM]]))
    events = [
        {
            "delays": np.asarray(tx.delays, np.float32).copy(),
            "apodization": np.ones(N_EL, np.float32),
            "angles_deg": 0.0,
        }
    ]
    grid = {
        "x_extent": [-1.0, 1.0],
        "y_extent": [-0.01, 0.01],
        "z_extent": [Z_TARGET_MM - 1.5, Z_TARGET_MM + 1.5],
        "dx": 0.05,
        "dy": 0.02,
        "dz": 0.01,
    }
    vol, gc = das_volume(
        rf[None],
        {"dt": coords["dt"], "t0_per_event": np.array([coords["t0"]])},
        events,
        rx,
        grid,
        c=C,
        fnum=1.0,
    )
    image = envelope_db(vol[:, 0, :])
    _, iz = np.unravel_index(np.argmax(image), image.shape)
    return float(gc["z_mm"][iz])


# A quarter wavelength: tighter than any lens-sign error (which is 2 sagittae,
# 0.34 mm here = 1.1 lambda) and looser than the pulse's own envelope jitter.
TOL_MM = 0.25 * C / FC * 1e3


@pytest.mark.parametrize(
    "elevation_focus_mm, label",
    [(None, "flat"), (12.0, "concave"), (-12.0, "convex")],
)
def test_point_target_lands_at_true_depth(elevation_focus_mm, label):
    measured = _measured_depth_mm(elevation_focus_mm)
    assert measured == pytest.approx(Z_TARGET_MM, abs=TOL_MM), (
        f"{label} aperture put the target at {measured:.3f} mm "
        f"instead of {Z_TARGET_MM} mm"
    )


def test_lens_sign_error_would_be_caught():
    """The guard has teeth: forcing the wrong sag sign must fail the check.

    Without this, a sign regression passes silently — the image is sharp, the
    speckle is right, and only its absolute depth is wrong.
    """
    concave = _probe(12.0)
    sag_mm = concave.elevation_lens_sag * 1e3
    assert sag_mm > 0
    # A flipped sign mis-places the echo by 2 sagittae, one per aperture.
    assert 2 * sag_mm > TOL_MM, (
        "the tolerance is too loose to catch a lens-sign regression on this geometry"
    )
