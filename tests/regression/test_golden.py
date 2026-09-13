"""Golden values: small canonical versions of the examples' physics, pinned to disk.

Each scenario mirrors one example family (field maps, transient wavefronts, attenuation,
curved apertures, PSFs, phantoms, beamformed lines) at a size that runs in seconds, with
both SIR sources where both exist. Any numerical change fails here. After an INTENTIONAL
change, regenerate and say why in the commit:

    uv run python tests/regression/test_golden.py
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from esdiva.emission import Emission
from esdiva.reception import Reception
from esdiva.transducers import (
    ConcaveCircularTransducer,
    LinearArrayTransducer,
    MatrixArrayTransducer,
)
from esdiva.utilities import make_phantom

GOLDEN = Path(__file__).with_name("golden.npz")
FS, FC = 100e6, 5e6
METHODS = ("spectral", "temporal")


def _pulse(fc=FC, fs=FS):
    t = np.arange(int(2 * fs / fc)) / fs
    return (np.sin(2 * np.pi * fc * t) * np.hanning(t.size)).astype(np.float32)


def _linear(n=32, focus=None, **kw):
    tx = LinearArrayTransducer(
        n_elements=n, element_width_mm=0.25, element_height_mm=5.0, kerf_mm=0.05,
        no_sub_x=2, no_sub_y=4, frequency_Hz=FC, **kw,
    )  # fmt: skip
    if focus is not None:
        tx.compute_delays(focus_mm=focus)
    return tx


def _plane(z=(5, 35), dz=1.0):
    return {"x_extent": [-4, 4], "y_extent": [0, 0], "z_extent": list(z),
            "dx": 0.5, "dy": 1.0, "dz": dz}  # fmt: skip


def emission_monochromatic_focused():  # example03 / example11
    tx = _linear(focus=[0, 0, 20])
    out = {}
    for m in METHODS:
        out[m] = Emission(tx, monochromatic=True, method=m, verbose=False)(_plane())[0]
        out[f"{m}_att"] = Emission(
            tx, monochromatic=True, alpha0=0.5, fast_attenuation=False, method=m,
            verbose=False,
        )(_plane())[0]  # fmt: skip
    return out


def emission_transient_diverging():  # example04
    tx = _linear(focus=[0, 0, -10])
    pts = np.array([[0, 0, 10], [3, 0, 15], [-5, 0, 20]], np.float32)
    out = {}
    for m in METHODS:
        p, co = Emission(tx, excitation=_pulse(), method=m, verbose=False)(pts)
        out[m], out[f"{m}_t0"] = p, np.array(co["t0"])
    return out


def emission_matrix_steered_pw():  # example05
    tx = MatrixArrayTransducer(
        n_elements_x=8, n_elements_y=8, element_width_mm=0.275, element_height_mm=0.275,
        kerf_x_mm=0.025, kerf_y_mm=0.025, no_sub_x=1, no_sub_y=1, frequency_Hz=3e6,
    )  # fmt: skip
    tx.compute_delays(angle_steering_deg=(10.0, 0.0))
    pts = np.array([[0, 0, 8], [2, 1, 10]], np.float32)
    return {
        m: Emission(tx, excitation=_pulse(3e6), method=m, verbose=False)(pts)[0]
        for m in METHODS
    }


def emission_attenuated_soft_baffle():  # example10 + soft baffle
    tx = _linear(focus=[0, 0, 20])
    tx.baffle = "soft"
    pts = np.array([[0, 0, 10], [0, 0, 20], [6, 0, 12]], np.float32)
    return {
        m: Emission(tx, excitation=_pulse(), alpha0=0.7, freq_power=1.1, method=m,
                    verbose=False)(pts)[0]
        for m in METHODS
    }  # fmt: skip


def emission_concave_bowl():  # example02 / example12
    tx = ConcaveCircularTransducer(
        diameter_mm=20.0, focus_mm=30.0, no_sub_diameter=12, frequency_Hz=1e6
    )
    grid = {"x_extent": [-3, 3], "y_extent": [0, 0], "z_extent": [20, 40],
            "dx": 0.5, "dy": 1.0, "dz": 1.0}  # fmt: skip
    return {m: Emission(tx, monochromatic=True, method=m, verbose=False)(grid)[0]
            for m in METHODS}  # fmt: skip


def reception_psf_per_element_drive():  # example06 / example07
    tx, rx = _linear(16, focus=[0, 0, 20]), _linear(16)
    tx.impulse_response = rx.impulse_response = _pulse()
    exc = _pulse()[:, None] * np.linspace(0.5, 1.5, 16).astype(np.float32)
    pts = np.array([[0, 0, 20], [3, 0, 25]], np.float32)
    return {
        m: Reception(
            tx, rx, fs=FS, excitation=exc, method=m, verbose=False
        ).pulse_echo_rf(pts, per_scatterer=True)[0]
        for m in METHODS
    }


def reception_phantom_attenuated():  # example20
    tx, rx = _linear(16, focus=[0, 0, 20]), _linear(16)
    tx.impulse_response = rx.impulse_response = _pulse()
    box = {"x_extent": [-3, 3], "y_extent": [-0.5, 0.5], "z_extent": [15, 25]}
    pos, amp = make_phantom(box, 300, seed=7)
    return {
        m: Reception(tx, rx, fs=FS, excitation=_pulse(), alpha0=0.5, method=m,
                     verbose=False).pulse_echo_rf(pos, amp)[0]
        for m in METHODS
    }  # fmt: skip


def reception_focused_line():  # example09 (B-mode line envelope)
    tx, rx = _linear(32), _linear(32)
    tx.impulse_response = rx.impulse_response = _pulse()
    pts = np.array([[0, 0, 15], [0, 0, 25], [1, 0, 30]], np.float32)
    sim = Reception(tx, rx, fs=FS, excitation=_pulse(), verbose=False)
    return {"envelope": sim.scan_focusline([0, 0, 25], pts, np.ones(3), FoverD=2.0)[0]}


SCENARIOS = {
    f.__name__: f
    for f in (
        emission_monochromatic_focused,
        emission_transient_diverging,
        emission_matrix_steered_pw,
        emission_attenuated_soft_baffle,
        emission_concave_bowl,
        reception_psf_per_element_drive,
        reception_phantom_attenuated,
        reception_focused_line,
    )
}


def _run(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # benign setup notices
        return {k: np.asarray(v) for k, v in SCENARIOS[name]().items()}


# Tolerance, measured: the same code compiled by numba for another CPU (fastmath float32
# vectorisation differs between AVX2 / AVX-512 / generic) moves the values by up to
# 1.8e-4 of peak — worst in the temporal phantom, where 300 random-sign echoes cancel.
# The smallest intentional change these values must catch was 2.3e-3 of peak (spectral
# dt-clamp removal); physics regressions are far larger. 1e-3 sits between the two.
ATOL_OF_PEAK = 1e-3


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_matches_golden(name):
    ref = np.load(GOLDEN)
    for key, value in _run(name).items():
        expected = ref[f"{name}/{key}"]
        assert value.shape == expected.shape, key
        np.testing.assert_allclose(
            value, expected, rtol=0, atol=ATOL_OF_PEAK * np.abs(expected).max(),
            err_msg=f"{name}/{key} drifted — regenerate only if intentional (see module)",
        )  # fmt: skip


if __name__ == "__main__":
    data = {f"{n}/{k}": v for n in sorted(SCENARIOS) for k, v in _run(n).items()}
    np.savez_compressed(GOLDEN, **data)
    print(f"wrote {len(data)} arrays to {GOLDEN}")
