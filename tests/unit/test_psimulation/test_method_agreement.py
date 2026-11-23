"""Every SIR source gives the same field and the same RF, at the common fs = 100 MHz.

`spectral` evaluates the closed-form SIR spectrum H(ω) exactly at the in-band
frequencies. The temporal sources (`sdi`, `fst`, `auto`) sample the trapezoidal SIR
h(t) in time, then transform. A trapezoid has sharp corners, so its spectrum reaches far
above Nyquist, and point sampling folds that energy back into the band (aliasing; the
sampled spectrum equals Σ_k H(f + k·fs) to 1e-8). Patches narrower than one sample are
also widened to one sample (a sinc(πf/fs) droop per aperture). At fs = 20·fc these
two effects leave the temporal result within about 1 % of the spectral one, in peak
amplitude and in RMS error over the whole trace — the gate below. The worst single
sample can reach about 1.3 % and is not gated. The error does not shrink smoothly with
fs: it depends on how each patch's ramp durations relate to the sample period.
"""

import warnings

import numpy as np
import pytest

from esdiva.emission import Emission
from esdiva.reception import Reception
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities import make_phantom

FS, FC = 100e6, 5e6
TEMPORAL = ("sdi", "fst", "auto")
TOL = 0.01  # peak-amplitude difference and RMS error, as a fraction of spectral

_FIELD_PTS = np.array([[0, 0, 10], [0, 0, 20], [6, 0, 12], [-3, 1, 22]], np.float32)
_SCAT = np.array([[0, 0, 18], [1, 0, 22], [-1.5, 0, 26], [6, 0, 12]], np.float32)
_AMP = np.array([1.0, 0.8, 1.2, 0.5], np.float32)


def _pulse():
    t = np.arange(int(2 * FS / FC)) / FS
    return (np.sin(2 * np.pi * FC * t) * np.hanning(t.size)).astype(np.float32)


def _linear(n, focus=None, baffle="rigid"):
    tx = LinearArrayTransducer(
        n_elements=n, element_width_mm=0.25, element_height_mm=5.0, kerf_mm=0.05,
        no_sub_x=2, no_sub_y=4, frequency_Hz=FC,
    )  # fmt: skip
    if focus is not None:
        tx.compute_delays(focus_mm=focus)
    tx.baffle = baffle
    return tx


def _drive_per_element(n):
    return _pulse()[:, None] * np.linspace(0.5, 1.5, n).astype(np.float32)


def _assert_agree(run, tol=TOL):
    """Run every method, compare each temporal one with spectral.

    The temporal path may return a few extra trailing samples (near-zero tail), so the
    common samples are compared; the time origin `t0` must be identical.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref, t0_ref = run("spectral")
        ref = np.asarray(ref, dtype=np.float64)
        peak = np.abs(ref).max()
        for m in TEMPORAL:
            out, t0 = run(m)
            out = np.asarray(out, dtype=np.float64)
            n = min(ref.shape[-1], out.shape[-1])
            a, b = ref[..., :n], out[..., :n]
            np.testing.assert_allclose(t0, t0_ref, rtol=0, atol=1e-12, err_msg=m)
            d_peak = abs(np.abs(b).max() - peak) / peak
            rms = np.sqrt(np.mean((b - a) ** 2) / np.mean(a**2))
            assert d_peak < tol, f"{m}: peak differs by {100 * d_peak:.2f} %"
            assert rms < tol, f"{m}: RMS error {100 * rms:.2f} %"


# Emission: field at four points; transient output is time-first, so put time last.
_EMISSION = {
    "mono": dict(monochromatic=True),
    "mono-att": dict(monochromatic=True, alpha0=0.5, fast_attenuation=False),
    "mono-soft": dict(monochromatic=True, baffle="soft"),
    "global": dict(excitation="global"),
    "global-diverging": dict(excitation="global", focus=[0, 0, -10]),
    "global-att": dict(excitation="global", alpha0=0.7, freq_power=1.1),
    "global-att-fast": dict(excitation="global", alpha0=0.7),
    "global-soft": dict(excitation="global", baffle="soft"),
    "per-element": dict(excitation="per_element"),
    "per-element-att": dict(
        excitation="per_element", alpha0=0.5, fast_attenuation=False
    ),
    "transfer-function": dict(
        excitation="global", transfer_function=lambda f: np.exp(-((f / 8e6) ** 2))
    ),
}


def _emission_run(kw, n=32):
    kw = dict(kw)
    tx = _linear(n, focus=kw.pop("focus", [0, 0, 20]), baffle=kw.pop("baffle", "rigid"))
    drive = kw.pop("excitation", None)
    if drive == "global":
        kw["excitation"] = _pulse()
    elif drive == "per_element":
        kw["excitation"] = _drive_per_element(n)

    def run(method):
        p, co = Emission(tx, fs=FS, method=method, verbose=False, **kw)(_FIELD_PTS)
        return (p.T if p.ndim > 1 else p), co.get("t0", 0.0)

    return run


@pytest.mark.parametrize("case", sorted(_EMISSION))
def test_emission_methods_agree(case):
    _assert_agree(_emission_run(_EMISSION[case]))


def test_emission_raw_sir_methods_agree():
    # No excitation: the raw, band-unlimited SIR keeps all its aliased energy, so the
    # gap is larger (measured 2.6 % peak, 1.6 % RMS) and does not vanish with fs.
    _assert_agree(_emission_run({}), tol=0.03)


# Reception: 16-element focused TX, 16-element RX, four scatterers.
_PHANTOM = make_phantom(
    {"x_extent": [-3, 3], "y_extent": [-0.5, 0.5], "z_extent": [15, 25]}, 300, seed=7
)


def _events(n):
    return [
        {"delays": np.zeros(n, np.float32)},
        {"delays": np.linspace(0, 2e-7, n).astype(np.float32)},
    ]


_RECEPTION = {
    "pulse-echo": (dict(), lambda s, n: s.pulse_echo_rf(_SCAT, _AMP)),
    "pulse-echo-no-ir": (dict(ir=False), lambda s, n: s.pulse_echo_rf(_SCAT, _AMP)),
    "pulse-echo-att": (dict(alpha0=0.5), lambda s, n: s.pulse_echo_rf(_SCAT, _AMP)),
    "pulse-echo-soft": (
        dict(baffle="soft"),
        lambda s, n: s.pulse_echo_rf(_SCAT, _AMP),
    ),
    "pulse-echo-per-element": (
        dict(per_element=True),
        lambda s, n: s.pulse_echo_rf(_SCAT, _AMP),
    ),
    "pulse-echo-per-element-att": (
        dict(per_element=True, alpha0=0.5),
        lambda s, n: s.pulse_echo_rf(_SCAT, _AMP),
    ),
    "psf-per-scatterer": (
        dict(),
        lambda s, n: s.pulse_echo_rf(_SCAT, _AMP, per_scatterer=True),
    ),
    "phantom": (dict(), lambda s, n: s.pulse_echo_rf(*_PHANTOM)),
    "phantom-att": (dict(alpha0=0.5), lambda s, n: s.pulse_echo_rf(*_PHANTOM)),
    "sequence-static": (dict(), lambda s, n: s.sequence_rf(_SCAT, _AMP, _events(n))),
    "sequence-moving": (
        dict(),
        lambda s, n: s.sequence_rf(
            np.stack([_SCAT, _SCAT + np.float32([0, 0, 0.1])]), _AMP, _events(n)
        ),
    ),
    "synthetic-aperture": (
        dict(),
        lambda s, n: s.synthetic_aperture_rf(
            _SCAT, _AMP, decimation=1, countdown=False
        ),
    ),
    "focused-line": (
        dict(n=32),
        lambda s, n: s.scan_focusline([0, 0, 22], _SCAT, _AMP, FoverD=2.0),
    ),
}


@pytest.mark.parametrize("case", sorted(_RECEPTION))
def test_reception_methods_agree(case):
    opts, call = _RECEPTION[case]
    n = opts.get("n", 16)

    def run(method):
        tx = _linear(n, focus=[0, 0, 20], baffle=opts.get("baffle", "rigid"))
        rx = _linear(n, baffle=opts.get("baffle", "rigid"))
        if opts.get("ir", True):
            tx.impulse_response = rx.impulse_response = _pulse()
        exc = _drive_per_element(n) if opts.get("per_element") else _pulse()
        sim = Reception(
            tx, rx, fs=FS, excitation=exc, alpha0=opts.get("alpha0"), method=method,
            verbose=False,
        )  # fmt: skip
        rf, co = call(sim, n)
        return rf, co.get("t0_per_event", co["t0"])

    _assert_agree(run)
