"""Tests for the spectral SIR kernels (``sir_spectral.py``)."""

import numpy as np
import pytest
from scipy.fft import rfft, rfftfreq

from esdiva.attenuation import causal_attenuation_tf, convert_alpha0_to_nepers
from esdiva.hsir import compute_h_sir, compute_h_sir_spectrum
from esdiva.hsir.sir_spectral import compute_twoway_spectrum_summed
from esdiva.transducers import LinearArrayTransducer
from esdiva.utilities.helper_functions import compute_sub_elem_attributes

C = 1540.0
PATCH = {
    "centers": np.zeros((1, 3), np.float32),
    "wx": np.array([0.3e-3], np.float32),
    "wy": np.array([0.3e-3], np.float32),
    "apod": np.ones(1, np.float32),
    "delays": np.zeros(1, np.float32),
}


def _h(points, omega, t0=0.0, soft=False):
    p = PATCH
    return compute_h_sir_spectrum(
        np.asarray(points, np.float32), p["centers"], p["wx"], p["wy"], p["apod"],
        p["delays"], 1 / C, t0, omega, soft_baffle=soft,
    )  # fmt: skip


@pytest.mark.parametrize("point", [[4e-3, 2.5e-3, 20e-3], [0.0, 0.0, 20e-3]])
def test_h_matches_fft_of_sampled_sir(point):
    """H(ω) = continuous FT of the trapezoidal SIR: dt·rfft(h[n]) at 1 GHz sampling."""
    fs = 1e9
    pts = np.array([point], np.float32)
    t0 = float(np.linalg.norm(pts) / C) - 50e-9
    T = 400
    tg = (t0 + np.arange(T) / fs).astype(np.float32)
    eu, ev = np.array([[1, 0, 0]], np.float32), np.array([[0, 1, 0]], np.float32)
    h, _ = compute_h_sir(
        1, 1, T, 1 / fs, tg, pts, PATCH["centers"], PATCH["wx"], PATCH["wy"], 1 / C,
        fs, PATCH["apod"], PATCH["delays"], 0, eu, ev,
    )  # fmt: skip
    f = rfftfreq(1 << 16, 1 / fs)
    band = (f > 1e6) & (f < 12e6)
    ref = rfft(h[0].astype(np.float64), 1 << 16)[band] / fs
    H = _h(pts, 2 * np.pi * f[band], t0)[0]
    assert np.abs(H - ref).max() < 2e-3 * np.abs(H).max()


def test_soft_baffle_is_cos_theta_and_zero_behind():
    """Soft baffle weights each patch by cosθ to its normal; nothing radiates backwards."""
    omega = 2 * np.pi * np.array([3e6, 5e6])
    front = np.array([[6e-3, 0.0, 8e-3]])  # cosθ = 0.8
    ratio = _h(front, omega, soft=True) / _h(front, omega)
    np.testing.assert_allclose(ratio, 0.8, rtol=1e-4)
    assert np.all(_h(-front, omega, soft=True) == 0)


def test_twoway_summed_matches_oneway_product():
    """Fused kernel == Σ_p a_p · H_TX · H_RX,e · H_att, per receive element."""
    rx = LinearArrayTransducer(
        n_elements=4, element_width_mm=0.25, element_height_mm=5.0, kerf_mm=0.05,
        no_sub_x=1, no_sub_y=2, frequency_Hz=5e6,
    )  # fmt: skip
    points = np.array([[0, 0, 20e-3], [2e-3, 0, 26e-3], [-1e-3, 0, 31e-3]], np.float32)
    amp = np.array([1.0, -0.7, 0.4])
    cen, apod, dl, _, wx, wy, idx = compute_sub_elem_attributes(rx)
    omega = 2 * np.pi * np.linspace(2e6, 8e6, 64)
    t0, n_el = 12e-6, 4
    ap = {"centers": cen, "wx": wx, "wy": wy, "apod": apod, "delays": dl, "t0": t0}
    order = np.argsort(idx, kind="stable")
    rx_ap = {k: (v[order] if k != "t0" else v) for k, v in ap.items()}
    ptr = np.concatenate([[0], np.cumsum(np.bincount(idx, minlength=n_el))])
    ecen = np.asarray(rx.element_centers, np.float64)
    atten = {"alpha0_np": convert_alpha0_to_nepers(0.5, 1.0), "freq_power": 1.0, "f0_hz": 5e6,
             "tx_ref": ecen.mean(0), "rx_ref": ecen}  # fmt: skip
    fused = compute_twoway_spectrum_summed(
        points, amp, ap, rx_ap, ptr, 1 / C, omega, atten=atten
    )

    h_tx = compute_h_sir_spectrum(points, cen, wx, wy, apod, dl, 1 / C, t0, omega)
    for e in range(n_el):
        m = idx == e
        h_rx = compute_h_sir_spectrum(
            points, cen[m], wx[m], wy[m], apod[m], dl[m], 1 / C, t0, omega
        )
        d = np.linalg.norm(points - ecen.mean(0), axis=1) + np.linalg.norm(
            points - ecen[e], axis=1
        )
        att = causal_attenuation_tf(omega / (2 * np.pi), d, 0.5, 1.0, 5e6)
        ref = amp @ (h_tx * h_rx * att)
        assert np.abs(fused[e] - ref).max() < 1e-3 * np.abs(ref).max()
