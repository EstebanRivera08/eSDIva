"""Spectral SIR kernels: the closed-form SIR spectrum H(r, ω), twin of `sir_temporal`.

A rectangular patch's far-field SIR is a trapezoid — the convolution of two rectangles
whose widths Δt1, Δt2 are the times the wavefront takes to cross the patch edges
(``Δt = w·|u|/c``, u the direction cosine along that edge). Its spectrum is closed form:

    H_patch(ω) = A/(2πl) · D(θ) · sinc(ωΔt1/2) · sinc(ωΔt2/2) · e^{-jω(t_c − t0)}

A = patch area × apodization, l = patch→point distance, t_c = l/c + delay (centre arrival),
t0 = window origin. The sincs are the rectangle's own directivity; D is the baffle
obliquity — rigid 1, soft ``cosθ = max(0, n·u)`` (n = patch normal, zero behind the patch).
Summing patches gives the aperture spectrum exactly: no time sampling, no FFT, no
sub-sample clamp. It links to the sampled SIR of `compute_h_sir` by ``rfft(h[n]) ≈ fs·H``.
The pulse-echo spectrum of a (TX, RX) pair is ``H_TX·H_RX`` (convolution ⇒ product).

- `compute_h_sir_spectrum` — one aperture's H at each point, ``(P, N_ω)``.
- `compute_twoway_spectrum_summed` — ``Σ_p a_p·H_TX·H_RX,e·H_att`` per receive element in
  one fused pass (H_TX built once per scatterer, reused for every element).
"""

import numpy as np
from numba import get_num_threads, njit, prange

from esdiva.attenuation.attenuation import _causal_atten_factor

from .helpers import identity_tangents, pack_tangents

_inv_2pi = 1.0 / (2.0 * np.pi)


@njit(inline="always")
def _phasor(x):
    """Unit phasor e^{-jx}."""
    return complex(np.cos(x), -np.sin(x))


@njit(inline="always")
def _accum_aperture(
    acc, px, py, pz, centers, wx, wy, t, apod, delays, lo, hi, inv_c, t0, omega, inv_w2,
    soft,
):  # fmt: skip
    """Add H of patches ``lo:hi`` at point (px, py, pz) into ``acc`` over ``omega``.

    ω is uniform, so each phasor advances by a constant factor per bin (one complex
    multiply, no per-bin sin/cos); ``sin(ωΔt/2)`` is read off a swept half-angle phasor
    and ``sinc·sinc = sin·sin / (ω²·h1·h2)`` uses the precomputed ``inv_w2 = 1/ω²``.
    """
    nb = omega.shape[0]
    k0 = 1 if omega[0] == 0.0 else 0  # DC: both sincs → 1, H(0) = amp
    w0 = omega[k0] if k0 < nb else 0.0
    dw = omega[1] - omega[0] if nb > 1 else 0.0
    for m in range(lo, hi):
        dx = px - centers[m, 0]
        dy = py - centers[m, 1]
        dz = pz - centers[m, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-12:
            continue
        amp = wx[m] * wy[m] * apod[m] * _inv_2pi / dist
        if soft:
            nx = t[m, 1] * t[m, 5] - t[m, 2] * t[m, 4]
            ny = t[m, 2] * t[m, 3] - t[m, 0] * t[m, 5]
            nz = t[m, 0] * t[m, 4] - t[m, 1] * t[m, 3]
            amp *= max(0.0, (dx * nx + dy * ny + dz * nz) / dist)
        if amp == 0.0:
            continue
        h1 = (
            0.5 * wx[m] * inv_c * abs(dx * t[m, 0] + dy * t[m, 1] + dz * t[m, 2]) / dist
        )
        h2 = (
            0.5 * wy[m] * inv_c * abs(dx * t[m, 3] + dy * t[m, 4] + dz * t[m, 5]) / dist
        )
        if k0:
            acc[0] += amp
        # A 1 ps floor keeps sin/(ωh) well defined on axis (sinc(ω·1ps) = 1 to 1e-9).
        h1 = max(h1, 1e-12)
        h2 = max(h2, 1e-12)
        amp_h = amp / (h1 * h2)
        tc = dist * inv_c + delays[m] - t0
        p, sp = _phasor(w0 * tc), _phasor(dw * tc)
        q1, s1 = _phasor(w0 * h1), _phasor(dw * h1)
        q2, s2 = _phasor(w0 * h2), _phasor(dw * h2)
        for k in range(k0, nb):
            acc[k] += (amp_h * inv_w2[k] * q1.imag * q2.imag) * p
            p *= sp
            q1 *= s1
            q2 *= s2


@njit(parallel=True, fastmath=True, cache=True)
def _h_points(points, centers, wx, wy, t, apod, delays, inv_c, t0, omega, inv_w2, soft):
    """H at every point, ``prange`` over points → (P, N_ω) complex128."""
    out = np.zeros((points.shape[0], omega.shape[0]), dtype=np.complex128)
    M = centers.shape[0]
    for p in prange(points.shape[0]):  # ty: ignore[not-iterable]
        _accum_aperture(
            out[p], points[p, 0], points[p, 1], points[p, 2],
            centers, wx, wy, t, apod, delays, 0, M, inv_c, t0, omega, inv_w2, soft,
        )  # fmt: skip
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _h_patches(point, centers, wx, wy, t, apod, delays, inv_c, t0, omega, inv_w2, soft):
    """H at ONE point, ``prange`` over patches (a PSF still saturates every core)."""
    M = centers.shape[0]
    buf = np.zeros((M, omega.shape[0]), dtype=np.complex128)
    for m in prange(M):  # ty: ignore[not-iterable]
        _accum_aperture(
            buf[m], point[0], point[1], point[2],
            centers, wx, wy, t, apod, delays, m, m + 1, inv_c, t0, omega, inv_w2, soft,
        )  # fmt: skip
    return buf.sum(axis=0).reshape(1, -1)


def _omega(omega):
    """Contiguous float64 ω and its ``1/ω²`` (0 at DC, handled apart by the kernel)."""
    omega = np.ascontiguousarray(omega, dtype=np.float64)
    return omega, np.divide(1.0, omega**2, out=np.zeros_like(omega), where=omega > 0)


def _tangents(M, eu, ev):
    if eu is None or ev is None:
        eu, ev = identity_tangents(M)
    return pack_tangents(
        np.asarray(eu, dtype=np.float32), np.asarray(ev, dtype=np.float32)
    )


def compute_h_sir_spectrum(
    points,
    centers,
    wx,
    wy,
    apod,
    delays,
    inv_c,
    t0,
    omega,
    *,
    eu=None,
    ev=None,
    soft_baffle=False,
):
    """Closed-form spatial impulse response spectrum ``H(r, ω)`` of one aperture.

    Sum over patches of ``A/(2πl)·D(θ)·sinc(ωΔt1/2)·sinc(ωΔt2/2)·e^{-jω(t_c−t0)}`` —
    the Fourier transform of the far-field trapezoidal SIR, evaluated exactly at the
    requested frequencies. ``rfft`` of the sampled SIR ``h[n]`` (origin ``t0``) ≈ ``fs·H``.

    Parameters
    ----------
    points : (P, 3) numpy.ndarray
        Field points in metres.
    centers : (M, 3) numpy.ndarray
        Patch centres in metres.
    wx, wy : (M,) numpy.ndarray
        Patch widths along the two in-plane axes (metres).
    apod : (M,) numpy.ndarray
        Apodization weight per patch.
    delays : (M,) numpy.ndarray
        Firing delay per patch (seconds).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        Time origin (seconds) the phase is referenced to.
    omega : (N_ω,) numpy.ndarray
        Angular frequencies 2πf (rad/s), uniformly spaced (one value is allowed).
    eu, ev : (M, 3) numpy.ndarray or None, default None
        Patch in-plane unit vectors; the normal is ``eu × ev``. None → flat patches in
        the xy-plane facing +z.
    soft_baffle : bool, default False
        Soft (pressure-release) baffle: weight each patch by ``cosθ = max(0, n·u)``.
        False is the rigid baffle (no obliquity).

    Returns
    -------
    (P, N_ω) numpy.ndarray
        SIR spectrum H in metres (complex64).
    """
    centers = np.asarray(centers, dtype=np.float32)
    args = (
        centers,
        np.asarray(wx, dtype=np.float32),
        np.asarray(wy, dtype=np.float32),
        _tangents(centers.shape[0], eu, ev),
        np.asarray(apod, dtype=np.float32),
        np.asarray(delays, dtype=np.float32),
        float(inv_c),
        float(t0),
        *_omega(omega),
        bool(soft_baffle),
    )
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 1:
        return _h_patches(points[0], *args).astype(np.complex64)
    return _h_points(points, *args).astype(np.complex64)


@njit(parallel=True, fastmath=True, cache=True)
def _twoway_summed_points(
    points,
    amps,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_t,
    tx_apod,
    tx_delays,
    tx_t0,
    tx_soft,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_t,
    rx_apod,
    rx_delays,
    rx_t0,
    rx_soft,
    rx_ptr,
    inv_c,
    omega,
    inv_w2,
    do_atten,
    tx_ref,
    rx_ref,
    alpha0_np,
    y,
    tan_y,
    f0,
    y_is_one,
    n_chunks,
):
    """Σ_p a_p·H_TX(p)·H_RX,e(p)·H_att(d_pe) → (n_out, N_ω), one parallel pass.

    Each thread chunk of scatterers owns a private ``(n_out, N_ω)`` buffer (race-free).
    Receive element ``e`` is the patch block ``rx_ptr[e]:rx_ptr[e+1]``. Attenuation uses
    the round trip ``d_pe = |r_p − tx_ref| + |r_p − rx_ref[e]|`` (aperture centres).
    """
    P = points.shape[0]
    nb = omega.shape[0]
    n_out = rx_ptr.shape[0] - 1
    Mtx = tx_centers.shape[0]
    buf = np.zeros((n_chunks, n_out, nb), dtype=np.complex128)
    for ci in prange(n_chunks):  # ty: ignore[not-iterable]
        h_tx = np.zeros(nb, dtype=np.complex128)
        h_rx = np.zeros(nb, dtype=np.complex128)
        for p in range(ci * P // n_chunks, (ci + 1) * P // n_chunks):
            px, py, pz = points[p, 0], points[p, 1], points[p, 2]
            h_tx[:] = 0.0
            _accum_aperture(
                h_tx, px, py, pz, tx_centers, tx_wx, tx_wy, tx_t, tx_apod,
                tx_delays, 0, Mtx, inv_c, tx_t0, omega, inv_w2, tx_soft,
            )  # fmt: skip
            d_tx = np.sqrt(
                (px - tx_ref[0]) ** 2 + (py - tx_ref[1]) ** 2 + (pz - tx_ref[2]) ** 2
            )
            for e in range(n_out):
                h_rx[:] = 0.0
                _accum_aperture(
                    h_rx, px, py, pz, rx_centers, rx_wx, rx_wy, rx_t, rx_apod,
                    rx_delays, rx_ptr[e], rx_ptr[e + 1], inv_c, rx_t0, omega, inv_w2,
                    rx_soft,
                )  # fmt: skip
                if do_atten:
                    d = d_tx + np.sqrt(
                        (px - rx_ref[e, 0]) ** 2
                        + (py - rx_ref[e, 1]) ** 2
                        + (pz - rx_ref[e, 2]) ** 2
                    )
                    for k in range(nb):
                        buf[ci, e, k] += (
                            amps[p]
                            * h_tx[k]
                            * h_rx[k]
                            * _causal_atten_factor(
                                omega[k], d, alpha0_np, y, tan_y, f0, y_is_one
                            )
                        )
                else:
                    for k in range(nb):
                        buf[ci, e, k] += amps[p] * h_tx[k] * h_rx[k]
    return buf.sum(axis=0)


def _aperture(a):
    """Aperture dict → the fused kernel's positional patch arrays.

    Parameters
    ----------
    a : dict
        Keys ``centers, wx, wy, apod, delays, t0`` and optional ``eu, ev, soft``.

    Returns
    -------
    tuple
        ``(centers, wx, wy, tangents, apod, delays, t0, soft)`` in kernel dtypes.
    """
    c = np.asarray(a["centers"], dtype=np.float32)
    return (
        c,
        np.asarray(a["wx"], dtype=np.float32),
        np.asarray(a["wy"], dtype=np.float32),
        _tangents(c.shape[0], a.get("eu"), a.get("ev")),
        np.asarray(a["apod"], dtype=np.float32),
        np.asarray(a["delays"], dtype=np.float32),
        float(a["t0"]),
        bool(a.get("soft", False)),
    )


def compute_twoway_spectrum_summed(
    points,
    amps,
    tx,
    rx,
    rx_ptr,
    inv_c,
    omega,
    *,
    atten=None,
):
    """Amplitude-summed pulse-echo SIR spectrum per receive element.

    Returns ``S_e(ω) = Σ_p amps[p]·H_TX(ω; r_p)·H_RX,e(ω; r_p)·H_att(ω; d_pe)`` — the
    two-way spectrum of the whole scatterer field, built without materialising any
    ``(P, N_ω)`` array. The caller multiplies the excitation/IR filter and inverse-FFTs.

    Parameters
    ----------
    points : (P, 3) numpy.ndarray
        Scatterer positions in metres.
    amps : (P,) numpy.ndarray
        Scattering amplitude per scatterer.
    tx, rx : dict
        Aperture patch arrays with keys ``centers, wx, wy, apod, delays, eu, ev`` (see
        `compute_h_sir_spectrum`), ``t0`` (phase origin, s) and ``soft`` (bool). The RX
        patches are laid out element by element.
    rx_ptr : (n_out + 1,) numpy.ndarray
        Offsets: receive element ``e`` owns RX patches ``rx_ptr[e]:rx_ptr[e+1]``.
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    omega : (N_ω,) numpy.ndarray
        Uniform angular frequencies (rad/s).
    atten : dict or None, default None
        Causal power-law attenuation along the round trip: keys ``alpha0_np``
        (Np/(Hz^y·m)), ``freq_power``, ``f0_hz``, ``tx_ref`` (3,) TX centre and
        ``rx_ref`` (n_out, 3) receive-element centres, in metres. None → lossless.

    Returns
    -------
    (n_out, N_ω) numpy.ndarray
        Summed two-way spectrum per receive element (complex128).
    """

    points = np.asarray(points, dtype=np.float32)
    n_out = len(rx_ptr) - 1
    at = atten or {}
    y = float(at.get("freq_power", 1.0))
    y_is_one = abs(y - 1.0) < 1e-10
    # Thread count resolved here so the cached kernel never touches the Numba runtime.
    n_chunks = max(1, min(get_num_threads(), points.shape[0]))
    return _twoway_summed_points(
        points,
        np.asarray(amps, dtype=np.float64),
        *_aperture(tx),
        *_aperture(rx),
        np.ascontiguousarray(rx_ptr, dtype=np.int64),
        float(inv_c),
        *_omega(omega),
        atten is not None,
        np.asarray(at.get("tx_ref", np.zeros(3)), dtype=np.float64),
        np.asarray(at.get("rx_ref", np.zeros((n_out, 3))), dtype=np.float64),
        float(at.get("alpha0_np", 0.0)),
        y,
        0.0 if y_is_one else float(np.tan(y * np.pi / 2.0)),
        float(at.get("f0_hz", 1.0)),
        y_is_one,
        n_chunks,
    )
