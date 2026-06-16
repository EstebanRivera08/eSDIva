"""Pulse-echo SDI kernels for the two analytic forms of `p_pe = v_pe ⊛ h_tx ⊛ h_rx`.

The pulse-echo response of a (TX patch, RX patch) pair is the time convolution of their
one-way spatial impulse responses, `h_tx ⊛ h_rx`. Each one-way SIR is a trapezoid whose
second time-derivative is four Dirac deltas, so the *product* of the two delta trains is
the raw two-way kernel `Δδ_pe = D²h_tx ⊛ D²h_rx` — 16 deltas per patch pair. The smooth
two-way SIR is recovered by integrating four times, `h_tx ⊛ h_rx = I⁴ Δδ_pe`.

PyField evaluates that pulse-echo signal two ways, each implemented here. They give the
same RF; they differ only in where the four integrations and the convolution are done:

Paired (TX×RX patch-pair enumeration, 16 deltas per pair — cost ∝ M_tx·M_rx):
- `compute_pe_complete` — push the four integrations onto the excitation once, forming the
  integrated pulse-echo waveform `w = I⁴ v_pe`, then for each of the 16 corner events of a
  pair splat a shifted, scaled copy of `w`. No FFT and no cumulative sum: the output is the
  final RF trace. Exact but O(len(w)) per pair, so it is the small-aperture / reference path.

Spectral (one-way TX and RX spectra built independently, then multiplied — cost ∝
M_tx + M_rx, no forward FFT):
- `compute_oneway_spectrum_band` — the closed-form Fourier transform of one aperture's
  corner-delta train, a sum of four phasors per patch, evaluated only on the in-band
  frequencies, `(P, N_band)`, with optional per-patch attenuation. The caller multiplies the
  TX and RX results (convolution ⇒ product) and applies I⁴ = ÷(jω)⁴ downstream.
- `compute_oneway_spectrum_band_batched` — the same spectrum for every receive element in a
  single call, `(E, P, N_band)`, so a multi-element receive aperture costs one kernel
  dispatch instead of one per element.

Each kernel parallelizes over scatterers (`prange` over P); a single field point (a PSF,
`P == 1`) instead parallelizes over patches so one point still saturates every core.
"""

import numpy as np
from numba import get_num_threads, njit, prange

from pyfield.attenuation.attenuation import _causal_atten_factor

from .helpers import (
    _compute_rectangle_SIR_params,
    _prep_pe_arrays,
    identity_tangents,
    pack_tangents,
)


@njit(inline="always")
def _patch_corner_times(
    px, py, pz, cx, cy, cz, eu0, eu1, eu2, ev0, ev1, ev2, wx, wy, inv_c, apod, delay, dt
):
    """Trapezoid corner times + slope of one patch seen from one field point.

    Projects the patch-to-point direction onto the patch local frame, then returns the
    trapezoidal SIR corners and its rising slope (= plateau height / rise time). The
    second derivative of that trapezoid is the delta train the SDI kernels place, scaled
    by this slope. ``slope == 0.0`` flags a degenerate patch (point on the patch, or a
    sub-threshold plateau) the caller should skip.
    """
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < np.float32(1e-12):
        return np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), 0.0
    inv_dist = np.float32(1.0) / dist
    xp = (dx * eu0 + dy * eu1 + dz * eu2) * inv_dist
    yp = (dx * ev0 + dy * ev1 + dz * ev2) * inv_dist
    t1, t2, t3, t4, h_max = _compute_rectangle_SIR_params(
        wx, wy, xp, yp, dist, inv_c, apod, delay, dt
    )
    if h_max < np.float32(1e-6):
        return t1, t2, t3, t4, 0.0
    return t1, t2, t3, t4, h_max / (t2 - t1)


# ---------------------------------------------------------------------------
# Factored SDI PE: closed-form one-way SIR spectrum (no forward FFT).
#
# The factored form never builds a time-domain SIR. The second derivative of one
# patch's trapezoidal SIR is four corner deltas, so the Fourier transform of that
# one-way delta train is exact and closed form — a sum of four complex exponentials:
#
#     Σ_one-way(ω) = Σ_m slope_m · [ e^{-jω t1} − e^{-jω t2} − e^{-jω t3} + e^{-jω t4} ]
#
# (corner signs +,−,−,+). The two-way SIR spectrum is then the product of the TX and
# RX one-way spectra (convolution ⇒ multiplication), and the four integrations I⁴ that
# turn ∂²h into h are applied downstream as ÷(jω)⁴. Because the received signal is
# band-limited by the excitation/impulse-response chain, the spectrum is only evaluated
# on the in-band frequencies handed in via ``omega`` — the rest of the band contributes
# nothing once multiplied by the (near-zero) out-of-band filter, so it is skipped.
#
# Per-patch one-way attenuation is folded in here for free: the patch-to-point distance
# is already needed, so each patch term is multiplied by its own causal attenuation
# factor before being summed. Doing it inside the patch sum (rather than once on the
# combined SIR) gives a true per-path round trip when TX and RX spectra are multiplied.
# ---------------------------------------------------------------------------


@njit(inline="always")
def _phasor(x):
    """Unit phasor e^{-jx} = cos x − j sin x."""
    return complex(np.cos(x), -np.sin(x))


@njit(parallel=True, fastmath=True, cache=True)
def _oneway_spectrum_points(
    points,
    centers,
    wx,
    wy,
    tangents,
    apod,
    delays,
    inv_c,
    t0,
    omega,
    dt,
    do_atten,
    alpha0_np,
    y,
    tan_y,
    f0,
    y_is_one,
):
    """Closed-form one-way SIR spectrum at each scatterer → (P, N_band) complex128.

    For each scatterer (``prange`` over P, race-free rows) sums the analytic four-corner
    delta spectrum of every patch over the in-band angular frequencies ``omega``. The
    frequency grid is uniform, so each corner's phasor ``e^{-jω(t_i-t0)}`` advances by a
    constant factor ``e^{-jΔω(t_i-t0)}`` from one bin to the next — it is swept by
    repeated complex multiplication instead of calling sin/cos at every bin.
    """
    P = points.shape[0]
    M = centers.shape[0]
    Nb = omega.shape[0]
    out = np.zeros((P, Nb), dtype=np.complex128)
    w0 = omega[0]
    dw = omega[1] - omega[0] if Nb > 1 else 0.0
    for p in prange(P):  # ty: ignore[not-iterable]
        px = points[p, 0]
        py = points[p, 1]
        pz = points[p, 2]
        for m in range(M):
            t1, t2, t3, t4, slope = _patch_corner_times(
                px,
                py,
                pz,
                centers[m, 0],
                centers[m, 1],
                centers[m, 2],
                tangents[m, 0],
                tangents[m, 1],
                tangents[m, 2],
                tangents[m, 3],
                tangents[m, 4],
                tangents[m, 5],
                wx[m],
                wy[m],
                inv_c,
                apod[m],
                delays[m],
                dt,
            )
            if slope == 0.0:
                continue
            # Distance for the per-path attenuation (only needed when attenuating).
            dist = 0.0
            if do_atten:
                dx = px - centers[m, 0]
                dy = py - centers[m, 1]
                dz = pz - centers[m, 2]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            # Corner times relative to the window origin, and the per-bin phasor steps.
            a1 = t1 - t0
            a2 = t2 - t0
            a3 = t3 - t0
            a4 = t4 - t0
            ph1 = slope * _phasor(w0 * a1)
            ph2 = slope * _phasor(w0 * a2)
            ph3 = slope * _phasor(w0 * a3)
            ph4 = slope * _phasor(w0 * a4)
            s1 = _phasor(dw * a1)
            s2 = _phasor(dw * a2)
            s3 = _phasor(dw * a3)
            s4 = _phasor(dw * a4)
            for k in range(Nb):
                val = ph1 - ph2 - ph3 + ph4
                if do_atten:
                    val *= _causal_atten_factor(
                        omega[k], dist, alpha0_np, y, tan_y, f0, y_is_one
                    )
                out[p, k] += val
                ph1 *= s1
                ph2 *= s2
                ph3 *= s3
                ph4 *= s4
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _oneway_spectrum_patches(
    point,
    centers,
    wx,
    wy,
    tangents,
    apod,
    delays,
    inv_c,
    t0,
    omega,
    dt,
    do_atten,
    alpha0_np,
    y,
    tan_y,
    f0,
    y_is_one,
):
    """Closed-form one-way SIR spectrum for one field point → (1, N_band) complex128.

    ``prange`` over patches (each writes its own row of an ``(M, N_band)`` buffer,
    race-free) so a single point (a PSF) still saturates every core; the rows are summed.
    """
    M = centers.shape[0]
    Nb = omega.shape[0]
    buf = np.zeros((M, Nb), dtype=np.complex128)
    px = point[0]
    py = point[1]
    pz = point[2]
    w0 = omega[0]
    dw = omega[1] - omega[0] if Nb > 1 else 0.0
    for m in prange(M):  # ty: ignore[not-iterable]
        t1, t2, t3, t4, slope = _patch_corner_times(
            px,
            py,
            pz,
            centers[m, 0],
            centers[m, 1],
            centers[m, 2],
            tangents[m, 0],
            tangents[m, 1],
            tangents[m, 2],
            tangents[m, 3],
            tangents[m, 4],
            tangents[m, 5],
            wx[m],
            wy[m],
            inv_c,
            apod[m],
            delays[m],
            dt,
        )
        if slope == 0.0:
            continue
        dist = 0.0
        if do_atten:
            dx = px - centers[m, 0]
            dy = py - centers[m, 1]
            dz = pz - centers[m, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        a1 = t1 - t0
        a2 = t2 - t0
        a3 = t3 - t0
        a4 = t4 - t0
        ph1 = slope * _phasor(w0 * a1)
        ph2 = slope * _phasor(w0 * a2)
        ph3 = slope * _phasor(w0 * a3)
        ph4 = slope * _phasor(w0 * a4)
        s1 = _phasor(dw * a1)
        s2 = _phasor(dw * a2)
        s3 = _phasor(dw * a3)
        s4 = _phasor(dw * a4)
        for k in range(Nb):
            val = ph1 - ph2 - ph3 + ph4
            if do_atten:
                val *= _causal_atten_factor(
                    omega[k], dist, alpha0_np, y, tan_y, f0, y_is_one
                )
            buf[m, k] += val
            ph1 *= s1
            ph2 *= s2
            ph3 *= s3
            ph4 *= s4
    out = np.zeros((1, Nb), dtype=np.complex128)
    for m in range(M):
        out[0] += buf[m]
    return out


def compute_oneway_spectrum_band(
    points,
    centers,
    wx,
    wy,
    apod,
    delays,
    inv_c,
    t0,
    omega,
    dt,
    *,
    eu=None,
    ev=None,
    alpha0_np=None,
    freq_power=1.0,
    f0_hz=0.0,
):
    """Closed-form one-way SIR spectrum on in-band frequencies (factored SDI form).

    Evaluates the analytic Fourier transform of one aperture's spatial-impulse-response
    second derivative — a sum over patches of four corner phasors,
    ``Σ_m slope_m (e^{-jωt1} − e^{-jωt2} − e^{-jωt3} + e^{-jωt4})`` — directly at the
    requested angular frequencies, with NO time sampling and NO forward FFT. The two-way
    SIR spectrum is the product of the TX and RX results; the integrations I⁴ are applied
    by the caller as ÷(jω)⁴. Optional per-patch causal attenuation
    ``exp(−α|f|^y d)·(K-K phase)`` is folded into each patch term using its own
    patch-to-point distance, so the TX×RX product carries the true round-trip loss.

    Parameters
    ----------
    points : (P, 3) numpy.ndarray
        Scatterer positions in metres.
    centers : (M, 3) numpy.ndarray
        Patch centres in metres (this aperture's patches, or one element's).
    wx, wy : (M,) numpy.ndarray
        Patch widths in the two in-plane directions (metres).
    apod : (M,) numpy.ndarray
        Apodization weight per patch.
    delays : (M,) numpy.ndarray
        Delay per patch (seconds).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        Window origin (seconds); corner times are referenced to it so the inverse
        transform lands the trace at the window start.
    omega : (N_band,) numpy.ndarray
        In-band angular frequencies 2πf (rad/s), uniformly spaced.
    dt : float
        Time step 1/fs (seconds); clamps sub-sample patch edge crossings.
    eu, ev : (M, 3) numpy.ndarray or None, default None
        Patch tangent frames; None → flat-patch identity tangents.
    alpha0_np : float or None, default None
        Absorption coefficient in Np/(Hz^y·m). None disables attenuation.
    freq_power : float, default 1.0
        Attenuation power-law exponent y.
    f0_hz : float, default 0.0
        Reference frequency (Hz) for the y = 1 dispersion term.

    Returns
    -------
    (P, N_band) numpy.ndarray
        One-way SIR-delta spectrum at each scatterer (complex64).
    """
    points = np.asarray(points, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    wx = np.asarray(wx, dtype=np.float32)
    wy = np.asarray(wy, dtype=np.float32)
    apod = np.asarray(apod, dtype=np.float32)
    delays = np.asarray(delays, dtype=np.float32)
    omega = np.ascontiguousarray(np.asarray(omega, dtype=np.float64))
    inv_c, t0, dt = float(inv_c), float(t0), float(dt)
    M = centers.shape[0]
    if eu is None or ev is None:
        eu, ev = identity_tangents(M)
    tangents = pack_tangents(
        np.asarray(eu, dtype=np.float32), np.asarray(ev, dtype=np.float32)
    )

    do_atten = alpha0_np is not None
    y = float(freq_power)
    y_is_one = abs(y - 1.0) < 1e-10
    tan_y = float(np.tan(y * np.pi / 2.0)) if not y_is_one else 0.0
    a0 = float(alpha0_np) if alpha0_np is not None else 0.0
    f0 = float(f0_hz)

    args = (
        centers,
        wx,
        wy,
        tangents,
        apod,
        delays,
        inv_c,
        t0,
        omega,
        dt,
        do_atten,
        a0,
        y,
        tan_y,
        f0,
        y_is_one,
    )
    if points.shape[0] == 1:
        return _oneway_spectrum_patches(points[0], *args).astype(np.complex64)
    return _oneway_spectrum_points(points, *args).astype(np.complex64)


@njit(parallel=True, fastmath=True, cache=True)
def _oneway_spectrum_batched(
    points, centers, wx, wy, tangents, apod, delays, inv_c, t0, omega, dt
):
    """Closed-form one-way SIR spectrum for EVERY receive element at once → (E, P, N_band).

    The single-element kernel `_oneway_spectrum_points` evaluated once per receive element
    pays a full kernel dispatch for each one, even though each element holds only ~M/E
    patches. Here the elements are stacked along a leading axis and swept in one parallel
    loop (`prange` over E, each element owning a race-free output block): same four-corner
    phasor sum per patch, same per-bin complex-multiply sweep across the uniform in-band
    grid (no per-bin sin/cos). Elements may have different patch counts; the array is padded
    to the largest with ``apod = 0`` rows, whose slope vanishes so they are skipped. No
    attenuation here — this serves the summed, attenuation-free receive path.
    """
    E = centers.shape[0]
    P = points.shape[0]
    M = centers.shape[1]
    Nb = omega.shape[0]
    out = np.zeros((E, P, Nb), dtype=np.complex128)
    w0 = omega[0]
    dw = omega[1] - omega[0] if Nb > 1 else 0.0
    for e in prange(E):  # ty: ignore[not-iterable]
        for p in range(P):
            px = points[p, 0]
            py = points[p, 1]
            pz = points[p, 2]
            for m in range(M):
                t1, t2, t3, t4, slope = _patch_corner_times(
                    px,
                    py,
                    pz,
                    centers[e, m, 0],
                    centers[e, m, 1],
                    centers[e, m, 2],
                    tangents[e, m, 0],
                    tangents[e, m, 1],
                    tangents[e, m, 2],
                    tangents[e, m, 3],
                    tangents[e, m, 4],
                    tangents[e, m, 5],
                    wx[e, m],
                    wy[e, m],
                    inv_c,
                    apod[e, m],
                    delays[e, m],
                    dt,
                )
                if slope == 0.0:
                    continue
                a1 = t1 - t0
                a2 = t2 - t0
                a3 = t3 - t0
                a4 = t4 - t0
                ph1 = slope * _phasor(w0 * a1)
                ph2 = slope * _phasor(w0 * a2)
                ph3 = slope * _phasor(w0 * a3)
                ph4 = slope * _phasor(w0 * a4)
                s1 = _phasor(dw * a1)
                s2 = _phasor(dw * a2)
                s3 = _phasor(dw * a3)
                s4 = _phasor(dw * a4)
                for k in range(Nb):
                    out[e, p, k] += ph1 - ph2 - ph3 + ph4
                    ph1 *= s1
                    ph2 *= s2
                    ph3 *= s3
                    ph4 *= s4
    return out


def compute_oneway_spectrum_band_batched(
    points, centers, wx, wy, apod, delays, eu, ev, inv_c, t0, omega, dt
):
    """`compute_oneway_spectrum_band` for all E receive elements in one call → (E, P, N_band).

    The receive-element patch sets are stacked into rectangular ``(E, m_max, …)`` arrays
    (pad rows carry ``apod = 0`` so they contribute nothing), so the whole receive aperture
    is evaluated with a single kernel dispatch instead of one per element. Same closed-form
    analytic spectrum as the single-aperture function; attenuation is not applied here (the
    summed receive path that uses it is attenuation-free).

    Parameters
    ----------
    points : (P, 3) numpy.ndarray
        Scatterer positions in metres.
    centers : (E, m_max, 3) numpy.ndarray
        Per-element patch centres in metres (zero-padded to the largest element).
    wx, wy : (E, m_max) numpy.ndarray
        Per-element patch widths in the two in-plane directions (metres).
    apod : (E, m_max) numpy.ndarray
        Per-element apodization weight per patch; pad rows must be 0.
    delays : (E, m_max) numpy.ndarray
        Per-element delay per patch (seconds).
    eu, ev : (E, m_max, 3) numpy.ndarray
        Per-element patch tangent frames (in-plane unit vectors).
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        Window origin (seconds) the corner phasors are referenced to.
    omega : (N_band,) numpy.ndarray
        In-band angular frequencies 2πf (rad/s), uniformly spaced.
    dt : float
        Time step 1/fs (seconds); clamps sub-sample patch edge crossings.

    Returns
    -------
    (E, P, N_band) numpy.ndarray
        One-way SIR-delta spectrum at each scatterer, per receive element (complex64).
    """
    points = np.asarray(points, dtype=np.float32)
    centers = np.ascontiguousarray(np.asarray(centers, dtype=np.float32))
    wx = np.ascontiguousarray(np.asarray(wx, dtype=np.float32))
    wy = np.ascontiguousarray(np.asarray(wy, dtype=np.float32))
    apod = np.ascontiguousarray(np.asarray(apod, dtype=np.float32))
    delays = np.ascontiguousarray(np.asarray(delays, dtype=np.float32))
    omega = np.ascontiguousarray(np.asarray(omega, dtype=np.float64))
    # Pack the two tangent vectors into one contiguous (E, m, 6) frame (cols 0-2 = u, 3-5 = v).
    E, m = centers.shape[:2]
    tangents = np.empty((E, m, 6), dtype=np.float32)
    tangents[..., :3] = eu
    tangents[..., 3:] = ev
    out = _oneway_spectrum_batched(
        points,
        centers,
        wx,
        wy,
        np.ascontiguousarray(tangents),
        apod,
        delays,
        float(inv_c),
        float(t0),
        omega,
        float(dt),
    )
    return out.astype(np.complex64)


# ---------------------------------------------------------------------------
# Complete SDI PE: splat w = I⁴ v_pe per pair (no FFT, no cumsum).
# Each of the 16 corner events convolves a 2-bin linear-interp delta with w, i.e. adds a
# shifted, scaled copy of w. Output is the final per-element RF trace (still ×scale and
# amplitude-weighted by the caller).
# ---------------------------------------------------------------------------


@njit(inline="always")
def _add_shifted_w(out, row, kf, gain, w, nfft):
    """Add ``gain``-scaled, 2-bin-interpolated copy of ``w`` at index ``kf``, wrapped mod nfft.

    Circular placement: the event at continuous index ``kf`` deposits ``w`` rolled to
    ``kf`` (plus the fractional ceil tap). ``w`` is the full-length (``nfft``) integrated
    exc/IR kernel — the zero-phase I⁴ filter is delocalized, so the convolution must be
    circular (then sliced to ``pe_T``) to match the FFT path exactly.
    """
    kf_floor = int(np.floor(kf))
    w_ceil = kf - kf_floor
    g_floor = gain * (np.float32(1.0) - w_ceil)
    g_ceil = gain * w_ceil
    base = kf_floor % nfft
    base1 = base + 1
    if base1 >= nfft:
        base1 -= nfft
    for li in range(nfft):
        wl = w[li]
        k = base + li
        if k >= nfft:
            k -= nfft
        out[row, k] += g_floor * wl
        k1 = base1 + li
        if k1 >= nfft:
            k1 -= nfft
        out[row, k1] += g_ceil * wl


@njit(inline="always")
def _place_pe_complete(
    out, row, t0, fs, t1e, t2e, t3e, t4e, t1r, t2r, t3r, t4r, weight, w, nfft
):
    """Splat w for the 16 PE corner events of one (m_e, m_r) pair into out[row, :]."""
    signp = np.float32(1.0)
    signn = np.float32(-1.0)
    for i_r in range(4):
        if i_r == 0:
            sr = signp
            tr = t1r
        elif i_r == 1:
            sr = signn
            tr = t2r
        elif i_r == 2:
            sr = signn
            tr = t3r
        else:
            sr = signp
            tr = t4r
        gp = signp * sr * weight
        gn = signn * sr * weight
        _add_shifted_w(out, row, (t1e + tr - t0) * fs, gp, w, nfft)
        _add_shifted_w(out, row, (t2e + tr - t0) * fs, gn, w, nfft)
        _add_shifted_w(out, row, (t3e + tr - t0) * fs, gn, w, nfft)
        _add_shifted_w(out, row, (t4e + tr - t0) * fs, gp, w, nfft)


@njit(parallel=True, fastmath=True, cache=True)
def _pe_complete_points(
    points,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_tangents,
    tx_apod,
    tx_delays,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_tangents,
    rx_apod,
    rx_delays,
    w,
    inv_c,
    t0,
    T,
    fs,
    dt,
):
    """Complete-SDI PE RF per scatterer → (P, T) float64. prange over scatterers.

    Splats ``w`` (full-length ``nfft`` integrated exc/IR kernel) circularly per pair,
    then returns the ``[:T]`` window — the FFT path's circular convolution done by hand.
    Exact but O(nfft) per pair, hence the slow reference path. Each scatterer owns its
    own output row.
    """
    P = points.shape[0]
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    nfft = w.shape[0]
    out = np.zeros((P, nfft), dtype=np.float64)
    for p in prange(P):  # ty: ignore[not-iterable]
        px = points[p, 0]
        py = points[p, 1]
        pz = points[p, 2]
        for m_r in range(M_r):
            t1r, t2r, t3r, t4r, slope_r = _patch_corner_times(
                px,
                py,
                pz,
                rx_centers[m_r, 0],
                rx_centers[m_r, 1],
                rx_centers[m_r, 2],
                rx_tangents[m_r, 0],
                rx_tangents[m_r, 1],
                rx_tangents[m_r, 2],
                rx_tangents[m_r, 3],
                rx_tangents[m_r, 4],
                rx_tangents[m_r, 5],
                rx_wx[m_r],
                rx_wy[m_r],
                inv_c,
                rx_apod[m_r],
                rx_delays[m_r],
                dt,
            )
            if slope_r == 0.0:
                continue
            for m_e in range(M_e):
                t1e, t2e, t3e, t4e, slope_e = _patch_corner_times(
                    px,
                    py,
                    pz,
                    tx_centers[m_e, 0],
                    tx_centers[m_e, 1],
                    tx_centers[m_e, 2],
                    tx_tangents[m_e, 0],
                    tx_tangents[m_e, 1],
                    tx_tangents[m_e, 2],
                    tx_tangents[m_e, 3],
                    tx_tangents[m_e, 4],
                    tx_tangents[m_e, 5],
                    tx_wx[m_e],
                    tx_wy[m_e],
                    inv_c,
                    tx_apod[m_e],
                    tx_delays[m_e],
                    dt,
                )
                if slope_e == 0.0:
                    continue
                _place_pe_complete(
                    out,
                    p,
                    t0,
                    fs,
                    t1e,
                    t2e,
                    t3e,
                    t4e,
                    t1r,
                    t2r,
                    t3r,
                    t4r,
                    slope_r * slope_e,
                    w,
                    nfft,
                )
    return out[:, :T]


@njit(parallel=True, fastmath=True, cache=True)
def _pe_complete_patches(
    point,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_tangents,
    tx_apod,
    tx_delays,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_tangents,
    rx_apod,
    rx_delays,
    w,
    inv_c,
    t0,
    T,
    fs,
    dt,
):
    """Complete-SDI PE RF for one field point → (T,) float64. prange over TX patches.

    With few scatterers `prange` over scatterers starves the cores, yet the cost per
    point is the full ``16·M_e·M_r`` pair sweep, each pair splatting the length-``nfft``
    kernel ``w`` — the analytic wall the complete form pays. Here each TX patch writes
    its own row of an ``(M_e, nfft)`` buffer (race-free), so one point saturates the box.
    Each pair splats ``w`` circularly (the I⁴ filter is zero-phase, delocalized); the
    summed buffer is sliced to ``[:T]``.
    """
    M_e = tx_centers.shape[0]
    M_r = rx_centers.shape[0]
    nfft = w.shape[0]
    px = point[0]
    py = point[1]
    pz = point[2]
    buf = np.zeros((M_e, nfft), dtype=np.float64)  # one row per TX patch, race-free.
    for m_e in prange(M_e):  # ty: ignore[not-iterable]
        t1e, t2e, t3e, t4e, slope_e = _patch_corner_times(
            px,
            py,
            pz,
            tx_centers[m_e, 0],
            tx_centers[m_e, 1],
            tx_centers[m_e, 2],
            tx_tangents[m_e, 0],
            tx_tangents[m_e, 1],
            tx_tangents[m_e, 2],
            tx_tangents[m_e, 3],
            tx_tangents[m_e, 4],
            tx_tangents[m_e, 5],
            tx_wx[m_e],
            tx_wy[m_e],
            inv_c,
            tx_apod[m_e],
            tx_delays[m_e],
            dt,
        )
        if slope_e == 0.0:
            continue
        for m_r in range(M_r):
            t1r, t2r, t3r, t4r, slope_r = _patch_corner_times(
                px,
                py,
                pz,
                rx_centers[m_r, 0],
                rx_centers[m_r, 1],
                rx_centers[m_r, 2],
                rx_tangents[m_r, 0],
                rx_tangents[m_r, 1],
                rx_tangents[m_r, 2],
                rx_tangents[m_r, 3],
                rx_tangents[m_r, 4],
                rx_tangents[m_r, 5],
                rx_wx[m_r],
                rx_wy[m_r],
                inv_c,
                rx_apod[m_r],
                rx_delays[m_r],
                dt,
            )
            if slope_r == 0.0:
                continue
            _place_pe_complete(
                buf,
                m_e,
                t0,
                fs,
                t1e,
                t2e,
                t3e,
                t4e,
                t1r,
                t2r,
                t3r,
                t4r,
                slope_r * slope_e,
                w,
                nfft,
            )
    return buf.sum(axis=0)[:T]


def compute_pe_complete(
    points,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_apod,
    tx_delays,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_apod,
    rx_delays,
    w,
    inv_c,
    t0,
    T,
    fs,
    dt,
    *,
    tx_eu=None,
    tx_ev=None,
    rx_eu=None,
    rx_ev=None,
):
    """Complete SDI PE: splat ``w = I⁴ v_pe`` per pair → ``(P, T)`` float32.

    Cumsum-free evaluation of ``p_pe = Σ_i Σ_j a_i a_j w(t − τ_i − τ_j)``: each of the
    16 corner events per (m_e, m_r) pair adds a 2-bin-interpolated, slope-weighted copy
    of ``w``, wrapped mod ``len(w)``. ``w`` is the FULL-length (``nfft``) integrated
    exc/IR kernel ``I⁴(e ⊛ ir_tx ⊛ ir_rx)``; the circular convolution is sliced to ``T``
    (= ``pe_T``), reproducing the FFT path exactly. The caller applies ``scale`` +
    amplitude weighting. Exact but O(nfft) per pair (slow reference path).

    Parallelizes over scatterers when ``P ≥ n_threads``; with fewer points it loops the
    points and parallelizes each over TX patches instead, so even a single point-spread
    scatterer keeps every core busy on the ``16·M_e·M_r`` sweep.

    Parameters
    ----------
    points : (P, 3) numpy.ndarray
        Scatterer positions in metres.
    tx_centers : (M_e, 3) numpy.ndarray
        TX patch centres in metres.
    tx_wx : (M_e,) numpy.ndarray
        TX patch width in x (metres).
    tx_wy : (M_e,) numpy.ndarray
        TX patch width in y (metres).
    tx_apod : (M_e,) numpy.ndarray
        TX apodization weight per patch.
    tx_delays : (M_e,) numpy.ndarray
        TX delay per patch (seconds).
    rx_centers : (M_r, 3) numpy.ndarray
        RX patch centres in metres (one element's patches).
    rx_wx : (M_r,) numpy.ndarray
        RX patch width in x (metres).
    rx_wy : (M_r,) numpy.ndarray
        RX patch width in y (metres).
    rx_apod : (M_r,) numpy.ndarray
        RX apodization weight per patch.
    rx_delays : (M_r,) numpy.ndarray
        RX delay per patch (seconds).
    w : (nfft,) numpy.ndarray
        Integrated exc/IR kernel ``I⁴(e ⊛ ir_tx ⊛ ir_rx)``; ``len(w)`` sets the
        circular-convolution period.
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    t0 : float
        Start of the time grid (seconds).
    T : int
        Number of output time samples (``pe_T``); the circular result is sliced to it.
    fs : float
        Sampling frequency (Hz).
    dt : float
        Time step 1/fs (seconds).
    tx_eu, tx_ev : (M_e, 3) numpy.ndarray or None, default None
        TX patch tangent vectors; None → flat-patch identity tangents.
    rx_eu, rx_ev : (M_r, 3) numpy.ndarray or None, default None
        RX patch tangent vectors; None → flat-patch identity tangents.

    Returns
    -------
    (P, T) numpy.ndarray
        Per-scatterer complete-SDI pulse-echo RF (float32), before ``scale``/amps.
    """
    (
        points,
        tx_centers,
        tx_wx,
        tx_wy,
        tx_apod,
        tx_delays,
        tx_tangents,
        rx_centers,
        rx_wx,
        rx_wy,
        rx_apod,
        rx_delays,
        rx_tangents,
        inv_c,
        t0,
        fs,
        dt,
        T,
    ) = _prep_pe_arrays(
        points,
        tx_centers,
        tx_wx,
        tx_wy,
        tx_apod,
        tx_delays,
        rx_centers,
        rx_wx,
        rx_wy,
        rx_apod,
        rx_delays,
        inv_c,
        t0,
        T,
        fs,
        dt,
        tx_eu,
        tx_ev,
        rx_eu,
        rx_ev,
    )
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    P = points.shape[0]

    # Few scatterers: prange over P starves the cores (only P run), but each point
    # carries the full 16·M_e·M_r pair sweep. Point-parallel wall ≈ W (one point's work,
    # P cores busy); patch-parallel looped ≈ P·W/n_threads — cheaper exactly when
    # P < n_threads. Below that crossover, loop the points and parallelize each over TX
    # patches so even a single scatterer (PSF) saturates the box.
    if P < get_num_threads():
        out = np.zeros((P, T), dtype=np.float64)
        for p in range(P):
            out[p] = _pe_complete_patches(
                points[p],
                tx_centers,
                tx_wx,
                tx_wy,
                tx_tangents,
                tx_apod,
                tx_delays,
                rx_centers,
                rx_wx,
                rx_wy,
                rx_tangents,
                rx_apod,
                rx_delays,
                w,
                inv_c,
                t0,
                T,
                fs,
                dt,
            )
        return out.astype(np.float32)

    return _pe_complete_points(
        points,
        tx_centers,
        tx_wx,
        tx_wy,
        tx_tangents,
        tx_apod,
        tx_delays,
        rx_centers,
        rx_wx,
        rx_wy,
        rx_tangents,
        rx_apod,
        rx_delays,
        w,
        inv_c,
        t0,
        T,
        fs,
        dt,
    ).astype(np.float32)
