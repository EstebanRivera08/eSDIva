"""Spectral SIR kernels: closed-form one-way SIR spectra, no time sampling, no FFT.

A rectangular patch's SIR is a trapezoid whose second derivative is four corner deltas,
so the Fourier transform of one aperture's corner train is a closed-form sum of phasors.
The pulse-echo spectrum of a (TX aperture, RX element) pair is the product of the two
one-way spectra (convolution ⇒ product), so the cost is linear in the patch count.

- `compute_oneway_spectrum_band` — one aperture's spectrum at each point, ``(P, N_band)``.
- `compute_twoway_spectrum_summed` — Σ_p a_p·Σ_TX·Σ_RX,e per receive element, one fused
  parallel pass (the TX spectrum is built once per scatterer and reused).

Each kernel parallelizes over scatterers (`prange` over P); a single field point (a PSF)
instead parallelizes over patches so one point still saturates every core.
"""

import numpy as np
from numba import get_num_threads, njit, prange

from esdiva.attenuation.attenuation import _causal_atten_factor

from .helpers import _patch_corner_times, identity_tangents, pack_tangents


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


@njit(inline="always")
def _accum_patch_band(
    acc,
    t1,
    t2,
    t3,
    t4,
    slope,
    dist,
    t0,
    omega,
    do_atten,
    alpha0_np,
    y,
    tan_y,
    f0,
    y_is_one,
):
    """Add one patch's closed-form one-way SIR spectrum into the 1-D ``acc`` over ``omega``.

    A rectangular patch's one-way SIR is a trapezoid with corner times
    ``t1,t2,t3,t4 = t1, t1+Δt1, t1+Δt2, t1+Δt1+Δt2``. The Fourier transform of its second
    derivative (a four-corner delta train) factors, via ``1-e^{-jx}=2j·sin(x/2)·e^{-jx/2}``,
    into a real envelope times a single phasor at the patch-centre arrival ``t_c=(t1+t4)/2``:

        S_patch(ω) = -4·slope·sin(ωΔt1/2)·sin(ωΔt2/2)·e^{-jω(t_c-t0)} .

    The two sines are read off the imaginary parts of swept half-angle phasors, so the term
    is never formed as a subtraction of near-equal phasors — it stays bounded by the physical
    plateau (``4·slope·sin(ωΔt1/2) → 2·h_max·ω`` for thin patches) and the sum is well
    conditioned in complex64/float32. The grid is uniform, so each phasor advances by a
    constant factor between bins (swept by one complex multiply, no per-bin sin/cos). Optional
    per-patch causal attenuation is folded in using ``dist``. Shared by every spectral kernel
    (points / patches / fused two-way) so the corner sweep lives in exactly one place.
    """
    Nb = omega.shape[0]
    w0 = omega[0]
    dw = omega[1] - omega[0] if Nb > 1 else 0.0
    half1 = np.float32(0.5) * (t2 - t1)
    half2 = np.float32(0.5) * (t3 - t1)
    tc = np.float32(0.5) * (t1 + t4) - t0
    amp = -4.0 * slope
    pc = _phasor(w0 * tc)
    q1 = _phasor(w0 * half1)
    q2 = _phasor(w0 * half2)
    spc = _phasor(dw * tc)
    sq1 = _phasor(dw * half1)
    sq2 = _phasor(dw * half2)
    for k in range(Nb):
        val = (amp * q1.imag * q2.imag) * pc  # -4·slope·sin(ωΔt1/2)·sin(ωΔt2/2)·phasor
        if do_atten:
            val *= _causal_atten_factor(
                omega[k], dist, alpha0_np, y, tan_y, f0, y_is_one
            )
        acc[k] += val
        pc *= spc
        q1 *= sq1
        q2 *= sq2


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
            _accum_patch_band(
                out[p],
                t1,
                t2,
                t3,
                t4,
                slope,
                dist,
                t0,
                omega,
                do_atten,
                alpha0_np,
                y,
                tan_y,
                f0,
                y_is_one,
            )
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
        _accum_patch_band(
            buf[m],
            t1,
            t2,
            t3,
            t4,
            slope,
            dist,
            t0,
            omega,
            do_atten,
            alpha0_np,
            y,
            tan_y,
            f0,
            y_is_one,
        )
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


# ---------------------------------------------------------------------------
# Fused two-way (pulse-echo) spectrum, summed over scatterers.
#
# The pulse-echo SIR spectrum of a (TX aperture, RX element) pair is the PRODUCT of their
# one-way SIR spectra (time convolution ⇒ frequency multiplication). For a field of point
# scatterers the recorded two-way spectrum per receive element is the amplitude-weighted
# sum over scatterers of that product:
#
#     S_e(ω) = Σ_p a_p · Σ_TX(ω; r_p) · Σ_RX,e(ω; r_p) .
#
# The fused kernel below evaluates this directly: for each scatterer it builds the TX
# one-way spectrum ONCE (shared by every receive element) and reuses it while sweeping the
# elements, accumulating straight into the per-element output. Nothing of size
# (P, N_band) is ever materialised, and the whole field of receive elements is produced in
# a single parallel pass — the fast path for the depth-binned spectral reception, where the
# per-bin work is otherwise dominated by per-element launch and memory traffic.
# ---------------------------------------------------------------------------


@njit(inline="always")
def _accum_oneway_band(
    acc,
    px,
    py,
    pz,
    centers,
    wx,
    wy,
    tangents,
    apod,
    delays,
    lo,
    hi,
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
    """Add one aperture's closed-form one-way SIR spectrum into ``acc`` over patches lo:hi.

    Walks patches ``lo:hi`` of one aperture, computing each patch's trapezoid corner times
    and (when attenuating) its patch-to-point distance, then delegating the per-patch
    closed-form spectrum sweep to `_accum_patch_band` — the shared, cancellation-free
    factored form ``S_patch(ω) = -4·slope·sin(ωΔt1/2)·sin(ωΔt2/2)·e^{-jω(t_c-t0)}``.
    """
    for m in range(lo, hi):
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
        _accum_patch_band(
            acc,
            t1,
            t2,
            t3,
            t4,
            slope,
            dist,
            t0,
            omega,
            do_atten,
            alpha0_np,
            y,
            tan_y,
            f0,
            y_is_one,
        )


@njit(parallel=True, fastmath=True, cache=True)
def _twoway_summed_points(
    points,
    amps,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_tangents,
    tx_apod,
    tx_delays,
    tx_t0,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_tangents,
    rx_apod,
    rx_delays,
    rx_t0,
    rx_ptr,
    inv_c,
    omega,
    dt,
    do_atten,
    alpha0_np,
    y,
    tan_y,
    f0,
    y_is_one,
    n_chunks,
):
    """Σ_p a_p · Σ_TX(p) · Σ_RX,e(p) → (n_out, N_band) complex128, one parallel pass.

    Parallelises over ``n_chunks`` scatterer chunks (each chunk owns a private
    ``(n_out, N_band)`` buffer, summed at the end — race-free; pass the thread count for
    chunks). Per scatterer the TX one-way spectrum is built ONCE and reused while sweeping
    the receive elements; element ``e`` covers the patch range ``rx_ptr[e]:rx_ptr[e+1]`` of
    the receive aperture laid out element-by-element. The complex128 accumulation is cheap
    (``N_band`` is the in-band bin count, small for a band-limited drive) and the factored
    one-way form keeps it well conditioned.
    """
    P = points.shape[0]
    Nb = omega.shape[0]
    n_out = rx_ptr.shape[0] - 1
    Mtx = tx_centers.shape[0]
    nthr = n_chunks
    buf = np.zeros((nthr, n_out, Nb), dtype=np.complex128)
    for ci in prange(nthr):  # ty: ignore[not-iterable]
        lo = ci * P // nthr
        hi = (ci + 1) * P // nthr
        tx_acc = np.zeros(Nb, dtype=np.complex128)
        rx_acc = np.zeros(Nb, dtype=np.complex128)
        for p in range(lo, hi):
            px = points[p, 0]
            py = points[p, 1]
            pz = points[p, 2]
            tx_acc[:] = 0.0
            _accum_oneway_band(
                tx_acc,
                px,
                py,
                pz,
                tx_centers,
                tx_wx,
                tx_wy,
                tx_tangents,
                tx_apod,
                tx_delays,
                0,
                Mtx,
                inv_c,
                tx_t0,
                omega,
                dt,
                do_atten,
                alpha0_np,
                y,
                tan_y,
                f0,
                y_is_one,
            )
            a = amps[p]
            for e in range(n_out):
                rx_acc[:] = 0.0
                _accum_oneway_band(
                    rx_acc,
                    px,
                    py,
                    pz,
                    rx_centers,
                    rx_wx,
                    rx_wy,
                    rx_tangents,
                    rx_apod,
                    rx_delays,
                    rx_ptr[e],
                    rx_ptr[e + 1],
                    inv_c,
                    rx_t0,
                    omega,
                    dt,
                    do_atten,
                    alpha0_np,
                    y,
                    tan_y,
                    f0,
                    y_is_one,
                )
                for k in range(Nb):
                    buf[ci, e, k] += a * tx_acc[k] * rx_acc[k]
    out = np.zeros((n_out, Nb), dtype=np.complex128)
    for ci in range(nthr):
        out += buf[ci]
    return out


def compute_twoway_spectrum_summed(
    points,
    amps,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_apod,
    tx_delays,
    tx_t0,
    rx_centers,
    rx_wx,
    rx_wy,
    rx_apod,
    rx_delays,
    rx_t0,
    rx_ptr,
    inv_c,
    omega,
    dt,
    *,
    tx_eu=None,
    tx_ev=None,
    rx_eu=None,
    rx_ev=None,
    alpha0_np=None,
    freq_power=1.0,
    f0_hz=0.0,
):
    """Amplitude-summed two-way (pulse-echo) SIR spectrum per receive element.

    The pulse-echo SIR spectrum of a (TX aperture, RX element) pair is the product of their
    one-way SIR spectra (time convolution ⇒ frequency product). This returns, per receive
    element ``e``, the scatterer sum

        S_e(ω) = Σ_p amps[p] · Σ_TX(ω; r_p) · Σ_RX,e(ω; r_p) ,

    on the in-band frequencies ``omega``, building each ``Σ_TX`` once per scatterer and
    reusing it across the receive elements (a single fused parallel pass; no ``(P, N_band)``
    intermediate). The caller multiplies by the shared filter ``I⁴·exc·IR`` and inverse-FFTs.
    Optional per-patch one-way attenuation is folded into both apertures, so the product
    carries the true round-trip loss.

    Parameters
    ----------
    points : (P, 3) numpy.ndarray
        Scatterer positions in metres.
    amps : (P,) numpy.ndarray
        Scattering amplitude per scatterer.
    tx_centers : (M_tx, 3) numpy.ndarray
        Transmit patch centres in metres.
    tx_wx, tx_wy : (M_tx,) numpy.ndarray
        Transmit patch widths in the two in-plane directions (metres).
    tx_apod : (M_tx,) numpy.ndarray
        Transmit apodization weight per patch.
    tx_delays : (M_tx,) numpy.ndarray
        Transmit delay per patch (seconds).
    tx_t0 : float
        Transmit window origin (seconds); corner times are referenced to it.
    rx_centers : (M_rx, 3) numpy.ndarray
        Receive patch centres in metres, laid out element-by-element (element ``e`` is the
        contiguous block ``rx_ptr[e]:rx_ptr[e+1]``).
    rx_wx, rx_wy : (M_rx,) numpy.ndarray
        Receive patch widths (metres).
    rx_apod : (M_rx,) numpy.ndarray
        Receive apodization weight per patch.
    rx_delays : (M_rx,) numpy.ndarray
        Receive delay per patch (seconds).
    rx_t0 : float
        Receive window origin (seconds).
    rx_ptr : (n_out + 1,) numpy.ndarray
        CSR offsets delimiting each receive element's patch block in the receive arrays.
    inv_c : float
        Inverse speed of sound 1/c (s/m).
    omega : (N_band,) numpy.ndarray
        In-band angular frequencies 2πf (rad/s), uniformly spaced.
    dt : float
        Time step 1/fs (seconds); clamps sub-sample patch edge crossings.
    tx_eu, tx_ev : (M_tx, 3) numpy.ndarray or None, default None
        Transmit patch tangent frames; None → flat-patch identity tangents.
    rx_eu, rx_ev : (M_rx, 3) numpy.ndarray or None, default None
        Receive patch tangent frames; None → flat-patch identity tangents.
    alpha0_np : float or None, default None
        Absorption coefficient in Np/(Hz^y·m). None disables attenuation.
    freq_power : float, default 1.0
        Attenuation power-law exponent y.
    f0_hz : float, default 0.0
        Reference frequency (Hz) for the y = 1 dispersion term.

    Returns
    -------
    (n_out, N_band) numpy.ndarray
        Amplitude-summed two-way SIR spectrum per receive element (complex128).
    """
    points = np.asarray(points, dtype=np.float32)
    amps = np.asarray(amps, dtype=np.float64)
    tx_centers = np.asarray(tx_centers, dtype=np.float32)
    tx_wx = np.asarray(tx_wx, dtype=np.float32)
    tx_wy = np.asarray(tx_wy, dtype=np.float32)
    tx_apod = np.asarray(tx_apod, dtype=np.float32)
    tx_delays = np.asarray(tx_delays, dtype=np.float32)
    rx_centers = np.asarray(rx_centers, dtype=np.float32)
    rx_wx = np.asarray(rx_wx, dtype=np.float32)
    rx_wy = np.asarray(rx_wy, dtype=np.float32)
    rx_apod = np.asarray(rx_apod, dtype=np.float32)
    rx_delays = np.asarray(rx_delays, dtype=np.float32)
    rx_ptr = np.ascontiguousarray(np.asarray(rx_ptr, dtype=np.int64))
    omega = np.ascontiguousarray(np.asarray(omega, dtype=np.float64))
    inv_c, tx_t0, rx_t0, dt = float(inv_c), float(tx_t0), float(rx_t0), float(dt)

    if tx_eu is None or tx_ev is None:
        tx_eu, tx_ev = identity_tangents(tx_centers.shape[0])
    if rx_eu is None or rx_ev is None:
        rx_eu, rx_ev = identity_tangents(rx_centers.shape[0])
    tx_tangents = pack_tangents(
        np.asarray(tx_eu, dtype=np.float32), np.asarray(tx_ev, dtype=np.float32)
    )
    rx_tangents = pack_tangents(
        np.asarray(rx_eu, dtype=np.float32), np.asarray(rx_ev, dtype=np.float32)
    )

    do_atten = alpha0_np is not None
    y = float(freq_power)
    y_is_one = abs(y - 1.0) < 1e-10
    tan_y = float(np.tan(y * np.pi / 2.0)) if not y_is_one else 0.0
    a0 = float(alpha0_np) if alpha0_np is not None else 0.0
    f0 = float(f0_hz)
    # Chunk count = thread count (resolved here, in Python, so the cached kernel never
    # references the get_num_threads runtime pointer — which would block Numba caching).
    n_chunks = max(1, min(get_num_threads(), points.shape[0]))

    return _twoway_summed_points(
        points,
        amps,
        tx_centers,
        tx_wx,
        tx_wy,
        tx_tangents,
        tx_apod,
        tx_delays,
        tx_t0,
        rx_centers,
        rx_wx,
        rx_wy,
        rx_tangents,
        rx_apod,
        rx_delays,
        rx_t0,
        rx_ptr,
        inv_c,
        omega,
        dt,
        do_atten,
        a0,
        y,
        tan_y,
        f0,
        y_is_one,
        n_chunks,
    )
