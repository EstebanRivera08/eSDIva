"""P4 prototype: fused numba kernel for the summed spectral two-way spectrum.

The current spectral path, per RX element, builds Σ_TX and Σ_RX as (P, N_band) arrays,
multiplies, and does amps @ (...) — materializing a (P, N_band) complex array and a
separate numba launch per element. This prototype fuses build · product · scatterer-sum
into ONE kernel that parallelizes over scatterers (prange over thread chunks, each writing
a private (N_band,) row, reduced at the end): no (P, N_band) materialization, one launch
for all elements of a group.

Compares, for one RX aperture (all patches → one summed spectrum, the focused-line case):
  (a) array path : compute_oneway_spectrum_band ×2 + amps @ (h_tx*h_rx)
  (b) fused path : single njit kernel, prange over scatterers
on speed (incl. thread scaling) and accuracy.
"""

import time

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from pyfield.hsir.transducer_sir_pe_sdi import (
    _patch_corner_times,
    _phasor,
    compute_oneway_spectrum_band,
)
from pyfield.reception.base import ReceptionBase  # noqa: F401  (ensure pkg import)
from pyfield.transducers import LinearArrayTransducer
from pyfield.utilities.helper_functions import compute_sub_elem_attributes


@njit(inline="always")
def _accum_oneway(acc, px, py, pz, centers, wx, wy, tg, apod, delays, inv_c, t0, omega, dt):
    """Add one aperture's factored-sin one-way SIR spectrum into acc[:] (length N_band)."""
    M = centers.shape[0]
    Nb = omega.shape[0]
    w0 = omega[0]
    dw = omega[1] - omega[0] if Nb > 1 else 0.0
    for m in range(M):
        t1, t2, t3, t4, slope = _patch_corner_times(
            px, py, pz,
            centers[m, 0], centers[m, 1], centers[m, 2],
            tg[m, 0], tg[m, 1], tg[m, 2], tg[m, 3], tg[m, 4], tg[m, 5],
            wx[m], wy[m], inv_c, apod[m], delays[m], dt,
        )
        if slope == 0.0:
            continue
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
            acc[k] += (amp * q1.imag * q2.imag) * pc
            pc *= spc
            q1 *= sq1
            q2 *= sq2


@njit(parallel=True, fastmath=True, cache=True)
def _fused_summed(
    points, amps,
    txc, txwx, txwy, txtg, txap, txdl, tx_t0,
    rxc, rxwx, rxwy, rxtg, rxap, rxdl, rx_t0,
    inv_c, omega, dt,
):
    """Σ_p a_p · Σ_TX(p) · Σ_RX(p) → (N_band,) complex128, fused over scatterers.

    prange over thread chunks of scatterers; each chunk accumulates into a private row
    (race-free), summed at the end. No (P, N_band) intermediate is ever materialized.
    """
    P = points.shape[0]
    Nb = omega.shape[0]
    nthreads = get_num_threads()
    buf = np.zeros((nthreads, Nb), dtype=np.complex128)
    for c in prange(nthreads):  # ty: ignore[not-iterable]
        lo = c * P // nthreads
        hi = (c + 1) * P // nthreads
        tx_acc = np.zeros(Nb, dtype=np.complex128)
        rx_acc = np.zeros(Nb, dtype=np.complex128)
        for p in range(lo, hi):
            tx_acc[:] = 0.0
            rx_acc[:] = 0.0
            px, py, pz = points[p, 0], points[p, 1], points[p, 2]
            _accum_oneway(tx_acc, px, py, pz, txc, txwx, txwy, txtg, txap, txdl,
                          inv_c, tx_t0, omega, dt)
            _accum_oneway(rx_acc, px, py, pz, rxc, rxwx, rxwy, rxtg, rxap, rxdl,
                          inv_c, rx_t0, omega, dt)
            a = amps[p]
            for k in range(Nb):
                buf[c, k] += a * tx_acc[k] * rx_acc[k]
    out = np.zeros(Nb, dtype=np.complex128)
    for c in range(nthreads):
        out += buf[c]
    return out


def pack_tg(eu, ev):
    from pyfield.hsir.helpers import pack_tangents
    return pack_tangents(np.asarray(eu, np.float32), np.asarray(ev, np.float32))


tx = LinearArrayTransducer(
    n_elements=64, element_width_mm=0.25, element_height_mm=8.0,
    kerf_mm=0.05, no_sub_x=4, no_sub_y=6, frequency_Hz=5e6,
)
c, fs = 1540.0, 100e6
inv_c = np.float32(1.0 / c)
txc, txap, txdl, txM, _, txwx, txwy, _ = compute_sub_elem_attributes(tx)
fr = tx.sub_patch_frames
txeu = np.asarray(fr["tangents_u"], np.float32)
txev = np.asarray(fr["tangents_v"], np.float32)
txtg = pack_tg(txeu, txev)

# One band slice (~5 MHz centred), modest length.
nfft = 1024
freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
band = (freqs > 2.5e6) & (freqs < 7.5e6)
omega = (2.0 * np.pi * freqs[band]).astype(np.float64)
print(f"M={txM}, N_band={omega.size}, threads={get_num_threads()}")

rng = np.random.default_rng(0)


def run_array(pts, amp):
    h_tx = compute_oneway_spectrum_band(pts, txc, txwx, txwy, txap, txdl, inv_c, 0.0,
                                        omega, 1.0 / fs, eu=txeu, ev=txev)
    h_rx = compute_oneway_spectrum_band(pts, txc, txwx, txwy, txap, txdl, inv_c, 0.0,
                                        omega, 1.0 / fs, eu=txeu, ev=txev)
    return amp.astype(np.float64) @ (h_tx * h_rx)


def run_fused(pts, amp):
    return _fused_summed(pts.astype(np.float32), amp.astype(np.float64),
                         txc, txwx, txwy, txtg, txap, txdl, 0.0,
                         txc, txwx, txwy, txtg, txap, txdl, 0.0,
                         inv_c, omega, 1.0 / fs)


# Warm up JIT.
p0 = np.array([[0.0, 0.0, 0.03]])
run_array(p0, np.ones(1)); run_fused(p0, np.ones(1))

print(f"\n{'P':>7} {'array[s]':>9} {'fused[s]':>9} {'speedup':>8} {'relerr':>9}")
for P in [2000, 10000, 40000]:
    pts = np.column_stack([rng.uniform(-6, 6, P), np.zeros(P),
                           rng.uniform(20, 90, P)]).astype(np.float64) * 1e-3
    amp = rng.standard_normal(P).astype(np.float32)
    ta = time.perf_counter(); ra = run_array(pts, amp); ta = time.perf_counter() - ta
    tf = time.perf_counter(); rf = run_fused(pts, amp); tf = time.perf_counter() - tf
    relerr = np.max(np.abs(ra - rf)) / max(np.max(np.abs(ra)), 1e-30)
    print(f"{P:>7} {ta:>9.3f} {tf:>9.3f} {ta/tf:>8.2f} {relerr:>9.2e}")

# Thread scaling of the fused kernel.
P = 40000
pts = np.column_stack([rng.uniform(-6, 6, P), np.zeros(P),
                       rng.uniform(20, 90, P)]).astype(np.float64) * 1e-3
amp = rng.standard_normal(P).astype(np.float32)
print(f"\nfused thread scaling (P={P}):")
maxt = get_num_threads()
for nt in sorted({1, 2, 4, maxt}):
    set_num_threads(nt)
    run_fused(p0, np.ones(1))  # re-warm at this thread count
    t = time.perf_counter(); run_fused(pts, amp); t = time.perf_counter() - t
    print(f"  {nt:>2} threads: {t:.3f} s")
set_num_threads(maxt)
