"""Prototype #1: multi-element fused spectral kernel vs the current per-element loop.

The current binned spectral path, per depth bin, builds Σ_TX once then LOOPS the 64 RX
elements — each a separate numba launch building a (Pb, N_band) array and an amps@(...)
matvec. At the binned operating point every piece is tiny (Pb~150, N_band~13), so the
64×n_bins kernel launches + (Pb,N_band) re-streaming of Σ_TX dominate.

Fused kernel: ONE launch per bin, prange over scatterers; Σ_TX computed once per scatterer
(kept in cache) and reused across all RX elements; no (Pb,N_band) temporaries. This times
both ways for producing the per-bin two-way spectra s_all (n_out, N_band), summed over the
real bins the class would use, and checks accuracy.
"""

import time

import numpy as np
from numba import get_num_threads, njit, prange

from pyfield.hsir.helpers import pack_tangents
from pyfield.hsir.transducer_sir_pe_sdi import (
    _patch_corner_times,
    _phasor,
    compute_oneway_spectrum_band,
)
from pyfield.reception import ReceptionSDI
from pyfield.transducers import LinearArrayTransducer


@njit(inline="always")
def _accum_oneway(acc, px, py, pz, c, wx, wy, tg, ap, dl, lo, hi, inv_c, t0, omega, dt):
    """Add patches [lo:hi] of one aperture's factored-sin one-way spectrum into acc[:]."""
    Nb = omega.shape[0]
    w0 = omega[0]
    dw = omega[1] - omega[0] if Nb > 1 else 0.0
    for m in range(lo, hi):
        t1, t2, t3, t4, slope = _patch_corner_times(
            px, py, pz, c[m, 0], c[m, 1], c[m, 2],
            tg[m, 0], tg[m, 1], tg[m, 2], tg[m, 3], tg[m, 4], tg[m, 5],
            wx[m], wy[m], inv_c, ap[m], dl[m], dt,
        )
        if slope == 0.0:
            continue
        half1 = np.float32(0.5) * (t2 - t1)
        half2 = np.float32(0.5) * (t3 - t1)
        tc = np.float32(0.5) * (t1 + t4) - t0
        amp = -4.0 * slope
        pc = _phasor(w0 * tc); spc = _phasor(dw * tc)
        q1 = _phasor(w0 * half1); sq1 = _phasor(dw * half1)
        q2 = _phasor(w0 * half2); sq2 = _phasor(dw * half2)
        for k in range(Nb):
            acc[k] += (amp * q1.imag * q2.imag) * pc
            pc *= spc; q1 *= sq1; q2 *= sq2


@njit(parallel=True, fastmath=True, cache=True)
def _fused_multi(
    points, amps,
    txc, txwx, txwy, txtg, txap, txdl, tx_t0,
    rxc, rxwx, rxwy, rxtg, rxap, rxdl, rx_t0, rx_ptr,
    inv_c, omega, dt,
):
    """Σ_p a_p · Σ_TX(p) · Σ_RX_e(p) → (n_out, N_band), one launch.

    prange over scatterer chunks; Σ_TX built once per scatterer and reused across all RX
    elements (CSR ranges rx_ptr[e]:rx_ptr[e+1]). No (P, N_band) intermediate.
    """
    P = points.shape[0]
    Nb = omega.shape[0]
    n_out = rx_ptr.shape[0] - 1
    nthr = get_num_threads()
    buf = np.zeros((nthr, n_out, Nb), dtype=np.complex128)
    for ci in prange(nthr):  # ty: ignore[not-iterable]
        lo = ci * P // nthr
        hi = (ci + 1) * P // nthr
        tx_acc = np.zeros(Nb, dtype=np.complex128)
        rx_acc = np.zeros(Nb, dtype=np.complex128)
        for p in range(lo, hi):
            px, py, pz = points[p, 0], points[p, 1], points[p, 2]
            tx_acc[:] = 0.0
            _accum_oneway(tx_acc, px, py, pz, txc, txwx, txwy, txtg, txap, txdl,
                          0, txc.shape[0], inv_c, tx_t0, omega, dt)
            a = amps[p]
            for e in range(n_out):
                rx_acc[:] = 0.0
                _accum_oneway(rx_acc, px, py, pz, rxc, rxwx, rxwy, rxtg, rxap, rxdl,
                              rx_ptr[e], rx_ptr[e + 1], inv_c, rx_t0, omega, dt)
                for k in range(Nb):
                    buf[ci, e, k] += a * tx_acc[k] * rx_acc[k]
    out = np.zeros((n_out, Nb), dtype=np.complex128)
    for ci in range(nthr):
        out += buf[ci]
    return out


# --- setup ---
fs, fc, c = 100e6, 5e6, 1540.0
t = np.arange(0, 4 / fc, 1 / fs)
exc = (np.hanning(t.size) * np.sin(2 * np.pi * fc * t)).astype(np.float32)
tx = LinearArrayTransducer(
    n_elements=64, element_width_mm=0.25, element_height_mm=8.0,
    kerf_mm=0.05, no_sub_x=4, no_sub_y=6, frequency_Hz=fc,
)
sim = ReceptionSDI(tx, tx, fs=fs, excitation=exc, method="spectral", verbose=False)
inv_c = np.float32(1.0 / c)

# TX patch arrays (all patches).
txc, txwx, txwy = sim._tx_centers, sim._tx_wx, sim._tx_wy
txap, txdl = sim._tx_apod, sim._tx_delays
txtg = pack_tangents(sim._tx_eu, sim._tx_ev)

# RX patches sorted by element → contiguous CSR ranges rx_ptr.
groups = sim._extract_rx_element_patches()
rxc = np.concatenate([g[0] for g in groups]).astype(np.float32)
rxwx = np.concatenate([g[1] for g in groups]).astype(np.float32)
rxwy = np.concatenate([g[2] for g in groups]).astype(np.float32)
rxap = np.concatenate([g[3] for g in groups]).astype(np.float32)
rxdl = np.concatenate([g[4] for g in groups]).astype(np.float32)
rxeu = np.concatenate([g[5] for g in groups]).astype(np.float32)
rxev = np.concatenate([g[6] for g in groups]).astype(np.float32)
rxtg = pack_tangents(rxeu, rxev)
rx_ptr = np.concatenate([[0], np.cumsum([g[0].shape[0] for g in groups])]).astype(np.int64)
n_out = len(groups)

rng = np.random.default_rng(0)


def per_bin_iter(points_m):
    """Yield (pts_b, amps_b, omega_band, tx_t0, rx_t0) for each real depth bin."""
    n_bins = sim._auto_depth_bins(points_m, max(n_out, 2))
    center = np.asarray(sim._tx_centers, np.float64).mean(axis=0)
    order = np.argsort(np.linalg.norm(points_m - center, axis=1))
    t0_g, dt, *_ = sim._compute_pe_time_grid(points_m)
    for idx in np.array_split(order, n_bins):
        if idx.size == 0:
            continue
        pts = points_m[idx]
        pe_t0_nat, _dt, pe_T_nat, tx_t0_b, _a, rx_t0_b, _b = sim._compute_pe_time_grid(pts)
        n0 = int(np.floor((pe_t0_nat - t0_g) / dt))
        shift = pe_t0_nat - (t0_g + n0 * dt)
        s = sim._pe_setup(
            pts, n_integrations=4, per_scatterer=False, focused_sum=False,
            label="x", grid_override=(t0_g + n0 * dt, dt, pe_T_nat + 1,
                                      tx_t0_b - shift, rx_t0_b),
        )
        yield idx, s["omega_band"], s["tx_t0"], s["rx_t0"], dt, n_bins


def current(points_m, amps):
    out = []
    for idx, omega, tx_t0, rx_t0, dt, _ in per_bin_iter(points_m):
        pts, am = points_m[idx].astype(np.float32), amps[idx].astype(np.float64)
        h_tx = compute_oneway_spectrum_band(pts, txc, txwx, txwy, txap, txdl, inv_c,
                                            tx_t0, omega, dt, eu=sim._tx_eu, ev=sim._tx_ev)
        s_all = np.zeros((n_out, omega.size), np.complex128)
        for e, g in enumerate(groups):
            h_rx = compute_oneway_spectrum_band(pts, g[0], g[1], g[2], g[3], g[4], inv_c,
                                                rx_t0, omega, dt, eu=g[5], ev=g[6])
            s_all[e] = am @ (h_tx * h_rx)
        out.append(s_all)
    return out


def fused(points_m, amps):
    out = []
    for idx, omega, tx_t0, rx_t0, dt, _ in per_bin_iter(points_m):
        pts, am = points_m[idx].astype(np.float32), amps[idx].astype(np.float64)
        out.append(_fused_multi(
            pts, am, txc, txwx, txwy, txtg, txap, txdl, tx_t0,
            rxc, rxwx, rxwy, rxtg, rxap, rxdl, rx_t0, rx_ptr, inv_c, omega, dt))
    return out


# Warm up JIT.
p0 = (np.column_stack([np.zeros(200), np.zeros(200),
                       np.linspace(20, 60, 200)]) * 1e-3)
a0 = np.ones(200, np.float32)
current(p0, a0); fused(p0, a0)

print(f"M={txc.shape[0]}, n_out={n_out}, threads={get_num_threads()}\n")
print(f"{'P':>7} {'bins':>5} {'cur[s]':>8} {'fused[s]':>9} {'speedup':>8} {'relerr':>9}")
for P, (zlo, zhi) in [(6000, (20, 90)), (20000, (20, 130)), (60000, (20, 150))]:
    pts = (np.column_stack([rng.uniform(-6, 6, P), np.zeros(P),
                            rng.uniform(zlo, zhi, P)]) * 1e-3).astype(np.float64)
    amp = rng.standard_normal(P).astype(np.float32)
    nb = sim._auto_depth_bins(pts, max(n_out, 2))

    t0 = time.perf_counter(); rc = current(pts, amp); tc = time.perf_counter() - t0
    t0 = time.perf_counter(); rf = fused(pts, amp); tf = time.perf_counter() - t0
    # accuracy on first bin
    relerr = np.max(np.abs(rc[0] - rf[0])) / max(np.max(np.abs(rc[0])), 1e-30)
    print(f"{P:>7} {nb:>5} {tc:>8.3f} {tf:>9.3f} {tc/tf:>8.2f} {relerr:>9.2e}")
