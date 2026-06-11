"""Isolate SIR-shape vs signal-chain as cause of PE smoothing.

Builds the concave PE response three ways, all on the SAME time axis and the
SAME signal chain (exc * ir * ir * (jw)^k), and compares to Field II:

  (T) trapezoid SIR  -> exactly what PyField Reception does (via compute_h_sir)
  (G) ground-truth numerical bowl SIR (Rayleigh histogram over true surface)

If (G) matches Field II much better than (T) -> trapezoid SIR is the culprit.
If (G) ~ (T) ~ 0.93 -> the signal chain (derivatives/IR) is the culprit.

Also sweeps the derivative power k in (jw)^k.
"""

import numpy as np
import scipy.io
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import hilbert

from pyfield.hsir.farfield_rect_patch import compute_h_sir
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
)

F0, FS, C = 3e6, 100e6, 1540.0
R_CURV = 80e-3
R_AP = 8e-3
ZS = 30e-3
X_MM = np.linspace(-10, 10, 101)

# ---- Field II reference ----
m = scipy.io.loadmat(
    "examples/rf_concave_psf.mat", simplify_cells=True
)["save_struct"]
fii_db = m["rf_env_dB"]  # (120,101)
fii_t0 = float(m["t0"])
fii_dt = 5.0 / FS
fii_t = fii_t0 + np.arange(fii_db.shape[0]) * fii_dt
fii_lin = 10 ** (fii_db / 20)


def mk_tx():
    return ConcaveCircularTransducer(
        diameter_mm=16.0, focus_mm=80.0, frequency_Hz=F0,
        refine_factor=1, no_sub_diameter=16,
    )


t_ir = np.arange(0, 2.0 / F0, 1.0 / FS)
EXC = np.sin(2 * np.pi * F0 * t_ir).astype(np.float64)
IR = (np.sin(2 * np.pi * F0 * t_ir) * np.hanning(len(t_ir))).astype(np.float64)

# ---- common field points ----
pts = np.column_stack(
    [X_MM * 1e-3, np.zeros(101), np.full(101, ZS)]
).astype(np.float32)
P = 101


def trapezoid_htx():
    """PyField trapezoid h_tx (P, T) for the concave, plus (t0, dt, T)."""
    tx = mk_tx()
    c_, a_, d_, M, _, wx, wy, idx = compute_sub_elem_attributes(tx)
    fr = tx.sub_patch_frames
    eu = np.asarray(fr["tangents_u"], np.float32)
    ev = np.asarray(fr["tangents_v"], np.float32)
    tg, t0, dt, T = compute_time_grid(
        P, M, pts, c_, float(wx.max()), float(wy.max()), C, FS,
        tx.delays, verbose=False,
    )
    h, _ = compute_h_sir(
        P, M, T, dt, tg, pts, c_, wx, wy, np.float32(1 / C), FS,
        a_, d_, 0, eu, ev,
    )
    return h, t0, dt, T


def bowl_points(n_r=400, n_th=400):
    """Fine point-source sampling of the true concave bowl surface (apex z=0)."""
    r = (np.arange(n_r) + 0.5) / n_r * R_AP
    th = (np.arange(n_th)) / n_th * 2 * np.pi
    R, TH = np.meshgrid(r, th)
    X = (R * np.cos(TH)).ravel()
    Y = (R * np.sin(TH)).ravel()
    Z = (R_CURV - np.sqrt(R_CURV**2 - R.ravel() ** 2))
    # area element of annulus ring: r dr dth
    dr = R_AP / n_r
    dth = 2 * np.pi / n_th
    dA = R.ravel() * dr * dth
    return np.column_stack([X, Y, Z]), dA


def ground_truth_htx(t0, dt, T):
    """Numerical Rayleigh h_tx (P, T) over the true bowl surface."""
    surf, dA = bowl_points()
    h = np.zeros((P, T), dtype=np.float64)
    for p in range(P):
        d = surf - pts[p]
        Rr = np.sqrt((d * d).sum(axis=1))
        tarr = Rr / C
        w = dA / (2 * np.pi * Rr)
        k = np.floor((tarr - t0) / dt).astype(int)
        valid = (k >= 0) & (k < T)
        np.add.at(h[p], k[valid], w[valid])
    h /= dt
    return h.astype(np.float32)


def chain(h_tx, t0, dt, T, k_deriv=3):
    """Full PE chain: h_pe = h_tx*h_tx, then *exc*ir*ir*(jw)^k. Returns (P,nf_t) env."""
    pe_T = 2 * T - 1
    L = len(EXC)
    nfft = 1 << int((pe_T + 3 * L - 1)).bit_length()
    freqs = rfftfreq(nfft, d=1.0 / FS)
    jw = 1j * 2 * np.pi * freqs
    H = rfft(h_tx.astype(np.float64), n=nfft, axis=1)
    H_pe = H * H  # h_tx conv h_rx (same aperture)
    Vp = rfft(EXC, nfft) * (jw**k_deriv)
    chain_f = Vp * rfft(IR, nfft) * rfft(IR, nfft)
    rf = irfft(H_pe * chain_f[None, :], n=nfft, axis=1)[:, :pe_T]
    pe_t0 = 2 * t0
    t = pe_t0 + np.arange(pe_T) * dt
    env = np.abs(hilbert(rf, axis=1))
    return env, t


def corr2d(env, t, label):
    """Resample env onto Field II grid, normalise, report 2D corr (full + >-30dB)."""
    e = np.zeros_like(fii_lin)
    for i in range(101):
        e[:, i] = np.interp(fii_t, t, env[i] / env.max(), left=0, right=0)
    a = e - e.mean()
    b = fii_lin - fii_lin.mean()
    full = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    mask = (fii_db > -30)
    am = e[mask] - e[mask].mean()
    bm = fii_lin[mask] - fii_lin[mask].mean()
    above = float((am * bm).sum() / np.sqrt((am * am).sum() * (bm * bm).sum()))
    pk = t[np.argmax(env[50])]
    print(f"  {label:24s} corr_full={full:.4f}  corr>-30dB={above:.4f}  "
          f"on-axis peak={pk*1e6:.3f}us")
    return full


if __name__ == "__main__":
    print("Building trapezoid h_tx ...")
    h_trap, t0, dt, T = trapezoid_htx()
    print("Building ground-truth bowl h_tx ...")
    h_gt = ground_truth_htx(t0, dt, T)

    print(f"\nField II on-axis peak = {fii_t[np.argmax(fii_lin[:,50])]*1e6:.3f}us\n")

    print("=== (jw)^3 chain ===")
    for h, lab in ((h_trap, "TRAPEZOID (PyField)"), (h_gt, "GROUND-TRUTH bowl")):
        env, t = chain(h, t0, dt, T, k_deriv=3)
        corr2d(env, t, lab)

    print("\n=== derivative power sweep (trapezoid SIR) ===")
    for k in (1, 2, 3, 4):
        env, t = chain(h_trap, t0, dt, T, k_deriv=k)
        corr2d(env, t, f"trap (jw)^{k}")

    print("\n=== derivative power sweep (ground-truth SIR) ===")
    for k in (1, 2, 3, 4):
        env, t = chain(h_gt, t0, dt, T, k_deriv=k)
        corr2d(env, t, f"gt (jw)^{k}")
