"""Validate depth-binned spectral reception: accuracy + speed.

  1. Binned spectral vs single-window spectral (n_depth_bins=1) at modest P — must match
     (binning is an exact reorganisation: each bin adds back at an integer sample offset).
  2. complex64 vs complex128 scatterer-sum accumulation accuracy in the binned path.
  3. Wall-clock at higher P: binned spectral (c128, c64) vs conventional (Field II-style).

Single-window spectral is O(N_band ∝ depth-span) and slow, so it is the ground truth
only at the small-P accuracy row; speed rows skip it.
"""

import time

import numpy as np

from pyfield.reception import ReceptionSDI
from pyfield.transducers import LinearArrayTransducer

rng = np.random.default_rng(0)
fs = 100e6
fc = 5e6
t = np.arange(0, 4 / fc, 1 / fs)
exc = (np.hanning(t.size) * np.sin(2 * np.pi * fc * t)).astype(np.float32)

tx = LinearArrayTransducer(
    n_elements=64, element_width_mm=0.25, element_height_mm=8.0,
    kerf_mm=0.05, no_sub_x=4, no_sub_y=6, frequency_Hz=fc,
)
M = 64 * 4 * 6


def corr(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return np.corrcoef(a[..., :n].ravel(), b[..., :n].ravel())[0, 1]


def rel_err(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return np.max(np.abs(a[..., :n] - b[..., :n])) / max(np.max(np.abs(b[..., :n])), 1e-30)


def run(method, pts, amp, *, accum=None, **kw):
    sim = ReceptionSDI(tx, tx, fs=fs, excitation=exc, method=method, verbose=False, **kw)
    if accum is not None:
        sim.spectral_accum_dtype = accum
    rf, _ = sim(pts, amp)  # warm up (numba JIT)
    t0 = time.perf_counter()
    rf, _ = sim(pts, amp)
    return time.perf_counter() - t0, rf


def field(P, zlo, zhi):
    pts = np.column_stack([
        rng.uniform(-6, 6, P), np.zeros(P), rng.uniform(zlo, zhi, P),
    ]).astype(np.float64)
    return pts, rng.standard_normal(P).astype(np.float32)


print(f"M={M}, fs={fs/1e6:.0f} MHz, fc={fc/1e6:.0f} MHz\n")

# --- 1+2. Accuracy at modest P (single-window ground truth) ---
pts, amp = field(1500, 20, 60)
_, rf1 = run("spectral", pts, amp, n_depth_bins=1)          # single window (truth)
_, rfB = run("spectral", pts, amp)                          # binned, c128
_, rfB64 = run("spectral", pts, amp, accum=np.complex64)    # binned, c64
nb = ReceptionSDI(tx, tx, fs=fs, excitation=exc, method="spectral", verbose=False)
n_bins = nb._auto_depth_bins(pts * 1e-3, max(int(tx.delays.shape[0]), 2))
print(f"accuracy (P=1500, {n_bins} bins):")
print(f"  corr(binned, single)      = {corr(rfB, rf1):.6f}")
print(f"  relerr(binned, single)    = {rel_err(rfB, rf1):.2e}")
print(f"  relerr(c64-bin, c128-bin) = {rel_err(rfB64, rfB):.2e}\n")

# --- 3. Speed at higher P ---
print(f"{'P':>7} {'z[mm]':>9} {'bins':>5} {'Nt':>6} "
      f"{'specB[s]':>9} {'specB64[s]':>10} {'conv[s]':>9} {'B/conv':>7} {'B64/conv':>9}")
for P, (zlo, zhi) in [(6000, (20, 90)), (20000, (20, 130))]:
    pts, amp = field(P, zlo, zhi)
    nbins = nb._auto_depth_bins(pts * 1e-3, max(int(tx.delays.shape[0]), 2))
    t_b, rf_b = run("spectral", pts, amp)
    t_b64, _ = run("spectral", pts, amp, accum=np.complex64)
    t_c, _ = run("conventional", pts, amp)
    print(f"{P:>7} {f'{zlo}-{zhi}':>9} {nbins:>5} {rf_b.shape[-1]:>6} "
          f"{t_b:>9.3f} {t_b64:>10.3f} {t_c:>9.3f} {t_b/t_c:>7.2f} {t_b64/t_c:>9.2f}")
