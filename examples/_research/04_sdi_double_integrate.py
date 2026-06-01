"""Test: can ReceptionSDI yield calc_hhp by double-integrating Dh_pe?

SDI bakes 3 derivatives into Dh_pe. Integrating twice removes 2 → 1-derivative
calc_hhp (pulse_echo_response). Integration commutes with the exc/ir LTI
convolutions, so we test by double-integrating the scattered_rf output and
comparing to the conventional Reception.pulse_echo_response (exact 1-deriv).

Two integration variants:
  (cumsum) two time-domain cumulative sums * dt^2
  (freq)   divide spectrum by (jω)^2, zero the f=0 bin

Reports correlation vs the exact conventional result and checks for drift.
"""

import numpy as np
import scipy.io
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import hilbert

from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers import ConcaveCircularTransducer

F0, FS, C = 3e6, 100e6, 1540.0
pts = np.array([[0, 0, 30.0], [3, 0, 30.0]], dtype=np.float32)


def mk(exc=False):
    tx = ConcaveCircularTransducer(diameter_mm=16.0, focus_mm=80.0,
                                   frequency_Hz=F0, refine_factor=1,
                                   no_sub_diameter=16)
    t = np.arange(0, 2 / F0, 1 / FS)
    tx.impulse_response = np.sin(2 * np.pi * F0 * t) * np.hanning(len(t))
    if exc:
        tx.excitation = np.sin(2 * np.pi * F0 * t)
    return tx


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30))


# Exact reference: conventional Reception, 1 derivative.
conv = Reception(mk(True), mk(), fs=FS, c=C, method="sdi", verbose=False)
pe_exact, co_e = conv.pulse_echo_response(pts, per_scatterer=True)

# SDI scattered_rf (3 derivatives).
sdi = ReceptionSDI(mk(True), mk(), fs=FS, c=C, verbose=False)
scat, co_s = sdi.scattered_rf(pts, per_scatterer=True)
dt = co_s["dt"]

print(f"dt={dt*1e6:.4f}us  pe_exact t0={co_e['t0']*1e6:.3f}  scat t0={co_s['t0']*1e6:.3f}")
print(f"shapes: pe_exact {pe_exact.shape}  scat {scat.shape}\n")

for p in range(pts.shape[0]):
    ex = pe_exact[p, :, 0].astype(np.float64)

    # variant 1: two time-domain cumsums
    s = scat[p, :, 0].astype(np.float64)
    i1 = np.cumsum(s) * dt
    i2_cumsum = np.cumsum(i1) * dt

    # variant 2: freq-domain /(jω)^2, f=0 zeroed
    n = len(s)
    nf = 1 << int(n - 1).bit_length()
    fq = rfftfreq(nf, d=1.0 / FS)
    jw2 = (1j * 2 * np.pi * fq) ** 2
    H = rfft(s, nf)
    Hd = np.zeros_like(H)
    Hd[1:] = H[1:] / jw2[1:]
    i2_freq = irfft(Hd, nf)[:n]

    # align lengths/peaks via envelope correlation (shapes differ by deriv const)
    L = min(len(ex), len(i2_cumsum))
    e_ex = np.abs(hilbert(ex))[:L]
    e_cs = np.abs(hilbert(i2_cumsum))[:L]
    e_fq = np.abs(hilbert(i2_freq))[:L]

    # drift metric: ratio of late-tail energy to peak (parabolic blowup detector)
    drift_cs = np.abs(i2_cumsum[-50:]).max() / (np.abs(i2_cumsum).max() + 1e-30)
    drift_fq = np.abs(i2_freq[-50:]).max() / (np.abs(i2_freq).max() + 1e-30)

    print(f"point {p} (x={pts[p,0]:.0f}mm):")
    print(f"  corr(env)  cumsum∫∫ vs exact = {corr(e_cs, e_ex):.4f}   "
          f"freq/(jω)² vs exact = {corr(e_fq, e_ex):.4f}")
    print(f"  raw corr   cumsum∫∫ vs exact = {corr(i2_cumsum[:L], ex[:L]):.4f}   "
          f"freq/(jω)² vs exact = {corr(i2_freq[:L], ex[:L]):.4f}")
    print(f"  tail/peak drift   cumsum = {drift_cs:.3f}   freq = {drift_fq:.3f}\n")
