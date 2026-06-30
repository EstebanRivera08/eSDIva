"""Amplitude ratio Reception(FST) / ReceptionSDI vs fs and fc.

The two engines compute h_tx (conv) h_rx differently:
  FST : FFT(h_tx)*FFT(h_rx)  -> discrete convolution (each FFT product is a
          discrete conv, missing a dt vs the continuous integral)
  SDI   : delta placement (continuous conv of the d2h delta trains) + cumsum
          (discrete integral, also a hidden dt)
A dt-power mismatch would show as ratio ~ fs^k. A constant offset shows as flat
ratio. Frequency dependence shows as ratio tracking fc. Sweep both to pinpoint.
"""

import numpy as np

from pyfield.reception import Reception, ReceptionSDI
from pyfield.transducers import FlatCircularTransducer

C = 1540.0
ZS = 20.0  # scatterer depth (mm)
pts = np.array([[0.0, 0.0, ZS]], dtype=np.float32)


def mk(fc, fs, exc=False):
    tx = FlatCircularTransducer(diameter_mm=10.0, no_sub_diameter=20,
                                refine_factor=2, frequency_Hz=fc)
    t = np.arange(0, 2 / fc, 1 / fs)
    tx.impulse_response = np.sin(2 * np.pi * fc * t) * np.hanning(len(t))
    if exc:
        tx.excitation = np.sin(2 * np.pi * fc * t)
    return tx


def peaks(fc, fs):
    FST = Reception(mk(fc, fs, True), mk(fc, fs), fs=fs, c=C, method="FST", verbose=False)
    sdi = ReceptionSDI(mk(fc, fs, True), mk(fc, fs), fs=fs, c=C, verbose=False)
    out = {}
    for label, meth in [("scat", "scattered_rf"), ("hhp", "pulse_echo_response")]:
        rn, _ = getattr(FST, meth)(pts, per_scatterer=True)
        rs, _ = getattr(sdi, meth)(pts, per_scatterer=True)
        out[label] = (np.abs(rn).max(), np.abs(rs).max())
    return out


print(f"{'fc(MHz)':>7} {'fs(MHz)':>7} | {'scat n/sdi':>12} {'/fs':>7} | "
      f"{'hhp n/sdi':>11} {'/fs':>7}")
print("-" * 64)
for fc in [2e6, 4e6]:
    for fs in [50e6, 100e6, 200e6]:
        p = peaks(fc, fs)
        rs = p["scat"][0] / (p["scat"][1] + 1e-30)
        rh = p["hhp"][0] / (p["hhp"][1] + 1e-30)
        print(f"{fc/1e6:7.1f} {fs/1e6:7.0f} | {rs:12.1f} {rs/fs:7.4f} | "
              f"{rh:11.1f} {rh/fs:7.4f}")

# After the dt fix in Reception: amplitudes agree directly (ratio -> 1). They are
# NOT bit-identical (different operators: sampled trapezoid + (jw)^n vs quantised
# delta train + cumsum), so a few-% residual / corr ~0.95 is expected, not a bug.
print("\n=== FST vs SDI, post-fix (scattered_rf, fc=3MHz fs=100MHz) ===")
fc, fs = 3e6, 100e6
FST = Reception(mk(fc, fs, True), mk(fc, fs), fs=fs, c=C, method="FST", verbose=False)
sdi = ReceptionSDI(mk(fc, fs, True), mk(fc, fs), fs=fs, c=C, verbose=False)
rn, _ = FST.scattered_rf(pts, per_scatterer=True)
rs, _ = sdi.scattered_rf(pts, per_scatterer=True)
a = rn[0, :, 0].astype(float)
b = rs[0, :, 0].astype(float)
n = min(len(a), len(b))
corr = np.corrcoef(a[:n], b[:n])[0, 1]
print(f"  peak(FST)/peak(SDI) = {np.abs(a).max() / np.abs(b).max():.5f}")
print(f"  waveform correlation  = {corr:.5f}")
