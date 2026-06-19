"""Does a longer time record T help the spectral form?

Fix the aperture (M) and the narrowband excitation, grow the record length T by
pushing the scatterers deeper (larger two-way TOF span -> more time samples), and
watch the spectral/conventional wall-clock ratio and the actual Nt.
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
M = 64 * 4 * 6  # 1536

P = 200


def run(method, pts, amp):
    sim = ReceptionSDI(tx, tx, fs=fs, excitation=exc, method=method, verbose=False)
    rf, _ = sim(pts, amp)  # warm up
    t0 = time.perf_counter()
    rf, _ = sim(pts, amp)
    return time.perf_counter() - t0, rf


print(f"M={M}, fs={fs / 1e6:.0f} MHz, fc={fc / 1e6:.0f} MHz, P={P}")
print(f"{'z_range[mm]':>12} {'Nt':>7} {'conv[s]':>9} {'spec[s]':>9} {'spec/conv':>10} {'corr':>7}")
for zlo, zhi in [(20, 40), (20, 80), (20, 140), (20, 220)]:
    pts = np.column_stack([
        rng.uniform(-5, 5, P),
        np.zeros(P),
        rng.uniform(zlo, zhi, P),
    ]).astype(np.float64)
    amp = rng.standard_normal(P).astype(np.float32)
    tc, rf_c = run("conventional", pts, amp)
    ts, rf_s = run("spectral", pts, amp)
    Nt = rf_c.shape[-1]
    corr = np.corrcoef(rf_c.ravel(), rf_s.ravel())[0, 1]
    print(f"{f'{zlo}-{zhi}':>12} {Nt:>7} {tc:>9.3f} {ts:>9.3f} {ts / tc:>10.2f} {corr:>7.4f}")
