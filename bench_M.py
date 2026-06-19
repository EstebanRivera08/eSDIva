"""Quick benchmark: spectral vs conventional reception cost as patch count M grows.

Sweeps the aperture (n_elements) → patch count M = n_elements*no_sub_x*no_sub_y,
times pulse_echo_rf for method="spectral" and method="conventional" at fixed
scatterer count P, and reports the ratio. Tests the heuristic that spectral loses
its forward-FFT saving as M grows (band DFT 4*M*N_b overtakes the saved transform).
"""

import time

import numpy as np

from pyfield.reception import ReceptionSDI
from pyfield.transducers import LinearArrayTransducer

rng = np.random.default_rng(0)
fs = 100e6
fc = 5e6
# Narrowband tone burst (small N_b) vs near-delta wideband (N_b -> N_fft).
t = np.arange(0, 4 / fc, 1 / fs)
exc_narrow = (np.hanning(t.size) * np.sin(2 * np.pi * fc * t)).astype(np.float32)
exc_wide = np.zeros(8, dtype=np.float32)
exc_wide[3] = 1.0  # near-delta

P = 200
pts = np.column_stack([
    rng.uniform(-5, 5, P),
    np.zeros(P),
    rng.uniform(20, 40, P),
]).astype(np.float64)
amp = rng.standard_normal(P).astype(np.float32)

NSX, NSY = 4, 6


def run(tx, exc, method):
    sim = ReceptionSDI(tx, tx, fs=fs, excitation=exc, method=method, verbose=False)
    sim(pts, amp)  # warm up (numba JIT)
    t0 = time.perf_counter()
    rf, _ = sim(pts, amp)
    return time.perf_counter() - t0, rf


for label, exc in [("narrowband (small N_b)", exc_narrow), ("wideband (N_b->N_fft)", exc_wide)]:
    print(f"\n== {label} ==")
    print(f"{'n_el':>5} {'M':>7} {'conv[s]':>9} {'spec[s]':>9} {'spec/conv':>10} {'corr':>7}")
    for n_el in [16, 32, 64, 128, 256]:
        tx = LinearArrayTransducer(
            n_elements=n_el, element_width_mm=0.25, element_height_mm=8.0,
            kerf_mm=0.05, no_sub_x=NSX, no_sub_y=NSY, frequency_Hz=fc,
        )
        M = n_el * NSX * NSY
        tc, rf_c = run(tx, exc, "conventional")
        ts, rf_s = run(tx, exc, "spectral")
        corr = np.corrcoef(rf_c.ravel(), rf_s.ravel())[0, 1]
        print(f"{n_el:>5} {M:>7} {tc:>9.3f} {ts:>9.3f} {ts / tc:>10.2f} {corr:>7.4f}")
