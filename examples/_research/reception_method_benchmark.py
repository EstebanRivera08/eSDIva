"""Benchmark the pulse-echo reception methods: speed + correctness across regimes.

`ReceptionSDI` evaluates the same pulse-echo RF equation `p_pe = v_pe ⊛ h_tx ⊛ h_rx`
three ways, which trade cost by regime:

* ``conventional`` — sample both one-way SIRs and FFT-convolve them. Patch-independent
  transform cost; the depth-binned reference.
* ``paired``       — enumerate the 16 corner events of every TX–RX patch pair and splat
  the integrated drive ``w = I⁴ v_pe`` at each (no FFT, no cumsum). Cost ∝ patch count
  squared and carries the kernel length per event, so it is the exact reference path —
  only cheap for a near-monoelement aperture.
* ``spectral``     — build each one-way SIR spectrum in closed form from the corner times
  and multiply them (no forward FFT, cost linear in patch count). Every receive element's
  spectrum is built in one batched kernel call. Exact; the fast default for compact and
  large apertures, and supports per-patch attenuation.

This script times each method on three regimes and reports, alongside the wall time, the
correlation against ``conventional`` (so speed is never read without confirming the
physics is unchanged). Run with ``uv run examples/_research/reception_method_benchmark.py``.
"""

import time

import numpy as np

from pyfield.reception import ReceptionSDI
from pyfield.transducers import LinearArrayTransducer

FS, FC, C = 100e6, 5e6, 1540.0
# A 2-cycle Hanning-windowed tone: a band-limited drive (so spectral is applicable).
_t = np.arange(0, 2.0 / FC, 1.0 / FS)
EXC = (np.sin(2 * np.pi * FC * _t) * np.hanning(len(_t))).astype(np.float32)

ALL_METHODS = ["conventional", "spectral", "paired"]
# ``paired`` splats the full kernel per patch pair, so it is impractical past tiny
# apertures; the large-array regime skips it.
FAST_METHODS = ["conventional", "spectral"]


def _array(n_elements, no_sub_x, no_sub_y):
    return LinearArrayTransducer(
        n_elements=n_elements,
        element_width_mm=0.25,
        element_height_mm=10.0,
        kerf_mm=0.05,
        no_sub_x=no_sub_x,
        no_sub_y=no_sub_y,
        frequency_Hz=FC,
    )


def _corr(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return float(np.corrcoef(a[..., :n].ravel(), b[..., :n].ravel())[0, 1])


def _time_methods(tx, rx, pos, amp, *, per_scatterer=False, methods=ALL_METHODS):
    """Time every method on one scenario; return {method: (seconds, corr_vs_conv)}."""
    results, ref = {}, None
    for method in methods:
        sim = ReceptionSDI(tx, rx, fs=FS, c=C, excitation=EXC, method=method, verbose=False)
        # Warm the Numba kernels before timing. Use the SAME scatterer count as the timed
        # call: a single point hits a different (patch-parallel) kernel than many points,
        # so warming with one point would leave the many-point kernel to compile mid-timing.
        warm = pos if per_scatterer else pos[: min(2, len(pos))]
        sim.pulse_echo_rf(warm, amp[: len(warm)], per_scatterer=per_scatterer)
        t0 = time.perf_counter()
        rf, _ = sim.pulse_echo_rf(pos, amp, per_scatterer=per_scatterer)
        dt = time.perf_counter() - t0
        if method == "conventional":
            ref = rf
        results[method] = (dt, rf)
    return {k: (dt, _corr(ref, rf)) for k, (dt, rf) in results.items()}


def _print_table(title, results):
    print(f"\n{title}")
    print(f"  {'method':16s} {'time [s]':>10s} {'corr vs conv':>14s}")
    fastest = min(results.values(), key=lambda v: v[0])[0]
    for label, (dt, corr) in results.items():
        mark = "  <- fastest" if dt == fastest else ""
        print(f"  {label:16s} {dt:10.3f} {corr:14.5f}{mark}")


def main():
    rng = np.random.RandomState(0)

    # 1. Point-spread function: one scatterer, per_scatterer output. spectral's home.
    tx = rx = _array(64, no_sub_x=2, no_sub_y=4)
    pos = np.array([[0.0, 0.0, 30.0]], dtype=np.float32)
    amp = np.ones(1, dtype=np.float32)
    _print_table(
        "PSF — 64 elements, 1 scatterer (per_scatterer)",
        _time_methods(tx, rx, pos, amp, per_scatterer=True),
    )

    # 2. Compact aperture, modest scatterer cloud.
    tx = rx = _array(16, no_sub_x=2, no_sub_y=2)
    pos = rng.uniform([-5, -0.5, 15], [5, 0.5, 45], (100, 3)).astype(np.float32)
    amp = np.ones(100, dtype=np.float32)
    _print_table(
        "Compact array — 16 elements, 100 scatterers (summed)",
        _time_methods(tx, rx, pos, amp),
    )

    # 3. Large aperture, scatterer cloud. paired blows up; spectral ≈ conventional.
    tx = rx = _array(128, no_sub_x=3, no_sub_y=6)
    pos = rng.uniform([-8, -0.5, 15], [8, 0.5, 45], (100, 3)).astype(np.float32)
    amp = np.ones(100, dtype=np.float32)
    _print_table(
        "Large array — 128 elements, 100 scatterers (summed)",
        _time_methods(tx, rx, pos, amp, methods=FAST_METHODS),
    )

    print(
        "\nTakeaway: with the batched receive-spectrum kernel, spectral is fastest or tied "
        "across compact and large summed fields and ties conventional on a single-point PSF; "
        "paired is the exact splat reference for near-monoelement apertures. All agree with "
        "conventional to corr ~1.0."
    )


if __name__ == "__main__":
    main()
