"""
Diagnostic: PE SDI absolute timing, leakage check, and reference comparison.
Run with:
    uv run examples/diag_pe_sdi_timing.py
"""

import warnings

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import hilbert

from pyfield.hsir.farfield_rect_patch import compute_h_sir as _compute_h_sir
from pyfield.hsir.transducer_sir_pe import compute_pe_sdi
from pyfield.reception import ReceptionSDI  # noqa: F401
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.utilities.helper_functions import (
    compute_sub_elem_attributes,
    compute_time_grid,
)

C = 1540.0
FS = 100e6
DT = 1.0 / FS
FC = 3e6
SCAT_Z_MM = 30.0


def _next_pow2(n):
    return 1 << (int(n - 1)).bit_length()


def make_hanning_pulse(cycles=2):
    n = int(cycles / FC * FS)
    t = np.arange(n) / FS
    return (np.sin(2 * np.pi * FC * t) * np.hanning(n)).astype(np.float32)


def _fft_diff_h(h, n_deriv, T, dt):
    """Return n_deriv-th derivative of h (P×T) via FFT."""
    from scipy.fft import irfft, rfft, rfftfreq
    nfft = h.shape[1]
    freqs = rfftfreq(nfft, d=dt)
    jw = 1j * 2.0 * np.pi * freqs
    H = rfft(h.astype(np.float64), n=nfft, axis=1)
    return irfft(H * jw**n_deriv, n=nfft, axis=1)[:, :T]


# ── CASE 1: single flat patch ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("CASE 1: single flat 1x1 mm patch at origin")
print("=" * 60)

WX = WY = 1e-3
scat_m = np.array([[0.0, 0.0, SCAT_Z_MM * 1e-3]], dtype=np.float32)
tx_centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
tx_wx = np.array([WX], dtype=np.float32)
tx_wy = np.array([WY], dtype=np.float32)
tx_apod = np.array([1.0], dtype=np.float32)
tx_delays = np.array([0.0], dtype=np.float32)
tx_delays_elem = np.array([0.0], dtype=np.float32)
inv_c = np.float32(1.0 / C)

_, tx_t0, _, tx_T = compute_time_grid(
    1, 1, scat_m, tx_centers, WX, WY, C, FS, tx_delays_elem, verbose=False
)
pe_t0 = 2 * tx_t0
pe_T = 2 * tx_T - 1

Dh_pe = compute_pe_sdi(
    scat_m,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_apod,
    tx_delays,
    tx_centers,
    tx_wx,
    tx_wy,
    tx_apod,
    tx_delays,
    inv_c,
    pe_t0,
    pe_T,
    FS,
    DT,
)
nz = np.nonzero(Dh_pe[0])[0]
t_first = (pe_t0 + nz[0] * DT) * 1e6
expected_rt = 2 * SCAT_Z_MM * 1e-3 / C
print(f"Dh_pe onset vs expected RT: {(t_first - expected_rt * 1e6) * 1e3:.2f} ns")

ir = make_hanning_pulse(cycles=2)
L = len(ir)
nfft = _next_pow2(pe_T + 3 * L - 3)
fft_dh = rfft(Dh_pe[0], n=nfft)
fft_ir = rfft(ir, n=nfft)
rf3 = irfft(fft_dh * fft_ir * fft_ir * fft_ir, n=nfft)[:pe_T]
env3 = np.abs(hilbert(rf3))
peak3 = np.argmax(env3)
t_peak3 = (pe_t0 + peak3 * DT) * 1e6
expected_triple = expected_rt * 1e6 + 3 * (L - 1) / (2 * FS) * 1e6
print(
    f"RF peak (triple exc): {t_peak3:.4f} us  expected: {expected_triple:.4f} us  delta: {(t_peak3 - expected_triple) * 1e3:.2f} ns"
)

# Leakage check: amplitude before onset
pre_onset_max = np.abs(rf3[: nz[0]]).max() if len(nz) else 0.0
print(
    f"Max RF amplitude before onset: {pre_onset_max:.3e}  (peak={env3.max():.3e},  ratio={pre_onset_max / env3.max():.2e})"
)

# ── CASE 2: reference comparison (PE SDI vs FFT-conv dh_tx*d2h_rx) ─────────
print("\n" + "=" * 60)
print("CASE 2: PE SDI vs dh_tx*d2h_rx reference (flat 4-element array)")
print("=" * 60)

# 4-patch flat array (simulate a multi-patch transducer)
N_pat = 4
xs = np.linspace(-1.5e-3, 1.5e-3, N_pat)
tx_c4 = np.column_stack([xs, np.zeros(N_pat), np.zeros(N_pat)]).astype(np.float32)
tx_wx4 = np.full(N_pat, 1e-3, dtype=np.float32)
tx_wy4 = np.full(N_pat, 1e-3, dtype=np.float32)
tx_ap4 = np.full(N_pat, 1.0, dtype=np.float32)
tx_dl4 = np.zeros(N_pat, dtype=np.float32)
tx_dl_elem4 = np.zeros(N_pat, dtype=np.float32)

time_grid4, tx_t04, _, tx_T4 = compute_time_grid(
    1, N_pat, scat_m, tx_c4, 1e-3, 1e-3, C, FS, tx_dl_elem4, verbose=False
)
rx_t04 = tx_t04
rx_T4 = tx_T4
pe_t04 = tx_t04 + rx_t04
pe_T4 = tx_T4 + rx_T4 - 1

Dh4 = compute_pe_sdi(
    scat_m,
    tx_c4,
    tx_wx4,
    tx_wy4,
    tx_ap4,
    tx_dl4,
    tx_c4,
    tx_wx4,
    tx_wy4,
    tx_ap4,
    tx_dl4,
    inv_c,
    pe_t04,
    pe_T4,
    FS,
    DT,
)

# Reference: dh_tx * d2h_rx via FFT differentiation of h_sir
h4, _ = _compute_h_sir(
    1, N_pat, tx_T4, DT, time_grid4,
    scat_m, tx_c4, tx_wx4, tx_wy4,
    inv_c, FS, tx_ap4, tx_dl4, 1, None, None,
)
dh_tx4 = _fft_diff_h(h4, 1, tx_T4, DT)
d2h_rx4 = _fft_diff_h(h4, 2, rx_T4, DT)
nfft_ref = _next_pow2(tx_T4 + rx_T4 + L - 2)
Dh_ref4 = irfft(
    rfft(dh_tx4.astype(np.float64), n=nfft_ref, axis=1)
    * rfft(d2h_rx4.astype(np.float64), n=nfft_ref, axis=1),
    n=nfft_ref,
    axis=1,
)[0, :pe_T4]

# Convolve both with excitation
nfft4 = _next_pow2(pe_T4 + 3 * L - 3)
rf_pe4 = irfft(rfft(Dh4[0], n=nfft4) * rfft(ir, n=nfft4) ** 3, n=nfft4)[:pe_T4]
rf_ref4 = irfft(
    rfft(Dh_ref4, n=nfft4) * rfft(ir.astype(np.float64), n=nfft4) ** 3, n=nfft4
)[:pe_T4]

env_pe4 = np.abs(hilbert(rf_pe4))
env_ref4 = np.abs(hilbert(rf_ref4))
peak_pe4 = np.argmax(env_pe4)
peak_ref4 = np.argmax(env_ref4)
t_pe4 = (pe_t04 + peak_pe4 * DT) * 1e6
t_ref4 = (pe_t04 + peak_ref4 * DT) * 1e6
print(f"PE SDI peak:  {t_pe4:.4f} us  (idx {peak_pe4})")
print(f"Reference peak: {t_ref4:.4f} us  (idx {peak_ref4})")
print(f"Timing difference: {(t_pe4 - t_ref4) * 1e3:.2f} ns")
peak_ratio = float(env_pe4.max()) / float(env_ref4.max())
print(f"Amplitude ratio PE/Ref: {peak_ratio:.4f}  (should be ~1.0)")

# ── CASE 3: Concave transducer leakage check ───────────────────────────────
print("\n" + "=" * 60)
print("CASE 3: ConcaveCircularTransducer Dh_pe leakage check")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    tx = ConcaveCircularTransducer(
        diameter_mm=16.0,
        focus_mm=80.0,
        frequency_Hz=FC,
        refine_factor=1,
        no_sub_diameter=16,
    )

c_sub, a_sub, d_sub, M, _, wx_sub, wy_sub, _ = compute_sub_elem_attributes(tx)
d_elem = tx.delays
print(f"Patches: {M}")

scat_m_conc = np.array([[0.0, 0.0, SCAT_Z_MM * 1e-3]], dtype=np.float32)
time_grid_c, tx_t0c, _, tx_Tc = compute_time_grid(
    1,
    M,
    scat_m_conc,
    c_sub,
    float(wx_sub.max()),
    float(wy_sub.max()),
    C,
    FS,
    d_elem,
    verbose=False,
)
pe_t0c = 2 * tx_t0c
pe_Tc = 2 * tx_Tc - 1

Dh_c = compute_pe_sdi(
    scat_m_conc,
    c_sub,
    wx_sub,
    wy_sub,
    a_sub,
    d_sub,
    c_sub,
    wx_sub,
    wy_sub,
    a_sub,
    d_sub,
    inv_c,
    pe_t0c,
    pe_Tc,
    FS,
    DT,
)

# Expected min round-trip
scat_pos3d = np.array([0.0, 0.0, SCAT_Z_MM * 1e-3])
dists_c = np.sqrt(((c_sub - scat_pos3d) ** 2).sum(axis=1))
min_dist_c = dists_c.min()
min_rt = 2 * min_dist_c / C
# Expected first event index in Dh_pe
exp_first_idx = int(np.floor((min_rt - pe_t0c) * FS + 1.5))  # approx

nz_c = np.nonzero(Dh_c[0])[0]
actual_first = nz_c[0] if len(nz_c) else -1
pre_min_max = float(np.abs(Dh_c[0, :exp_first_idx]).max()) if exp_first_idx > 0 else 0.0
post_peak = float(np.abs(Dh_c[0]).max())

print(f"pe_t0        = {pe_t0c * 1e6:.4f} us")
print(f"Min round-trip = {min_rt * 1e6:.4f} us")
print(
    f"Expected first Dh_pe event at ~idx {exp_first_idx} (t={pe_t0c * 1e6 + exp_first_idx / FS * 1e6:.4f} us)"
)
print(
    f"Dh_pe first nonzero at idx {actual_first} (t={pe_t0c * 1e6 + actual_first * DT * 1e6:.4f} us)"
)
print(f"Max Dh_pe before expected onset: {pre_min_max:.3e}")
print(f"Max Dh_pe peak:                  {post_peak:.3e}")
print(
    f"Leakage ratio (before/peak):     {pre_min_max / post_peak if post_peak > 0 else 'N/A':.2e}"
)

# Tail check: Dh_pe should return to ~0 after max round-trip
max_dist_c = dists_c.max()
max_rt = 2 * max_dist_c / C
exp_last_idx = int(np.floor((max_rt - pe_t0c) * FS + 2.5))
tail_val = float(Dh_c[0, min(exp_last_idx + 20, pe_Tc - 1)])
print(
    f"Dh_pe tail value at idx {exp_last_idx + 20}: {tail_val:.3e}  (should be ~0 if no DC leak)"
)

# RF signal amplitude check
ir_pulse = make_hanning_pulse(cycles=2)
nfft_c = _next_pow2(pe_Tc + 3 * L - 3)
fft_irc = rfft(ir_pulse, n=nfft_c)
rf_c = irfft(rfft(Dh_c[0], n=nfft_c) * fft_irc**3, n=nfft_c)[:pe_Tc]
env_c = np.abs(hilbert(rf_c))
peak_c_idx = np.argmax(env_c)
t_peak_c = (pe_t0c + peak_c_idx * DT) * 1e6
pre_onset_rf = float(np.abs(rf_c[:exp_first_idx]).max())

print(f"\nRF peak: {t_peak_c:.4f} us  (idx {peak_c_idx})")
print(
    f"Max RF before expected onset: {pre_onset_rf:.3e}  peak: {env_c.max():.3e}  ratio: {pre_onset_rf / env_c.max():.2e}"
)

# ── CASE 4: Comparison of PyField vs reference for CONCAVE ────────────────
print("\n" + "=" * 60)
print("CASE 4: Concave Dh_pe vs dh_tx*d2h_rx reference")
print("=" * 60)

h_c, _ = _compute_h_sir(
    1, M, tx_Tc, DT, time_grid_c,
    scat_m_conc, c_sub, wx_sub, wy_sub,
    inv_c, FS, a_sub, d_sub, 1, None, None,
)
dh_tx_c = _fft_diff_h(h_c, 1, tx_Tc, DT)
d2h_rx_c = _fft_diff_h(h_c, 2, tx_Tc, DT)
nfft_refc = _next_pow2(2 * tx_Tc + L - 2)
Dh_ref_c = irfft(
    rfft(dh_tx_c.astype(np.float64), n=nfft_refc, axis=1)
    * rfft(d2h_rx_c.astype(np.float64), n=nfft_refc, axis=1),
    n=nfft_refc,
    axis=1,
)[0, :pe_Tc]

rf_pe_c = irfft(rfft(Dh_c[0], n=nfft_c) * fft_irc**3, n=nfft_c)[:pe_Tc]
rf_ref_c = irfft(
    rfft(Dh_ref_c, n=nfft_c) * rfft(ir_pulse.astype(np.float64), n=nfft_c) ** 3,
    n=nfft_c,
)[:pe_Tc]

env_pec = np.abs(hilbert(rf_pe_c))
env_refc = np.abs(hilbert(rf_ref_c))
peak_pec = np.argmax(env_pec)
peak_refc = np.argmax(env_refc)
t_pec = (pe_t0c + peak_pec * DT) * 1e6
t_refc = (pe_t0c + peak_refc * DT) * 1e6
print(f"PE SDI peak:    {t_pec:.4f} us  (idx {peak_pec})")
print(f"Reference peak: {t_refc:.4f} us  (idx {peak_refc})")
print(f"Timing delta:   {(t_pec - t_refc) * 1e3:.2f} ns")
amp_ratio = float(env_pec.max()) / float(env_refc.max())
print(f"Amplitude ratio PE/Ref: {amp_ratio:.4f}")
