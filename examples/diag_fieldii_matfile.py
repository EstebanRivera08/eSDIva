"""
Diagnostic: Inspect Field II mat file and compare absolute timing.
Run with:
    uv run examples/diag_fieldii_matfile.py
"""

from pathlib import Path
import numpy as np
import scipy.io
from scipy.signal import hilbert

MAT_FILE = Path(__file__).parent / "rf_concave_psf.mat"
FS = 100e6
# Mat file stores every-5th-sample envelope (RF_data(1:5:600,:) in Matlab).
# Actual sample spacing is 5/FS = 50 ns, not 1/FS.
DT = 5.0 / FS
C = 1540.0
SCAT_Z_MM = 30.0

print("=" * 60)
print("Field II mat file inspection")
print("=" * 60)
mat = scipy.io.loadmat(str(MAT_FILE), simplify_cells=True)
print("Keys:", list(mat.keys()))
ss = mat["save_struct"]
print("save_struct keys:", list(ss.keys()) if isinstance(ss, dict) else type(ss))

if isinstance(ss, dict):
    for k, v in ss.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.3g}, {v.max():.3g}]")
        else:
            print(f"  {k}: {v}")
else:
    print(ss.dtype)
    for name in ss.dtype.names:
        v = ss[name]
        if hasattr(v, 'shape') and v.ndim > 0:
            print(f"  {name}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.3g}, {v.max():.3g}]")
        else:
            print(f"  {name}: {v}")

print()

# Extract fields
fii_env_db = np.array(ss["rf_env_dB"] if isinstance(ss, dict) else ss["rf_env_dB"]).astype(np.float64)
fii_t0 = float(ss["t0"] if isinstance(ss, dict) else ss["t0"])

print(f"rf_env_dB shape: {fii_env_db.shape}")
print(f"t0 = {fii_t0*1e6:.4f} us")
print(f"Nt = {fii_env_db.shape[0]}, N_lat = {fii_env_db.shape[1]}")

fii_Nt = fii_env_db.shape[0]
fii_t_us = (fii_t0 + np.arange(fii_Nt) * DT) * 1e6
print(f"Time axis: {fii_t_us[0]:.4f} to {fii_t_us[-1]:.4f} us ({fii_Nt} samples)")

# Peak location
peak_t, peak_x = np.unravel_index(np.argmax(fii_env_db), fii_env_db.shape)
print(f"\nPeak: t_idx={peak_t}, x_idx={peak_x}")
print(f"Peak time: {fii_t_us[peak_t]:.4f} us")
print(f"Peak dB: {fii_env_db[peak_t, peak_x]:.2f} dB")

# Center column (on-axis)
center_col = fii_env_db.shape[1] // 2
print(f"\nCenter column (x_idx={center_col}) peak:")
peak_t_center = np.argmax(fii_env_db[:, center_col])
print(f"  t_idx={peak_t_center}, time={fii_t_us[peak_t_center]:.4f} us, dB={fii_env_db[peak_t_center, center_col]:.2f}")

print("\nTime profile at center column:")
for i in range(0, fii_Nt, max(1, fii_Nt//20)):
    print(f"  t={fii_t_us[i]:.3f} us  -> {fii_env_db[i, center_col]:.1f} dB")

print()
# Physics sanity check
expected_rt = 2 * SCAT_Z_MM * 1e-3 / C
print(f"Expected round-trip (flat at z=0): {expected_rt*1e6:.4f} us")
print(f"Expected round-trip (bowl pole at z=-0.4mm): {(expected_rt + 2*0.4e-3/C)*1e6:.4f} us")
print(f"FII t0 vs expected flat RT: {(fii_t0 - expected_rt)*1e9:.0f} ns offset")
print(f"FII peak vs expected flat RT: {(fii_t_us[peak_t_center] - expected_rt*1e6)*1e3:.2f} ns offset")

# Check if mat might store rf_raw (not envelope)
print("\nIs rf_env_dB truly in dB? Max value:", fii_env_db.max())
print("If max=0 and values negative -> yes (log-compressed envelope).")
print("If arbitrary -> might be raw RF or something else.")
