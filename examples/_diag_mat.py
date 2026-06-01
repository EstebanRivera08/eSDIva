import numpy as np, scipy.io
from pathlib import Path
m=scipy.io.loadmat("examples/rf_concave_psf.mat", simplify_cells=True)["save_struct"]
print("keys:", list(m.keys()))
env=m["rf_env_dB"]; t0=float(m["t0"])
FS=100e6; dt=5/FS
t_us=(t0+np.arange(env.shape[0])*dt)*1e6
print("env shape", env.shape, "t0", t0*1e6, "t_end", t_us[-1])
ci=env.shape[1]//2
col=env[:,ci]
print("on-axis col peak idx", int(np.argmax(col)), "-> t", t_us[int(np.argmax(col))], "us  val", col.max())
# show where envelope > -6 dB on axis
strong=np.where(col>-6)[0]
print("on-axis >-6dB samples at t_us:", t_us[strong])
print(".m display window: 38.41 - 39.6 us")
# print first 40 samples on axis
for i in range(0,40):
    print(f"  t={t_us[i]:.3f}us  dB={col[i]:.1f}")
