import numpy as np, scipy.io
from scipy.signal import hilbert
from pyfield.reception import Reception
from pyfield.transducers import ConcaveCircularTransducer
F0,FS,C,ZS=3e6,100e6,1540.0,30.0
def mk(exc=False):
    tx=ConcaveCircularTransducer(diameter_mm=16.0,focus_mm=80.0,frequency_Hz=F0,refine_factor=1,no_sub_diameter=16)
    t=np.arange(0,2.0/F0,1.0/FS)
    tx.impulse_response=np.sin(2*np.pi*F0*t)*np.hanning(len(t))
    if exc: tx.excitation=np.sin(2*np.pi*F0*t)
    return tx
pts=np.array([[0,0,ZS]],dtype=np.float32)
sim=Reception(mk(True),mk(),fs=FS,c=C,method="sdi",verbose=False)
rf,co=sim.scattered_rf(pts,per_scatterer=True)
sig=rf[0,:,0]; env=np.abs(hilbert(sig.astype(float))); env/=env.max()
t_us=(co["t0"]+np.arange(len(sig))*co["dt"])*1e6
m=scipy.io.loadmat("examples/rf_concave_psf.mat",simplify_cells=True)["save_struct"]
fe=m["rf_env_dB"][:,50]; fe_lin=10**(fe/20); ft=(float(m["t0"])+np.arange(len(fe))*5/FS)*1e6
# PyField onset/peak
above=np.where(env>10**(-60/20))[0]
print(f"PyField: onset(-60dB)={t_us[above[0]]:.3f}us  peak={t_us[np.argmax(env)]:.3f}us")
print(f"FieldII: onset(-60dB)={ft[np.where(fe>-60)[0][0]]:.3f}us  peak={ft[np.argmax(fe_lin)]:.3f}us")
print("\nPyField on-axis envelope (dB), every ~50ns:")
step=5
for i in range(above[0]-2, np.argmax(env)+25, step):
    if 0<=i<len(env):
        db=20*np.log10(max(env[i],1e-6))
        print(f"  t={t_us[i]:.3f}us  dB={db:6.1f}")
