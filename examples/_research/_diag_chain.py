import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import hilbert
from pyfield.transducers import ConcaveCircularTransducer
from pyfield.hsir.farfield_rect_patch import compute_h_sir
from pyfield.utilities.helper_functions import compute_sub_elem_attributes, compute_time_grid

F0, FS, C, ZS = 3e6, 100e6, 1540.0, 30.0
def mk():
    tx = ConcaveCircularTransducer(diameter_mm=16.0, focus_mm=80.0, frequency_Hz=F0,
                                    refine_factor=1, no_sub_diameter=16)
    return tx
tx = mk(); rx = mk()
t = np.arange(0, 2.0/F0, 1.0/FS)
exc = np.sin(2*np.pi*F0*t).astype(np.float32)
ir  = (np.sin(2*np.pi*F0*t)*np.hanning(len(t))).astype(np.float32)

pts = np.array([[0.0,0.0,ZS*1e-3]], dtype=np.float32)
(c_,a_,d_,M,_,wx,wy,idx) = compute_sub_elem_attributes(tx)
wxm,wym=float(wx.max()),float(wy.max())
fr=tx.sub_patch_frames; eu=np.asarray(fr["tangents_u"],np.float32); ev=np.asarray(fr["tangents_v"],np.float32)
tg,t0,dt,T = compute_time_grid(1,M,pts,c_,wxm,wym,C,FS,tx.delays,verbose=False)
h,_=compute_h_sir(1,M,T,dt,tg,pts,c_,wx,wy,np.float32(1/C),FS,a_,d_,None,eu,ev)
h=h[0]  # (T,)

def pk(sig, base_t0):
    env=np.abs(hilbert(np.asarray(sig,float)))
    i=int(np.argmax(env)); return (base_t0+i*dt)*1e6

pe_t0 = 2*t0
# h_tx alone
print(f"t0(one-way)={t0*1e6:.3f}us  pe_t0={pe_t0*1e6:.3f}us  T={T}")
print(f"[1] h_tx peak: {pk(h,t0):.3f}us  (peak-t0={pk(h,t0)-t0*1e6:.3f})")

# h_pe = h_tx conv h_rx  (same h)
def conv(a,b):
    n=len(a)+len(b)-1; nf=1<<((n-1).bit_length())
    return irfft(rfft(a,nf)*rfft(b,nf),nf)[:n]
hpe=conv(h,h)
print(f"[2] h_pe=h*h peak: {pk(hpe,pe_t0):.3f}us  (peak-pe_t0={pk(hpe,pe_t0)-pe_t0*1e6:.3f})")

# build freq chain on hpe
n=len(hpe); 
def addfft(sig, extra_len):
    nf=1<<((len(sig)+extra_len-1).bit_length()); return nf
nf=1<<((len(hpe)+3*len(exc))-1).bit_length()
fr_=rfftfreq(nf,1/FS); jw=1j*2*np.pi*fr_
H=rfft(hpe,nf)
for k,label in [(0,"hpe"),(1,"+jw"),(2,"+jw^2"),(3,"+jw^3")]:
    Hk=H*(jw**k)
    s=irfft(Hk,nf)[:len(hpe)]
    print(f"[3.{k}] {label:7s} peak: {pk(s,pe_t0):.3f}us  (gd={pk(s,pe_t0)-pe_t0*1e6:.3f})")

# now jw^3 + exc + ir + ir
H3=H*(jw**3)*rfft(exc,nf)
print(f"[4] jw3+exc           gd={pk(irfft(H3,nf)[:len(hpe)],pe_t0)-pe_t0*1e6:.3f}")
H3b=H3*rfft(ir,nf)
print(f"[5] jw3+exc+ir         gd={pk(irfft(H3b,nf)[:len(hpe)],pe_t0)-pe_t0*1e6:.3f}")
H3c=H3b*rfft(ir,nf)
print(f"[6] jw3+exc+ir+ir(full) gd={pk(irfft(H3c,nf)[:len(hpe)],pe_t0)-pe_t0*1e6:.3f}  peak={pk(irfft(H3c,nf)[:len(hpe)],pe_t0):.3f}us")
# compare jw^1 and jw^2 full chains
for k in (1,2):
    Hx=H*(jw**k)*rfft(exc,nf)*rfft(ir,nf)*rfft(ir,nf)
    s=irfft(Hx,nf)[:len(hpe)]
    print(f"    jw^{k}+exc+ir+ir   gd={pk(s,pe_t0)-pe_t0*1e6:.3f}  peak={pk(s,pe_t0):.3f}us")
print("\nFieldII: peak=40.220us, gd(=peak-t0_FII38.420)=1.800us")
