"""Compare sliding-mode tracking on original vs projected W, same setup."""
import numpy as np

n=6; ny=3; lam=1.0; A=-lam*np.eye(n)
C=np.zeros((ny,n))
for j in range(ny): C[j,j]=0.6+j*0.1; C[j,j+ny]=0.4

W0_raw=np.array([[1.20,0.40,0.25,-0.80,-0.20,-0.10],[0.40,1.10,0.30,-0.20,-0.70,-0.15],[0.25,0.30,0.90,-0.10,-0.15,-0.60],[0.90,0.30,0.20,-0.60,-0.10,-0.05],[0.30,0.85,0.25,-0.10,-0.55,-0.08],[0.20,0.25,0.80,-0.05,-0.08,-0.45]])
W0_raw=W0_raw/np.linalg.norm(W0_raw)*4.4
P_rangeC=C.T@np.linalg.solve(C@C.T,C)
W0_proj=P_rangeC@W0_raw; W0_proj=W0_proj/np.linalg.norm(W0_proj)*4.4

def S(v): return 1/(1+np.exp(-4*np.clip(v,-7.5,7.5)))
dt=1e-3; N_SUB=10; T=10.0; Nt=int(T/dt); stride=10
Lj=5; sig=0.01

for name,W0 in [("Original",15*W0_raw),("Projected",15*W0_proj)]:
    rng=np.random.default_rng(7)
    V=rng.uniform(-0.3,0.3,n); z1=np.zeros(ny); z2=np.zeros(ny)
    smo_err=np.zeros(Nt//stride); v_norms=np.zeros(Nt//stride)
    rec=0
    for k in range(Nt):
        yt=C@V; ym=yt+sig*rng.standard_normal(ny)
        dt_s=dt/N_SUB
        for _ in range(N_SUB):
            z1n=z1.copy(); z2n=z2.copy()
            for j in range(ny):
                e1=ym[j]-z1n[j]
                z1n[j]+=dt_s*(-lam*z1n[j]+z2n[j]+Lj*np.sign(e1)*np.sqrt(abs(e1)))
                e1_n=ym[j]-z1n[j]
                z2n[j]+=dt_s*(Lj*Lj*np.sign(e1_n))
            z1,z2=z1n,z2n
        if k%stride==0:
            smo_err[rec]=np.linalg.norm(z2-C@(W0@S(V)))
            v_norms[rec]=np.linalg.norm(V)
            rec+=1
        V=V+dt*(A@V+W0@S(V)+sig*rng.standard_normal(n))
    ss=int(0.6*len(smo_err))
    rms=np.sqrt(np.mean(smo_err[ss:]**2))
    print(f"{name}: RMS_smo={rms:.4f}  ||V||_avg={np.mean(v_norms[ss:]):.1f}")
