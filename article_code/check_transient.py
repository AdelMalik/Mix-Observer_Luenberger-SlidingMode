"""Check if L_j=4 combined observer is still settling at t=6-10s."""
import numpy as np

n=6; ny=3; lam=1.0; A=-lam*np.eye(n)
C=np.zeros((ny,n))
for j in range(ny): C[j,j]=0.6+j*0.1; C[j,j+ny]=0.4
W0=np.array([[1.20,0.40,0.25,-0.80,-0.20,-0.10],[0.40,1.10,0.30,-0.20,-0.70,-0.15],[0.25,0.30,0.90,-0.10,-0.15,-0.60],[0.90,0.30,0.20,-0.60,-0.10,-0.05],[0.30,0.85,0.25,-0.10,-0.55,-0.08],[0.20,0.25,0.80,-0.05,-0.08,-0.45]])
W0=W0/np.linalg.norm(W0)*4.4
W=15*W0
def S(v): return 1/(1+np.exp(-4*np.clip(v,-7.5,7.5)))
def rk4(f,x,dt,*a):
    k1=f(x,*a); k2=f(x+dt/2*k1,*a); k3=f(x+dt/2*k2,*a); k4=f(x+dt*k3,*a)
    return x+dt/6*(k1+2*k2+2*k3+k4)

dt=1e-3; N_SUB=10; T=20.0; Nt=int(T/dt); stride=20; sig=0.01; Lj=4

# LMI gains (pre-computed from simulate_observers.py)
Kc=np.array([...])  # need to compute
# Actually let me just use the sliding-mode tracking error which is the bottleneck
rng=np.random.default_rng(7)
V=rng.uniform(-0.3,0.3,n); z1=np.zeros(ny); z2=np.zeros(ny)
errs=np.zeros(Nt//stride); rec=0
for k in range(Nt):
    yt=C@V; ym=yt+sig*rng.standard_normal(ny)
    dt_s=dt/N_SUB
    for _ in range(N_SUB):
        for j in range(ny):
            e1=ym[j]-z1[j]
            z1[j]+=dt_s*(-lam*z1[j]+z2[j]+Lj*np.sign(e1)*np.sqrt(abs(e1)))
            z2[j]+=dt_s*(Lj*Lj*np.sign(e1))
    if k%stride==0:
        errs[rec]=np.linalg.norm(z2-C@(W@S(V))); rec+=1
    V=rk4(lambda v,n_:A@v+W@S(v)+n_,V,dt,sig*rng.standard_normal(n))

# RMS over different windows
for t0_pct in [0.3,0.5,0.6,0.7,0.8]:
    ss=int(t0_pct*len(errs))
    rms=np.sqrt(np.mean(errs[ss:]**2))
    print(f'sliding-mode RMS (t>{t0_pct*T:.0f}s): {rms:.4f}')
