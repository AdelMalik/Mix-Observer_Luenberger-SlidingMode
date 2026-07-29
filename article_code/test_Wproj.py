"""Quick test: W = P_rangeC @ W0, K' fixed, check gains and plant."""
import numpy as np
import cvxpy as cp

n=6; ny=3; A=-np.eye(n)
C=np.zeros((ny,n))
for j in range(ny): C[j,j]=0.6+j*0.1; C[j,j+ny]=0.4

W0=np.array([[1.20,0.40,0.25,-0.80,-0.20,-0.10],[0.40,1.10,0.30,-0.20,-0.70,-0.15],[0.25,0.30,0.90,-0.10,-0.15,-0.60],[0.90,0.30,0.20,-0.60,-0.10,-0.05],[0.30,0.85,0.25,-0.10,-0.55,-0.08],[0.20,0.25,0.80,-0.05,-0.08,-0.45]])
W0=W0/np.linalg.norm(W0)*4.4
P_rangeC=C.T@np.linalg.solve(C@C.T,C)
W0_new=P_rangeC@W0; W0_new=W0_new/np.linalg.norm(W0_new)*4.4
Kp=C.T@np.linalg.solve(C@C.T,np.eye(ny))
print(f"||(I-K'C)W|| = {np.linalg.norm((np.eye(n)-Kp@C)@W0_new):.2e}")
print(f"||W0||={np.linalg.norm(W0_new):.1f}")

Gamma=np.eye(n); Q_CAP=10; EPS_P=1e-4; EPS_LV=1e-4
def solve_lmi(W,combined):
    P=cp.Variable((n,n),symmetric=True); R1=cp.Variable((n,ny))
    lv=cp.Variable(n,nonneg=True); q=cp.Variable(nonneg=True)
    if combined: PW_eff=P@W-P@Kp@(C@W)
    else: PW_eff=P@W
    PAcl=P@A-R1@C
    M11=PAcl+PAcl.T+q*np.eye(n); M12=PW_eff+Gamma@cp.diag(lv); M22=-2.0*cp.diag(lv)
    M=cp.bmat([[M11,M12],[M12.T,M22]])
    cons=[M<<0,P>>EPS_P*np.eye(n),lv>=EPS_LV,q>=0,q<=Q_CAP]
    prob=cp.Problem(cp.Maximize(q),cons)
    prob.solve(solver=cp.CLARABEL,verbose=False)
    if prob.status not in ('optimal','optimal_inaccurate'): return None,None,None
    Pv=P.value; K=np.linalg.solve(Pv,R1.value)
    return (K,Kp,q.value) if combined else (K,None,q.value)

print(f"{'s':>5}  {'||W||':>7}  {'q_std':>6}  {'||K_std||':>9}  {'q_comb':>6}  {'||K_comb||':>9}")
print("-"*55)
for sv in [0.5,1,2,3,5,7,10,15,20]:
    Wv=sv*W0_new; ws=np.linalg.norm(Wv)
    Ks,_,qs=solve_lmi(Wv,False)
    Kc,_,qc=solve_lmi(Wv,True)
    ks_s=f"{np.linalg.norm(Ks):.1f}" if Ks is not None else "INFEAS"
    kc_s=f"{np.linalg.norm(Kc):.1f}" if Kc is not None else "INFEAS"
    print(f"{sv:>5.1f}  {ws:>7.1f}  {qs:>6.2f}  {ks_s:>9}  {qc:>6.2f}  {kc_s:>9}")
