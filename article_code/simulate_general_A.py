"""
General-A model: A is Hurwitz (not -λI).  Sliding-mode observer reconstructs
the full y₂ = CAV + CWS(V).  The combined observer uses the full form with
K' acting on both A and W in the (1,1) and (1,2) blocks.
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Model ────────────────────────────────────────────────────────────────────
n   = 6
ny  = 3

# Same C as the original Wilson--Cowan model
C = np.zeros((ny, n))
for j in range(ny):
    C[j, j]      = 0.6 + j * 0.1
    C[j, j + ny] = 0.4

# General Hurwitz A (not -λI)
rng_a = np.random.default_rng(42)
A = rng_a.normal(0, 1, (n, n)) - 2.0 * np.eye(n)
print(f"max Re(eig(A)) = {np.max(np.real(np.linalg.eigvals(A))):.2f}")

# Original W0
W0 = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0 = W0 / np.linalg.norm(W0) * 4.4

def S(v, lam_sig=1.0):
    return 1.0 / (1.0 + np.exp(-lam_sig * np.clip(v, -30/lam_sig, 30/lam_sig)))

LAM_SIG = 4.0
Gamma_tight = (LAM_SIG / 4.0) * np.eye(n)
Gamma_cons  = np.eye(n)
Q_CAP  = 10.0
EPS_P  = 1e-4
EPS_LV = 1e-4

# ── Time settings ────────────────────────────────────────────────────────────
dt     = 1e-3
stride = 10
N_SUB  = 10

# ── LMI solver ───────────────────────────────────────────────────────────────
# K' is fixed: K' = Cᵀ(CCᵀ)⁻¹  (projector onto range(Cᵀ))
KPRIME = C.T @ np.linalg.solve(C @ C.T, np.eye(ny))

def solve_lmi(W, combined, Gamma_mat=None):
    if Gamma_mat is None:
        Gamma_mat = Gamma_cons
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))
    lv = cp.Variable(n, nonneg=True)
    q  = cp.Variable(nonneg=True)
    if combined:
        # Full form: (I-K'C)A - KC in (1,1), (I-K'C)W in (1,2)
        PAcl   = P @ A - P @ KPRIME @ (C @ A) - R1 @ C
        PW_eff = P @ W - P @ KPRIME @ (C @ W)
    else:
        PAcl   = P @ A - R1 @ C
        PW_eff = P @ W
    M11 = PAcl + PAcl.T + q * np.eye(n)
    M12 = PW_eff + Gamma_mat @ cp.diag(lv)
    M22 = -2.0 * cp.diag(lv)
    M   = cp.bmat([[M11, M12], [M12.T, M22]])
    cons = [M << 0, P >> EPS_P * np.eye(n), lv >= EPS_LV, q >= 0, q <= Q_CAP]
    prob = cp.Problem(cp.Maximize(q), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LMI failed ({prob.status})")
    Pv = P.value
    K  = np.linalg.solve(Pv, R1.value)
    if combined:
        return K, KPRIME, q.value
    return K, None, q.value

# ── Gain scaling table ───────────────────────────────────────────────────────
print("=" * 75)
print("General-A model  —  Gain norms vs coupling  (K'=C^T(CC^T)^{-1} fixed)")
print("=" * 75)
h = f"{'s':>5}  {'||W||':>7}  {'q_std':>6}  {'||K_std||':>9}  {'q_comb':>6}  {'||K_comb||':>9}"
print(h)
print("-" * 55)
for sv in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
    Wv = sv * W0
    ws = np.linalg.norm(Wv)
    try:
        Ks, _, qs = solve_lmi(Wv, False)
        ks = np.linalg.norm(Ks)
    except:
        ks, qs = float("inf"), 0
    try:
        Kc, _, qc = solve_lmi(Wv, True)
        kc = np.linalg.norm(Kc)
    except:
        kc, qc = float("inf"), 0
    ks_s = f"{ks:.1f}" if np.isfinite(ks) else "INFEAS"
    kc_s = f"{kc:.1f}" if np.isfinite(kc) else "INFEAS"
    print(f"{sv:>5.1f}  {ws:>7.1f}  {qs:>6.2f}  {ks_s:>9}  {qc:>6.2f}  {kc_s:>9}")

# ── Sliding-mode observer (full form: reconstructs CAV + CWS(V)) ────────────
def sliding_observer_substep(z1, z2, ym, L, dt_s):
    z1n = z1.copy(); z2n = z2.copy()
    for j in range(ny):
        e1 = ym[j] - z1n[j]
        # No -λ·z₁ term — z₂ must track the full CAV + CWS(V)
        z1n[j] += dt_s * (z2n[j] + L[j] * np.sign(e1) * np.sqrt(abs(e1)))
        z2n[j] += dt_s * (L[j] * L[j] * np.sign(e1))
    return z1n, z2n

def sliding_observer_step(z1, z2, ym, L, n_sub=N_SUB):
    dt_s = dt / n_sub
    for _ in range(n_sub):
        z1, z2 = sliding_observer_substep(z1, z2, ym, L, dt_s)
    return z1, z2

def rk4(f, x, dt, *a):
    k1 = f(x, *a); k2 = f(x + dt/2*k1, *a)
    k3 = f(x + dt/2*k2, *a); k4 = f(x + dt*k3, *a)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# ── Observer RHS ─────────────────────────────────────────────────────────────
def rhs_classical(vh, A, Wv, K, ym, lam_sig=1.0):
    return A @ vh + Wv @ S(vh, lam_sig) + K @ (ym - C @ vh)

def rhs_combined(vh, A, Wv, K, Kp, ym, z2_est, lam_sig=1.0):
    # Full form: injects both y and the reconstruction of CAV + CWS(V)
    y2_pred = C @ (A @ vh + Wv @ S(vh, lam_sig))
    return (A @ vh + Wv @ S(vh, lam_sig)
            + K @ (ym - C @ vh)
            - Kp @ (y2_pred - z2_est))

# ── Noise sweep ──────────────────────────────────────────────────────────────
S_SW = 15.0
T_SW = 10.0
Nt_sw = int(T_SW / dt)
L_smo = 6.0 * np.ones(ny)   # slightly higher for general A

try:
    Ks_sw, _, _ = solve_lmi(S_SW * W0, False, Gamma_tight)
except:
    Ks_sw = None
try:
    Kc_sw, Kpc_sw, _ = solve_lmi(S_SW * W0, True, Gamma_tight)
except:
    Kc_sw, Kpc_sw = None, None
W_sw = S_SW * W0

print(f"\nNoise sweep — general-A  s={S_SW}  ||W||={np.linalg.norm(W_sw):.1f}  lam_sig={LAM_SIG}")
if Ks_sw is not None:
    print(f"  Classical: ||K||={np.linalg.norm(Ks_sw):.1f}")
else:
    print(f"  Classical: INFEAS")
if Kc_sw is not None:
    print(f"  Combined: ||K||={np.linalg.norm(Kc_sw):.1f}  ||K'||={np.linalg.norm(Kpc_sw):.2f}")
else:
    print(f"  Combined: INFEAS")
if Ks_sw is None or Kc_sw is None:
    print("Cannot run noise sweep — one or both LMIs infeasible.")
    exit(1)

print(f"  L_j={L_smo[0]:.0f}, n_sub={N_SUB}")
print(f"{'sigma':>8}  {'RMS_std':>10}  {'RMS_comb':>12}  {'ratio':>8}")
print("-" * 44)

rms_rows = []
for sig in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 1.0]:
    rng = np.random.default_rng(7)
    V     = rng.uniform(-0.3, 0.3, n)
    Vh_s  = np.zeros(n)
    Vh_c  = np.zeros(n)
    z1    = np.zeros(ny)
    z2    = np.zeros(ny)
    nr    = Nt_sw // stride
    e_s   = np.zeros(nr)
    e_c   = np.zeros(nr)
    rec = 0
    for k in range(Nt_sw):
        yt = C @ V
        nm = sig * rng.standard_normal(ny)
        ym = yt + nm
        z1, z2 = sliding_observer_step(z1, z2, ym, L_smo)
        if k % stride == 0:
            e_s[rec] = np.linalg.norm(Vh_s - V)
            e_c[rec] = np.linalg.norm(Vh_c - V)
            rec += 1
        V = rk4(lambda v, n_: A @ v + W_sw @ S(v, LAM_SIG) + n_,
                V, dt, sig * rng.standard_normal(n))
        Vh_s = rk4(rhs_classical, Vh_s, dt, A, W_sw, Ks_sw, ym, LAM_SIG)
        Vh_c = rk4(rhs_combined, Vh_c, dt, A, W_sw, Kc_sw, Kpc_sw, ym, z2.copy(), LAM_SIG)
    ss = int(0.6 * len(e_s))
    rs = np.sqrt(np.mean(e_s[ss:]**2))
    rc = np.sqrt(np.mean(e_c[ss:]**2))
    r  = rs / rc if rc > 1e-15 else float("inf")
    print(f"{sig:>8.0e}  {rs:>10.5f}  {rc:>12.5f}  {r:>8.2f}")
    rms_rows.append((sig, rs, rc, r))

# ── Figure ───────────────────────────────────────────────────────────────────
OUT = (r"c:\Perso\VSCODE WORKSPACE\Thesis_Manuscript-07-02-2026"
       r"\Articles Source\Article_LMI_Lure_Observer")

sa = np.array([x[0] for x in rms_rows])
fig, ax = plt.subplots(figsize=(4.5, 3.0))
ax.loglog(sa, [x[1] for x in rms_rows], "b-o", ms=4, lw=1.0,
           label=rf"Classical  ($\|K\|={np.linalg.norm(Ks_sw):.0f}$)")
ax.loglog(sa, [x[2] for x in rms_rows], "r-s", ms=4, lw=1.0,
           label=rf"Combined  ($\|K\|={np.linalg.norm(Kc_sw):.1f}$, "
           rf"$\|K'\|={np.linalg.norm(Kpc_sw):.1f}$)")
ax.set_xlabel(r"$\sigma$", fontsize=9)
ax.set_ylabel(r"RMS $\|e\|_{\rm ss}$", fontsize=9)
ax.set_title(f"General-A model ($s={S_SW}$, $\\lambda_{{\\rm sig}}={LAM_SIG:.0f}$)", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}\\general_A_noise_sweep.pdf")
plt.close(fig)
print(f"\nFigure saved to {OUT}\\general_A_noise_sweep.pdf")
print("Done.")
