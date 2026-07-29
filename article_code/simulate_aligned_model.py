"""
Aligned-coupling model: W = β·CᵀC + W_residual.
K' = Cᵀ(CCᵀ)⁻¹ eliminates the β·CᵀC term completely from the LMI,
leaving only P_kerC·W in the off-diagonal.  This showcases the combined
observer on a model where the coupling cancellation via K' is structural.
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Model ────────────────────────────────────────────────────────────────────
n   = 6
ny  = 3
lam = 1.0
A   = -lam * np.eye(n)

# Same C as the original Wilson--Cowan model
C = np.zeros((ny, n))
for j in range(ny):
    C[j, j]      = 0.6 + j * 0.1
    C[j, j + ny] = 0.4

# ── Model: W = P_rangeC @ W0, columns projected onto range(Cᵀ) ────────────
# K' = Cᵀ(CCᵀ)⁻¹ then satisfies (I-K'C)W = 0 exactly.
# Plant dynamics stay bounded (||V|| ~ original WC).
W0_raw = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0_raw = W0_raw / np.linalg.norm(W0_raw) * 4.4

P_rangeC = C.T @ np.linalg.solve(C @ C.T, C)
W0_aligned = P_rangeC @ W0_raw
W0_aligned = W0_aligned / np.linalg.norm(W0_aligned) * 4.4

print(f"||(I-K'C)W|| = {np.linalg.norm((np.eye(n)-C.T@np.linalg.solve(C@C.T,C))@W0_aligned):.2e}")

def S(v, lam_sig=1.0):
    return 1.0 / (1.0 + np.exp(-lam_sig * np.clip(v, -30/lam_sig, 30/lam_sig)))

# Sector bounds
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
KPRIME_FIXED = C.T @ np.linalg.solve(C @ C.T, np.eye(ny))

def solve_lmi(W, combined, Gamma_mat=None):
    if Gamma_mat is None:
        Gamma_mat = Gamma_cons
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))
    lv = cp.Variable(n, nonneg=True)
    q  = cp.Variable(nonneg=True)
    if combined:
        PW_eff = P @ W - P @ KPRIME_FIXED @ (C @ W)
    else:
        PW_eff = P @ W
    PAcl   = P @ A - R1 @ C
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
        return K, KPRIME_FIXED, q.value
    return K, None, q.value

# ── Gain scaling table ───────────────────────────────────────────────────────
print("=" * 75)
print("Projected-W model  —  Gain norms vs coupling  (K'=C^T(CC^T)^{-1}, (I-K'C)W=0)")
print("=" * 75)
h = f"{'s':>5}  {'||Ws||':>7}  {'q_std':>6}  {'||K_std||':>9}  {'q_comb':>6}  {'||K_comb||':>9}  {'||K''||':>8}"
print(h)
print("-" * 70)

for sv in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
    Wv = sv * W0_aligned
    ws = np.linalg.norm(Wv)
    try:
        Ks, _, qs = solve_lmi(Wv, False)
        ks = np.linalg.norm(Ks)
    except:
        ks, qs = float("inf"), 0
    try:
        Kc, Kpc, qc = solve_lmi(Wv, True)
        kc  = np.linalg.norm(Kc)
        kpc = np.linalg.norm(Kpc)
    except:
        kc, kpc, qc = float("inf"), float("inf"), 0
    ks_s  = f"{ks:.1f}"  if np.isfinite(ks)  else "INFEAS"
    kc_s  = f"{kc:.1f}"  if np.isfinite(kc)  else "INFEAS"
    kpc_s = f"{kpc:.2f}" if np.isfinite(kpc) else "INFEAS"
    print(f"{sv:>5.1f}  {ws:>7.1f}  {qs:>6.2f}  {ks_s:>9}  {qc:>6.2f}  {kc_s:>9}  {kpc_s:>8}")

# ── Sliding-mode observer ────────────────────────────────────────────────────
def sliding_observer_substep(z1, z2, ym, L, dt_s):
    z1n = z1.copy(); z2n = z2.copy()
    for j in range(ny):
        e1 = ym[j] - z1n[j]
        z1n[j] += dt_s * (-lam * z1n[j] + z2n[j]
                          + L[j] * np.sign(e1) * np.sqrt(abs(e1)))
        e1_n = ym[j] - z1n[j]
        z2n[j] += dt_s * (L[j] * L[j] * np.sign(e1_n))
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

def rhs_classical(vh, A, Wv, K, ym, lam_sig=1.0):
    return A @ vh + Wv @ S(vh, lam_sig) + K @ (ym - C @ vh)

def rhs_combined(vh, A, Wv, K, Kp, ym, z2_est, lam_sig=1.0):
    return (A @ vh + Wv @ S(vh, lam_sig)
            + K @ (ym - C @ vh)
            - Kp @ (C @ (Wv @ S(vh, lam_sig)) - z2_est))

# ── Noise sweep ──────────────────────────────────────────────────────────────
S_SW = 30.0
T_SW = 10.0
Nt_sw  = int(T_SW / dt)
L_smo  = 3.0 * np.ones(ny)

try:
    Ks_sw, _, _ = solve_lmi(S_SW * W0_aligned, False, Gamma_tight)
except:
    Ks_sw = None
try:
    Kc_sw, Kpc_sw, _ = solve_lmi(S_SW * W0_aligned, True, Gamma_tight)
except:
    Kc_sw, Kpc_sw = None, None
W_sw = S_SW * W0_aligned

print(f"\nNoise sweep — aligned model  s={S_SW}  ||W||={np.linalg.norm(W_sw):.1f}  lam_sig={LAM_SIG}")
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
print(f"{'sigma':>8}  {'RMS_std':>10}  {'RMS_comb':>12}  {'ratio':>8}  {'RMS_smo':>10}")
print("-" * 55)

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
    e_smo = np.zeros(nr)
    rec = 0
    for k in range(Nt_sw):
        yt = C @ V
        nm = sig * rng.standard_normal(ny)
        ym = yt + nm
        z1, z2 = sliding_observer_step(z1, z2, ym, L_smo)
        y2_true = C @ (W_sw @ S(V, LAM_SIG))
        if k % stride == 0:
            e_s[rec] = np.linalg.norm(Vh_s - V)
            e_c[rec] = np.linalg.norm(Vh_c - V)
            e_smo[rec] = np.linalg.norm(z2 - y2_true)
            rec += 1
        V = rk4(lambda v, n_: A @ v + W_sw @ S(v, LAM_SIG) + n_,
                V, dt, sig * rng.standard_normal(n))
        Vh_s = rk4(rhs_classical, Vh_s, dt, A, W_sw, Ks_sw, ym, LAM_SIG)
        Vh_c = rk4(rhs_combined, Vh_c, dt, A, W_sw, Kc_sw, Kpc_sw, ym, z2.copy(), LAM_SIG)
    ss = int(0.6 * len(e_s))
    rs = np.sqrt(np.mean(e_s[ss:]**2))
    rc = np.sqrt(np.mean(e_c[ss:]**2))
    rsm = np.sqrt(np.mean(e_smo[ss:]**2))
    r  = rs / rc if rc > 1e-15 else float("inf")
    print(f"{sig:>8.0e}  {rs:>10.5f}  {rc:>12.5f}  {r:>8.2f}  {rsm:>10.5f}")
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
ax.set_title(f"Projected-W model ($s={S_SW}$, $\\lambda_{{\\rm sig}}={LAM_SIG:.0f}$)", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}\\aligned_noise_sweep.pdf")
plt.close(fig)
print(f"\nFigure saved to {OUT}\\aligned_noise_sweep.pdf")
print("Done.")
