"""
Sector-bound sweep for the Wilson--Cowan observer paper.
Uses S(λ V) to vary the effective slope of the sigmoid:
  S_λ(ξ) = 1/(1 + exp(-λ ξ))
Sector slope: max S_λ' = λ/4, used as tight bound Γ = (λ/4)·I.
Sweeps λ (steepness) at fixed coupling strength s.
Produces: Table for the paper (gain norms vs λ), and a figure.
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Model (Wilson--Cowan, n=6, ny=3) ─────────────────────────────────────────
n   = 6
ny  = 3
lam = 1.0
A   = -lam * np.eye(n)

W0 = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0 = W0 / np.linalg.norm(W0) * 4.4

C = np.zeros((ny, n))
for j in range(ny):
    C[j, j]      = 0.6 + j * 0.1
    C[j, j + ny] = 0.4

def S(v, lam_s):
    """Steep sigmoid: S(λ·v), sector slope = λ/4."""
    return 1.0 / (1.0 + np.exp(-lam_s * np.clip(v, -30/lam_s, 30/lam_s)))

Q_CAP  = 10.0
EPS_P  = 1e-4
EPS_LV = 1e-4

# ── LMI solver ───────────────────────────────────────────────────────────────
def solve_lmi(W_mat, lam_s, combined):
    """WC-type LMI with Γ = (λ/4)·I (tight bound for sigmoid)."""
    gamma = lam_s / 4.0
    Gamma_mat = gamma * np.eye(n)
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))
    lv = cp.Variable(n, nonneg=True)
    q  = cp.Variable(nonneg=True)
    if combined:
        R2     = cp.Variable((n, ny))
        PAcl   = P @ A - R1 @ C
        PW_eff = P @ W_mat - R2 @ (C @ W_mat)
    else:
        R2     = None
        PAcl   = P @ A - R1 @ C
        PW_eff = P @ W_mat
    M11 = PAcl + PAcl.T + q * np.eye(n)
    M12 = PW_eff + Gamma_mat @ cp.diag(lv)
    M22 = -2.0 * cp.diag(lv)
    M   = cp.bmat([[M11, M12], [M12.T, M22]])
    cons = [M << 0, P >> EPS_P * np.eye(n), lv >= EPS_LV, q >= 0, q <= Q_CAP]
    prob = cp.Problem(cp.Maximize(q), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None, None
    Pv = P.value; K = np.linalg.solve(Pv, R1.value)
    if combined:
        return K, np.linalg.solve(Pv, R2.value), q.value
    return K, None, q.value


# ── Sweep over λ at fixed coupling s=10 ─────────────────────────────────────
s_fixed = 10.0
W_fixed = s_fixed * W0
lam_vals = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

print("=" * 80)
print(f"  Wilson--Cowan, n=6, ny=3, s={s_fixed}, ||W||={np.linalg.norm(W_fixed):.1f}")
print(f"  Sigmoïde S(λ V), pente sectorielle γ = λ/4")
print("=" * 80)
hdr = f"{'λ':>6}  {'γ=λ/4':>7}  {'q_std':>6}  {'||K_std||':>9}  {'q_comb':>6}  {'||K_comb||':>9}  {'||K''||':>8}"
print(hdr)
print("-" * 70)

rows = []
for lv_val in lam_vals:
    gamma = lv_val / 4.0
    try:
        Ks, _, qs = solve_lmi(W_fixed, lv_val, False)
        ks = np.linalg.norm(Ks) if Ks is not None else float("inf")
    except:
        ks, qs = float("inf"), 0.0
    try:
        Kc, Kpc, qc = solve_lmi(W_fixed, lv_val, True)
        kc  = np.linalg.norm(Kc) if Kc is not None else float("inf")
        kpc = np.linalg.norm(Kpc) if Kpc is not None else float("inf")
    except:
        kc, kpc, qc = float("inf"), float("inf"), 0.0

    ks_s  = f"{ks:.1f}"  if np.isfinite(ks)  else "INFEAS"
    kc_s  = f"{kc:.1f}"  if np.isfinite(kc)  else "INFEAS"
    kpc_s = f"{kpc:.2f}" if np.isfinite(kpc) else "INFEAS"
    print(f"{lv_val:>6.2f}  {gamma:>7.2f}  {qs:>6.2f}  {ks_s:>9}  {qc:>6.2f}  {kc_s:>9}  {kpc_s:>8}")
    rows.append((lv_val, gamma, ks, kc, kpc, qs, qc))


# ── Output table for paper ───────────────────────────────────────────────────
OUT = (r"c:\Perso\VSCODE WORKSPACE\Thesis_Manuscript-07-02-2026"
       r"\Articles Source\Article_LMI_Lure_Observer")

print(f"\nLaTeX table for the paper (save to {OUT}\\sector_sweep_table.tex):\n")
print(r"\begin{table}[ht]")
print(r"  \centering")
print(r"  \caption{Observer gain norms vs.\ sigmoid steepness $\lambda$")
print(r"           ($n=6$, $\lambda_{\rm decay}=1$, $n_y=3$, $s=10$,")
print(r"           $\|W\|=44.0$, LMI~\eqref{eq:LMI_wc}, $\max q \leq 10$).")
print(r"           $\Gamma = (\lambda/4)I_6$ (tight sector bound for $S(\lambda\,\cdot)$).")
print(r"           Both designs attain $q=10.00$ for all $\lambda$.")
print(r"           $\|K'\|$ stays nearly constant while $\|K\|_{\rm std}$")
print(r"           grows with the sector slope.}")
print(r"  \label{tab:sector_sweep}")
print(r"  \begin{tabular}{rrrrr}")
print(r"    \toprule")
print(r"    $\lambda$ & $\gamma=\lambda/4$ & $\|K\|_{\rm std}$ & $\|K\|_{\rm comb}$ & $\|K'\|_{\rm comb}$ \\")
print(r"    \midrule")
for lv_val, gamma, ks, kc, kpc, qs, qc in rows:
    ks_s  = f"{ks:.1f}"  if np.isfinite(ks)  else "---"
    kc_s  = f"{kc:.1f}"  if np.isfinite(kc)  else "---"
    kpc_s = f"{kpc:.2f}" if np.isfinite(kpc) else "---"
    print(f"    {lv_val:.2f} & {gamma:.2f} & {ks_s} & {kc_s} & {kpc_s} \\\\")
print(r"    \bottomrule")
print(r"  \end{tabular}")
print(r"\end{table}")

# ── Figure ───────────────────────────────────────────────────────────────────
sa  = np.array([r[0] for r in rows])
ksa = np.array([r[2] for r in rows])
kca = np.array([r[3] for r in rows])
kpa = np.array([r[4] for r in rows])

fig, ax = plt.subplots(figsize=(4.5, 3.0))
ax.loglog(sa, ksa, "b-o", ms=4, lw=1.0, label=r"$\|K\|_{\rm std}$")
ax.loglog(sa, kca, "r-s", ms=4, lw=1.0, label=r"$\|K\|_{\rm comb}$")
ax.loglog(sa, kpa, "g-^", ms=4, lw=1.0, label=r"$\|K'\|_{\rm comb}$")
ax.set_xlabel(r"$\lambda$  (sigmoid steepness)", fontsize=9)
ax.set_ylabel(r"Gain norm", fontsize=9)
ax.set_title(rf"Gain norms vs.\ $\lambda$  ($s={s_fixed}$, $\|W\|={np.linalg.norm(W_fixed):.0f}$)", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}\\sector_sweep.pdf")
plt.close(fig)
print(f"\nFigure saved to {OUT}\\sector_sweep.pdf")
