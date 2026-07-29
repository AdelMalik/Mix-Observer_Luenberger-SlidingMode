"""
Test combined observer under varying sector-bound conservatism.
n=2, ny=1.  Vary both coupling strength (r) and sigmoid steepness (α).
Classical LMI: He{P(A-KC)}+qI in (1,1), PW+ΓΛ in (1,2).
Combined LMI: uses K' to cancel row(C) coupling.
"""

import numpy as np
import cvxpy as cp

# ── System ──────────────────────────────────────────────────────────────────
n  = 2
ny = 1
lam = 1.0
A   = -lam * np.eye(n)

# rank-1 coupling: w = [1, 1]^T, W = r * w w^T
w  = np.ones(n) / np.sqrt(n)   # ||w|| = 1
W0 = np.outer(w, w)
C  = np.array([[1.0, 0.0]])    # scalar output, sees only x1

Q_CAP  = 10.0
EPS_P  = 1e-4
EPS_LV = 1e-4


def S(v, alpha):
    """Sigmoid with steepness alpha.  Sector slope = alpha/4."""
    return 1.0 / (1.0 + np.exp(-alpha * np.clip(v, -30/alpha, 30/alpha)))


def solve_lmi(r, alpha, combined):
    """Maximise q (<= Q_CAP) subject to WC-type LMI.
       Gamma = (alpha/4) * I_n  (tight bound for logistic).
    """
    gamma = alpha / 4.0
    Gamma_mat = gamma * np.eye(n)
    W_mat = r * W0

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
        return None, None, None, None
    Pv = P.value
    K  = np.linalg.solve(Pv, R1.value)
    if combined:
        Kp = np.linalg.solve(Pv, R2.value)
        return K, Kp, q.value, np.linalg.eigvalsh(Pv)
    return K, None, q.value, np.linalg.eigvalsh(Pv)


# ── Sweep ────────────────────────────────────────────────────────────────────
r_vals   = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
alpha_vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

print("=" * 90)
print("  n=2, ny=1  —  Sweep over coupling r and sigmoid steepness α")
print("  Γ = (α/4)·I   (tight sector slope for logistic)")
print("=" * 90)

for alpha in alpha_vals:
    gamma = alpha / 4.0
    print(f"\n── α = {alpha:.0f}  (γ = {gamma:.2f}) ──")
    hdr = f"{'r':>6}  {'||W||':>7}  {'q_std':>6}  {'||K_std||':>9}  {'q_comb':>6}  {'||K_comb||':>9}  {'||K''||':>8}"
    print(hdr)
    print("-" * 70)
    for r in r_vals:
        Wv = r * W0
        ws = np.linalg.norm(Wv)
        try:
            Ks, _, qs, _ = solve_lmi(r, alpha, False)
            ks = np.linalg.norm(Ks) if Ks is not None else float("inf")
        except:
            ks, qs = float("inf"), 0.0
        try:
            Kc, Kpc, qc, _ = solve_lmi(r, alpha, True)
            kc  = np.linalg.norm(Kc) if Kc is not None else float("inf")
            kpc = np.linalg.norm(Kpc) if Kpc is not None else float("inf")
        except:
            kc, kpc, qc = float("inf"), float("inf"), 0.0

        ks_s  = f"{ks:.1f}"  if np.isfinite(ks)  else "INFEAS"
        qs_s  = f"{qs:.2f}"  if np.isfinite(qs)  else "  --"
        kc_s  = f"{kc:.1f}"  if np.isfinite(kc)  else "INFEAS"
        qc_s  = f"{qc:.2f}"  if np.isfinite(qc)  else "  --"
        kpc_s = f"{kpc:.2f}" if np.isfinite(kpc) else "INFEAS"
        print(f"{r:>6.1f}  {ws:>7.2f}  {qs_s:>6}  {ks_s:>9}  {qc_s:>6}  {kc_s:>9}  {kpc_s:>8}")


# ── Summary: ratio of ||K_std|| / ||K_comb|| at max r ────────────────────────
print("\n" + "=" * 90)
print("  Ratio  ||K_std|| / ||K_comb||  at r = 100")
print("=" * 90)
print(f"{'α':>6}  {'γ':>6}  {'||K_std||':>10}  {'||K_comb||':>10}  {'ratio':>8}")
print("-" * 50)
for alpha in alpha_vals:
    Ks, _, qs, _ = solve_lmi(100.0, alpha, False)
    Kc, Kpc, qc, _ = solve_lmi(100.0, alpha, True)
    if Ks is not None and Kc is not None:
        ks = np.linalg.norm(Ks)
        kc = np.linalg.norm(Kc)
        print(f"{alpha:>6.0f}  {alpha/4:>6.2f}  {ks:>10.1f}  {kc:>10.1f}  {ks/kc:>8.1f}")
    else:
        print(f"{alpha:>6.0f}  {alpha/4:>6.2f}  {'INFEAS':>10}  {'INFEAS':>10}  {'--':>8}")
