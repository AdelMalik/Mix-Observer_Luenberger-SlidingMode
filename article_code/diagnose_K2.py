"""
Diagnostic 2: test BOTH classical and combined LMI, check q values,
and run the exact same code as simulate_direct_output.py.
"""
import numpy as np
import cvxpy as cp

n   = 6
ny  = 3
rng_a = np.random.default_rng(42)
A = rng_a.normal(0, 1, (n, n)) - 2.0 * np.eye(n)
C = np.zeros((ny, n))
C[:ny, :ny] = np.eye(ny)

EPS_REG = 1e-3
EPS_P   = 1e-4
EPS_LV  = 1e-4
Q_CAP   = 10.0

W0 = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0 = W0 / np.linalg.norm(W0) * 4.4

def solve_lmi(W_mat, combined):
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))
    lv = cp.Variable(n, nonneg=True)
    q  = cp.Variable(nonneg=True)
    PAcl = P @ A - R1 @ C
    if combined:
        R2 = cp.Variable((n, ny))
        PW_eff = P @ W_mat - R2 @ C
    else:
        PW_eff = P @ W_mat
    M11 = PAcl + PAcl.T + q * np.eye(n)
    M12 = PW_eff + cp.diag(lv)
    M22 = -EPS_REG * np.eye(n)
    M = cp.bmat([[M11, M12], [M12.T, M22]])
    cons = [M << 0, P >> EPS_P * np.eye(n), lv >= EPS_LV, q >= 0, q <= Q_CAP]
    prob = cp.Problem(cp.Maximize(q), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LMI failed ({prob.status})")
    Pv = P.value; K = np.linalg.solve(Pv, R1.value)
    if combined:
        return K, np.linalg.solve(Pv, R2.value), q.value, np.linalg.cond(Pv)
    return K, None, q.value, np.linalg.cond(Pv)

print("=" * 100)
print("Case 1 — Both classical AND combined LMIs")
print(f"max Re(eig(A)) = {np.max(np.real(np.linalg.eigvals(A))):.4f}")
print(f"EPS_REG = {EPS_REG}, EPS_P = {EPS_P}")
print("=" * 100)
hdr = f"{'s':>5}  {'||W||':>7}  {'q_std':>8}  {'||K_std||':>10}  {'cond(P)_std':>12}  {'q_comb':>8}  {'||K_comb||':>10}  {'||K''||':>8}  {'cond(P)_comb':>12}"
print(hdr)
print("-" * 100)

for sv in [0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0]:
    Wv = sv * W0; ws = np.linalg.norm(Wv)
    try:
        Ks, _, qs, conds = solve_lmi(Wv, False)
        ks = np.linalg.norm(Ks)
        ks_s = f"{ks:.1f}"
        qs_s = f"{qs:.4f}"
        cs_s = f"{conds:.1e}"
    except Exception as e:
        ks_s = "INFEAS"
        qs_s = "----"
        cs_s = "----"
    try:
        Kc, Kpc, qc, condc = solve_lmi(Wv, True)
        kc_s  = f"{np.linalg.norm(Kc):.1f}"
        kpc_s = f"{np.linalg.norm(Kpc):.1f}"
        qc_s  = f"{qc:.4f}"
        cc_s  = f"{condc:.1e}"
    except Exception as e:
        kc_s, kpc_s = "INFEAS", "---"
        qc_s, cc_s = "----", "----"
    print(f"{sv:>5.1f}  {ws:>7.1f}  {qs_s:>8}  {ks_s:>10}  {cs_s:>12}  {qc_s:>8}  {kc_s:>10}  {kpc_s:>8}  {cc_s:>12}")

# ── Now: what if we fix q=0 and minimize trace(P) instead? ──────────────────
print("\n" + "=" * 100)
print("MINIMIZING trace(P) instead of maximizing q (classical only)")
print("=" * 100)

def solve_lmi_mintrace(W_mat):
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))
    lv = cp.Variable(n, nonneg=True)
    q  = cp.Variable(nonneg=True)
    PAcl = P @ A - R1 @ C
    PW_eff = P @ W_mat
    M11 = PAcl + PAcl.T + q * np.eye(n)
    M12 = PW_eff + cp.diag(lv)
    M22 = -EPS_REG * np.eye(n)
    M = cp.bmat([[M11, M12], [M12.T, M22]])
    cons = [M << 0, P >> EPS_P * np.eye(n), lv >= EPS_LV, q >= 0, q <= Q_CAP]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None, None, None
    Pv = P.value; K = np.linalg.solve(Pv, R1.value)
    return K, q.value, np.linalg.cond(Pv), np.trace(Pv)

for sv in [0.5, 1.0, 2.0, 5.0, 7.0, 10.0]:
    Wv = sv * W0; ws = np.linalg.norm(Wv)
    try:
        K, qv, condv, trP = solve_lmi_mintrace(Wv)
        print(f"  s={sv:.1f}: ||K||={np.linalg.norm(K):.1f}, q={qv:.4f}, cond(P)={condv:.1e}, tr(P)={trP:.2f}")
    except Exception as e:
        print(f"  s={sv:.1f}: INFEAS ({e})")

print("\nDone.")
