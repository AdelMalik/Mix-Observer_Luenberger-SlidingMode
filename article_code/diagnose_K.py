"""
Diagnostic: investigate why ||K||_std is anomalously large at s=0.5 in Case 1.
Hypothesis: the LMI is very loose at low coupling, and the solver picks an
arbitrary P among many that achieve q=10.  We check P's condition number
and test whether penalizing trace(P) yields a smaller K.
"""
import numpy as np
import cvxpy as cp

# ── Same model as simulate_direct_output.py ──────────────────────────────────
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

# ── LMI solver (classical only, increasing-only) ─────────────────────────────
def solve_lmi(W_mat, penalize_trace=False, trace_weight=0.0):
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

    if penalize_trace:
        obj = cp.Maximize(q - trace_weight * cp.trace(P))
    else:
        obj = cp.Maximize(q)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None, None, None, prob.status

    Pv = P.value
    K  = np.linalg.solve(Pv, R1.value)
    cond_P = np.linalg.cond(Pv)
    return K, q.value, Pv, cond_P, prob.status

# ── W0 ──────────────────────────────────────────────────────────────────────
W0 = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0 = W0 / np.linalg.norm(W0) * 4.4

print("=" * 80)
print("DIAGNOSTIC: Case 1 — classical increasing-only LMI")
print(f"max Re(eig(A)) = {np.max(np.real(np.linalg.eigvals(A))):.4f}")
print(f"EPS_REG = {EPS_REG}, Q_CAP = {Q_CAP}")
print("=" * 80)

for sv in [0.5, 1.0, 5.0, 10.0]:
    Wv = sv * W0
    ws = np.linalg.norm(Wv)
    print(f"\n--- s = {sv:.1f}, ||W|| = {ws:.1f} ---")

    # Without trace penalty
    K, q_val, Pv, cond_P, status = solve_lmi(Wv, penalize_trace=False)
    if K is not None:
        print(f"  No penalty:  ||K|| = {np.linalg.norm(K):.1f},  q = {q_val:.2f},  "
              f"cond(P) = {cond_P:.1e},  status = {status}")
        # Check eigenvalues of P
        eigP = np.sort(np.linalg.eigvalsh(Pv))
        print(f"  eig(P): min={eigP[0]:.2e}, max={eigP[-1]:.2e}")

        # Check the Schur complement residual
        PAcl = Pv @ A - Pv @ K @ C
        M11 = PAcl + PAcl.T + q_val * np.eye(n)
        lv_val = np.diag(np.linalg.solve(Pv, Pv))  # not quite right, need to extract lv
        # Actually let's recompute with the LMI variables
    else:
        print(f"  No penalty:  INFEASIBLE ({status})")

    # With trace penalty — sweep over weights
    for tw in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        K2, q2, Pv2, cond_P2, status2 = solve_lmi(Wv, penalize_trace=True, trace_weight=tw)
        if K2 is not None:
            print(f"  trace_pen={tw:.0e}: ||K|| = {np.linalg.norm(K2):.1f},  q = {q2:.2f},  "
                  f"cond(P) = {cond_P2:.1e}")
            if abs(q2 - Q_CAP) < 0.01:
                # We are at the same q=10, so this is a valid comparison
                pass

# ── Now solve the actual LMI and extract all variables ───────────────────────
print("\n" + "=" * 80)
print("FULL LMI VARIABLE EXTRACTION at s=0.5")
print("=" * 80)

Wv = 0.5 * W0
P  = cp.Variable((n, n), symmetric=True)
R1 = cp.Variable((n, ny))
lv = cp.Variable(n, nonneg=True)
q  = cp.Variable(nonneg=True)

PAcl = P @ A - R1 @ C
PW_eff = P @ Wv

M11 = PAcl + PAcl.T + q * np.eye(n)
M12 = PW_eff + cp.diag(lv)
M22 = -EPS_REG * np.eye(n)
M = cp.bmat([[M11, M12], [M12.T, M22]])

cons = [M << 0, P >> EPS_P * np.eye(n), lv >= EPS_LV, q >= 0, q <= Q_CAP]
prob = cp.Problem(cp.Maximize(q), cons)
prob.solve(solver=cp.CLARABEL, verbose=False)

Pv = P.value
Kv = np.linalg.solve(Pv, R1.value)
lv_val = lv.value
q_val = q.value

print(f"  q = {q_val:.4f}")
print(f"  ||K|| = {np.linalg.norm(Kv):.1f}")
print(f"  ||lv|| = {np.linalg.norm(lv_val):.4f}")
print(f"  cond(P) = {np.linalg.cond(Pv):.1e}")

# Check the (1,2) block norm
M12_val = Pv @ Wv + np.diag(lv_val)
print(f"  ||P@W + Lambda|| = {np.linalg.norm(M12_val):.4f}")
print(f"  ||P@W|| = {np.linalg.norm(Pv @ Wv):.4f}")

# Check that the LMI is actually satisfied
M_full = np.block([
    [Pv@A - Pv@Kv@C + (Pv@A - Pv@Kv@C).T + q_val*np.eye(n), Pv@Wv + np.diag(lv_val)],
    [(Pv@Wv + np.diag(lv_val)).T, -EPS_REG*np.eye(n)]
])
eigM = np.sort(np.linalg.eigvalsh(M_full))
print(f"  max eig(LMI) = {eigM[-1]:.6e} (should be < 0)")

# Now re-solve with a minimal-trace P
print("\n--- With trace penalty at s=0.5 ---")
for tw in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    prob2 = cp.Problem(cp.Maximize(q - tw * cp.trace(P)), cons)
    prob2.solve(solver=cp.CLARABEL, verbose=False)
    if prob2.status in ("optimal", "optimal_inaccurate"):
        Pv2 = P.value
        Kv2 = np.linalg.solve(Pv2, R1.value)
        q2 = q.value
        print(f"  trace_pen={tw:.0e}: ||K|| = {np.linalg.norm(Kv2):.1f},  q = {q2:.4f},  "
              f"cond(P) = {np.linalg.cond(Pv2):.1e}")
    else:
        print(f"  trace_pen={tw:.0e}: FAILED ({prob2.status})")

print("\nDone.")
