"""
Full closed-loop test: differentiator + observer with steep sigmoid.
n=2, ny=1, WC-type system.
Varies α (sigmoid steepness), solves LMI for both designs,
then simulates the full architecture (differentiator running in parallel).
"""

import numpy as np
import cvxpy as cp

# ── System (n=2, ny=1) ──────────────────────────────────────────────────────
n  = 2
ny = 1
lam = 1.0
A   = -lam * np.eye(n)
w   = np.ones(n) / np.sqrt(n)
W0  = np.outer(w, w)
C   = np.array([[1.0, 0.0]])
r   = 20.0       # coupling strength
Wv  = r * W0

def S(v, alpha):
    return 1.0 / (1.0 + np.exp(-alpha * np.clip(v, -30/alpha, 30/alpha)))

Q_CAP  = 10.0
EPS_P  = 1e-4
EPS_LV = 1e-4

# ── LMI solver ───────────────────────────────────────────────────────────────
def solve_lmi(W_mat, alpha, combined):
    gamma = alpha / 4.0
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


# ── Simulation parameters ────────────────────────────────────────────────────
dt      = 1e-3
T_end   = 15.0
Nt      = int(T_end / dt)
stride  = 10
N_SUB   = 10
L_diff  = 5.0 * np.ones(ny)
sigma   = 0.01           # measurement noise

# ── Differentiator (WC form: A = -λI, reconstructs CWS(V)) ──────────────────
def diff_substep(z1, z2, ym, L, dt_s, alpha):
    z1n = z1.copy(); z2n = z2.copy()
    for j in range(ny):
        e1 = ym[j] - z1n[j]
        z1n[j] += dt_s * (-lam * z1n[j] + z2n[j]
                          + L[j] * np.sign(e1) * np.sqrt(abs(e1)))
        e1_n = ym[j] - z1n[j]
        z2n[j] += dt_s * (L[j] * L[j] * np.sign(e1_n))
    return z1n, z2n

def diff_step(z1, z2, ym, L, alpha, n_sub=N_SUB):
    dt_s = dt / n_sub
    for _ in range(n_sub):
        z1, z2 = diff_substep(z1, z2, ym, L, dt_s, alpha)
    return z1, z2

def rk4(f, x, dt, *a):
    k1 = f(x, *a); k2 = f(x + dt/2*k1, *a)
    k3 = f(x + dt/2*k2, *a); k4 = f(x + dt*k3, *a)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def rhs_classical(vh, K, ym, alpha):
    return A @ vh + Wv @ S(vh, alpha) + K @ (ym - C @ vh)

def rhs_combined(vh, K, Kp, ym, z2_est, alpha):
    return (A @ vh + Wv @ S(vh, alpha)
            + K @ (ym - C @ vh)
            - Kp @ (C @ (Wv @ S(vh, alpha)) - z2_est))


# ── Run ──────────────────────────────────────────────────────────────────────
alpha_vals = [1, 4, 16]

print("=" * 80)
print(f"  Full closed-loop test: n=2, ny=1, r={r}, ||W||={np.linalg.norm(Wv):.1f}")
print(f"  differentiator L={L_diff[0]:.0f}, σ={sigma}, T={T_end}s")
print("=" * 80)

for alpha in alpha_vals:
    gamma = alpha / 4.0
    print(f"\n── α = {alpha}  (γ = {gamma:.2f}) ──")

    Ks, _, qs = solve_lmi(Wv, alpha, False)
    Kc, Kpc, qc = solve_lmi(Wv, alpha, True)
    print(f"  Classical:        ||K|| = {np.linalg.norm(Ks):.1f},  q = {qs:.2f}")
    print(f"  Combined:  ||K|| = {np.linalg.norm(Kc):.1f},  ||K'|| = {np.linalg.norm(Kpc):.2f},  q = {qc:.2f}")

    # Time-domain simulation with differentiator running in parallel
    rng = np.random.default_rng(42)
    V     = rng.uniform(-0.3, 0.3, n)
    Vh_s  = np.zeros(n)
    Vh_c  = np.zeros(n)
    z1    = np.zeros(ny)
    z2    = np.zeros(ny)
    Nr    = Nt // stride
    t_r   = np.zeros(Nr)
    e_s   = np.zeros(Nr)
    e_c   = np.zeros(Nr)
    y2_true = np.zeros(Nr)
    z2_rec  = np.zeros(Nr)
    rec = 0

    for k in range(Nt):
        yt = C @ V
        nm = sigma * rng.standard_normal(ny)
        ym = yt + nm

        # Run differentiator on noisy output
        z1, z2 = diff_step(z1, z2, ym, L_diff, alpha)

        if k % stride == 0:
            t_r[rec]   = k * dt
            e_s[rec]   = np.linalg.norm(Vh_s - V)
            e_c[rec]   = np.linalg.norm(Vh_c - V)
            y2_true[rec] = (C @ (Wv @ S(V, alpha)))[0]
            z2_rec[rec]  = z2[0]
            rec += 1

        # Advance plant
        V = rk4(lambda v, n_: A @ v + Wv @ S(v, alpha) + n_,
                V, dt, sigma * rng.standard_normal(n))
        # Advance observers
        Vh_s = rk4(rhs_classical, Vh_s, dt, Ks, ym, alpha)
        Vh_c = rk4(rhs_combined, Vh_c, dt, Kc, Kpc, ym, z2.copy(), alpha)

    # Steady-state RMS (last 40%)
    ss = int(0.6 * len(e_s))
    rms_s = np.sqrt(np.mean(e_s[ss:]**2))
    rms_c = np.sqrt(np.mean(e_c[ss:]**2))

    # Differentiator tracking error (last 40%)
    diff_rms = np.sqrt(np.mean((y2_true[ss:] - z2_rec[ss:])**2))

    print(f"  RMS ||e|| (steady):  classical = {rms_s:.4f},  combined = {rms_c:.4f},  ratio = {rms_s/rms_c:.2f}")
    print(f"  RMS diff error (z₂ - y₂): {diff_rms:.4f}")

print("\nDone.")
