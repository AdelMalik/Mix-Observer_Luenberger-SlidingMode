"""
Case 2 — Hoelder nonlinearity + sliding-mode bank.
Requires reconstruction of CWS(V) via homogeneous differentiator.
S(xi) = |xi|^{1/2} sign(xi) / (1 + |xi|^{1/2})  — no sector bound.

LMI: increasing-only (eps-regularized), WC form (A = -lambda*I).

Observer:  dV̂/dt = A V̂ + W S(V̂) + K(y - C V̂) - K'(C W S(V̂) - ẑ₂)
           with ẑ₂ reconstructed from y via sliding-mode bank.

Result: classical increasing LMI → INFEAS or explosive gains.
        combined increasing LMI → flat, moderate gains (||K'|| ~ 2.3).
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Model (WC: A = -λI, n=6, ny=3) ───────────────────────────────────────────
n   = 6
ny  = 3
lam = 1.0
A   = -lam * np.eye(n)

C = np.zeros((ny, n))
for j in range(ny):
    C[j, j]      = 0.6 + j * 0.1
    C[j, j + ny] = 0.4 - j * 0.05

W0 = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0 = W0 / np.linalg.norm(W0) * 4.4

# ── Hölder nonlinearity ──────────────────────────────────────────────────────
ALPHA = 0.5
def S(v):
    av = np.abs(v); sv = np.sign(v)
    p  = av ** ALPHA
    return sv * p / (1.0 + p)

# ── LMI parameters ───────────────────────────────────────────────────────────
Q_CAP   = 10.0
EPS_P   = 1e-4
EPS_LV  = 1e-4
EPS_REG = 1e-3

dt     = 1e-3
stride = 10
N_SUB  = 10


# ═══════════════════════════════════════════════════════════════════════════════
# LMI: increasing-only, WC form — P(I-K'C)W in (1,2)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_lmi(W_mat, combined):
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))
    lv = cp.Variable(n, nonneg=True)
    q  = cp.Variable(nonneg=True)
    PAcl = P @ A - R1 @ C
    if combined:
        R2     = cp.Variable((n, ny))
        PW_eff = P @ W_mat - R2 @ (C @ W_mat)
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
        return K, np.linalg.solve(Pv, R2.value), q.value
    return K, None, q.value


# ═══════════════════════════════════════════════════════════════════════════════
# Table — Gain scaling
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Case 2 — Hoelder S + sliding-mode bank")
print("Increasing-only LMI  (no Gamma)")
print("=" * 70)
h = f"{'s':>5}  {'||W||':>7}  {'classical':>12}  {'combined':>12}  {'||Kp||':>8}"
print(h); print("-" * 55)

for sv in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0]:
    Wv = sv * W0; ws = np.linalg.norm(Wv)
    try:
        Ks, _, _ = solve_lmi(Wv, False); ks = np.linalg.norm(Ks)
        ks_s = f"{ks:.1f}"
    except:
        ks_s = "INFEAS"
    try:
        Kc, Kpc, _ = solve_lmi(Wv, True)
        kc_s  = f"{np.linalg.norm(Kc):.1f}"
        kpc_s = f"{np.linalg.norm(Kpc):.1f}"
    except:
        kc_s, kpc_s = "INFEAS", "---"
    print(f"{sv:>5.1f}  {ws:>7.1f}  {ks_s:>12}  {kc_s:>12}  {kpc_s:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# Noise sweep
# ═══════════════════════════════════════════════════════════════════════════════
S_SW  = 10.0
T_SW  = 10.0
Nt_sw = int(T_SW / dt)
W_sw  = S_SW * W0
L_smo = 3.0 * np.ones(ny)

try:
    Ks, _, _ = solve_lmi(W_sw, False)
    ks_ok = True; ks_norm = np.linalg.norm(Ks)
except:
    ks_ok = False
try:
    Kc, Kpc, _ = solve_lmi(W_sw, True)
    kc_ok = True; kc_norm = np.linalg.norm(Kc); kpc_norm = np.linalg.norm(Kpc)
except:
    kc_ok = False

print(f"\n{'='*70}")
print(f"Noise sweep — s={S_SW}, ||W||={np.linalg.norm(W_sw):.1f}, L_j={L_smo[0]:.0f}")
if ks_ok:
    print(f"  Classical:  ||K||={ks_norm:.0f}")
else:
    print(f"  Classical:  INFEASIBLE")
if kc_ok:
    print(f"  Combined:   ||K||={kc_norm:.1f}  ||K'||={kpc_norm:.1f}")
else:
    print(f"  Combined:   INFEASIBLE")
print(f"{'='*70}")

# ── Sliding-mode observer ────────────────────────────────────────────────────
K1 = 1.5; K2 = 1.1

def sliding_substep(z1, z2, ym, L, dt_s):
    z1n = z1.copy(); z2n = z2.copy()
    for j in range(ny):
        e1 = ym[j] - z1n[j]
        z1n[j] += dt_s * (-lam * z1n[j] + z2n[j]
                          + L[j] * K1 * np.sign(e1) * np.sqrt(abs(e1)))
        z2n[j] += dt_s * (L[j] * L[j] * K2 * np.sign(e1))
    return z1n, z2n

def sliding_step(z1, z2, ym, L):
    dt_s = dt / N_SUB
    for _ in range(N_SUB):
        z1, z2 = sliding_substep(z1, z2, ym, L, dt_s)
    return z1, z2

def rk4(f, x, dt, *a):
    k1 = f(x, *a); k2 = f(x + dt/2*k1, *a)
    k3 = f(x + dt/2*k2, *a); k4 = f(x + dt*k3, *a)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def rhs_classical(vh, K, ym):
    return A @ vh + W_sw @ S(vh) + K @ (ym - C @ vh)

def rhs_combined(vh, K, Kp, ym, z2_est):
    return (A @ vh + W_sw @ S(vh)
            + K @ (ym - C @ vh)
            - Kp @ (C @ (W_sw @ S(vh)) - z2_est))

sigmas = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 1.0]

if kc_ok:
    if ks_ok:
        print(f"\n{'sigma':>8}  {'RMS_cls':>10}  {'RMS_cmb':>12}  {'ratio':>8}  {'RMS_smo':>10}")
    else:
        print(f"\n  Classical INFEAS — showing combined only:")
        print(f"  {'sigma':>8}  {'RMS_cmb':>12}  {'RMS_smo':>10}")
    print("-" * (55 if ks_ok else 36))

    rows = []
    for sig in sigmas:
        rng = np.random.default_rng(7)
        V     = rng.uniform(-0.3, 0.3, n)
        Vh_s  = np.zeros(n) if ks_ok else None
        Vh_c  = np.zeros(n)
        z1    = np.zeros(ny); z2 = np.zeros(ny)
        nr    = Nt_sw // stride
        e_s   = np.zeros(nr) if ks_ok else None
        e_c   = np.zeros(nr); e_smo = np.zeros(nr)
        rec   = 0
        for k in range(Nt_sw):
            yt = C @ V
            nm = sig * rng.standard_normal(ny)
            ym = yt + nm
            z1, z2 = sliding_step(z1, z2, ym, L_smo)
            y2_true = C @ (W_sw @ S(V))
            if k % stride == 0:
                if ks_ok:
                    e_s[rec] = np.linalg.norm(Vh_s - V)
                e_c[rec]   = np.linalg.norm(Vh_c - V)
                e_smo[rec] = np.linalg.norm(z2 - y2_true)
                rec += 1
            V = rk4(lambda v, n_: A @ v + W_sw @ S(v) + n_,
                    V, dt, sig * rng.standard_normal(n))
            if ks_ok:
                Vh_s = rk4(rhs_classical, Vh_s, dt, Ks, ym)
            Vh_c = rk4(rhs_combined, Vh_c, dt, Kc, Kpc, ym, z2.copy())
        ss  = int(0.6 * len(e_c))
        rc  = np.sqrt(np.mean(e_c[ss:]**2))
        rsm = np.sqrt(np.mean(e_smo[ss:]**2))
        if ks_ok:
            rs = np.sqrt(np.mean(e_s[ss:]**2))
            r  = rs / rc if rc > 1e-15 else float("inf")
            print(f"{sig:>8.0e}  {rs:>10.5f}  {rc:>12.5f}  {r:>8.2f}  {rsm:>10.5f}")
        else:
            rs = float("nan"); r = float("nan")
            print(f"  {sig:>8.0e}  {rc:>12.5f}  {rsm:>10.5f}")
        rows.append((sig, rs, rc, r, rsm))

    # ── Figure ───────────────────────────────────────────────────────────────
    OUT = (r"c:\Perso\VSCODE WORKSPACE\Thesis_Manuscript-07-02-2026"
           r"\Articles Source\Article_LMI_Lure_Observer")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sa = np.array([r[0] for r in rows])
    if ks_ok:
        ax.loglog(sa, [r[1] for r in rows], "b-o", ms=4, lw=1.0,
                   label=rf"Classical ($\|K\|={ks_norm:.0f}$)")
    ax.loglog(sa, [r[2] for r in rows], "r-s", ms=4, lw=1.2,
               label=rf"Combined ($\|K\|={kc_norm:.1f}$, $\|K'\|={kpc_norm:.1f}$)")
    ax.set_xlabel(r"$\sigma$", fontsize=10)
    ax.set_ylabel(r"RMS $\|e\|_{\rm ss}$", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}\\holder_smo_noise_sweep.pdf")
    plt.close(fig)
    print(f"\nFigure saved to {OUT}\\holder_smo_noise_sweep.pdf")

    # ── Time-domain error trajectory ─────────────────────────────────────
    T_END = 8.0; Nt_td = int(T_END / dt)
    sig_td = 0.1
    rng = np.random.default_rng(42)
    V     = rng.uniform(-0.3, 0.3, n)
    Vh_s  = np.zeros(n); Vh_c = np.zeros(n)
    z1    = np.zeros(ny); z2 = np.zeros(ny)
    nr    = Nt_td // stride
    t     = np.zeros(nr); e_s = np.zeros(nr); e_c = np.zeros(nr)
    rec   = 0
    for k in range(Nt_td):
        yt = C @ V; ym = yt + sig_td * rng.standard_normal(ny)
        z1, z2 = sliding_step(z1, z2, ym, L_smo)
        if k % stride == 0:
            t[rec] = k * dt; e_s[rec] = np.linalg.norm(Vh_s - V)
            e_c[rec] = np.linalg.norm(Vh_c - V); rec += 1
        V = rk4(lambda v, n_: A @ v + W_sw @ S(v) + n_, V, dt,
                sig_td * rng.standard_normal(n))
        if ks_ok:
            Vh_s = rk4(rhs_classical, Vh_s, dt, Ks, ym)
        Vh_c = rk4(rhs_combined, Vh_c, dt, Kc, Kpc, ym, z2.copy())
    fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
    if ks_ok:
        ax2.semilogy(t, np.maximum(e_s, 1e-8), "b-", lw=0.8, alpha=0.6,
                      label=rf"Classical ($\|K\|={ks_norm:.0f}$)")
    ax2.semilogy(t, np.maximum(e_c, 1e-8), "r-", lw=1.2,
                  label=rf"Combined ($\|K\|={kc_norm:.1f}$, $\|K'\|={kpc_norm:.1f}$)")
    ax2.set_xlabel(r"$t$ (s)", fontsize=9)
    ax2.set_ylabel(r"$\|e(t)\|$", fontsize=9)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(f"{OUT}\\holder_error_trajectory.pdf")
    plt.close(fig2)
    print(f"Figure saved to {OUT}\\holder_error_trajectory.pdf")
else:
    print("\nCombined LMI infeasible — no noise sweep.")

print("Done.")
