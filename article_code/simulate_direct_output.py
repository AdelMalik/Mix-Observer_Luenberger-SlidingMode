"""
Case 1 — Direct output, non-sector-bounded nonlinearity.
C = [I_{ny}  0]  =>  CS(V) = S(CV) = S(y) directly.  No sliding-mode bank.

Nonlinearity: S(xi) = |xi|^{1/2} sign(xi) / (1 + |xi|^{1/2})
  — bounded in [0,1], strictly increasing, S'(0) = +inf — no sector bound.

LMI: increasing-only (eps-regularized), no Gamma needed.

Observer:  dV̂/dt = A V̂ + W S(V̂) + K(y - C V̂)  -  K'(C S(V̂) - S(y))

Result: classical LMI → INFEAS at moderate s.
        combined LMI → works with moderate gains.
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Model ────────────────────────────────────────────────────────────────────
n   = 6
ny  = 3

rng_a = np.random.default_rng(42)
A = rng_a.normal(0, 1, (n, n)) - 2.0 * np.eye(n)
print(f"max Re(eig(A)) = {np.max(np.real(np.linalg.eigvals(A))):.2f}")

C = np.zeros((ny, n))
C[:ny, :ny] = np.eye(ny)

W0 = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0 = W0 / np.linalg.norm(W0) * 4.4

# ── Hölder nonlinearity — no global slope bound ──────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# LMI: increasing-only — P(W - K'C) in (1,2), eps-regularized (2,2)
# ═══════════════════════════════════════════════════════════════════════════════

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
        return K, np.linalg.solve(Pv, R2.value), q.value
    return K, None, q.value


# ═══════════════════════════════════════════════════════════════════════════════
# Table — Classical vs Combined, increasing-only LMI
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Case 1 — Direct output, Hoelder S (no sector bound)")
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
# Noise sweep — same noise seed for both observers
# ═══════════════════════════════════════════════════════════════════════════════
S_SW  = 10.0
T_SW  = 10.0
Nt_sw = int(T_SW / dt)
W_sw  = S_SW * W0

try:
    Ks, _, _ = solve_lmi(W_sw, False)
    ks_ok = True
    ks_norm = np.linalg.norm(Ks)
except:
    ks_ok = False
try:
    Kc, Kpc, _ = solve_lmi(W_sw, True)
    kc_ok = True
    kc_norm = np.linalg.norm(Kc); kpc_norm = np.linalg.norm(Kpc)
except:
    kc_ok = False

print(f"\n{'='*70}")
print(f"Noise sweep — s={S_SW}, ||W||={np.linalg.norm(W_sw):.1f}")
if ks_ok:
    print(f"  Classical:  ||K||={ks_norm:.0f}")
else:
    print(f"  Classical:  INFEASIBLE")
if kc_ok:
    print(f"  Combined:   ||K||={kc_norm:.1f}  ||K'||={kpc_norm:.1f}")
else:
    print(f"  Combined:   INFEASIBLE")
print(f"{'='*70}")

def rk4(f, x, dt, *a):
    k1 = f(x, *a); k2 = f(x + dt/2*k1, *a)
    k3 = f(x + dt/2*k2, *a); k4 = f(x + dt*k3, *a)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def rhs_classical(vh, K, ym):
    return A @ vh + W_sw @ S(vh) + K @ (ym - C @ vh)

def rhs_combined(vh, K, Kp, ym):
    Sy = S(ym)
    return (A @ vh + W_sw @ S(vh)
            + K @ (ym - C @ vh)
            - Kp @ (C @ S(vh) - Sy))

sigmas = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 1.0]

if kc_ok:
    if ks_ok:
        print(f"\n{'sigma':>8}  {'RMS_cls':>10}  {'RMS_cmb':>12}  {'ratio':>8}")
    else:
        print(f"\n  Classical INFEAS — showing combined only:")
        print(f"  {'sigma':>8}  {'RMS_cmb':>12}")
    print("-" * (44 if ks_ok else 24))

    rows = []
    for sig in sigmas:
        rng = np.random.default_rng(7)
        V     = rng.uniform(-0.3, 0.3, n)
        Vh_s  = np.zeros(n) if ks_ok else None
        Vh_c  = np.zeros(n)
        nr    = Nt_sw // stride
        e_s   = np.zeros(nr) if ks_ok else None
        e_c   = np.zeros(nr)
        rec   = 0
        for k in range(Nt_sw):
            yt = C @ V
            nm = sig * rng.standard_normal(ny)
            ym = yt + nm
            if k % stride == 0:
                if ks_ok:
                    e_s[rec] = np.linalg.norm(Vh_s - V)
                e_c[rec] = np.linalg.norm(Vh_c - V)
                rec += 1
            V = rk4(lambda v, n_: A @ v + W_sw @ S(v) + n_,
                    V, dt, sig * rng.standard_normal(n))
            if ks_ok:
                Vh_s = rk4(rhs_classical, Vh_s, dt, Ks, ym)
            Vh_c = rk4(rhs_combined, Vh_c, dt, Kc, Kpc, ym)
        ss  = int(0.6 * len(e_c))
        rc  = np.sqrt(np.mean(e_c[ss:]**2))
        if ks_ok:
            rs = np.sqrt(np.mean(e_s[ss:]**2))
            r  = rs / rc if rc > 1e-15 else float("inf")
            print(f"{sig:>8.0e}  {rs:>10.5f}  {rc:>12.5f}  {r:>8.2f}")
        else:
            rs = float("nan"); r = float("nan")
            print(f"  {sig:>8.0e}  {rc:>12.5f}")
        rows.append((sig, rs, rc, r))

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
    fig.savefig(f"{OUT}\\direct_output_holder_noise_sweep.pdf")
    plt.close(fig)
    print(f"\nFigure saved to {OUT}\\direct_output_holder_noise_sweep.pdf")

    # ── Time-domain error trajectory ─────────────────────────────────────
    T_END = 8.0; Nt_td = int(T_END / dt)
    sig_td = 0.1
    rng = np.random.default_rng(42)
    V    = rng.uniform(-0.3, 0.3, n)
    Vh_c = np.zeros(n)
    nr   = Nt_td // stride
    t    = np.zeros(nr); e_c = np.zeros(nr)
    rec  = 0
    for k in range(Nt_td):
        yt = C @ V; ym = yt + sig_td * rng.standard_normal(ny)
        if k % stride == 0:
            t[rec] = k * dt; e_c[rec] = np.linalg.norm(Vh_c - V); rec += 1
        V = rk4(lambda v, n_: A @ v + W_sw @ S(v) + n_, V, dt,
                sig_td * rng.standard_normal(n))
        Vh_c = rk4(rhs_combined, Vh_c, dt, Kc, Kpc, ym)
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    ax2.semilogy(t, np.maximum(e_c, 1e-8), "r-", lw=1.0,
                  label=rf"Combined ($\|K\|={kc_norm:.1f}$, $\|K'\|={kpc_norm:.1f}$)")
    ax2.set_xlabel(r"$t$ (s)", fontsize=9)
    ax2.set_ylabel(r"$\|e(t)\|$", fontsize=9)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(f"{OUT}\\direct_output_error_trajectory.pdf")
    plt.close(fig2)
    print(f"Figure saved to {OUT}\\direct_output_error_trajectory.pdf")
else:
    print("\nCombined LMI infeasible — no noise sweep.")

print("Done.")
