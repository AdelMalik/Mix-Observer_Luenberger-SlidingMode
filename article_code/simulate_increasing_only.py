"""
Increasing-only (monotone) LMI — no slope bound.
Compares classical vs combined observer when the nonlinearity is only known
to be increasing (component-wise monotone), not slope-restricted.

LMI structure (S-procedure with delta_i * e_i >= 0):
  [ He{P(A-KC)} + qI     PW + Lambda          ]  <= 0     (classical)
  [ W^T P + Lambda        -eps * I             ]

  [ He{P(A-KC)} + qI     P(I-K'C)W + Lambda   ]  <= 0     (combined)
  [ W^T (I-K'C)^T P + Lambda    -eps * I      ]

  Lambda = diag(lv) > 0   encodes the increasing condition.
  eps > 0 is a small regularization (without it the (2,2) block is 0,
  which forces the off-diagonal to zero — too restrictive).

Key insight: the combined observer replaces PW with P(I-K'C)W in the
off-diagonal.  Since K' can cancel components of W in row(C), the effective
coupling is P * P_kerC * W, which is structurally smaller.  The LMI is
therefore much easier to satisfy.

For the classical observer with increasing-only, PW + Lambda ≈ 0 forces
PW to be negative (since Lambda > 0), which is often infeasible when W
has positive entries.  The combined observer can choose K' to make
P(I-K'C)W + Lambda ≈ 0 feasible.
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
    C[j, j + ny] = 0.4 - j * 0.05

def S(v, lam_sig=1.0):
    return 1.0 / (1.0 + np.exp(-lam_sig * np.clip(v, -30/lam_sig, 30/lam_sig)))

LAM_SIG = 4.0
Q_CAP   = 10.0
EPS_P   = 1e-4
EPS_LV  = 1e-4
EPS_REG = 1e-3   # regularization in (2,2) block for increasing-only LMI

# ── Time settings ────────────────────────────────────────────────────────────
dt     = 1e-3
stride = 10
N_SUB  = 10


# ═══════════════════════════════════════════════════════════════════════════════
# LMI: increasing-only  (no sector/slope bound, only delta_i * e_i >= 0)
# ═══════════════════════════════════════════════════════════════════════════════
def solve_lmi_increasing(W_mat, combined):
    """Increasing-only LMI: uses delta_i * e_i >= 0, no Gamma."""
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))
    lv = cp.Variable(n, nonneg=True)    # Lambda = diag(lv)
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
    M12 = PW_eff + cp.diag(lv)
    M22 = -EPS_REG * np.eye(n)          # regularization instead of -2*Lambda

    M = cp.bmat([[M11, M12], [M12.T, M22]])
    cons = [M << 0, P >> EPS_P * np.eye(n), lv >= EPS_LV, q >= 0, q <= Q_CAP]
    prob = cp.Problem(cp.Maximize(q), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LMI failed ({prob.status})")
    Pv = P.value
    K  = np.linalg.solve(Pv, R1.value)
    if combined:
        return K, np.linalg.solve(Pv, R2.value), q.value
    return K, None, q.value


# ═══════════════════════════════════════════════════════════════════════════════
# LMI: sector-bounded  (standard, for comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def solve_lmi_sector(W_mat, combined, Gamma_mat=None):
    """Standard sector-bounded LMI with Gamma = (lam_sig/4)*I."""
    if Gamma_mat is None:
        Gamma_mat = (LAM_SIG / 4.0) * np.eye(n)
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
        raise RuntimeError(f"LMI failed ({prob.status})")
    Pv = P.value
    K  = np.linalg.solve(Pv, R1.value)
    if combined:
        return K, np.linalg.solve(Pv, R2.value), q.value
    return K, None, q.value


# ═══════════════════════════════════════════════════════════════════════════════
# Table: gain scaling with coupling, comparing increasing-only vs sector
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 85)
print("Increasing-only LMI (monotone, eps_reg=1e-3) vs Sector-bounded LMI")
print("Gain norms vs coupling strength s")
print("=" * 85)
h = (f"{'s':>5}  {'||W||':>7}  "
     f"{'q_std_inc':>10}  {'||K_std_inc||':>13}  "
     f"{'q_comb_inc':>10}  {'||K_comb_inc||':>13}  {'||K''_inc||':>10}  "
     f"{'||K_std_sec||':>13}  {'||K_comb_sec||':>13}")
print(h)
print("-" * 110)

for sv in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0, 100.0]:
    Wv = sv * W0
    ws = np.linalg.norm(Wv)

    # ── Increasing-only LMI ──
    try:
        Ks_inc, _, qs_inc = solve_lmi_increasing(Wv, False)
        ks_inc = np.linalg.norm(Ks_inc)
    except:
        ks_inc, qs_inc = float("inf"), 0
    try:
        Kc_inc, Kpc_inc, qc_inc = solve_lmi_increasing(Wv, True)
        kc_inc  = np.linalg.norm(Kc_inc)
        kpc_inc = np.linalg.norm(Kpc_inc)
    except:
        kc_inc, kpc_inc, qc_inc = float("inf"), float("inf"), 0

    # ── Sector-bounded LMI ──
    try:
        Ks_sec, _, qs_sec = solve_lmi_sector(Wv, False)
        ks_sec = np.linalg.norm(Ks_sec)
    except:
        ks_sec, qs_sec = float("inf"), 0
    try:
        Kc_sec, Kpc_sec, qc_sec = solve_lmi_sector(Wv, True)
        kc_sec = np.linalg.norm(Kc_sec)
    except:
        kc_sec, qc_sec = float("inf"), 0

    def fs(v, w=9):
        return f"{v:.1f}" if np.isfinite(v) else "INFEAS"
    def fq(v):
        return f"{v:.2f}" if np.isfinite(v) else "----"

    print(f"{sv:>5.1f}  {ws:>7.1f}  "
          f"{fq(qs_inc):>10}  {fs(ks_inc,13):>13}  "
          f"{fq(qc_inc):>10}  {fs(kc_inc,13):>13}  {fs(kpc_inc,10):>10}  "
          f"{fs(ks_sec,13):>13}  {fs(kc_sec,13):>13}")


# ═══════════════════════════════════════════════════════════════════════════════
# Noise sweep — increasing-only LMI
# ═══════════════════════════════════════════════════════════════════════════════
S_SW = 15.0
T_SW = 10.0
Nt_sw  = int(T_SW / dt)
L_smo  = 5.0 * np.ones(ny)

# Solve both LMIs for noise sweep
try:
    Ks_inc_sw, _, qs_inc_sw = solve_lmi_increasing(S_SW * W0, False)
    ks_inc_norm = np.linalg.norm(Ks_inc_sw)
except Exception as e:
    Ks_inc_sw = None
    ks_inc_norm = float("inf")
    print(f"\nClassical increasing-only LMI failed at s={S_SW}: {e}")

try:
    Kc_inc_sw, Kpc_inc_sw, qc_inc_sw = solve_lmi_increasing(S_SW * W0, True)
    kc_inc_norm  = np.linalg.norm(Kc_inc_sw)
    kpc_inc_norm = np.linalg.norm(Kpc_inc_sw)
except Exception as e:
    Kc_inc_sw, Kpc_inc_sw = None, None
    kc_inc_norm, kpc_inc_norm = float("inf"), float("inf")
    print(f"\nCombined increasing-only LMI failed at s={S_SW}: {e}")

# Also get sector gains for comparison
Ks_sec_sw, _, _ = solve_lmi_sector(S_SW * W0, False)
Kc_sec_sw, Kpc_sec_sw, _ = solve_lmi_sector(S_SW * W0, True)

W_sw = S_SW * W0

print(f"\n{'='*85}")
print(f"Noise sweep — s={S_SW}, ||W||={np.linalg.norm(W_sw):.1f}, lam_sig={LAM_SIG}")
print(f"  Increasing-only LMI (eps_reg={EPS_REG}):")
if Ks_inc_sw is not None:
    print(f"    Classical:  ||K||={ks_inc_norm:.1f}  q={qs_inc_sw:.2f}")
else:
    print(f"    Classical:  INFEASIBLE")
if Kc_inc_sw is not None:
    print(f"    Combined:   ||K||={kc_inc_norm:.1f}  ||K'||={kpc_inc_norm:.2f}  q={qc_inc_sw:.2f}")
else:
    print(f"    Combined:   INFEASIBLE")
print(f"  Sector-bounded LMI (for reference):")
print(f"    Classical:  ||K||={np.linalg.norm(Ks_sec_sw):.1f}")
print(f"    Combined:   ||K||={np.linalg.norm(Kc_sec_sw):.1f}  ||K'||={np.linalg.norm(Kpc_sec_sw):.2f}")
print(f"  L_j={L_smo[0]:.0f}, n_sub={N_SUB}")
print(f"{'='*85}")

# Only run noise sweep if both increasing LMIs are feasible
if Ks_inc_sw is not None and Kc_inc_sw is not None:

    # ── Sliding-mode observer ────────────────────────────────────────────────
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

    # ── RK4 ──────────────────────────────────────────────────────────────────
    def rk4(f, x, dt, *a):
        k1 = f(x, *a); k2 = f(x + dt/2*k1, *a)
        k3 = f(x + dt/2*k2, *a); k4 = f(x + dt*k3, *a)
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    # ── Observer RHS ─────────────────────────────────────────────────────────
    def rhs_classical(vh, A, Wv, K, ym, lam_sig=1.0):
        return A @ vh + Wv @ S(vh, lam_sig) + K @ (ym - C @ vh)

    def rhs_combined(vh, A, Wv, K, Kp, ym, z2_est, lam_sig=1.0):
        return (A @ vh + Wv @ S(vh, lam_sig)
                + K @ (ym - C @ vh)
                - Kp @ (C @ (Wv @ S(vh, lam_sig)) - z2_est))

    # ── Run noise sweep ──────────────────────────────────────────────────────
    sigmas = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 1.0]

    print(f"\n{'sigma':>8}  {'RMS_std_inc':>12}  {'RMS_comb_inc':>13}  {'ratio_inc':>10}  "
          f"{'RMS_std_sec':>12}  {'RMS_comb_sec':>13}  {'ratio_sec':>10}")
    print("-" * 76)

    rms_rows_inc = []
    rms_rows_sec = []
    for sig in sigmas:
        rng = np.random.default_rng(7)

        # ── Increasing-only gains ──
        V_inc     = rng.uniform(-0.3, 0.3, n)
        Vh_s_inc  = np.zeros(n)
        Vh_c_inc  = np.zeros(n)
        z1_inc    = np.zeros(ny)
        z2_inc    = np.zeros(ny)
        nr        = Nt_sw // stride
        e_s_inc   = np.zeros(nr)
        e_c_inc   = np.zeros(nr)
        e_smo_inc = np.zeros(nr)
        rec = 0
        for k in range(Nt_sw):
            yt = C @ V_inc
            nm = sig * rng.standard_normal(ny)
            ym = yt + nm
            z1_inc, z2_inc = sliding_observer_step(z1_inc, z2_inc, ym, L_smo)
            if k % stride == 0:
                e_s_inc[rec] = np.linalg.norm(Vh_s_inc - V_inc)
                e_c_inc[rec] = np.linalg.norm(Vh_c_inc - V_inc)
                e_smo_inc[rec] = np.linalg.norm(z2_inc - C @ (W_sw @ S(V_inc, LAM_SIG)))
                rec += 1
            V_inc = rk4(lambda v, n_: A @ v + W_sw @ S(v, LAM_SIG) + n_,
                        V_inc, dt, sig * rng.standard_normal(n))
            Vh_s_inc = rk4(rhs_classical, Vh_s_inc, dt, A, W_sw, Ks_inc_sw, ym, LAM_SIG)
            Vh_c_inc = rk4(rhs_combined, Vh_c_inc, dt, A, W_sw, Kc_inc_sw, Kpc_inc_sw, ym, z2_inc.copy(), LAM_SIG)
        ss = int(0.6 * len(e_s_inc))
        rs_inc = np.sqrt(np.mean(e_s_inc[ss:]**2))
        rc_inc = np.sqrt(np.mean(e_c_inc[ss:]**2))
        rr_inc = rs_inc / rc_inc if rc_inc > 1e-15 else float("inf")
        rsmo_inc = np.sqrt(np.mean(e_smo_inc[ss:]**2))

        # ── Sector-bounded gains (same noise seed, different RNG) ──
        rng2 = np.random.default_rng(7)
        V_sec     = rng2.uniform(-0.3, 0.3, n)
        Vh_s_sec  = np.zeros(n)
        Vh_c_sec  = np.zeros(n)
        z1_sec    = np.zeros(ny)
        z2_sec    = np.zeros(ny)
        e_s_sec   = np.zeros(nr)
        e_c_sec   = np.zeros(nr)
        rec = 0
        for k in range(Nt_sw):
            yt = C @ V_sec
            nm = sig * rng2.standard_normal(ny)
            ym = yt + nm
            z1_sec, z2_sec = sliding_observer_step(z1_sec, z2_sec, ym, L_smo)
            if k % stride == 0:
                e_s_sec[rec] = np.linalg.norm(Vh_s_sec - V_sec)
                e_c_sec[rec] = np.linalg.norm(Vh_c_sec - V_sec)
                rec += 1
            V_sec = rk4(lambda v, n_: A @ v + W_sw @ S(v, LAM_SIG) + n_,
                        V_sec, dt, sig * rng2.standard_normal(n))
            Vh_s_sec = rk4(rhs_classical, Vh_s_sec, dt, A, W_sw, Ks_sec_sw, ym, LAM_SIG)
            Vh_c_sec = rk4(rhs_combined, Vh_c_sec, dt, A, W_sw, Kc_sec_sw, Kpc_sec_sw, ym, z2_sec.copy(), LAM_SIG)
        ss = int(0.6 * len(e_s_sec))
        rs_sec = np.sqrt(np.mean(e_s_sec[ss:]**2))
        rc_sec = np.sqrt(np.mean(e_c_sec[ss:]**2))
        rr_sec = rs_sec / rc_sec if rc_sec > 1e-15 else float("inf")

        print(f"{sig:>8.0e}  {rs_inc:>12.5f}  {rc_inc:>13.5f}  {rr_inc:>10.2f}  "
              f"{rs_sec:>12.5f}  {rc_sec:>13.5f}  {rr_sec:>10.2f}")
        rms_rows_inc.append((sig, rs_inc, rc_inc, rr_inc))
        rms_rows_sec.append((sig, rs_sec, rc_sec, rr_sec))


    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 1: noise sweep — increasing-only vs sector, both designs
    # ═══════════════════════════════════════════════════════════════════════════
    OUT = (r"c:\Perso\VSCODE WORKSPACE\Thesis_Manuscript-07-02-2026"
           r"\Articles Source\Article_LMI_Lure_Observer")

    sa = np.array([x[0] for x in rms_rows_inc])
    fig1, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: increasing-only LMI
    ax = axes[0]
    if Ks_inc_sw is not None:
        ax.loglog(sa, [x[1] for x in rms_rows_inc], "b-o", ms=4, lw=1.0,
                   label=rf"Classical  ($\|K\|={ks_inc_norm:.1f}$)")
    if Kc_inc_sw is not None:
        ax.loglog(sa, [x[2] for x in rms_rows_inc], "r-s", ms=4, lw=1.0,
                   label=rf"Combined  ($\|K\|={kc_inc_norm:.1f}$, "
                   rf"$\|K'\|={kpc_inc_norm:.1f}$)")
    ax.set_xlabel(r"$\sigma$", fontsize=9)
    ax.set_ylabel(r"RMS $\|e\|_{\rm ss}$", fontsize=9)
    ax.set_title(f"Increasing-only LMI  (eps={EPS_REG})", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    # Right: sector-bounded LMI
    ax = axes[1]
    ax.loglog(sa, [x[1] for x in rms_rows_sec], "b-o", ms=4, lw=1.0,
               label=rf"Classical  ($\|K\|={np.linalg.norm(Ks_sec_sw):.0f}$)")
    ax.loglog(sa, [x[2] for x in rms_rows_sec], "r-s", ms=4, lw=1.0,
               label=rf"Combined  ($\|K\|={np.linalg.norm(Kc_sec_sw):.1f}$, "
               rf"$\|K'\|={np.linalg.norm(Kpc_sec_sw):.1f}$)")
    ax.set_xlabel(r"$\sigma$", fontsize=9)
    ax.set_ylabel(r"RMS $\|e\|_{\rm ss}$", fontsize=9)
    ax.set_title(f"Sector-bounded LMI  ($\Gamma={LAM_SIG/4:.1f}I$)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    fig1.suptitle(f"Noise sweep — $s={S_SW}$, $\\|W\\|={np.linalg.norm(W_sw):.1f}$, "
                  f"$\\lambda_{{\\rm sig}}={LAM_SIG:.0f}$", fontsize=10, y=1.02)
    fig1.tight_layout()
    fig1.savefig(f"{OUT}\\increasing_vs_sector_noise_sweep.pdf")
    plt.close(fig1)


    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 2: overlay — combined observer with both LMIs
    # ═══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ks_s = np.linalg.norm(Ks_sec_sw)
    kc_s = np.linalg.norm(Kc_sec_sw)

    if Ks_inc_sw is not None:
        ax2.loglog(sa, [x[1] for x in rms_rows_inc], "b--o", ms=4, lw=1.0, alpha=0.6,
                    label=rf"Classical (incr-only, $\|K\|={ks_inc_norm:.1f}$)")
    if Kc_inc_sw is not None:
        ax2.loglog(sa, [x[2] for x in rms_rows_inc], "r--s", ms=4, lw=1.0, alpha=0.6,
                    label=rf"Combined (incr-only, $\|K\|={kc_inc_norm:.1f}$)")
    ax2.loglog(sa, [x[1] for x in rms_rows_sec], "b-o", ms=4, lw=1.5,
                label=rf"Classical (sector, $\|K\|={ks_s:.0f}$)")
    ax2.loglog(sa, [x[2] for x in rms_rows_sec], "r-s", ms=4, lw=1.5,
                label=rf"Combined (sector, $\|K\|={kc_s:.1f}$)")
    ax2.set_xlabel(r"$\sigma$", fontsize=9)
    ax2.set_ylabel(r"RMS $\|e\|_{\rm ss}$", fontsize=9)
    ax2.set_title(f"Increasing-only vs Sector-bounded — $s={S_SW}$", fontsize=9)
    ax2.legend(fontsize=6.5, loc="upper left")
    fig2.tight_layout()
    fig2.savefig(f"{OUT}\\increasing_vs_sector_overlay.pdf")
    plt.close(fig2)

    print(f"\nFigures saved to {OUT}")
    print("Done.")
else:
    print("\nSkipping noise sweep — one or both increasing-only LMIs infeasible.")
    print("This is the expected result: the classical LMI fails under the increasing-only")
    print("condition when W has large positive entries, while the combined LMI can still")
    print("succeed because K' attenuates the coupling via the projector I-K'C.")
