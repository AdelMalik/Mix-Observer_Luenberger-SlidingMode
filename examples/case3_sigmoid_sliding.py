"""
Case 3 — Wilson–Cowan neural mass, sigmoid nonlinearity + sliding-mode bank.

Reproduces the figures from the article *"Observer design for Lur'e systems
via injection of a reconstructed nonlinear output"*, Section 5.3.

Uses the lure_observer library — only the model definition is case-specific.
"""

import numpy as np
import matplotlib.pyplot as plt
from lure_observer import solve_lmi, is_aligned
from lure_observer import ClassicalObserver, CombinedObserver, SlidingModeBank

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Define the Lur'e model
# ═══════════════════════════════════════════════════════════════════════════════
n   = 6          # 3 excitatory + 3 inhibitory
ny  = 3
lam = 1.0
A   = -lam * np.eye(n)

# Coupling matrix (block E-I structure)
W0 = np.array([
    [ 1.20,  0.40,  0.25, -0.80, -0.20, -0.10],
    [ 0.40,  1.10,  0.30, -0.20, -0.70, -0.15],
    [ 0.25,  0.30,  0.90, -0.10, -0.15, -0.60],
    [ 0.90,  0.30,  0.20, -0.60, -0.10, -0.05],
    [ 0.30,  0.85,  0.25, -0.10, -0.55, -0.08],
    [ 0.20,  0.25,  0.80, -0.05, -0.08, -0.45],
])
W0 = W0 / np.linalg.norm(W0) * 4.4

# LFP-type measurement: each channel mixes one E and one I component
C = np.zeros((ny, n))
for j in range(ny):
    C[j, j]      = 0.6 + j * 0.1
    C[j, j + ny] = 0.4 - j * 0.05

# Sigmoid nonlinearity
LAM_SIG = 4.0

def S(v):
    return 1.0 / (1.0 + np.exp(-LAM_SIG * np.clip(v, -30 / LAM_SIG, 30 / LAM_SIG)))

# Sector bound: 0 ≤ S' ≤ LAM_SIG/4  →  Γ = (LAM_SIG/4) I
Gamma = (LAM_SIG / 4.0) * np.eye(n)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Check alignment and solve LMIs
# ═══════════════════════════════════════════════════════════════════════════════
print(f"CA = MC ?  {is_aligned(A, C)}  (expected True: A = -λI)")

print(f"\n{'s':>5}  {'||W||':>7}  {'||K_std||':>10}  {'||K_comb||':>10}  {'||K''||':>8}")
print("-" * 50)

for sv in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0, 100.0]:
    Ws = sv * W0
    try:
        _, Ks, _, _ = solve_lmi(A, Ws, C, mode="sector", combined=False,
                                Gamma=Gamma)
        ks = np.linalg.norm(Ks)
        ks_s = f"{ks:.1f}"
    except RuntimeError:
        ks_s = "INFEAS"
    try:
        _, Kc, Kpc, _ = solve_lmi(A, Ws, C, mode="sector", combined=True,
                                   Gamma=Gamma)
        kc_s  = f"{np.linalg.norm(Kc):.1f}"
        kpc_s = f"{np.linalg.norm(Kpc):.1f}"
    except RuntimeError:
        kc_s, kpc_s = "INFEAS", "---"
    print(f"{sv:>5.1f}  {np.linalg.norm(Ws):>7.1f}  {ks_s:>10}  {kc_s:>10}  {kpc_s:>8}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Noise sweep — compare classical vs combined
# ═══════════════════════════════════════════════════════════════════════════════
S_SW  = 15.0
W_sw  = S_SW * W0
dt    = 1e-3
T_sim = 10.0
Nt    = int(T_sim / dt)
L_smo = 3.0 * np.ones(ny)

# Solve LMIs at the chosen coupling
_, K_std, _, _   = solve_lmi(A, W_sw, C, mode="sector", combined=False, Gamma=Gamma)
_, K_cmb, Kp, _  = solve_lmi(A, W_sw, C, mode="sector", combined=True,  Gamma=Gamma)

print(f"\nNoise sweep — s={S_SW}, ||W||={np.linalg.norm(W_sw):.1f}")
print(f"  Classical:  ||K|| = {np.linalg.norm(K_std):.0f}")
print(f"  Combined:   ||K|| = {np.linalg.norm(K_cmb):.1f}   ||K'|| = {np.linalg.norm(Kp):.1f}")

# Build observers
smo     = SlidingModeBank(ny, L_smo, A=A, C=C, dt=dt, n_sub=10)
obs_std = ClassicalObserver(A, W_sw, None, C, K_std, S)
obs_cmb = CombinedObserver(A, W_sw, None, C, K_cmb, Kp, S, aligned=True)


def plant_rhs(v, noise):
    return A @ v + W_sw @ S(v) + noise


def rk4_step(v, noise, dt):
    k1 = plant_rhs(v, noise)
    k2 = plant_rhs(v + dt / 2 * k1, noise)
    k3 = plant_rhs(v + dt / 2 * k2, noise)
    k4 = plant_rhs(v + dt * k3, noise)
    return v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


sigmas = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 1.0]
stride = 10

print(f"\n{'sigma':>8}  {'RMS_cls':>10}  {'RMS_cmb':>10}  {'ratio':>8}")
print("-" * 42)

rows = []
for sig in sigmas:
    rng = np.random.default_rng(7)
    V     = rng.uniform(-0.3, 0.3, n)
    Vh_s  = np.zeros(n)
    Vh_c  = np.zeros(n)
    smo.reset()

    nr = Nt // stride
    e_s = np.zeros(nr)
    e_c = np.zeros(nr)
    rec = 0

    for k in range(Nt):
        yt = C @ V
        nm = sig * rng.standard_normal(ny)
        ym = yt + nm
        y2 = smo.step(ym)

        if k % stride == 0:
            e_s[rec] = np.linalg.norm(Vh_s - V)
            e_c[rec] = np.linalg.norm(Vh_c - V)
            rec += 1

        V     = rk4_step(V, sig * rng.standard_normal(n), dt)
        Vh_s  = obs_std.step(Vh_s, 0.0, ym, dt)
        Vh_c  = obs_cmb.step(Vh_c, 0.0, ym, y2, dt)

    ss = int(0.6 * len(e_c))
    rs = np.sqrt(np.mean(e_s[ss:] ** 2))
    rc = np.sqrt(np.mean(e_c[ss:] ** 2))
    r  = rs / rc if rc > 1e-15 else float("inf")
    print(f"{sig:>8.0e}  {rs:>10.5f}  {rc:>10.5f}  {r:>8.2f}")
    rows.append((sig, rs, rc, r))

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Figures
# ═══════════════════════════════════════════════════════════════════════════════

# ── Noise sweep ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
sa = np.array([r[0] for r in rows])
ax.loglog(sa, [r[1] for r in rows], "b-o", ms=4, lw=1.0,
          label=rf"Classical ($\|K\|={np.linalg.norm(K_std):.0f}$)")
ax.loglog(sa, [r[2] for r in rows], "r-s", ms=4, lw=1.2,
          label=rf"Combined ($\|K\|={np.linalg.norm(K_cmb):.1f}$, "
                rf"$\|K'\|={np.linalg.norm(Kp):.1f}$)")
ax.set_xlabel(r"$\sigma$", fontsize=10)
ax.set_ylabel(r"RMS $\|e\|_{\rm ss}$", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("case3_noise_sweep.pdf")
print("\nSaved case3_noise_sweep.pdf")

# ── Time-domain error trajectory ─────────────────────────────────────────────
T_td  = 8.0
Nt_td = int(T_td / dt)
sig_td = 0.3

rng = np.random.default_rng(42)
V     = rng.uniform(-0.3, 0.3, n)
Vh_s  = np.zeros(n)
Vh_c  = np.zeros(n)
smo.reset()

nr_td = Nt_td // stride
t_arr = np.zeros(nr_td)
e_s_td = np.zeros(nr_td)
e_c_td = np.zeros(nr_td)
rec = 0

for k in range(Nt_td):
    yt = C @ V
    ym = yt + sig_td * rng.standard_normal(ny)
    y2 = smo.step(ym)

    if k % stride == 0:
        t_arr[rec]    = k * dt
        e_s_td[rec]   = np.linalg.norm(Vh_s - V)
        e_c_td[rec]   = np.linalg.norm(Vh_c - V)
        rec += 1

    V     = rk4_step(V, sig_td * rng.standard_normal(n), dt)
    Vh_s  = obs_std.step(Vh_s, 0.0, ym, dt)
    Vh_c  = obs_cmb.step(Vh_c, 0.0, ym, y2, dt)

fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
ax2.semilogy(t_arr, np.maximum(e_s_td, 1e-8), "b-", lw=0.8, alpha=0.6,
             label=rf"Classical ($\|K\|={np.linalg.norm(K_std):.0f}$)")
ax2.semilogy(t_arr, np.maximum(e_c_td, 1e-8), "r-", lw=1.2,
             label=rf"Combined ($\|K\|={np.linalg.norm(K_cmb):.1f}$, "
                   rf"$\|K'\|={np.linalg.norm(Kp):.1f}$)")
ax2.set_xlabel(r"$t$ (s)", fontsize=9)
ax2.set_ylabel(r"$\|e(t)\|$", fontsize=9)
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("case3_error_trajectory.pdf")
print("Saved case3_error_trajectory.pdf")

print("\nDone.")
