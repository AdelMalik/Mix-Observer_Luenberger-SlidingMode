"""
LMI solver for the combined Luenberger observer.

Implements Theorem 1 (sector-bounded) and Remark 5 (increasing-only)
from the article *"Observer design for Lur'e systems via injection of
a reconstructed nonlinear output"*.

Four LMI variants:
  mode="sector"       →  slope-bounded, needs Γ  (Theorem 1)
  mode="increasing"   →  only δᵢeᵢ ≥ 0           (Remark 5)

Each × {classical, combined}, where combined adds the K' correction channel.

Two structural cases:
  aligned=False  →  general A:      y₂ = CAV + CWS(V)  
  aligned=True   →  CA = MC:        y₂ = CWS(V)  (CAV = My known from y)
                    (WC with A=-λI is a special case, M=-λI)
"""

import numpy as np
import cvxpy as cp


def is_aligned(A, C, tol=1e-9):
    """Return True if CA = MC for some M, i.e. rows of CA lie in row(C).

    For WC models A = -λI, this is always True (M = -λI).
    """
    CA = C @ A
    CT = C.T
    # Solve Cᵀ Mᵀ = AᵀCᵀ  →  M = (C A) C†
    M, residuals, rank, _ = np.linalg.lstsq(CT, CA.T, rcond=None)
    return np.allclose(CT @ M, CA.T, atol=tol)


def solve_lmi(A, W, C, *,
              mode="sector",
              combined=True,
              Gamma=None,
              aligned=None,
              Kprime_fixed=None,
              q_cap=10.0,
              eps_p=1e-4,
              eps_lv=1e-4,
              eps_reg=1e-3,
              solver="CLARABEL",
              verbose=False):
    """Solve the observer-design LMI for a Lur'e system.

    Parameters
    ----------
    A : (n, n) array
        Linear dynamics matrix.
    W : (n, n) array
        Nonlinear coupling matrix.
    C : (ny, n) array
        Measurement matrix.
    mode : str
        "sector"      — slope-bounded (Theorem 1), requires Gamma.
        "increasing"  — only δᵢ eᵢ ≥ 0 (Remark 5), no Gamma.
    combined : bool
        True  → optimises over K' (combined observer).
        False → K' = 0 (classical Luenberger observer).
    Gamma : (n, n) diagonal array or None
        Sector bound: 0 ≤ S'ᵢ ≤ γᵢ. Required for mode="sector".
    aligned : bool or None
        If None, auto-detected via `is_aligned(A, C)`.
        True  → CA = MC: y₂ = CWS(V) only, (1,1) block is P(A-KC).
        False → general A: y₂ = CAV + CWS(V), (1,1) block is
                P((I-K'C)A - KC).
    Kprime_fixed : (n, ny) array or None
        If given, K' is fixed (e.g. Cᵀ(CCᵀ)⁻¹, the projector onto range(Cᵀ))
        and not optimised. Only used when combined=True.
    q_cap : float
        Upper bound on the convergence rate q (numerical stability).
    eps_p, eps_lv : float
        Regularisation: P ≻ ε_p I, Λ ≻ ε_lv I.
    eps_reg : float
        For mode="increasing": replaces -2Λ with -ε_reg I in (2,2).
    solver : str
        CVXPY solver name (default "CLARABEL").
    verbose : bool
        Print solver output.

    Returns
    -------
    P : (n, n) array       — Lyapunov matrix
    K : (n, ny) array       — linear injection gain
    Kprime : (n, ny) or None — nonlinear injection gain (None if classical)
    q : float               — convergence rate
    """
    if mode not in ("sector", "increasing"):
        raise ValueError(f"mode must be 'sector' or 'increasing', got '{mode}'")
    if mode == "sector" and Gamma is None:
        raise ValueError("Gamma is required for mode='sector'")

    n = A.shape[0]
    ny = C.shape[0]

    if aligned is None:
        aligned = is_aligned(A, C)

    # ── Decision variables ─────────────────────────────────────────────────
    P  = cp.Variable((n, n), symmetric=True)
    R1 = cp.Variable((n, ny))           # R1 = P K
    lv = cp.Variable(n, nonneg=True)    # Λ = diag(lv)
    q  = cp.Variable(nonneg=True)

    # K' variable — created whenever combined=True and not fixed by user
    R2 = None
    if combined and Kprime_fixed is None:
        R2 = cp.Variable((n, ny))

    # ── (1,1) block: P * A_cl ──────────────────────────────────────────────
    if combined and not aligned:
        # General A: A_cl = (I - K'C)A - KC   →   P·A_cl = P(I-K'C)A - R1·C
        if Kprime_fixed is not None:
            PA_cl = P @ A - P @ Kprime_fixed @ (C @ A) - R1 @ C
        else:
            PA_cl = P @ A - R2 @ (C @ A) - R1 @ C
    else:
        # Aligned (CA=MC) or classical: A_cl = A - KC   →   P·A_cl = P·A - R1·C
        PA_cl = P @ A - R1 @ C

    M11 = PA_cl + PA_cl.T + q * np.eye(n)

    # ── (1,2) block: effective coupling ────────────────────────────────────
    if combined:
        if Kprime_fixed is not None:
            PW_eff = P @ W - P @ Kprime_fixed @ (C @ W)
        else:
            PW_eff = P @ W - R2 @ (C @ W)
    else:
        PW_eff = P @ W

    if mode == "sector":
        M12 = PW_eff + Gamma @ cp.diag(lv)
        M22 = -2.0 * cp.diag(lv)
    else:  # increasing-only
        M12 = PW_eff + cp.diag(lv)
        M22 = -eps_reg * np.eye(n)

    # ── Assemble and solve ─────────────────────────────────────────────────
    M = cp.bmat([[M11, M12], [M12.T, M22]])
    constraints = [
        M << 0,
        P >> eps_p * np.eye(n),
        lv >= eps_lv,
        q >= 0,
        q <= q_cap,
    ]
    prob = cp.Problem(cp.Maximize(q), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LMI infeasible (status: {prob.status})")

    Pv = P.value
    Kv = np.linalg.solve(Pv, R1.value)

    if not combined:
        return Pv, Kv, None, q.value

    Kprime_val = Kprime_fixed if Kprime_fixed is not None else np.linalg.solve(Pv, R2.value)
    return Pv, Kv, Kprime_val, q.value
