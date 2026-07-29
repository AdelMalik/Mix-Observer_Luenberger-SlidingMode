"""Super-twisting differentiator bank for virtual-output reconstruction.

Reconstructs y₂ from y in finite time via second-order sliding modes.

For general A (aligned=False):
  y₂ = CAV + CWf(V)        — full nonlinear output
  ż̂₁ⱼ = ẑ₂ⱼ + [CBu]ⱼ + Lⱼ k₁ ⌈yⱼ − ẑ₁ⱼ⌋^{½}
  ż̂₂ⱼ ∈ Lⱼ² k₂ sign(yⱼ − ẑ₁ⱼ)

For aligned CA = MC (aligned=True):
  y₂ = CWf(V)              — CAV = My already known from output
  ż̂₁ⱼ = [My]ⱼ + ẑ₂ⱼ + [CBu]ⱼ + Lⱼ k₁ ⌈yⱼ − ẑ₁ⱼ⌋^{½}
  ż̂₂ⱼ ∈ Lⱼ² k₂ sign(yⱼ − ẑ₁ⱼ)

After finite time T₀:  ẑ₁ = y,  ẑ₂ = y₂.
"""

import numpy as np


class SlidingModeBank:
    """Bank of ny super-twisting differentiators, one per output channel.

    Parameters
    ----------
    ny : int
        Number of output channels.
    L : (ny,) array
        Per-channel gains.  Must satisfy Lⱼ > sup |ÿⱼ| / k₂ (see Proposition).
    A, C, B : arrays or None
        System matrices.  A, C required for aligned mode; B for input feedforward.
    k1 : float
        Super-twisting constant (default 1.5).
    k2 : float
        Super-twisting constant (default 1.1).
    dt : float
        Macro time step.
    n_sub : int
        Substeps per macro step for stiff integration.
    aligned : bool or None
        If None, auto-detected.  Affects the linear drift term.
    """

    def __init__(self, ny, L, *, A=None, C=None, B=None,
                 k1=1.5, k2=1.1, dt=1e-3, n_sub=10, aligned=None):
        self.ny = ny
        self.L = np.atleast_1d(L).astype(float)
        self.k1, self.k2 = k1, k2
        self.dt_s = dt / n_sub
        self.n_sub = n_sub
        self.z1 = np.zeros(ny)
        self.z2 = np.zeros(ny)

        # ── Detect alignment ───────────────────────────────────────────────
        if aligned is None and A is not None and C is not None:
            from .lmi import is_aligned
            aligned = is_aligned(A, C)
        self.aligned = aligned if aligned is not None else False

        # ── Precompute feedforward terms ────────────────────────────────────
        self._CB = None
        if B is not None and C is not None:
            self._CB = (C @ B).ravel()

        # For aligned mode:  CA = MC  →  My is the known linear drift.
        self._M = None
        if self.aligned and A is not None and C is not None:
            CA = C @ A
            CT = C.T
            # M = (CA) C†   (least-squares, exact since rows of CA ⊂ rows of C)
            self._M = np.linalg.lstsq(CT, CA.T, rcond=None)[0].T  # (ny, ny)

    def step(self, y, u=0.0):
        """Advance one macro step and return the reconstructed virtual output.

        Parameters
        ----------
        y : (ny,) array — current noisy measurement
        u : scalar or (nu,) array — control input

        Returns
        -------
        z2 : (ny,) array — reconstructed y₂
        """
        y_arr = np.atleast_1d(y)
        for _ in range(self.n_sub):
            for j in range(self.ny):
                e1 = y_arr[j] - self.z1[j]

                # Linear drift (known from output when aligned)
                if self.aligned and self._M is not None:
                    drift = self._M[j] @ y_arr
                else:
                    drift = 0.0

                # Input feedforward
                inp = 0.0
                if self._CB is not None:
                    inp = self._CB[j] * (u if np.ndim(u) == 0 else np.atleast_1d(u)[0])

                self.z1[j] += self.dt_s * (
                    drift + self.z2[j] + inp
                    + self.L[j] * self.k1 * np.sign(e1) * np.sqrt(abs(e1))
                )
                self.z2[j] += self.dt_s * (
                    self.L[j] ** 2 * self.k2 * np.sign(e1)
                )
        return self.z2.copy()

    def reset(self):
        """Reset internal states to zero."""
        self.z1.fill(0.0)
        self.z2.fill(0.0)
