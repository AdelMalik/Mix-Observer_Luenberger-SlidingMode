"""Classical and combined Luenberger observers for Lur'e systems.

  Classical:   dV̂/dt = A V̂ + W S(V̂) + B u + K (y − C V̂)
  Combined:    dV̂/dt = A V̂ + W S(V̂) + B u + K(y−CV̂) − K'(CAV̂ + CWS(V̂) − y₂)
               (general A)  or  − K'(CWS(V̂) − y₂)  (aligned, CA=MC)
"""

import numpy as np


def _rk4(f, x, dt, *args):
    """Classical RK4 step."""
    k1 = f(x, *args)
    k2 = f(x + dt / 2 * k1, *args)
    k3 = f(x + dt / 2 * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class ClassicalObserver:
    """Standard Luenberger observer for a Lur'e system.

    dV̂/dt = A V̂ + W S(V̂) + B u + K (y − C V̂)

    Parameters
    ----------
    A, W : (n, n) arrays
    B : (n, nu) array or None
    C : (ny, n) array
    K : (n, ny) array — from solve_lmi
    S : callable — S(v) → array, component-wise nonlinearity
    """

    def __init__(self, A, W, B, C, K, S):
        self.A, self.W = A, W
        self.B = B if B is not None else np.zeros((A.shape[0], 1))
        self.C, self.K, self.S = C, K, S

    def _rhs(self, v_hat, u, y):
        Sv = self.S(v_hat)
        return (
            self.A @ v_hat
            + self.W @ Sv
            + self.B @ np.atleast_1d(u)
            + self.K @ (y - self.C @ v_hat)
        )

    def step(self, v_hat, u, y, dt):
        """Advance one time step.  u may be scalar or (nu,) array."""
        return _rk4(self._rhs, v_hat, dt, u, y)


class CombinedObserver:
    """Combined observer with virtual-output injection.

    General A (aligned=False):
      dV̂/dt = A V̂ + W S(V̂) + B u + K(y−CV̂) − K'(CAV̂ + CWS(V̂) − y₂)

    Aligned, CA = MC (aligned=True):
      dV̂/dt = A V̂ + W S(V̂) + B u + K(y−CV̂) − K'(CWS(V̂) − y₂)

    Parameters
    ----------
    A, W : (n, n) arrays
    B : (n, nu) array or None
    C : (ny, n) array
    K, Kprime : (n, ny) arrays — from solve_lmi
    S : callable — S(v) → array
    aligned : bool
        False → y₂ = CAV + CWS(V) (full reconstruction)
        True  → y₂ = CWS(V) only (CAV = My known from output)
    """

    def __init__(self, A, W, B, C, K, Kprime, S, *, aligned=False):
        self.A, self.W = A, W
        self.B = B if B is not None else np.zeros((A.shape[0], 1))
        self.C, self.K, self.Kprime, self.S = C, K, Kprime, S
        self.aligned = aligned

    def _rhs(self, v_hat, u, y, y2):
        Sv = self.S(v_hat)
        if self.aligned:
            y2_pred = self.C @ (self.W @ Sv)              # CWS(V̂) only
        else:
            y2_pred = self.C @ (self.A @ v_hat + self.W @ Sv)  # CAV̂ + CWS(V̂)
        return (
            self.A @ v_hat
            + self.W @ Sv
            + self.B @ np.atleast_1d(u)
            + self.K @ (y - self.C @ v_hat)
            - self.Kprime @ (y2_pred - y2)
        )

    def step(self, v_hat, u, y, y2, dt):
        """Advance one time step.  y2 is the reconstructed virtual output."""
        return _rk4(self._rhs, v_hat, dt, u, y, y2)
