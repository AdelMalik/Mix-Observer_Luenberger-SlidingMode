# Combined LMI-Based Observer for Lur'e Systems

> *"Observer design for Lur'e-type systems via injection of a reconstructed nonlinear output."*

## Overview

When the nonlinear coupling in a Lur'e system is strong, the classical Luenberger observer LMI becomes infeasible or demands impractically large gains.  
This repository provides a **combined observer** that adds a second correction channel fed by a **virtual output** — key nonlinear terms reconstructed in finite time from the measurements themselves via a bank of super-twisting differentiators.

```
  ┌─────────────────────────────────────────────────────────┐
  │              Classical (top row)                        │
  │   System ──y──► Luenberger Observer ──► V̂               │
  │                                                        │
  │              Proposed (bottom row)                      │
  │                   ┌──────────────────────┐              │
  │   System ──y──►   │ Homogeneous Observer │──ẑ₂──┐       │
  │               │   │  (virtual output)    │      │       │
  │               │   └──────────────────────┘      │       │
  │               └──────────y──────────────────────┤       │
  │                                                 ▼       │
  │                        Luenberger Observer 2 ──► V̂      │
  └─────────────────────────────────────────────────────────┘
```

The combined observer attenuates the effective nonlinear coupling from $W$ to $(I-K'C)W$ in the error dynamics, keeping the LMI feasible and the gains moderate even when the classical design fails.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9, NumPy, SciPy, Matplotlib, and [CVXPY](https://www.cvxpy.org/) with the CLARABEL solver.

## Quickstart

```python
from lure_observer import solve_lmi, CombinedObserver, SlidingModeBank
import numpy as np

# 1. Define your Lur'e model
A = ...       # (n,n)  — linear dynamics
W = ...       # (n,n)  — nonlinear coupling
C = ...       # (ny,n) — measurement matrix
def f(v):     # nonlinearity
    return 1 / (1 + np.exp(-v))

Gamma = np.eye(n)   # sector bound (required for slope-bounded LMI)

# 2. Solve the LMI
P, K, Kprime, q = solve_lmi(A, W, C, mode="sector", combined=True, Gamma=Gamma)
print(f"||K|| = {np.linalg.norm(K):.1f},  ||K'|| = {np.linalg.norm(Kprime):.1f}")

# 3. Build the observer
smo = SlidingModeBank(ny=3, L=np.ones(3)*3, A=A, C=C, dt=1e-3)
obs = CombinedObserver(A, W, None, C, K, Kprime, f)

# 4. Run
for t in ...:
    y  = measure()          # noisy measurement
    y2 = smo.step(y)        # reconstructed virtual output
    Vh = obs.step(Vh, u=0, y=y, y2=y2, dt=1e-3)
```

## LMI variants

| `mode` | Condition | Needs `Gamma`? | (2,2) block |
|---|---|---|---|
| `"sector"` | $0 \leq S_i' \leq \gamma_i$ | Yes | $-2\Lambda$ |
| `"increasing"` | $\delta_i e_i \geq 0$ only | No | $-\varepsilon I$ |

Each `mode` works with `combined=True` (Theorem 1 / Remark 5) or `combined=False` (classical Luenberger).

### Aligned vs general $A$

- **Aligned** ($CA = MC$): the virtual output is $y_2 = CWS(V)$ only. Detected automatically.
- **General $A$**: the sliding-mode bank reconstructs the full $y_2 = CAV + CWS(V)$. Set `aligned=False` explicitly for non-aligned models.

## Repository structure

```
lure-observer/
├── src/lure_observer/         # Reusable library
│   ├── lmi.py                 #   LMI solver (all 4 variants)
│   ├── observer.py            #   ClassicalObserver + CombinedObserver
│   └── sliding_mode.py        #   Super-twisting differentiator bank
├── article_code/              # Exact scripts from the article
├── examples/
│   └── case3_sigmoid_sliding.py   # Case study using the library
└── docs/
```

## Examples

```bash
cd examples
python case3_sigmoid_sliding.py
```

Reproduces Case 3 from the article: Wilson-Cowan network ($n=6$, $n_y=3$), sigmoid nonlinearity, sector-bounded LMI, sliding-mode bank. Produces two figures:
- **Noise sweep**: RMS steady-state error vs measurement noise $\sigma$, comparing classical and combined observers.
- **Error trajectory**: $\|e(t)\|$ at fixed noise level.

## Reference

Annabi, A.M. *"Observer design for Lur'e-type systems via injection of a reconstructed nonlinear output."* [arXiv:2606.24656](https://arxiv.org/abs/2606.24656).
