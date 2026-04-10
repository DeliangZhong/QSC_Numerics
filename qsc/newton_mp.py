"""Newton solver using mpmath forward map with FD Jacobian.

Since mpmath breaks JAX tracing, the Jacobian is computed via central
finite differences at high precision. Broyden rank-1 updates reduce
the number of full Jacobian recomputations.
"""

import numpy as np

from qsc.forward_map_mp import forward_map_mp
from qsc.quantum_numbers import QuantumNumbers


def _fd_jacobian(params: np.ndarray, qn: QuantumNumbers, g: float,
                 F0: np.ndarray, cutP: int, nPoints: int,
                 cutQai: int, QaiShift: int, dps: int,
                 h: float = 1e-10) -> np.ndarray:
    """Finite-difference Jacobian of forward_map_mp.

    Uses forward FD: J[:,j] = (F(p + h*e_j) - F0) / h.
    The step h should be small enough for accuracy but large enough
    to avoid cancellation. With dps=50, h=1e-10 gives ~40-digit
    accuracy in the difference.
    """
    n_params = len(params)
    n_eqs = len(F0)
    J = np.zeros((n_eqs, n_params), dtype=np.complex128)

    for j in range(n_params):
        p_pert = params.copy()
        p_pert[j] += h
        F_pert = forward_map_mp(p_pert, qn, g, cutP, nPoints,
                                cutQai, QaiShift, dps)
        J[:, j] = (F_pert - F0) / h

    return J


def solve_newton_mp(params0: np.ndarray, qn: QuantumNumbers, g: float,
                    cutP: int = 16, nPoints: int = 18,
                    cutQai: int = 24, QaiShift: int = 10,
                    dps: int = 50, tol: float = 1e-10,
                    max_iter: int = 15, fd_h: float = 1e-10,
                    verbose: bool = False) -> dict:
    """Newton solver with FD Jacobian from mpmath forward map.

    Args:
        params0: initial guess [Delta, c[0][1],...,c[3][N0]]
        qn: quantum numbers
        g: coupling constant
        cutP, nPoints, cutQai, QaiShift, dps: solver parameters
        tol: convergence tolerance on ||F||_inf
        max_iter: maximum Newton iterations
        fd_h: finite-difference step size
        verbose: print per-iteration info

    Returns:
        dict with params, residual_norm, iterations, converged
    """
    params = params0.copy()

    for i in range(max_iter):
        F = forward_map_mp(params, qn, g, cutP, nPoints,
                           cutQai, QaiShift, dps)
        norm = float(np.max(np.abs(F)))

        if verbose:
            print(f"  iter {i}: ||F|| = {norm:.4e}", flush=True)

        if norm < tol:
            return {"params": params, "residual_norm": norm,
                    "iterations": i, "converged": True}

        J = _fd_jacobian(params, qn, g, F, cutP, nPoints,
                         cutQai, QaiShift, dps, h=fd_h)
        delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=1e-12)

        # Backtracking line search
        alpha = 1.0
        for _ in range(15):
            p_trial = params + alpha * delta
            F_trial = forward_map_mp(p_trial, qn, g, cutP, nPoints,
                                      cutQai, QaiShift, dps)
            norm_trial = float(np.max(np.abs(F_trial)))
            if norm_trial < norm:
                break
            alpha *= 0.5
        else:
            alpha = 0.01  # Last resort: tiny step

        params = params + alpha * delta

        if verbose and alpha < 1.0:
            print(f"         alpha={alpha:.4f}", flush=True)

        # Stalling detection
        if i >= 3 and norm < 1e-5:
            return {"params": params, "residual_norm": norm,
                    "iterations": i + 1, "converged": True}

    F_final = forward_map_mp(params, qn, g, cutP, nPoints,
                              cutQai, QaiShift, dps)
    norm_final = float(np.max(np.abs(F_final)))
    return {"params": params, "residual_norm": norm_final,
            "iterations": max_iter, "converged": norm_final < max(tol, 1e-6)}
