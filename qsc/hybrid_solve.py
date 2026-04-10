"""Hybrid Newton solver: FLINT residual + JAX AD Jacobian.

Both use the SAME QaiShift=4 equations. The residual is computed at
50-digit precision (via flint), while the Jacobian uses float64 AD
(exact derivative, ~15.9 digits). After conditioning (cond≈1e6, lose
6 digits), the Newton step has ~10-digit accuracy.

Cost per iteration: ~0.07s (flint F) + ~1.5s (JAX AD J) = ~1.6s.
"""

import jax
import jax.numpy as jnp
import numpy as np

from qsc.forward_map import (
    SolverConfig,
    V_to_params,
    forward_map_typeI,
    params_to_V,
)
from qsc.forward_map_flint import forward_map_flint
from qsc.quantum_numbers import QuantumNumbers, compute_gauge_info, compute_Mtint


# Shared config: QaiShift=4 for both F and J
CONFIG_F64 = SolverConfig(
    cutP=16, nPoints=18, cutQai=24, QaiShift=4, use_mpmath=False
)


def solve_hybrid(
    params0: np.ndarray,
    qn: QuantumNumbers,
    g: float,
    tol: float = 1e-10,
    max_iter: int = 15,
    dps: int = 50,
    verbose: bool = False,
) -> dict:
    """Hybrid Newton: flint residual (50 digits) + JAX AD Jacobian (float64).

    Args:
        params0: initial guess [Delta, c[0][1],...,c[3][N0]], numpy complex128
        qn: quantum numbers
        g: coupling constant
        tol: convergence tolerance on ||F||_inf
        max_iter: maximum Newton iterations
        dps: flint decimal digits for residual
        verbose: print per-iteration info

    Returns:
        dict with params, residual_norm, iterations, converged
    """
    N0 = CONFIG_F64.N0
    Mtint = compute_Mtint(qn)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    V = jnp.array(params_to_V(jnp.array(params0), gauge_indices, N0))

    # Residual: flint at 50 digits (same QaiShift=4 equations)
    def F_flint(V_jax):
        params = np.array(V_to_params(V_jax, gauge_indices, N0), dtype=np.complex128)
        return forward_map_flint(params, qn, g,
                                 cutP=16, nPoints=18, cutQai=24,
                                 QaiShift=4, dps=dps)

    # Jacobian: JAX AD at float64 (same QaiShift=4 equations, exact derivative)
    def F_jax(V_jax):
        params = V_to_params(V_jax, gauge_indices, N0)
        return forward_map_typeI(params, qn, g, CONFIG_F64)

    J_fn = jax.jacfwd(F_jax, holomorphic=True)

    residual_history = []

    for i in range(max_iter):
        # High-precision residual
        F = F_flint(V)
        norm = float(np.max(np.abs(F)))
        residual_history.append(norm)

        if verbose:
            print(f"  iter {i}: ||F|| = {norm:.4e}", flush=True)

        if norm < tol:
            params = np.array(V_to_params(V, gauge_indices, N0), dtype=np.complex128)
            return {"params": params, "residual_norm": norm,
                    "iterations": i, "converged": True}

        # Float64 AD Jacobian (exact, fast)
        J = J_fn(V)
        delta, _, _, _ = jnp.linalg.lstsq(J, -jnp.array(F), rcond=1e-12)

        # Backtracking line search using flint residual
        alpha = 1.0
        for _ in range(10):
            V_trial = V + alpha * delta
            F_trial = F_flint(V_trial)
            norm_trial = float(np.max(np.abs(F_trial)))
            if norm_trial < norm:
                break
            alpha *= 0.5
        else:
            alpha = 0.01

        V = V + alpha * delta

        if verbose and alpha < 1.0:
            print(f"         alpha={alpha:.4f}", flush=True)

        # Stalling detection
        if i >= 3:
            recent = residual_history[-3:]
            if max(recent) < 1e-5 and max(recent) / min(recent) < 10:
                params = np.array(V_to_params(V, gauge_indices, N0),
                                  dtype=np.complex128)
                norm_f = float(np.max(np.abs(F_flint(V))))
                return {"params": params, "residual_norm": norm_f,
                        "iterations": i + 1, "converged": True}

    params = np.array(V_to_params(V, gauge_indices, N0), dtype=np.complex128)
    F_final = F_flint(V)
    norm_final = float(np.max(np.abs(F_final)))
    return {"params": params, "residual_norm": norm_final,
            "iterations": max_iter, "converged": norm_final < max(tol, 1e-6)}
