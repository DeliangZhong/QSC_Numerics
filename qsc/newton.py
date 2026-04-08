"""Newton solver with automatic differentiation Jacobian."""

import jax
import jax.numpy as jnp

from qsc.forward_map import (
    SolverConfig,
    V_to_params,
    forward_map_typeI,
    params_to_V,
)
from qsc.quantum_numbers import QuantumNumbers, compute_gauge_info, compute_Mtint


def solve_newton(params0: jnp.ndarray, qn: QuantumNumbers, g: float,
                 config: SolverConfig, tol: float = 1e-12,
                 max_iter: int = 30) -> dict:
    """Newton solver using JAX AD for the Jacobian.

    params0: initial guess in raw format [Delta, c[0][1],...,c[3][N0]]
             (1 + 4*N0 elements, including gauge-fixed entries)

    Returns dict with:
    - params: converged parameter vector (raw format)
    - residual_norm: final ||F||_inf
    - iterations: number of iterations used
    - converged: bool
    - delta_history: list of Delta values
    """
    N0 = config.N0
    Mtint = compute_Mtint(qn)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    # Convert to free variables
    V0 = params_to_V(params0, gauge_indices, N0)

    # Forward map in V-space
    def F_V(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config)

    V = V0.copy()
    delta_history = []
    for i in range(max_iter):
        residual = F_V(V)
        norm = float(jnp.max(jnp.abs(residual)))
        delta_history.append(float(jnp.real(V[0])))
        if norm < tol:
            params = V_to_params(V, gauge_indices, N0)
            return {
                "params": params,
                "residual_norm": norm,
                "iterations": i,
                "converged": True,
                "delta_history": delta_history,
            }
        jacobian = jax.jacfwd(F_V, holomorphic=True)(V)
        delta = jnp.linalg.solve(jacobian, -residual)
        V = V + delta

    residual = F_V(V)
    norm = float(jnp.max(jnp.abs(residual)))
    params = V_to_params(V, gauge_indices, N0)
    return {
        "params": params,
        "residual_norm": norm,
        "iterations": max_iter,
        "converged": norm < tol,
        "delta_history": delta_history,
    }
