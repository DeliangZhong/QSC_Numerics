"""Newton solver with automatic differentiation Jacobian."""

import jax
import jax.numpy as jnp

from qsc.forward_map import SolverConfig, forward_map_typeI
from qsc.quantum_numbers import QuantumNumbers


def solve_newton(params0: jnp.ndarray, qn: QuantumNumbers, g: float,
                 config: SolverConfig, tol: float = 1e-12,
                 max_iter: int = 30) -> dict:
    """Newton solver using JAX AD for the Jacobian.

    Returns dict with:
    - params: converged parameter vector
    - residual_norm: final ||F||
    - iterations: number of iterations used
    - converged: bool
    """
    F = lambda p: forward_map_typeI(p, qn, g, config)
    J_fn = jax.jacfwd(F)

    params = params0.copy()
    for i in range(max_iter):
        residual = F(params)
        norm = jnp.max(jnp.abs(residual))
        if norm < tol:
            return {
                "params": params,
                "residual_norm": float(norm),
                "iterations": i,
                "converged": True,
            }
        jacobian = J_fn(params)
        # Take real parts for the linear solve if needed
        delta = jnp.linalg.solve(jacobian, -residual)
        params = params + delta

    residual = F(params)
    norm = jnp.max(jnp.abs(residual))
    return {
        "params": params,
        "residual_norm": float(norm),
        "iterations": max_iter,
        "converged": float(norm) < tol,
    }
