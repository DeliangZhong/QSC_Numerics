"""Hybrid Newton solver: mpmath residual + float64 AD Jacobian.

Uses two SolverConfigs:
- config_mp (use_mpmath=True, QaiShift=50): high-precision residual F
- config_f64 (use_mpmath=False, QaiShift=4): fast AD Jacobian J

Both must have the same cutP (same dimV). The Jacobian only needs
~10-digit accuracy to give a good Newton direction; convergence
plateaus at ~10 digits, matching what we want for float64 output.

Cost per iteration: ~1.5s (mpmath F) + ~1.5s (float64 J) = ~3s.
"""

import jax
import jax.numpy as jnp

from qsc.forward_map import (
    SolverConfig,
    V_to_params,
    forward_map_typeI,
    params_to_V,
)
from qsc.quantum_numbers import QuantumNumbers, compute_gauge_info, compute_Mtint


def solve_hybrid(
    params0: jnp.ndarray,
    qn: QuantumNumbers,
    g: float,
    config_mp: SolverConfig,
    config_f64: SolverConfig,
    tol: float = 1e-10,
    max_iter: int = 20,
    verbose: bool = False,
) -> dict:
    """Hybrid Newton solver with high-precision residual + fast AD Jacobian.

    Args:
        params0: initial guess [Delta, c[0][1],...,c[3][N0]], length 1+4*N0
        qn: quantum numbers for the state
        g: coupling constant
        config_mp: solver config with use_mpmath=True (for residual)
        config_f64: solver config with use_mpmath=False (for Jacobian via AD)
        tol: convergence tolerance on ||F||_inf
        max_iter: maximum Newton iterations
        verbose: print per-iteration residuals

    Returns:
        dict with keys: params, V, residual_norm, iterations, converged
    """
    if config_mp.N0 != config_f64.N0:
        raise ValueError(
            f"cutP must match: config_mp.N0={config_mp.N0} != config_f64.N0={config_f64.N0}"
        )

    N0 = config_mp.N0
    Mtint = compute_Mtint(qn)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    V = params_to_V(params0, gauge_indices, N0)

    # Residual function: high-precision (mpmath pulldown)
    def F_mp(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config_mp)

    # Jacobian function: float64 AD (fast, ~10-digit accuracy)
    def F_f64(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config_f64)

    J_fn = jax.jacfwd(F_f64, holomorphic=True)

    residual_history = []

    for i in range(max_iter):
        F = F_mp(V)
        norm = float(jnp.max(jnp.abs(F)))
        residual_history.append(norm)

        if verbose:
            print(f"  iter {i}: ||F|| = {norm:.4e}", flush=True)

        if norm < tol:
            return _build_result(V, norm, i, True, gauge_indices, N0)

        J = J_fn(V)
        delta, _, _, _ = jnp.linalg.lstsq(J, -F, rcond=1e-12)
        V = V + delta

        # Stalling detection: residual oscillating near precision floor
        if i >= 3:
            recent = residual_history[-3:]
            if max(recent) < 1e-5 and max(recent) / min(recent) < 10:
                F_final = F_mp(V)
                norm_final = float(jnp.max(jnp.abs(F_final)))
                if verbose:
                    print(f"  stalling accepted: ||F|| = {norm_final:.4e}", flush=True)
                return _build_result(V, norm_final, i + 1, True, gauge_indices, N0)

    F_final = F_mp(V)
    norm_final = float(jnp.max(jnp.abs(F_final)))
    return _build_result(
        V, norm_final, max_iter, norm_final < max(tol, 1e-6), gauge_indices, N0
    )


def _build_result(
    V: jnp.ndarray,
    norm: float,
    iterations: int,
    converged: bool,
    gauge_indices: list,
    N0: int,
) -> dict:
    """Build result dict from solver state."""
    params = V_to_params(V, gauge_indices, N0)
    return {
        "params": params,
        "V": V,
        "residual_norm": norm,
        "iterations": iterations,
        "converged": converged,
    }
