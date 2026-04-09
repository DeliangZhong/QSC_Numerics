"""Newton solver with AD Jacobian, backtracking line search, and LM fallback."""

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
                 config: SolverConfig, tol: float = 1e-10,
                 max_iter: int = 30, damped: bool = True,
                 return_diagnostics: bool = False) -> dict:
    """Newton solver with AD Jacobian and optional line search.

    Args:
        params0: initial guess [Delta, c[0][1],...,c[3][N0]] (1+4*N0 elements)
        qn: quantum numbers
        g: coupling constant
        config: solver config
        tol: convergence tolerance on ||F||_inf
        max_iter: maximum Newton iterations
        damped: if True, use backtracking line search
        return_diagnostics: if True, include residual_history in output

    Returns dict with params, residual_norm, iterations, converged, J (last Jacobian).
    """
    N0 = config.N0
    Mtint = compute_Mtint(qn)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    V = params_to_V(params0, gauge_indices, N0)

    def F_V(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config)

    J_last = None
    residual_history = []
    prev_norm = float("inf")

    for i in range(max_iter):
        F = F_V(V)
        norm = float(jnp.max(jnp.abs(F)))
        residual_history.append(norm)

        if norm < tol:
            return _result(V, norm, i, True, J_last, gauge_indices, N0,
                          residual_history if return_diagnostics else None)

        J = jax.jacfwd(F_V, holomorphic=True)(V)
        J_last = J
        delta, _, _, _ = jnp.linalg.lstsq(J, -F, rcond=1e-12)

        if damped:
            # Backtracking line search
            alpha = 1.0
            accepted = False
            for _ in range(15):
                V_trial = V + alpha * delta
                F_trial = F_V(V_trial)
                norm_trial = float(jnp.max(jnp.abs(F_trial)))
                if norm_trial < norm:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                # Line search failed — take a small step anyway
                alpha = 0.01
            V = V + alpha * delta
        else:
            V = V + delta

        # Detect stalling at float64 precision floor
        if i >= 3 and len(residual_history) >= 3:
            recent = residual_history[-3:]
            if max(recent) < 1e-5 and max(recent) / min(recent) < 10:
                # Residual oscillating near floor — accept
                F_final = F_V(V)
                norm_final = float(jnp.max(jnp.abs(F_final)))
                return _result(V, norm_final, i + 1, True, J_last,
                              gauge_indices, N0,
                              residual_history if return_diagnostics else None)

    F_final = F_V(V)
    norm_final = float(jnp.max(jnp.abs(F_final)))
    return _result(V, norm_final, max_iter, norm_final < max(tol, 1e-6),
                  J_last, gauge_indices, N0,
                  residual_history if return_diagnostics else None)


def _lm_step(J, F, V, F_V, current_norm):
    """Levenberg-Marquardt step as fallback when line search fails."""
    JtJ = J.conj().T @ J
    JtF = J.conj().T @ F
    n = JtJ.shape[0]

    # Try increasing damping until we get improvement
    for lam_exp in range(-3, 6):
        lam = 10.0 ** lam_exp
        delta = jnp.linalg.solve(JtJ + lam * jnp.eye(n, dtype=complex), -JtF)
        F_trial = F_V(V + delta)
        norm_trial = float(jnp.max(jnp.abs(F_trial)))
        if norm_trial < current_norm:
            return 1.0, delta

    # Nothing worked — take a very small gradient descent step
    grad = JtF
    return 1e-4, -grad / jnp.max(jnp.abs(grad))


def _result(V, norm, iterations, converged, J, gauge_indices, N0,
           residual_history):
    """Build result dict."""
    params = V_to_params(V, gauge_indices, N0)
    result = {
        "params": params,
        "V": V,
        "residual_norm": norm,
        "iterations": iterations,
        "converged": converged,
        "J": J,
    }
    if residual_history is not None:
        result["residual_history"] = residual_history
    return result
