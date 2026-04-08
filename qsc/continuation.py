"""Predictor-corrector continuation in coupling g with adaptive step size."""

import jax
import jax.numpy as jnp
from functools import partial

from qsc.forward_map import (
    SolverConfig,
    V_to_params,
    forward_map_typeI,
    params_to_V,
)
from qsc.quantum_numbers import (
    QuantumNumbers,
    compute_gauge_info,
    compute_kettoLAMBDA,
    compute_Mt,
    compute_Mtint,
)


def _make_F(qn: QuantumNumbers, config: SolverConfig,
            gauge_indices: list, N0: int):
    """Build the forward map in V-space, parameterized by g."""
    def F_Vg(V, g):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config)
    return F_Vg


def solve_at_g(V0: jnp.ndarray, qn: QuantumNumbers, g: float,
               config: SolverConfig, gauge_indices: list, N0: int,
               tol: float = 1e-10, max_iter: int = 15) -> dict:
    """Newton solve at a single coupling g, starting from V0.

    Returns dict with V, norm, iterations, converged, and J_inv (if converged).
    """
    def F_V(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config)

    V = V0.copy()
    J_last = None
    prev_Delta = float(jnp.real(V[0]))
    for i in range(max_iter):
        E = F_V(V)
        norm = float(jnp.max(jnp.abs(E)))
        if norm < tol:
            return {"V": V, "norm": norm, "iterations": i,
                    "converged": True, "J": J_last}
        if jnp.isnan(E[0]) or norm > 1e10:
            return {"V": V, "norm": norm, "iterations": i,
                    "converged": False, "J": None}
        J = jax.jacfwd(F_V, holomorphic=True)(V)
        J_last = J
        delta, _, _, _ = jnp.linalg.lstsq(J, -E, rcond=1e-12)
        V = V + delta
        new_Delta = float(jnp.real(V[0]))
        if i >= 2 and abs(new_Delta - prev_Delta) < tol:
            E_final = F_V(V)
            return {"V": V, "norm": float(jnp.max(jnp.abs(E_final))),
                    "iterations": i + 1, "converged": True, "J": J_last}
        prev_Delta = new_Delta

    E = F_V(V)
    norm = float(jnp.max(jnp.abs(E)))
    return {"V": V, "norm": norm, "iterations": max_iter,
            "converged": norm < max(tol, 1e-5), "J": J_last}


def _internal_to_physical(params: jnp.ndarray, g: float,
                          Mt: jnp.ndarray, N0: int) -> jnp.ndarray:
    """Convert internal params to physical convention (smooth in g)."""
    phys = jnp.zeros(1 + 4 * N0)
    phys = phys.at[0].set(jnp.real(params[0]))
    for a in range(4):
        start = 1 + a * N0
        block = params[start:start + N0]
        if a == 0 or a == 2:
            phys = phys.at[start:start + N0].set(jnp.imag(block) * g ** Mt[a])
        else:
            phys = phys.at[start:start + N0].set(jnp.real(block) * g ** Mt[a])
    return phys


def _physical_to_internal(phys: jnp.ndarray, g: float,
                          Mt: jnp.ndarray, N0: int) -> jnp.ndarray:
    """Convert physical params to internal convention."""
    internal = jnp.zeros(1 + 4 * N0, dtype=complex)
    internal = internal.at[0].set(phys[0] + 0j)
    for a in range(4):
        start = 1 + a * N0
        block = phys[start:start + N0]
        denorm = block / g ** Mt[a]
        if a == 0 or a == 2:
            internal = internal.at[start:start + N0].set(1j * denorm)
        else:
            internal = internal.at[start:start + N0].set(denorm + 0j)
    return internal


def predictor_step(V: jnp.ndarray, J: jnp.ndarray, qn: QuantumNumbers,
                   g: float, dg: float, config: SolverConfig,
                   gauge_indices: list, N0: int,
                   Mt: jnp.ndarray) -> jnp.ndarray:
    """Predictor step: extrapolate solution from g to g+dg.

    Works in physical convention where coefficients are smooth in g,
    then converts back to internal convention at g+dg.
    """
    # Convert current solution to physical convention
    params_int = V_to_params(V, gauge_indices, N0)
    phys = _internal_to_physical(params_int, g, Mt, N0)

    # Use tangent prediction in physical space via dF/dg
    def F_at_g(g_val):
        int_p = _physical_to_internal(phys, g_val, Mt, N0)
        return forward_map_typeI(int_p, qn, g_val, config)

    h = 1e-7 * max(abs(g), 0.01)
    dFdg = (F_at_g(g + h) - F_at_g(g - h)) / (2 * h)

    # dV/dg = -J^{-1} @ dF/dg (in V-space at current g)
    dVdg, _, _, _ = jnp.linalg.lstsq(J, -dFdg, rcond=1e-12)

    # Extrapolate V in V-space, then convert to internal at new g
    V_pred_at_g = V + dg * dVdg

    # But we need to re-internalize at g+dg, not at g
    params_pred = V_to_params(V_pred_at_g, gauge_indices, N0)
    phys_pred = _internal_to_physical(params_pred, g, Mt, N0)
    # Now convert to internal at g+dg
    internal_at_gnew = _physical_to_internal(phys_pred, g + dg, Mt, N0)
    return params_to_V(internal_at_gnew, gauge_indices, N0)


def scan_predictor_corrector(params0: jnp.ndarray, qn: QuantumNumbers,
                             g_start: float, g_end: float,
                             config: SolverConfig,
                             dg_init: float = 0.005,
                             tol: float = 1e-10,
                             max_iter: int = 8,
                             verbose: bool = True) -> list[dict]:
    """Scan Delta(g) using predictor-corrector continuation.

    Predictor: tangent extrapolation using J^{-1} and dF/dg
    Corrector: Newton iteration
    Adaptive step: double if ≤3 iter, halve if >6 or fail
    """
    N0 = config.N0
    Mtint = compute_Mtint(qn)
    kettoLAMBDA = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kettoLAMBDA)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    V = params_to_V(params0, gauge_indices, N0)
    g = g_start
    dg = dg_init
    results = []
    direction = 1.0 if g_end > g_start else -1.0

    # Solve at starting point
    result0 = solve_at_g(V, qn, g, config, gauge_indices, N0, tol, max_iter)
    if not result0["converged"] and result0["norm"] > 1e-4:
        if verbose:
            print(f"Failed to converge at starting point g={g}", flush=True)
        return results

    V = result0["V"]
    J = result0["J"]
    Delta = float(jnp.real(V[0])) + qn.Delta0
    results.append({"g": g, "Delta": Delta, "norm": result0["norm"],
                    "converged": True, "iterations": result0["iterations"]})
    if verbose:
        print(f"  g={g:.4f}: Delta={Delta:.10f} (start)", flush=True)

    while (direction > 0 and g < g_end) or (direction < 0 and g > g_end):
        g_new = g + direction * dg
        if (direction > 0 and g_new > g_end):
            g_new = g_end
        if (direction < 0 and g_new < g_end):
            g_new = g_end

        # Predictor step (tangent extrapolation)
        if J is not None:
            V_pred = predictor_step(V, J, qn, g, direction * dg, config,
                                   gauge_indices, N0, Mt)
        else:
            V_pred = V  # fallback: no prediction

        # Corrector step (Newton)
        result = solve_at_g(V_pred, qn, g_new, config, gauge_indices, N0,
                           tol, max_iter)

        if result["converged"] or result["norm"] < 1e-4:
            V = result["V"]
            J = result["J"]
            g = g_new
            Delta = float(jnp.real(V[0])) + qn.Delta0
            results.append({
                "g": g, "Delta": Delta, "norm": result["norm"],
                "converged": result["converged"],
                "iterations": result["iterations"],
            })
            if verbose:
                print(f"  g={g:.4f}: Delta={Delta:.10f}, iter={result['iterations']}, "
                      f"||E||={result['norm']:.1e}, dg={dg:.4f}", flush=True)

            # Adaptive step: increase if easy convergence
            if result["iterations"] <= 3:
                dg = min(dg * 2, 0.1)
            elif result["iterations"] <= 5:
                pass  # keep dg
            else:
                dg = max(dg * 0.7, 1e-4)
        else:
            # Failed: halve step and retry
            dg /= 2
            if verbose:
                print(f"  g={g_new:.4f}: FAIL (||E||={result['norm']:.1e}), "
                      f"halving to dg={dg:.4f}", flush=True)
            if dg < 1e-5:
                if verbose:
                    print(f"  Step too small, stopping at g={g:.4f}", flush=True)
                break

    return results
