"""Continuation in coupling g — scan Delta(g) for a given state."""

import jax
import jax.numpy as jnp

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


def solve_at_g(V0: jnp.ndarray, qn: QuantumNumbers, g: float,
               config: SolverConfig, gauge_indices: list, N0: int,
               tol: float = 1e-10, max_iter: int = 15) -> dict:
    """Newton solve at a single coupling g, starting from V0."""
    def F_V(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config)

    V = V0.copy()
    prev_Delta = float(jnp.real(V[0]))
    for i in range(max_iter):
        E = F_V(V)
        norm = float(jnp.max(jnp.abs(E)))
        if norm < tol:
            return {"V": V, "norm": norm, "iterations": i, "converged": True}
        if jnp.isnan(E[0]) or norm > 1e10:
            return {"V": V, "norm": norm, "iterations": i, "converged": False}
        J = jax.jacfwd(F_V, holomorphic=True)(V)
        delta, _, _, _ = jnp.linalg.lstsq(J, -E, rcond=1e-12)
        V = V + delta
        # Check Delta convergence (more robust than residual norm at float64)
        new_Delta = float(jnp.real(V[0]))
        if i >= 2 and abs(new_Delta - prev_Delta) < tol:
            E_final = F_V(V)
            return {"V": V, "norm": float(jnp.max(jnp.abs(E_final))),
                    "iterations": i + 1, "converged": True}
        prev_Delta = new_Delta

    E = F_V(V)
    norm = float(jnp.max(jnp.abs(E)))
    return {"V": V, "norm": norm, "iterations": max_iter,
            "converged": norm < max(tol, 1e-5)}


def scan_coupling(params0: jnp.ndarray, qn: QuantumNumbers,
                  g_values: list[float], config: SolverConfig,
                  tol: float = 1e-10, max_iter: int = 15,
                  verbose: bool = True) -> list[dict]:
    """Scan Delta(g) over a list of coupling values.

    Uses the solution at g_k as initial guess for g_{k+1}.
    Returns list of {g, Delta, norm, converged, iterations} dicts.
    """
    N0 = config.N0
    Mtint = compute_Mtint(qn)
    kettoLAMBDA = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kettoLAMBDA)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    V = params_to_V(params0, gauge_indices, N0)
    prev_g = g_values[0]
    results = []

    for g in g_values:
        # Rescale c-coefficients when g changes: c_new = c_old * (g_old/g_new)^Mt[a]
        # This accounts for the g-dependent denormalization c_cpp = c_phys / g^Mt
        if g != prev_g and len(results) > 0 and results[-1]["converged"]:
            params_old = V_to_params(V, gauge_indices, N0)
            # Rescale c[a][n] for n >= 1 (c[a][0] is recomputed from A)
            params_new = params_old.copy()
            for a in range(4):
                scale = (prev_g / g) ** Mt[a]
                start = 1 + a * N0
                params_new = params_new.at[start:start + N0].mul(scale)
            V = params_to_V(params_new, gauge_indices, N0)
            prev_g = g

        result = solve_at_g(V, qn, g, config, gauge_indices, N0, tol, max_iter)
        Delta = float(jnp.real(result["V"][0])) + qn.Delta0
        entry = {
            "g": g,
            "Delta": Delta,
            "norm": result["norm"],
            "converged": result["converged"],
            "iterations": result["iterations"],
        }
        results.append(entry)
        if verbose:
            status = "OK" if result["converged"] else "FAIL"
            print(f"  g={g:.4f}: Delta={Delta:.10f}, ||E||={result['norm']:.2e}, "
                  f"iter={result['iterations']}, {status}", flush=True)
        if result["converged"]:
            V = result["V"]  # use as next initial guess
        else:
            if verbose:
                print(f"  WARNING: not converged at g={g}, stopping scan", flush=True)
            break

    return results
