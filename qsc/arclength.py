"""Pseudo-arc-length continuation for tracking solutions through coupling g.

Standard Newton finds the nearest root — which can be a spurious root from
truncation. Arc-length continuation tracks the solution CURVE, using an
arclength constraint to prevent branch-jumping.

Algorithm per step:
1. Compute tangent (t_V, t_g) via AD: J * t_V = -F_g, t_g = 1
2. Predict along tangent: (V, g) += ds * (t_V, t_g) / ||(t_V, t_g)||
3. Correct with augmented Newton (arclength constraint in last row)
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


def _make_F_Vg(qn: QuantumNumbers, config: SolverConfig,
               gauge_indices: list, N0: int):
    """Build the residual function F(V, g) for AD."""

    def F_Vg(V: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, qn, g, config)

    return F_Vg


def compute_tangent(V: jnp.ndarray, g: float, F_Vg,
                    t_V_prev: jnp.ndarray = None,
                    t_g_prev: float = None):
    """Compute tangent direction along the solution curve.

    Returns (t_V, t_g) normalized so ||(t_V, t_g)|| = 1.
    Sign chosen for consistency with previous tangent.
    """
    g_jnp = jnp.array(g, dtype=complex)

    # J = dF/dV (Jacobian w.r.t. free variables)
    J = jax.jacfwd(F_Vg, argnums=0, holomorphic=True)(V, g_jnp)

    # F_g = dF/dg (sensitivity to coupling)
    F_g = jax.jacfwd(F_Vg, argnums=1, holomorphic=True)(V, g_jnp)
    F_g = F_g.ravel()  # ensure 1D

    # Tangent: J * t_V = -F_g, t_g = 1
    t_V, _, _, _ = jnp.linalg.lstsq(J, -F_g, rcond=1e-12)
    t_g = 1.0 + 0j

    # Normalize
    norm_t = jnp.sqrt(jnp.sum(jnp.abs(t_V) ** 2) + jnp.abs(t_g) ** 2)
    t_V = t_V / norm_t
    t_g = t_g / norm_t

    # Sign consistency with previous tangent (prevent direction reversal)
    if t_V_prev is not None:
        dot = jnp.real(jnp.sum(t_V.conj() * t_V_prev) + t_g * t_g_prev)
        sign = jnp.where(dot < 0, -1.0, 1.0)
        t_V = t_V * sign
        t_g = t_g * sign

    return t_V, t_g, J


def corrector_step(V: jnp.ndarray, g: float,
                   V_pred: jnp.ndarray, g_pred: float,
                   t_V: jnp.ndarray, t_g: complex,
                   F_Vg, tol: float = 1e-8, max_iter: int = 8,
                   verbose: bool = False):
    """Augmented Newton corrector with arclength constraint.

    Solves the bordered system:
    [J    F_g] [dV]   -[F(V,g)   ]
    [t_V  t_g] [dg] = -[arc_resid]

    where arc_resid = Re(t_V† · (V - V_pred)) + Re(t_g* · (g - g_pred))
    """
    V_c = V.copy()
    g_c = g
    dimV = V.shape[0]

    for i in range(max_iter):
        g_c_jnp = jnp.array(g_c, dtype=complex)

        # Residual
        F = F_Vg(V_c, g_c_jnp)
        norm_F = float(jnp.max(jnp.abs(F)))

        # Arclength constraint residual
        arc_res = jnp.real(
            jnp.sum(t_V.conj() * (V_c - V_pred)) + t_g.conj() * (g_c - g_pred)
        )

        if verbose:
            print(f"    corr {i}: ||F||={norm_F:.3e}  arc={float(arc_res):.3e}",
                  flush=True)

        if norm_F < tol:
            return V_c, g_c, norm_F, i + 1, True

        # Jacobian and F_g
        J = jax.jacfwd(F_Vg, argnums=0, holomorphic=True)(V_c, g_c_jnp)
        F_g = jax.jacfwd(F_Vg, argnums=1, holomorphic=True)(V_c, g_c_jnp)
        F_g = F_g.ravel()

        # Build bordered system: (dimV+1) x (dimV+1)
        # Top block: [J | F_g], bottom: [t_V^T | t_g]
        A_top = jnp.concatenate([J, F_g[:, None]], axis=1)      # (dimV, dimV+1)
        A_bot = jnp.concatenate([t_V[None, :], jnp.array([[t_g]])],
                                axis=1)                          # (1, dimV+1)
        A_aug = jnp.concatenate([A_top, A_bot], axis=0)          # (dimV+1, dimV+1)

        # RHS
        rhs = jnp.concatenate([-F, jnp.array([-arc_res + 0j])])  # (dimV+1,)

        # Solve
        delta, _, _, _ = jnp.linalg.lstsq(A_aug, rhs, rcond=1e-12)

        dV = delta[:dimV]
        dg = float(jnp.real(delta[dimV]))

        V_c = V_c + dV
        g_c = g_c + dg

    # Final residual
    F_final = F_Vg(V_c, jnp.array(g_c, dtype=complex))
    norm_final = float(jnp.max(jnp.abs(F_final)))
    return V_c, g_c, norm_final, max_iter, norm_final < tol


def continuation_step(V: jnp.ndarray, g: float, ds: float,
                      F_Vg, t_V_prev: jnp.ndarray = None,
                      t_g_prev: float = None,
                      tol: float = 1e-8, max_corr: int = 8,
                      verbose: bool = False):
    """One pseudo-arc-length continuation step.

    Args:
        V: current free-variable vector (gauge-fixed)
        g: current coupling
        ds: arc-length step size
        F_Vg: residual function F(V, g)
        t_V_prev, t_g_prev: previous tangent (for sign consistency)
        tol: corrector convergence tolerance
        max_corr: max corrector iterations
        verbose: print corrector progress

    Returns:
        dict with V, g, t_V, t_g, residual_norm, corr_iters, converged
    """
    # 1. Tangent
    t_V, t_g, _ = compute_tangent(V, g, F_Vg, t_V_prev, t_g_prev)

    # 2. Predict
    V_pred = V + ds * t_V
    g_pred = g + float(jnp.real(ds * t_g))

    if verbose:
        print(f"  predict: g={g:.6f} -> {g_pred:.6f} (ds={ds:.5f})", flush=True)

    # 3. Correct
    V_new, g_new, norm, n_corr, converged = corrector_step(
        V_pred, g_pred, V_pred, g_pred, t_V, t_g,
        F_Vg, tol=tol, max_iter=max_corr, verbose=verbose,
    )

    return {
        "V": V_new,
        "g": g_new,
        "t_V": t_V,
        "t_g": t_g,
        "residual_norm": norm,
        "corr_iters": n_corr,
        "converged": converged,
    }


def scan_arclength(params0: jnp.ndarray, g_start: float,
                   g_end: float, ds: float,
                   qn: QuantumNumbers, config: SolverConfig,
                   tol: float = 1e-8, max_corr: int = 8,
                   ds_min: float = 1e-5, ds_max: float = 0.01,
                   verbose: bool = True):
    """Scan coupling g using pseudo-arc-length continuation.

    Args:
        params0: converged solution at g_start (full params, 1+4*N0)
        g_start: starting coupling
        g_end: target coupling
        ds: initial arc-length step size
        qn: quantum numbers
        config: solver config
        tol: convergence tolerance
        max_corr: max corrector iterations per step
        ds_min: minimum ds before giving up
        ds_max: maximum ds
        verbose: print progress

    Returns:
        list of dicts with g, params, residual_norm for each converged point
    """
    N0 = config.N0
    Mtint = compute_Mtint(qn)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    F_Vg = _make_F_Vg(qn, config, gauge_indices, N0)
    V = params_to_V(params0, gauge_indices, N0)

    results = [{
        "g": g_start,
        "params": params0,
        "residual_norm": float(jnp.max(jnp.abs(
            F_Vg(V, jnp.array(g_start, dtype=complex))
        ))),
    }]

    t_V_prev = None
    t_g_prev = None
    success_count = 0
    g = g_start

    while (g_end > g_start and g < g_end) or (g_end < g_start and g > g_end):
        step = continuation_step(
            V, g, ds, F_Vg,
            t_V_prev=t_V_prev, t_g_prev=t_g_prev,
            tol=tol, max_corr=max_corr, verbose=(verbose and ds < 0.002),
        )

        if step["converged"]:
            V = step["V"]
            g = step["g"]
            t_V_prev = step["t_V"]
            t_g_prev = step["t_g"]
            params = V_to_params(V, gauge_indices, N0)

            results.append({
                "g": g,
                "params": params,
                "residual_norm": step["residual_norm"],
            })
            success_count += 1

            # Adaptive ds: grow after consecutive successes
            if success_count > 3 and ds < ds_max:
                ds = min(ds * 1.3, ds_max)
                success_count = 0

            if verbose and len(results) % 10 == 0:
                D = float(jnp.real(params[0])) + 2
                print(f"g={g:.5f}: D={D:.10f} ||E||={step['residual_norm']:.1e} "
                      f"ds={ds:.5f} [{len(results)}pts]", flush=True)
        else:
            ds /= 2
            success_count = 0
            if ds < ds_min:
                if verbose:
                    print(f"STUCK: ds={ds:.2e} < ds_min at g={g:.5f}", flush=True)
                break

    return results
