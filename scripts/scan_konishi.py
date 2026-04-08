"""Scan Konishi Delta(g) — physical convention continuation like C++."""

import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from qsc.continuation import solve_at_g
from qsc.forward_map import SolverConfig, params_to_V, V_to_params
from qsc.quantum_numbers import (
    KONISHI,
    compute_gauge_info,
    compute_kettoLAMBDA,
    compute_Mt,
    compute_Mtint,
)


def physical_to_internal(phys: np.ndarray, g: float, Mt: np.ndarray,
                         N0: int) -> jnp.ndarray:
    """Physical (Mathematica) → internal (C++ denormalized) convention."""
    internal = jnp.zeros(1 + 4 * N0, dtype=complex)
    internal = internal.at[0].set(phys[0] + 0j)
    for a in range(4):
        s = 1 + a * N0
        block = phys[s:s + N0] / g ** Mt[a]
        if a == 0 or a == 2:
            internal = internal.at[s:s + N0].set(1j * block)
        else:
            internal = internal.at[s:s + N0].set(block + 0j)
    return internal


def internal_to_physical(params: jnp.ndarray, g: float, Mt: np.ndarray,
                         N0: int) -> np.ndarray:
    """Internal (C++ denormalized) → physical (Mathematica) convention."""
    phys = np.zeros(1 + 4 * N0)
    phys[0] = float(jnp.real(params[0]))
    for a in range(4):
        s = 1 + a * N0
        block = params[s:s + N0]
        if a == 0 or a == 2:
            phys[s:s + N0] = np.array(jnp.imag(block)) * g ** Mt[a]
        else:
            phys[s:s + N0] = np.array(jnp.real(block)) * g ** Mt[a]
    return phys


def interpolate_physical(saved_gs: list, saved_phys: list, g_new: float,
                         order: int = 4) -> np.ndarray:
    """Polynomial extrapolation in physical convention from saved solutions.

    Like InterpolateIn from the Mathematica package.
    """
    n_saved = len(saved_gs)
    if n_saved == 0:
        raise ValueError("No saved solutions for interpolation")
    if n_saved == 1:
        return saved_phys[0].copy()

    # Use nearest saved points
    use = min(order, n_saved)
    dists = [abs(g - g_new) for g in saved_gs]
    indices = sorted(range(n_saved), key=lambda i: dists[i])[:use]
    indices.sort()  # keep in order

    gs = np.array([saved_gs[i] for i in indices])
    phys_arr = np.array([saved_phys[i] for i in indices])

    # Polynomial fit for each parameter component
    result = np.zeros_like(saved_phys[0])
    n_params = len(result)

    for j in range(n_params):
        values = phys_arr[:, j]
        # Polynomial fit of degree min(use-1, 3)
        deg = min(use - 1, 3)
        coeffs = np.polyfit(gs, values, deg)
        result[j] = np.polyval(coeffs, g_new)

    result[0] = max(result[0], 0)  # anomalous dim should be positive
    return result


def main():
    with open("tests/fixtures/konishi_cpp_internal.json") as f:
        cpp = json.load(f)

    N0 = 8
    Mtint_arr = np.array([3, 2, 1, 0])
    kL = int(1 - Mtint_arr[0] - Mtint_arr[3])
    Mt = Mtint_arr + kL / 2.0
    Mtint = jnp.array(Mtint_arr)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    # Build initial params from C++ internal at g=0.1
    c_full = jnp.zeros((4, N0 + 1), dtype=complex)
    for a in range(4):
        raw = cpp[f"c_internal_{a}"]
        for n in range(N0 + 1):
            val = raw[n]
            if a == 0 or a == 2:
                val = 1j * val
            c_full = c_full.at[a, n].set(val)
    params_int0 = jnp.concatenate([
        jnp.array([cpp["anomalous_delta"] + 0j]),
        c_full[0, 1:], c_full[1, 1:], c_full[2, 1:], c_full[3, 1:],
    ])

    config = SolverConfig(cutP=16, nPoints=18, cutQai=24, QaiShift=4)

    # Convert initial solution to physical convention
    g0 = 0.1
    phys0 = internal_to_physical(params_int0, g0, Mt, N0)

    # Storage for continuation (in physical convention)
    saved_gs = [g0]
    saved_phys = [phys0]
    results = [{"g": g0, "Delta": float(jnp.real(params_int0[0])) + 2}]

    # Adaptive continuation like C++
    g = g0
    dg = 0.005  # start with reasonable step
    success_count = 0
    t_start = time.time()
    max_time = 540  # 9 minutes

    print(f"Scanning Konishi Delta(g) from g={g:.3f}")
    print(f"Config: cutP={config.cutP}, QaiShift={config.QaiShift}")
    print(f"  g={g:.4f}: Delta={results[0]['Delta']:.10f} (start)", flush=True)

    while g < 1.0 and time.time() - t_start < max_time:
        g_new = g + dg

        # Predict initial guess via polynomial extrapolation in physical space
        phys_pred = interpolate_physical(saved_gs, saved_phys, g_new)

        # Convert to internal convention at g_new
        params_int = physical_to_internal(phys_pred, g_new, Mt, N0)
        V = params_to_V(params_int, gauge_indices, N0)

        # Newton corrector
        result = solve_at_g(V, KONISHI, g_new, config, gauge_indices, N0,
                           tol=1e-10, max_iter=8)

        if result["converged"] or result["norm"] < 5e-4:
            V_sol = result["V"]
            g = g_new
            params_sol = V_to_params(V_sol, gauge_indices, N0)
            phys_sol = internal_to_physical(params_sol, g, Mt, N0)
            Delta = float(jnp.real(V_sol[0])) + 2

            saved_gs.append(g)
            saved_phys.append(phys_sol)
            results.append({"g": g, "Delta": Delta, "norm": result["norm"],
                           "iter": result["iterations"]})

            success_count += 1
            # Adaptive step: double after 4 successes, max 0.05
            if success_count > 3 and dg < 0.05:
                dg = min(dg * 2, 0.05)
                success_count = 0

            if len(results) % 5 == 0:
                print(f"  g={g:.4f}: Delta={Delta:.10f}, iter={result['iterations']}, "
                      f"||E||={result['norm']:.1e}, dg={dg:.4f}", flush=True)
        else:
            dg /= 2
            success_count = 0
            if dg < 1e-4:
                print(f"  Step too small at g={g_new:.4f}, stopping", flush=True)
                break
            if len(results) % 5 == 0:
                print(f"  g={g_new:.4f}: FAIL, dg→{dg:.4f}", flush=True)

    dt = time.time() - t_start
    g_max = results[-1]["g"]
    print(f"\n{len(results)} points in {dt:.0f}s, g=[{results[0]['g']:.3f}, {g_max:.3f}]")

    # Compare with reference
    with open("tests/fixtures/reference_spectral_data.json") as f:
        ref = json.load(f)
    konishi_ref = ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]
    ref_dict = {round(r[0], 4): r[1] for r in konishi_ref}

    print(f"\n{'g':>6s}  {'Delta':>14s}  {'Ref':>14s}  {'diff':>10s}  {'digits':>6s}")
    print("-" * 58)
    for r in results:
        g_r = round(r["g"], 2)
        if abs(r["g"] - g_r) < 0.005:
            ref_val = ref_dict.get(g_r)
            if ref_val:
                diff = abs(r["Delta"] - ref_val)
                digits = -math.log10(max(diff / abs(ref_val), 1e-16))
                print(f"{g_r:6.2f}  {r['Delta']:14.10f}  {ref_val:14.10f}  "
                      f"{diff:10.2e}  {digits:6.1f}")


if __name__ == "__main__":
    main()
