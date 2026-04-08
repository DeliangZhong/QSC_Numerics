"""Scan Konishi Delta(g) using Newton solver with continuation in g."""

import json
import math
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qsc.continuation import solve_at_g
from qsc.forward_map import SolverConfig, params_to_V, V_to_params
from qsc.quantum_numbers import (
    KONISHI,
    compute_A,
    compute_B,
    compute_BB,
    compute_alfa,
    compute_gauge_info,
    compute_kettoLAMBDA,
    compute_Mhat,
    compute_Mhat0,
    compute_Mt,
    compute_Mtint,
)


def internal_to_physical(params: jnp.ndarray, g: float, Mt: jnp.ndarray,
                         N0: int) -> jnp.ndarray:
    """Convert internal (denormalized) params to physical (Mathematica) convention.

    Physical: c_phys[a][n] = c_internal[a][n] * g^Mt[a]
    For a=0,2 (imaginary): c_phys = Im(c_internal) * g^Mt (real output)
    For a=1,3 (real): c_phys = Re(c_internal) * g^Mt
    """
    phys = jnp.zeros(1 + 4 * N0)
    phys = phys.at[0].set(jnp.real(params[0]))  # anomalous Delta (real)
    for a in range(4):
        start = 1 + a * N0
        block = params[start:start + N0]
        if a == 0 or a == 2:
            # Internal has i * real_value, physical wants the real_value * g^Mt
            phys = phys.at[start:start + N0].set(jnp.imag(block) * g ** Mt[a])
        else:
            phys = phys.at[start:start + N0].set(jnp.real(block) * g ** Mt[a])
    return phys


def physical_to_internal(phys: jnp.ndarray, g: float, Mt: jnp.ndarray,
                         N0: int) -> jnp.ndarray:
    """Convert physical (Mathematica) params to internal (denormalized) convention."""
    internal = jnp.zeros(1 + 4 * N0, dtype=complex)
    internal = internal.at[0].set(phys[0] + 0j)  # anomalous Delta
    for a in range(4):
        start = 1 + a * N0
        block = phys[start:start + N0]
        denorm = block / g ** Mt[a]
        if a == 0 or a == 2:
            internal = internal.at[start:start + N0].set(1j * denorm)
        else:
            internal = internal.at[start:start + N0].set(denorm + 0j)
    return internal


def main():
    # Load initial params from C++ internal values at g=0.1
    with open("tests/fixtures/konishi_cpp_internal.json") as f:
        cpp = json.load(f)

    N0 = 8
    Mtint = compute_Mtint(KONISHI)
    kL = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kL)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    c_full = jnp.zeros((4, N0 + 1), dtype=complex)
    for a in range(4):
        raw = cpp[f"c_internal_{a}"]
        for n in range(N0 + 1):
            val = raw[n]
            if a == 0 or a == 2:
                val = 1j * val
            c_full = c_full.at[a, n].set(val)

    params_internal = jnp.concatenate([
        jnp.array([cpp["anomalous_delta"] + 0j]),
        c_full[0, 1:], c_full[1, 1:], c_full[2, 1:], c_full[3, 1:],
    ])

    config = SolverConfig(cutP=16, nPoints=18, cutQai=24, QaiShift=4)

    # Convert to physical convention for continuation
    g = 0.1
    phys = internal_to_physical(params_internal, g, Mt, N0)

    results = []
    dg = 0.001
    success_count = 0
    t_start = time.time()

    print(f"Scanning Konishi Delta(g) from g={g:.3f}", flush=True)
    print(f"Config: cutP={config.cutP}, QaiShift={config.QaiShift}", flush=True)

    while g <= 1.0 and time.time() - t_start < 600:
        # Convert physical params to internal at current g
        params_int = physical_to_internal(phys, g, Mt, N0)
        V = params_to_V(params_int, gauge_indices, N0)

        result = solve_at_g(V, KONISHI, g, config, gauge_indices, N0,
                           tol=1e-10, max_iter=8)

        if result["converged"] or result["norm"] < 5e-4:
            V_sol = result["V"]
            Delta = float(jnp.real(V_sol[0])) + KONISHI.Delta0
            results.append({"g": g, "Delta": Delta, "norm": result["norm"]})

            # Convert solution back to physical for next step
            params_sol = V_to_params(V_sol, gauge_indices, N0)
            phys = internal_to_physical(params_sol, g, Mt, N0)

            success_count += 1
            if success_count > 3 and dg < 0.02:
                dg = min(dg * 1.5, 0.02)
                success_count = 0

            g += dg
        else:
            dg /= 2
            success_count = 0
            if dg < 1e-5:
                print(f"  Stuck at g={g:.6f}, dg too small", flush=True)
                break
            # Don't advance g, retry with smaller step
            continue

        if len(results) % 10 == 0:
            print(f"  {len(results)} pts, g={g:.4f}, Delta={Delta:.8f}, "
                  f"dg={dg:.4f}, ||E||={result['norm']:.1e}", flush=True)

    dt = time.time() - t_start
    print(f"\n{len(results)} points in {dt:.0f}s", flush=True)
    if results:
        print(f"g range: [{results[0]['g']:.3f}, {results[-1]['g']:.3f}]", flush=True)

    # Compare with reference
    with open("tests/fixtures/reference_spectral_data.json") as f:
        ref = json.load(f)
    konishi_ref = ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]
    ref_dict = {round(r[0], 4): r[1] for r in konishi_ref}

    print(f"\n{'g':>6s}  {'Delta':>14s}  {'Ref':>14s}  {'diff':>10s}  {'digits':>6s}")
    for r in results:
        g_r = round(r["g"], 2)
        if abs(r["g"] - g_r) < 0.002:
            ref_val = ref_dict.get(g_r)
            if ref_val:
                diff = abs(r["Delta"] - ref_val)
                digits = -math.log10(max(diff / abs(ref_val), 1e-16))
                print(f"{g_r:6.2f}  {r['Delta']:14.10f}  {ref_val:14.10f}  "
                      f"{diff:10.2e}  {digits:6.1f}")


if __name__ == "__main__":
    main()
