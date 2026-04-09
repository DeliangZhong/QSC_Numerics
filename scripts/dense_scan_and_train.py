"""Dense JAX scan from g=0.05 to g=1.0, save data, train ML predictor."""

import json
import math
import time

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate GPU/CPU memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"   # Limit to 30% of available

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from qsc.forward_map import SolverConfig, forward_map_typeI, params_to_V, V_to_params
from qsc.newton import solve_newton
from qsc.perturbative import perturbative_params
from qsc.quantum_numbers import KONISHI, compute_gauge_info, compute_Mtint

config = SolverConfig(cutP=16, nPoints=18, cutQai=24, QaiShift=4, use_mpmath=False)
N0 = config.N0
Mt = np.array([2., 1., 0., -1.])
Mtint = compute_Mtint(KONISHI)
gauge_info = compute_gauge_info(Mtint, N0)
gauge_indices = gauge_info["gauge_indices"]


def i2p(params, g):
    phys = np.zeros(1 + 4 * N0)
    phys[0] = float(jnp.real(params[0]))
    for a in range(4):
        s = 1 + a * N0
        block = params[s:s + N0]
        if a in (0, 2):
            phys[s:s + N0] = np.array(jnp.imag(block)) * g ** Mt[a]
        else:
            phys[s:s + N0] = np.array(jnp.real(block)) * g ** Mt[a]
    return phys


def p2i(phys, g):
    internal = jnp.zeros(1 + 4 * N0, dtype=complex)
    internal = internal.at[0].set(phys[0] + 0j)
    for a in range(4):
        s = 1 + a * N0
        block = phys[s:s + N0] / g ** Mt[a]
        if a in (0, 2):
            internal = internal.at[s:s + N0].set(1j * block)
        else:
            internal = internal.at[s:s + N0].set(block + 0j)
    return internal


def main():
    # Load reference for validation
    with open("tests/fixtures/reference_spectral_data.json") as f:
        ref = json.load(f)
    ref_dict = {round(r[0], 4): r[1]
                for r in ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]}

    # Resume from saved data if available
    scan_file = "data/konishi_dense_scan.npz"
    if os.path.exists(scan_file):
        saved = np.load(scan_file)
        solved_g = list(saved["g"])
        solved_Delta = list(saved["Delta"])
        solved_phys = list(saved["phys"])
        g = solved_g[-1]
        phys = solved_phys[-1].copy()
        phys_prev = solved_phys[-2].copy() if len(solved_phys) >= 2 else None
        print(f"Resumed from {scan_file}: {len(solved_g)} points, g_max={g:.4f}", flush=True)
    else:
        # Start fresh: perturbative at g=0.05 → Newton converge
        g = 0.05
        params = perturbative_params(g, N0)
        result = solve_newton(params, KONISHI, g, config, tol=1e-10, max_iter=15, damped=True)
        print(f"Start: g={g}, ||E||={result['residual_norm']:.1e}, "
              f"D={float(jnp.real(result['params'][0]))+2:.10f}", flush=True)

        phys = i2p(result["params"], g)
        phys_prev = None
        solved_g = [g]
        solved_Delta = [float(jnp.real(result["params"][0])) + 2]
        solved_phys = [phys.copy()]

    # Dense scan with adaptive dg
    dg = 0.002
    success_count = 0
    t_start = time.time()

    while g < 1.0:
        g_new = round(g + dg, 6)

        # Linear extrapolation in physical space
        if phys_prev is not None:
            slope = (phys - phys_prev) / (solved_g[-1] - solved_g[-2])
            pred = phys + slope * (g_new - solved_g[-1])
        else:
            pred = phys.copy()

        params_pred = p2i(pred, g_new)
        res = solve_newton(params_pred, KONISHI, g_new, config,
                          tol=1e-10, max_iter=12, damped=True)

        if res["converged"] or res["residual_norm"] < 1e-4:
            phys_prev = phys.copy()
            phys = i2p(res["params"], g_new)
            g = g_new
            D = float(jnp.real(res["params"][0])) + 2
            solved_g.append(g)
            solved_Delta.append(D)
            solved_phys.append(phys.copy())
            success_count += 1

            if success_count > 4 and dg < 0.01:
                dg = min(dg * 1.5, 0.01)
                success_count = 0

            # Report at reference points
            ref_val = ref_dict.get(round(g, 2))
            if ref_val and abs(g - round(g, 2)) < 0.003:
                diff = abs(D - ref_val)
                digits = -math.log10(max(diff / abs(ref_val), 1e-16))
                dt = time.time() - t_start
                print(f"g={g:.2f}: D={D:.10f} ref={ref_val:.10f} "
                      f"dig={digits:.1f} ||E||={res['residual_norm']:.1e} "
                      f"dg={dg:.4f} [{len(solved_g)}pts {dt:.0f}s]", flush=True)
            elif len(solved_g) % 20 == 0:
                dt = time.time() - t_start
                print(f"g={g:.4f}: D={D:.8f} [{len(solved_g)}pts {dt:.0f}s dg={dg:.4f}]",
                      flush=True)
        else:
            dg /= 2
            success_count = 0
            if dg < 1e-4:
                print(f"STUCK at g={g_new:.4f}, dg<1e-4", flush=True)
                break

        # Save frequently (every 10 points) for crash resilience
        if len(solved_g) % 10 == 0:
            _save(solved_g, solved_Delta, solved_phys, ref_dict)

    dt = time.time() - t_start
    print(f"\nDone: {len(solved_g)} points in {dt:.0f}s, "
          f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.3f}]", flush=True)

    _save(solved_g, solved_Delta, solved_phys, ref_dict)


def _save(solved_g, solved_Delta, solved_phys, ref_dict):
    """Save scan results."""
    np.savez("data/konishi_dense_scan.npz",
             g=np.array(solved_g),
             Delta=np.array(solved_Delta),
             phys=np.array(solved_phys))
    print(f"  [saved {len(solved_g)} points to data/konishi_dense_scan.npz]", flush=True)


if __name__ == "__main__":
    main()
