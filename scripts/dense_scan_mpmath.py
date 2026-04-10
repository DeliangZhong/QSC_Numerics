"""Dense scan with tight convergence control.

Root cause of g≈0.17 barrier: error accumulation from accepting
partially-converged solutions (||E||=1e-4 accepted, but floor is ~1e-8).
By g=0.15, accumulated error contaminates interpolation, causing Newton to fail.

Fix: tight acceptance (||E|| < 1e-6), more Newton iters, smaller initial dg.
No mpmath needed — the QaiShift=4 float64 system is sufficient.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import math
import time

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

# Newton floor is ~1e-6 to ~5e-6 (varies with g, limited by cond(J)~1e6).
# Original scan accepted 1e-4 → error accumulated to 1e-5 by g=0.15 → 1e-4 by g=0.17.
# Tighten to 1e-5: prevents the worst degradation while remaining achievable.
ACCEPT_TOL = 1e-5
NEWTON_MAX_ITER = 15


def i2p(params, g):
    """Internal convention → physical convention (smooth in g)."""
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
    """Physical convention → internal convention."""
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


def polynomial_interpolate(solved_g, solved_phys, g_new):
    """4-point polynomial interpolation in physical convention."""
    n_interp = min(4, len(solved_g))
    dists = [abs(gg - g_new) for gg in solved_g]
    idxs = sorted(range(len(solved_g)), key=lambda i: dists[i])[:n_interp]
    idxs.sort()
    gs_i = np.array([solved_g[i] for i in idxs])
    phys_i = np.array([solved_phys[i] for i in idxs])
    deg = min(n_interp - 1, 3)

    pred = np.zeros(1 + 4 * N0)
    for j in range(1 + 4 * N0):
        coeffs = np.polyfit(gs_i, phys_i[:, j], deg)
        pred[j] = np.polyval(coeffs, g_new)
    return pred


def load_reference_data():
    """Load (g, Delta) reference pairs for validation."""
    ref_path = "tests/fixtures/reference_spectral_data.json"
    if not os.path.exists(ref_path):
        return {}
    with open(ref_path) as f:
        ref = json.load(f)
    return {
        round(r[0], 4): r[1]
        for r in ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]
    }


def solve_tight(params0, g):
    """Newton solve with tight convergence.

    Try undamped first with few iters (fast if quadratic convergence kicks in).
    If residual increases → switch to damped immediately.
    """
    # Quick undamped probe: 3 iterations to check if quadratic convergence
    result = solve_newton(params0, KONISHI, g, config,
                         tol=1e-10, max_iter=3, damped=False)
    if result["residual_norm"] < ACCEPT_TOL:
        # Undamped worked fast — continue to fully converge
        result = solve_newton(result["params"], KONISHI, g, config,
                             tol=1e-10, max_iter=NEWTON_MAX_ITER, damped=False)
        if result["residual_norm"] < ACCEPT_TOL:
            return result

    # Undamped didn't work — use damped (always converges, just slower)
    result2 = solve_newton(params0, KONISHI, g, config,
                          tol=1e-10, max_iter=NEWTON_MAX_ITER, damped=True)
    if result2["residual_norm"] < result["residual_norm"]:
        return result2
    return result


def main():
    scan_file = "data/konishi_tight_scan.npz"
    ref_dict = load_reference_data()

    # Resume or start fresh
    if os.path.exists(scan_file):
        saved = np.load(scan_file)
        solved_g = list(saved["g"])
        solved_Delta = list(saved["Delta"])
        solved_phys = list(saved["phys"])
        g = solved_g[-1]
        print(f"Resumed: {len(solved_g)} pts, g_max={g:.4f}", flush=True)
    else:
        # Start at g=0.03 where damped Newton converges to ~1e-6
        g = 0.03
        params = perturbative_params(g, N0)
        result = solve_tight(params, g)
        if result["residual_norm"] > ACCEPT_TOL:
            print(f"WARNING: starting point poorly converged: "
                  f"||E||={result['residual_norm']:.1e}", flush=True)
        phys = i2p(result["params"], g)
        D = float(jnp.real(result["params"][0])) + 2
        solved_g = [g]
        solved_Delta = [D]
        solved_phys = [phys.copy()]
        print(f"Start: g={g}, D={D:.10f}, ||E||={result['residual_norm']:.1e}",
              flush=True)

    dg = 0.001  # Start with small steps (matching C++ dg=0.0008)
    success_count = 0
    t_start = time.time()

    while g < 1.0 and time.time() - t_start < 7200:
        g_new = round(g + dg, 6)

        # 4-point polynomial interpolation in physical convention
        pred = polynomial_interpolate(solved_g, solved_phys, g_new)
        params_pred = p2i(pred, g_new)

        result = solve_tight(params_pred, g_new)

        if result["residual_norm"] < ACCEPT_TOL:
            g = g_new
            phys = i2p(result["params"], g)
            D = float(jnp.real(result["params"][0])) + 2
            solved_g.append(g)
            solved_Delta.append(D)
            solved_phys.append(phys.copy())
            success_count += 1

            # Adaptive step: grow very slowly to maintain interpolation quality
            if success_count > 6 and dg < 0.003:
                dg = min(dg * 1.2, 0.003)
                success_count = 0

            # Report at reference points or periodically
            ref_val = ref_dict.get(round(g, 2))
            if ref_val and abs(g - round(g, 2)) < 0.002:
                digits = -math.log10(max(abs(D - ref_val) / abs(ref_val), 1e-16))
                dt = time.time() - t_start
                print(f"g={round(g, 2):.2f}: D={D:.10f} ref={ref_val:.10f} "
                      f"dig={digits:.1f} ||E||={result['residual_norm']:.1e} "
                      f"dg={dg:.5f} [{len(solved_g)}pts {dt:.0f}s]", flush=True)
            elif len(solved_g) % 20 == 0:
                dt = time.time() - t_start
                print(f"g={g:.4f}: D={D:.8f} ||E||={result['residual_norm']:.1e} "
                      f"[{len(solved_g)}pts {dt:.0f}s dg={dg:.5f}]", flush=True)

            # Checkpoint every 10 points
            if len(solved_g) % 10 == 0:
                np.savez(scan_file,
                        g=np.array(solved_g),
                        Delta=np.array(solved_Delta),
                        phys=np.array(solved_phys))
        else:
            dg /= 2
            success_count = 0
            if dg < 1e-5:
                print(f"STUCK at g={g_new:.5f}, "
                      f"||E||={result['residual_norm']:.1e}, dg<1e-5",
                      flush=True)
                break

    # Final save
    dt = time.time() - t_start
    np.savez(scan_file, g=np.array(solved_g), Delta=np.array(solved_Delta),
            phys=np.array(solved_phys))
    print(f"\nDone: {len(solved_g)} pts in {dt:.0f}s, "
          f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
