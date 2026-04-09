"""Dense JAX scan with GD warmup + Newton, 4-pt polynomial interpolation."""

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


def solve_with_gd_warmup(params0, g, gd_steps=30, gd_lr=1e-6,
                         newton_max_iter=8):
    """Gradient descent warmup on ||F||² then Newton.

    GD has wider basin of attraction than Newton.
    """
    V = params_to_V(params0, gauge_indices, N0)

    def F_V(V):
        return forward_map_typeI(V_to_params(V, gauge_indices, N0),
                                KONISHI, g, config)

    # Phase 1: GD on ||F||² (wide basin, slow convergence)
    loss_fn = lambda V: jnp.sum(jnp.abs(F_V(V)) ** 2)
    grad_fn = jax.grad(loss_fn, holomorphic=True)

    best_V = V
    best_norm = float(jnp.max(jnp.abs(F_V(V))))

    for step in range(gd_steps):
        F = F_V(V)
        norm = float(jnp.max(jnp.abs(F)))
        if norm < best_norm:
            best_V = V
            best_norm = norm
        if norm < 0.01:  # Close enough for Newton
            break
        grad = grad_fn(V)
        grad_norm = float(jnp.max(jnp.abs(grad)))
        if grad_norm > 0:
            V = V - gd_lr * grad / grad_norm * norm  # Normalized step

    # Phase 2: Newton (narrow basin, fast convergence)
    params_gd = V_to_params(best_V, gauge_indices, N0)
    result = solve_newton(params_gd, KONISHI, g, config,
                         tol=1e-10, max_iter=newton_max_iter, damped=False)
    return result


def main():
    with open("tests/fixtures/reference_spectral_data.json") as f:
        ref = json.load(f)
    ref_dict = {round(r[0], 4): r[1]
                for r in ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]}

    # Resume or start fresh
    scan_file = "data/konishi_dense_v2.npz"
    if os.path.exists(scan_file):
        saved = np.load(scan_file)
        solved_g = list(saved["g"])
        solved_Delta = list(saved["Delta"])
        solved_phys = list(saved["phys"])
        g = solved_g[-1]
        print(f"Resumed: {len(solved_g)} pts, g_max={g:.4f}", flush=True)
    else:
        g = 0.02
        params = perturbative_params(g, N0)
        result = solve_newton(params, KONISHI, g, config, tol=1e-10,
                             max_iter=15, damped=True)
        phys = i2p(result["params"], g)
        solved_g = [g]
        solved_Delta = [float(jnp.real(result["params"][0])) + 2]
        solved_phys = [phys.copy()]
        print(f"Start: g={g}, D={solved_Delta[0]:.10f}", flush=True)

    dg = 0.002
    success_count = 0
    t_start = time.time()

    while g < 1.0 and time.time() - t_start < 3600:
        g_new = round(g + dg, 6)

        # 4-point polynomial interpolation
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

        params_pred = p2i(pred, g_new)

        # Try Newton first (fast)
        result = solve_newton(params_pred, KONISHI, g_new, config,
                             tol=1e-10, max_iter=5, damped=False)

        if not (result["converged"] or result["residual_norm"] < 1e-4):
            # Newton failed — try GD warmup + Newton
            result = solve_with_gd_warmup(params_pred, g_new)

        if result["converged"] or result["residual_norm"] < 1e-4:
            g = g_new
            phys = i2p(result["params"], g)
            D = float(jnp.real(result["params"][0])) + 2
            solved_g.append(g)
            solved_Delta.append(D)
            solved_phys.append(phys.copy())
            success_count += 1

            if success_count > 4 and dg < 0.01:
                dg = min(dg * 1.5, 0.01)
                success_count = 0

            ref_val = ref_dict.get(round(g, 2))
            if ref_val and abs(g - round(g, 2)) < 0.003:
                digits = -math.log10(max(abs(D - ref_val) / abs(ref_val), 1e-16))
                dt = time.time() - t_start
                print(f"g={round(g, 2):.2f}: D={D:.10f} ref={ref_val:.10f} "
                      f"dig={digits:.1f} ||E||={result['residual_norm']:.1e} "
                      f"dg={dg:.4f} [{len(solved_g)}pts {dt:.0f}s]", flush=True)
            elif len(solved_g) % 20 == 0:
                dt = time.time() - t_start
                print(f"g={g:.4f}: D={D:.8f} ||E||={result['residual_norm']:.1e} "
                      f"[{len(solved_g)}pts {dt:.0f}s dg={dg:.4f}]", flush=True)

            if len(solved_g) % 10 == 0:
                np.savez(scan_file, g=np.array(solved_g),
                        Delta=np.array(solved_Delta),
                        phys=np.array(solved_phys))
        else:
            dg /= 2
            success_count = 0
            if dg < 1e-4:
                print(f"STUCK at g={g_new:.5f}, ||E||={result['residual_norm']:.1e}, "
                      f"dg<1e-4", flush=True)
                break

    dt = time.time() - t_start
    np.savez(scan_file, g=np.array(solved_g), Delta=np.array(solved_Delta),
            phys=np.array(solved_phys))
    print(f"\nDone: {len(solved_g)} pts in {dt:.0f}s, "
          f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
