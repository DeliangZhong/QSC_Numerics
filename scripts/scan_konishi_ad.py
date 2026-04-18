"""Konishi scan with hybrid flint F + JAX AD Jacobian.

Uses:
  - flint forward map at dps=100 for the residual (accurate)
  - JAX float64 forward map at QS=12 for the Jacobian (via jax.jacfwd)

The JAX f64 Jacobian matches the flint Jacobian to ~5e-5 relative despite
the noisy f64 primal — the sensitivity (dF/dparams) is well-conditioned
even when the primal has pulldown amplification.

Usage: python scripts/scan_konishi_ad.py [--trim-to=G] [--fresh]
"""

import os
import sys
import json
import math
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from qsc.forward_map import (
    SolverConfig, V_to_params, params_to_V, forward_map_typeI,
)
from qsc.forward_map_flint import forward_map_flint
from qsc.quantum_numbers import KONISHI, compute_gauge_info, compute_Mtint


# --- Config ---
CUTP = 16
NPOINTS = 18
CUTQAI = 24
QAISHIFT = 12     # optimal per Implementation-23 diagnostic
DPS = 100

N0 = CUTP // 2
Mt = np.array([2., 1., 0., -1.])

ACCEPT_TOL = 5e-5    # same as QS=4 scan for fair comparison
DG_INIT = 0.001
DG_MIN = 1e-5
DG_MAX = 0.005
MAX_ITER = 6         # Newton iterations per point
SCAN_FILE = "data/konishi_ad_scan.npz"

Mtint = compute_Mtint(KONISHI)
gauge_info = compute_gauge_info(Mtint, N0)
gauge_indices = gauge_info["gauge_indices"]

# JAX configs
CFG_AD = SolverConfig(cutP=CUTP, nPoints=NPOINTS, cutQai=CUTQAI,
                       QaiShift=QAISHIFT, use_mpmath=False)


def i2p(params, g):
    phys = np.zeros(1 + 4 * N0)
    phys[0] = float(np.real(params[0]))
    for a in range(4):
        s = 1 + a * N0
        block = params[s:s + N0]
        if a in (0, 2):
            phys[s:s + N0] = np.imag(block) * g ** Mt[a]
        else:
            phys[s:s + N0] = np.real(block) * g ** Mt[a]
    return phys


def p2i(phys, g):
    internal = np.zeros(1 + 4 * N0, dtype=np.complex128)
    internal[0] = phys[0] + 0j
    for a in range(4):
        s = 1 + a * N0
        block = phys[s:s + N0] / g ** Mt[a]
        if a in (0, 2):
            internal[s:s + N0] = 1j * block
        else:
            internal[s:s + N0] = block + 0j
    return internal


def poly_interp(solved_g, solved_phys, g_new):
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


def make_F_and_J(g):
    """Build (F_flint_V, J_ad_V) closures for given g."""
    def F_flint_V(V):
        p = np.array(V_to_params(V, gauge_indices, N0), dtype=np.complex128)
        return forward_map_flint(p, KONISHI, g, cutP=CUTP, nPoints=NPOINTS,
                                  cutQai=CUTQAI, QaiShift=QAISHIFT, dps=DPS)

    def F_jax_V(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, KONISHI, g, CFG_AD)

    J_ad_V = jax.jacfwd(F_jax_V, holomorphic=True)
    return F_flint_V, J_ad_V


def newton_ad(V0, g, max_iter=MAX_ITER):
    """Newton with flint F + JAX AD J + backtracking.

    Returns (V, norm, n_iter, converged).
    """
    F_V, J_V = make_F_and_J(g)
    V = V0.copy()

    for i in range(max_iter):
        F_flint = F_V(V)
        norm = float(np.max(np.abs(F_flint)))
        if norm < 1e-12:
            return V, norm, i, True

        J = J_V(jnp.array(V))
        delta, _, _, _ = jnp.linalg.lstsq(J, -jnp.array(F_flint), rcond=1e-12)
        delta_np = np.array(delta)

        # Backtracking
        best_alpha, best_norm, best_F = 0, norm, F_flint
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.01]:
            V_trial = V + alpha * delta_np
            F_trial = F_V(V_trial)
            n_trial = float(np.max(np.abs(F_trial)))
            if n_trial < best_norm:
                best_alpha, best_norm, best_F = alpha, n_trial, F_trial

        if best_alpha == 0:
            return V, norm, i, norm < ACCEPT_TOL

        V = V + best_alpha * delta_np

    F_final = F_V(V)
    norm_final = float(np.max(np.abs(F_final)))
    return V, norm_final, max_iter, norm_final < ACCEPT_TOL


def load_reference_data():
    ref_path = "tests/fixtures/reference_spectral_data.json"
    if not os.path.exists(ref_path):
        return {}
    with open(ref_path) as f:
        ref = json.load(f)
    return {
        round(r[0], 4): r[1]
        for r in ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]
    }


def main():
    ref_dict = load_reference_data()

    trim_to = None
    for arg in sys.argv[1:]:
        if arg.startswith("--trim-to="):
            trim_to = float(arg.split("=")[1])

    # Resume or start from existing flint scan data
    if os.path.exists(SCAN_FILE) and "--fresh" not in sys.argv:
        saved = np.load(SCAN_FILE)
        solved_g = list(saved["g"])
        solved_Delta = list(saved["Delta"])
        solved_phys = list(saved["phys"])
        print(f"Resumed AD scan: {len(solved_g)} pts, "
              f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)
    else:
        # Bootstrap from flint FD scan (QS=4 data, up to g=0.183)
        base_file = "data/konishi_mp_scan.npz"
        base = np.load(base_file)
        solved_g = list(base["g"])
        solved_Delta = list(base["Delta"])
        solved_phys = list(base["phys"])
        print(f"Bootstrapped from {base_file}: {len(solved_g)} pts", flush=True)

    # Optional trim
    if trim_to is not None:
        n_before = len(solved_g)
        keep = [i for i, gg in enumerate(solved_g) if gg <= trim_to + 1e-6]
        solved_g = [solved_g[i] for i in keep]
        solved_Delta = [solved_Delta[i] for i in keep]
        solved_phys = [solved_phys[i] for i in keep]
        print(f"Trimmed: {n_before} -> {len(solved_g)} pts", flush=True)

    g = solved_g[-1]
    dg = DG_INIT
    success_count = 0
    t_start = time.time()

    while g < 1.0:
        g_new = round(g + dg, 6)

        pred = poly_interp(solved_g, solved_phys, g_new)
        params_pred = p2i(pred, g_new)
        V_pred = np.array(params_to_V(params_pred, gauge_indices, N0),
                          dtype=np.complex128)

        V_new, norm, n_iter, converged = newton_ad(V_pred, g_new)

        if converged or norm < ACCEPT_TOL:
            g = g_new
            params_new = np.array(V_to_params(V_new, gauge_indices, N0),
                                  dtype=np.complex128)
            phys = i2p(params_new, g)
            D = float(np.real(params_new[0])) + 2
            solved_g.append(g)
            solved_Delta.append(D)
            solved_phys.append(phys.copy())
            success_count += 1

            if success_count > 4 and dg < DG_MAX:
                dg = min(dg * 1.3, DG_MAX)
                success_count = 0

            ref_val = ref_dict.get(round(g, 2))
            if ref_val and abs(g - round(g, 2)) < 0.003:
                digits = -math.log10(max(abs(D - ref_val) / abs(ref_val), 1e-16))
                dt = time.time() - t_start
                print(f"g={round(g, 2):.2f}: D={D:.10f} ref={ref_val:.10f} "
                      f"dig={digits:.1f} ||E||={norm:.1e} dg={dg:.4f} "
                      f"[{len(solved_g)}pts {dt:.0f}s]", flush=True)
            elif len(solved_g) % 5 == 0:
                dt = time.time() - t_start
                print(f"g={g:.4f}: D={D:.8f} ||E||={norm:.1e} "
                      f"dg={dg:.4f} [{len(solved_g)}pts {dt:.0f}s]", flush=True)

            if len(solved_g) % 10 == 0:
                np.savez(SCAN_FILE, g=np.array(solved_g),
                         Delta=np.array(solved_Delta),
                         phys=np.array(solved_phys))
        else:
            dg /= 2
            success_count = 0
            if dg < DG_MIN:
                print(f"STUCK g={g_new:.5f} ||E||={norm:.1e} dg<{DG_MIN:.0e}",
                      flush=True)
                break

    np.savez(SCAN_FILE, g=np.array(solved_g),
             Delta=np.array(solved_Delta),
             phys=np.array(solved_phys))
    dt = time.time() - t_start
    print(f"\nDone: {len(solved_g)} pts in {dt:.0f}s, "
          f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
