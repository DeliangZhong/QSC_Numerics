"""Konishi scan with mpmath FD Jacobian + Broyden acceleration.

Phase 1: Use JAX float64 scan data (g<0.15) as interpolation base
Phase 2: Continue with mpmath FD Newton past the g≈0.157 barrier
Phase 3: Broyden rank-1 updates after first FD Jacobian (13s/pt vs 170s/pt)

Usage: python scripts/scan_konishi_mp.py [--resume]
"""

import os
import sys
import json
import math
import time

import numpy as np

# Ensure float64 for quantum_numbers
import jax
jax.config.update("jax_enable_x64", True)

try:
    from qsc.forward_map_flint import forward_map_flint as forward_map_mp
except ImportError:
    from qsc.forward_map_mp import forward_map_mp
from qsc.forward_map import params_to_V, V_to_params
from qsc.quantum_numbers import KONISHI, compute_gauge_info, compute_Mtint

# --- Config ---
CUTP = 16
NPOINTS = 18
CUTQAI = 24
QAISHIFT = 4
DPS = 50
FD_H = 1e-10

N0 = CUTP // 2
Mt = np.array([2., 1., 0., -1.])

ACCEPT_TOL = 5e-5
DG_INIT = 0.001
DG_MIN = 1e-5
DG_MAX = 0.005
MAX_BROYDEN_AGE = 5  # force FD refresh after this many g-points
SCAN_FILE = "data/konishi_mp_scan.npz"

# Gauge info (computed once)
Mtint = compute_Mtint(KONISHI)
gauge_info = compute_gauge_info(Mtint, N0)
gauge_indices = gauge_info["gauge_indices"]


def i2p(params, g):
    """Internal → physical convention."""
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
    """Physical → internal convention."""
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


def F_V(V, g):
    """Forward map in gauge-reduced V-space."""
    params = np.array(V_to_params(V, gauge_indices, N0), dtype=np.complex128)
    return forward_map_mp(params, KONISHI, g,
                          cutP=CUTP, nPoints=NPOINTS,
                          cutQai=CUTQAI, QaiShift=QAISHIFT, dps=DPS)


def fd_jacobian(V, g, F0):
    """Finite-difference Jacobian in V-space (square: dimV × dimV)."""
    n = len(V)
    m = len(F0)
    J = np.zeros((m, n), dtype=np.complex128)
    for j in range(n):
        V_pert = V.copy()
        V_pert[j] += FD_H
        J[:, j] = (F_V(V_pert, g) - F0) / FD_H
    return J


def newton_solve(V0, g, J_inv_init=None, max_iter=10):
    """Newton in V-space with adaptive Broyden/FD refresh.

    If J_inv_init is provided, starts with Broyden. If a Broyden step
    fails to reduce ||F|| by 50%, immediately recomputes full FD Jacobian.

    Returns (V, norm, n_iter, converged, J_inv, refreshed).
    refreshed=True means a fresh FD Jacobian was computed.
    """
    V = V0.copy()
    Fval = F_V(V, g)
    norm = float(np.max(np.abs(Fval)))
    refreshed = False

    if J_inv_init is not None:
        J_inv = J_inv_init.copy()
    else:
        J = fd_jacobian(V, g, Fval)
        J_inv = np.linalg.inv(J)
        refreshed = True

    for i in range(max_iter):
        if norm < 1e-10:
            return V, norm, i, True, J_inv, refreshed

        delta = -J_inv @ Fval

        # Take step and check quality
        V_new = V + delta
        F_new = F_V(V_new, g)
        norm_new = float(np.max(np.abs(F_new)))

        # If step didn't help, try FD refresh or damping
        if norm_new > 0.5 * norm:
            if not refreshed:
                # Broyden drifted — recompute fresh FD Jacobian
                J = fd_jacobian(V, g, Fval)
                J_inv = np.linalg.inv(J)
                refreshed = True
                delta = -J_inv @ Fval
                V_new = V + delta
                F_new = F_V(V_new, g)
                norm_new = float(np.max(np.abs(F_new)))

            # If still not helping, try damped steps
            if norm_new > 0.5 * norm:
                for alpha in [0.5, 0.25, 0.1, 0.01]:
                    V_trial = V + alpha * delta
                    F_trial = F_V(V_trial, g)
                    n_trial = float(np.max(np.abs(F_trial)))
                    if n_trial < norm:
                        V_new, F_new, norm_new = V_trial, F_trial, n_trial
                        break

        # Broyden rank-1 update of J_inv
        dx = V_new - V
        df = F_new - Fval
        denom = dx @ (J_inv @ df)
        if abs(denom) > 1e-50:
            u = dx - J_inv @ df
            J_inv = J_inv + np.outer(u, dx @ J_inv) / denom

        V, Fval, norm = V_new, F_new, norm_new

        # Stalling: accept if small enough
        if i >= 2 and norm < ACCEPT_TOL:
            return V, norm, i + 1, True, J_inv, refreshed

    return V, norm, max_iter, norm < ACCEPT_TOL, J_inv, refreshed


def load_reference_data():
    """Load (g, Delta) reference pairs."""
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

    # Resume or load JAX base data
    if os.path.exists(SCAN_FILE) and "--fresh" not in sys.argv:
        saved = np.load(SCAN_FILE)
        solved_g = list(saved["g"])
        solved_Delta = list(saved["Delta"])
        solved_phys = list(saved["phys"])
        print(f"Resumed: {len(solved_g)} pts, g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]",
              flush=True)
    else:
        # Load JAX scan data as base (47 good points up to g≈0.152)
        jax_data = np.load("data/konishi_dense_v2.npz")
        solved_g = list(jax_data["g"][:47])
        solved_Delta = list(jax_data["Delta"][:47])
        solved_phys = list(jax_data["phys"][:47])
        print(f"Loaded {len(solved_g)} JAX base points, "
              f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)

    g = solved_g[-1]
    dg = DG_INIT
    success_count = 0
    J_inv_current = None  # will be computed on first point
    broyden_age = 0
    t_start = time.time()

    while g < 1.0:
        g_new = round(g + dg, 6)

        # 4-pt polynomial interpolation → V-space
        pred = poly_interp(solved_g, solved_phys, g_new)
        params_pred = p2i(pred, g_new)
        V_pred = np.array(params_to_V(params_pred, gauge_indices, N0),
                          dtype=np.complex128)

        # Use Broyden if we have a recent J_inv, otherwise FD
        if J_inv_current is None or broyden_age >= MAX_BROYDEN_AGE:
            V_new, norm, _, converged, J_inv_new, _ = newton_solve(
                V_pred, g_new, J_inv_init=None, max_iter=10
            )
            J_inv_current = J_inv_new
            broyden_age = 0
            mode = "FD"
        else:
            V_new, norm, _, converged, J_inv_new, refreshed = newton_solve(
                V_pred, g_new, J_inv_init=J_inv_current, max_iter=8
            )
            J_inv_current = J_inv_new
            if refreshed:
                broyden_age = 0
                mode = "FD*"
            else:
                mode = "Br"

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
            broyden_age += 1

            # Adaptive dg: grow after successes
            if success_count > 4 and dg < DG_MAX:
                dg = min(dg * 1.3, DG_MAX)
                success_count = 0

            # Report
            ref_val = ref_dict.get(round(g, 2))
            if ref_val and abs(g - round(g, 2)) < 0.003:
                digits = -math.log10(max(abs(D - ref_val) / abs(ref_val), 1e-16))
                dt_total = time.time() - t_start
                print(f"g={round(g, 2):.2f}: D={D:.10f} ref={ref_val:.10f} "
                      f"dig={digits:.1f} ||E||={norm:.1e} dg={dg:.4f} "
                      f"[{mode} {len(solved_g)}pts {dt_total:.0f}s]", flush=True)
            elif len(solved_g) % 5 == 0:
                dt_total = time.time() - t_start
                print(f"g={g:.4f}: D={D:.8f} ||E||={norm:.1e} "
                      f"dg={dg:.4f} [{mode} {len(solved_g)}pts {dt_total:.0f}s]",
                      flush=True)

            # Checkpoint every 10 points
            if len(solved_g) % 10 == 0:
                np.savez(SCAN_FILE,
                         g=np.array(solved_g),
                         Delta=np.array(solved_Delta),
                         phys=np.array(solved_phys))
        else:
            dg /= 2
            success_count = 0
            # Force J refresh on failure
            J_inv_current = None
            broyden_age = 0
            if dg < DG_MIN:
                print(f"STUCK g={g_new:.5f} ||E||={norm:.1e} dg<{DG_MIN:.0e}",
                      flush=True)
                break

    # Final save
    np.savez(SCAN_FILE,
             g=np.array(solved_g),
             Delta=np.array(solved_Delta),
             phys=np.array(solved_phys))
    dt_total = time.time() - t_start
    print(f"\nDone: {len(solved_g)} pts in {dt_total:.0f}s, "
          f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
