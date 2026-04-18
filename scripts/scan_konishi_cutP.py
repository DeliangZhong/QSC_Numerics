"""Konishi scan with larger cutP (N0=10, i.e., 10 c-coefficients per a).

Discovery: at g=0.18, the real barrier is cutP=16 (not QaiShift). Increasing
to cutP=20 drops ||E|| from 3e-5 to 1e-7 after Newton.

Uses flint forward map with FD Jacobian (matching Implementation-23 approach
but at cutP=20). Bootstraps from existing QS=4 scan data (cutP=16), padding
c-coefficients with zeros for the new high-n components.

Usage: python scripts/scan_konishi_cutP.py [--cutP=N] [--trim-to=G] [--fresh]
"""

import os
import sys
import json
import math
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from qsc.forward_map import V_to_params, params_to_V
from qsc.forward_map_flint import forward_map_flint
from qsc.quantum_numbers import KONISHI, compute_gauge_info, compute_Mtint


# --- Config ---
CUTP = 20  # was 16 in the original scan (N0=10 vs N0=8)
NPOINTS = CUTP + 2
CUTQAI = 24
QAISHIFT = 8
DPS = 50
FD_H = 1e-6  # need moderate h for cutP=20

N0 = CUTP // 2  # 10
Mt = np.array([2., 1., 0., -1.])

ACCEPT_TOL = 5e-5
DG_INIT = 0.001
DG_MIN = 1e-5
DG_MAX = 0.005
MAX_BROYDEN_AGE = 5
SCAN_FILE = "data/konishi_cutp20_scan.npz"

Mtint = compute_Mtint(KONISHI)
gauge_info = compute_gauge_info(Mtint, N0)
gauge_indices = gauge_info["gauge_indices"]


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


def pad_phys_from_cutP16(phys_old, g, N0_old=8):
    """Pad phys from cutP=16 (N0=8) to current N0, zero-filling new entries."""
    phys_new = np.zeros(1 + 4 * N0)
    phys_new[0] = phys_old[0]
    for a in range(4):
        s_old = 1 + a * N0_old
        s_new = 1 + a * N0
        phys_new[s_new:s_new + N0_old] = phys_old[s_old:s_old + N0_old]
    return phys_new


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


def F_V(V, g):
    params = np.array(V_to_params(V, gauge_indices, N0), dtype=np.complex128)
    return forward_map_flint(params, KONISHI, g,
                              cutP=CUTP, nPoints=NPOINTS,
                              cutQai=CUTQAI, QaiShift=QAISHIFT, dps=DPS)


def fd_jacobian(V, g, F0):
    n = len(V)
    m = len(F0)
    J = np.zeros((m, n), dtype=np.complex128)
    for j in range(n):
        Vp = V.copy()
        Vp[j] += FD_H
        J[:, j] = (F_V(Vp, g) - F0) / FD_H
    return J


def newton_solve(V0, g, J_inv_init=None, max_iter=10):
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
        V_new = V + delta
        F_new = F_V(V_new, g)
        norm_new = float(np.max(np.abs(F_new)))

        if norm_new > 0.5 * norm:
            if not refreshed:
                J = fd_jacobian(V, g, Fval)
                J_inv = np.linalg.inv(J)
                refreshed = True
                delta = -J_inv @ Fval
                V_new = V + delta
                F_new = F_V(V_new, g)
                norm_new = float(np.max(np.abs(F_new)))

            if norm_new > 0.5 * norm:
                for alpha in [0.5, 0.25, 0.1, 0.01]:
                    V_trial = V + alpha * delta
                    F_trial = F_V(V_trial, g)
                    n_trial = float(np.max(np.abs(F_trial)))
                    if n_trial < norm:
                        V_new, F_new, norm_new = V_trial, F_trial, n_trial
                        break

        dx = V_new - V
        df = F_new - Fval
        denom = dx @ (J_inv @ df)
        if abs(denom) > 1e-50:
            u = dx - J_inv @ df
            J_inv = J_inv + np.outer(u, dx @ J_inv) / denom

        V, Fval, norm = V_new, F_new, norm_new

        if i >= 2 and norm < ACCEPT_TOL:
            return V, norm, i + 1, True, J_inv, refreshed

    return V, norm, max_iter, norm < ACCEPT_TOL, J_inv, refreshed


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

    # Resume or bootstrap from cutP=16 scan
    if os.path.exists(SCAN_FILE) and "--fresh" not in sys.argv:
        saved = np.load(SCAN_FILE)
        solved_g = list(saved["g"])
        solved_Delta = list(saved["Delta"])
        solved_phys = list(saved["phys"])
        print(f"Resumed cutP={CUTP} scan: {len(solved_g)} pts, "
              f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)
    else:
        # Bootstrap from cutP=16 scan, padding with zeros
        base = np.load("data/konishi_mp_scan.npz")
        solved_g = list(base["g"])
        solved_Delta = list(base["Delta"])
        solved_phys = [pad_phys_from_cutP16(p, g)
                       for p, g in zip(base["phys"], base["g"])]
        print(f"Bootstrapped from cutP=16: {len(solved_g)} pts, "
              f"padded to cutP={CUTP}", flush=True)

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
    J_inv_current = None
    broyden_age = 0
    t_start = time.time()

    while g < 1.0:
        g_new = round(g + dg, 6)

        pred = poly_interp(solved_g, solved_phys, g_new)
        params_pred = p2i(pred, g_new)
        V_pred = np.array(params_to_V(params_pred, gauge_indices, N0),
                          dtype=np.complex128)

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
            mode = "FD*" if refreshed else "Br"
            if refreshed:
                broyden_age = 0

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

            if success_count > 4 and dg < DG_MAX:
                dg = min(dg * 1.3, DG_MAX)
                success_count = 0

            ref_val = ref_dict.get(round(g, 2))
            if ref_val and abs(g - round(g, 2)) < 0.003:
                digits = -math.log10(max(abs(D - ref_val) / abs(ref_val), 1e-16))
                dt = time.time() - t_start
                print(f"g={round(g, 2):.2f}: D={D:.10f} ref={ref_val:.10f} "
                      f"dig={digits:.1f} ||E||={norm:.1e} dg={dg:.4f} "
                      f"[{mode} {len(solved_g)}pts {dt:.0f}s]", flush=True)
            elif len(solved_g) % 5 == 0:
                dt = time.time() - t_start
                print(f"g={g:.4f}: D={D:.8f} ||E||={norm:.1e} "
                      f"dg={dg:.4f} [{mode} {len(solved_g)}pts {dt:.0f}s]", flush=True)

            if len(solved_g) % 10 == 0:
                np.savez(SCAN_FILE, g=np.array(solved_g),
                         Delta=np.array(solved_Delta),
                         phys=np.array(solved_phys))
        else:
            dg /= 2
            success_count = 0
            J_inv_current = None
            broyden_age = 0
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
