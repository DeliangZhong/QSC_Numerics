"""Adaptive-cutP Konishi scan, matching the C++ BoostShift strategy.

Starts at cutP=16 (C++ default). When Newton residual exceeds tolerance,
bumps cutP by 2 (up to CUTP_MAX). Base data is padded with zeros for new
high-n coefficients.

The insight from Implementation-27: at g=0.18, cutP=16 has floor 10^-5 but
cutP=20 has floor 10^-7. The C++ handles this with its adaptive cutP+=2
mechanism (run_konishi.py lines 189-194, 219-223).

Usage: python scripts/scan_konishi_adaptive.py [--trim-to=G] [--fresh]
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
CUTP_INIT = 16
CUTP_MAX = 128
CUTQAI = 24
QAISHIFT = 8
DPS = 50   # dps=100 test gave bit-identical results → floor is
           # truncation-limited, not arithmetic-limited. Keep at 50 for speed.
FD_H = 1e-6

Mt = np.array([2., 1., 0., -1.])
DG_INIT = 0.001
DG_MIN = 1e-5
DG_MAX = 0.005
MAX_BROYDEN_AGE = 10        # Phase 15B: amortize AD Jacobian over more points.
CUTP_BUMP_STEP = 4          # Phase 15B: bump +4 (not +2) for head room.
# Pre-bump disabled: jumping cutP too far at once causes Newton basin mismatch
# (large gap between poly_interp prediction and the actual solution manifold).
# Reactive bumps with CUTP_BUMP_STEP=4 climb more robustly through the g=0.25+
# regime. target_cutP still used as a guard for bidirectional reduction.
ENABLE_PREBUMP = False
CUTP_HEADROOM = 0
# Implementation-33 finding: cutP past target_cutP(g) + CUTP_CEIL_MARGIN
# introduces truncation noise rather than improving Δ. Cap reactive bumps so
# the scan accepts the intrinsic (QS=8, dps=50) floor instead of chasing it.
CUTP_CEIL_MARGIN = 12
SCAN_FILE = "data/konishi_adaptive_scan.npz"

# Phase 15A + empirical refit: cutP schedule calibrated to QS=8/dps=50 reality.
TARGET_DIGITS = 4
TOL_BASE = 5e-5   # Phase 14 baseline — never go tighter than this.


def target_cutP(g: float) -> int:
    """Empirical fit for QS=8/dps=50 setup. Observed g→cutP data:
      g=0.10 → 16, g=0.15 → 20, g=0.18 → 30, g=0.25 → 72.
    Exponential fit: 16 * exp(10*(g-0.1)) for g ≥ 0.1, capped at CUTP_MAX.
    For g < 0.10, cutP=16 is adequate (Phase 14 evidence)."""
    if g <= 0.10:
        base = CUTP_INIT   # 16
    else:
        base = int(math.ceil(CUTP_INIT * math.exp(10.0 * (g - 0.10))))
    base = max(CUTP_INIT, min(CUTP_MAX, base))
    return base + (base & 1)


def accept_tol(g: float, target_digits: int = TARGET_DIGITS) -> float:
    """Loosen tolerance at strong coupling: we only need target_digits in Δ.
    Floor at TOL_BASE (Phase 14's 5e-5), loosening to 10^-3 at g=1.0."""
    return max(TOL_BASE, 10.0 ** (-target_digits - 1 + 2.0 * g))


# Fallback floor used when a function needs a conservative tol regardless of g.
ACCEPT_TOL_FLOOR = TOL_BASE

Mtint = compute_Mtint(KONISHI)


def i2p(params, g, N0):
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


def p2i(phys, g, N0):
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


def pad_phys(phys_old, N0_old, N0_new):
    """Pad phys from N0_old to N0_new, zero-filling new high-n entries."""
    if N0_new == N0_old:
        return phys_old.copy()
    phys_new = np.zeros(1 + 4 * N0_new)
    phys_new[0] = phys_old[0]
    n_copy = min(N0_old, N0_new)
    for a in range(4):
        s_old = 1 + a * N0_old
        s_new = 1 + a * N0_new
        phys_new[s_new:s_new + n_copy] = phys_old[s_old:s_old + n_copy]
    return phys_new


def poly_interp(solved_g, solved_phys, g_new, N0):
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


class Scanner:
    def __init__(self, cutP, use_ad=True):
        self.use_ad = use_ad
        self.set_cutP(cutP)

    def set_cutP(self, cutP):
        self.cutP = cutP
        self.N0 = cutP // 2
        self.nPoints = cutP + 2
        gauge_info = compute_gauge_info(Mtint, self.N0)
        self.gauge_indices = gauge_info["gauge_indices"]
        if self.use_ad:
            self.cfg_ad = SolverConfig(
                cutP=self.cutP, nPoints=self.nPoints,
                cutQai=CUTQAI, QaiShift=QAISHIFT, use_mpmath=False
            )

    def F_V(self, V, g):
        """Forward map F(V, g). Returns NaN array if flint hits a degenerate
        (singular b-matrix) region — caller treats this as Newton failure."""
        p = np.array(V_to_params(V, self.gauge_indices, self.N0),
                     dtype=np.complex128)
        try:
            return forward_map_flint(
                p, KONISHI, g,
                cutP=self.cutP, nPoints=self.nPoints,
                cutQai=CUTQAI, QaiShift=QAISHIFT, dps=DPS)
        except ZeroDivisionError:
            # Singular b-coefficient matrix — return NaN so Newton backtracks.
            dimF = 1 + 4 * self.N0
            return np.full(dimF, np.nan, dtype=np.complex128)

    def _F_jax(self, V, g):
        p = V_to_params(V, self.gauge_indices, self.N0)
        return forward_map_typeI(p, KONISHI, g, self.cfg_ad)

    def ad_jacobian(self, V, g):
        import jax
        self.ad_calls = getattr(self, "ad_calls", 0) + 1
        J = jax.jacfwd(lambda V_: self._F_jax(V_, g),
                       holomorphic=True)(jnp.array(V))
        return np.array(J)

    def fd_jacobian(self, V, g, F0):
        n = len(V)
        J = np.zeros((len(F0), n), dtype=np.complex128)
        for j in range(n):
            Vp = V.copy()
            Vp[j] += FD_H
            J[:, j] = (self.F_V(Vp, g) - F0) / FD_H
        return J

    def get_jacobian(self, V, g, F0):
        if self.use_ad:
            return self.ad_jacobian(V, g)
        return self.fd_jacobian(V, g, F0)

    def newton_solve(self, V0, g, J_inv_init=None, max_iter=10,
                     tol=ACCEPT_TOL_FLOOR):
        V = V0.copy()
        Fval = self.F_V(V, g)
        norm = float(np.max(np.abs(Fval)))
        if not np.isfinite(norm):
            # Initial point already singular — cannot proceed.
            return V, float("inf"), 0, False, np.eye(len(V), dtype=np.complex128), True
        refreshed = False
        if J_inv_init is not None:
            J_inv = J_inv_init.copy()
        else:
            J = self.get_jacobian(V, g, Fval)
            J_inv = np.linalg.inv(J)
            refreshed = True

        for i in range(max_iter):
            if norm < 1e-10:
                return V, norm, i, True, J_inv, refreshed
            delta = -J_inv @ Fval
            V_new = V + delta
            F_new = self.F_V(V_new, g)
            norm_new = float(np.max(np.abs(F_new)))
            # Treat NaN (singular) as worse than current norm so we backtrack.
            step_bad = (not np.isfinite(norm_new)) or norm_new > 0.5 * norm
            if step_bad:
                if not refreshed:
                    J = self.get_jacobian(V, g, Fval)
                    J_inv = np.linalg.inv(J)
                    refreshed = True
                    delta = -J_inv @ Fval
                    V_new = V + delta
                    F_new = self.F_V(V_new, g)
                    norm_new = float(np.max(np.abs(F_new)))
                    step_bad = (not np.isfinite(norm_new)) or norm_new > 0.5 * norm
                if step_bad:
                    for alpha in [0.5, 0.25, 0.1, 0.01]:
                        V_trial = V + alpha * delta
                        F_trial = self.F_V(V_trial, g)
                        n_trial = float(np.max(np.abs(F_trial)))
                        if np.isfinite(n_trial) and n_trial < norm:
                            V_new, F_new, norm_new = V_trial, F_trial, n_trial
                            step_bad = False
                            break
                    if step_bad:
                        # All backtracks hit singular or worse — declare failure.
                        return V, norm, i, False, J_inv, refreshed
            dx = V_new - V
            df = F_new - Fval
            denom = dx @ (J_inv @ df)
            if abs(denom) > 1e-50 and np.isfinite(denom):
                u = dx - J_inv @ df
                J_inv = J_inv + np.outer(u, dx @ J_inv) / denom
            V, Fval, norm = V_new, F_new, norm_new
            if i >= 2 and norm < tol:
                return V, norm, i + 1, True, J_inv, refreshed
        return V, norm, max_iter, norm < tol, J_inv, refreshed


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


def save_state(solved_g, solved_Delta, solved_phys, cutP, cutP_hist=None):
    kwargs = dict(g=np.array(solved_g),
                  Delta=np.array(solved_Delta),
                  phys=np.array(solved_phys),
                  cutP=np.array(cutP))
    if cutP_hist is not None:
        kwargs["cutP_hist"] = np.array(cutP_hist)
    np.savez(SCAN_FILE, **kwargs)


def main():
    ref_dict = load_reference_data()
    trim_to = None
    for arg in sys.argv[1:]:
        if arg.startswith("--trim-to="):
            trim_to = float(arg.split("=")[1])

    if os.path.exists(SCAN_FILE) and "--fresh" not in sys.argv:
        saved = np.load(SCAN_FILE)
        cutP = int(saved["cutP"])
        solved_g = list(saved["g"])
        solved_Delta = list(saved["Delta"])
        solved_phys = list(saved["phys"])
        if "cutP_hist" in saved.files:
            solved_cutP = list(saved["cutP_hist"])
        else:
            solved_cutP = [cutP] * len(solved_g)
        print(f"Resumed: cutP={cutP}, {len(solved_g)} pts, "
              f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)
    else:
        cutP = CUTP_INIT
        base = np.load("data/konishi_mp_scan.npz")
        solved_g = list(base["g"])
        solved_Delta = list(base["Delta"])
        N0_base = 8
        N0_init = cutP // 2
        solved_phys = [pad_phys(p, N0_base, N0_init) for p in base["phys"]]
        solved_cutP = [cutP] * len(solved_g)
        print(f"Bootstrapped from cutP={N0_base*2}: "
              f"cutP={cutP}, {len(solved_g)} pts", flush=True)

    if trim_to is not None:
        keep = [i for i, gg in enumerate(solved_g) if gg <= trim_to + 1e-6]
        n_before = len(solved_g)
        solved_g = [solved_g[i] for i in keep]
        solved_Delta = [solved_Delta[i] for i in keep]
        solved_phys = [solved_phys[i] for i in keep]
        solved_cutP = [solved_cutP[i] for i in keep]
        print(f"Trimmed: {n_before} -> {len(solved_g)} pts", flush=True)

    scanner = Scanner(cutP)
    g = solved_g[-1]
    dg = DG_INIT
    success_count = 0
    J_inv_current = None
    broyden_age = 0
    consecutive_bumps = 0   # track bumps at current g_new
    cutP_change_ago = 99    # successful points since last cutP change (high = allow reduce now)
    V_cur = None            # current Newton state between retries
    t_start = time.time()

    while g < 1.0:
        g_new = round(g + dg, 6)
        tol_here = accept_tol(g_new)

        # Pre-bump conditionally disabled: reactive bump+4 climbs more safely.
        if ENABLE_PREBUMP:
            target_with_room = target_cutP(g_new) + CUTP_HEADROOM
            target_with_room = min(target_with_room + (target_with_room & 1),
                                   CUTP_MAX)
            if scanner.cutP < target_with_room:
                N0_old = scanner.N0
                N0_new = target_with_room // 2
                dt = time.time() - t_start
                print(f"[{dt:.0f}s] cutP PROMOTE: {scanner.cutP} -> "
                      f"{target_with_room}", flush=True)
                solved_phys = [pad_phys(p, N0_old, N0_new) for p in solved_phys]
                scanner.set_cutP(target_with_room)
                J_inv_current = None
                broyden_age = 0
                cutP_change_ago = 0
                V_cur = None

        if V_cur is None or consecutive_bumps == 0:
            pred = poly_interp(solved_g, solved_phys, g_new, scanner.N0)
            params_pred = p2i(pred, g_new, scanner.N0)
            V_pred = np.array(params_to_V(params_pred, scanner.gauge_indices,
                                           scanner.N0), dtype=np.complex128)
        else:
            V_pred = V_cur

        iters_budget = 20 if consecutive_bumps > 0 else 10
        if J_inv_current is None or broyden_age >= MAX_BROYDEN_AGE or consecutive_bumps > 0:
            V_new, norm, _, converged, J_inv_new, _ = scanner.newton_solve(
                V_pred, g_new, J_inv_init=None,
                max_iter=iters_budget, tol=tol_here
            )
            J_inv_current = J_inv_new
            broyden_age = 0
            mode = "J" if consecutive_bumps == 0 else f"J+{consecutive_bumps}"
        else:
            V_new, norm, _, converged, J_inv_new, refreshed = scanner.newton_solve(
                V_pred, g_new, J_inv_init=J_inv_current,
                max_iter=8, tol=tol_here
            )
            J_inv_current = J_inv_new
            mode = "J*" if refreshed else "Br"
            if refreshed:
                broyden_age = 0

        # Soft-accept: if cutP has hit the ceiling for this g, accept at 100×tol.
        # Past g=0.25, the truncation-limited floor dominates; strict tol is
        # unreachable without raising dps/QS. Loose soft-accept keeps scan
        # moving while digit accuracy is tracked in the ref-point log.
        at_ceiling = scanner.cutP >= min(target_cutP(g_new) + CUTP_CEIL_MARGIN,
                                          CUTP_MAX)
        soft_tol = 100 * tol_here if at_ceiling else tol_here
        if converged or norm < tol_here or (at_ceiling and norm < soft_tol):
            g = g_new
            params_new = np.array(V_to_params(V_new, scanner.gauge_indices,
                                              scanner.N0), dtype=np.complex128)
            phys = i2p(params_new, g, scanner.N0)
            D = float(np.real(params_new[0])) + 2
            solved_g.append(g)
            solved_Delta.append(D)
            solved_phys.append(phys.copy())
            solved_cutP.append(scanner.cutP)
            success_count += 1
            broyden_age += 1
            cutP_change_ago += 1
            consecutive_bumps = 0
            V_cur = None
            if at_ceiling and norm >= tol_here:
                dt = time.time() - t_start
                print(f"[{dt:.0f}s] SOFT-ACCEPT at g={g:.4f}: cutP="
                      f"{scanner.cutP} hit ceiling, ||E||={norm:.1e} "
                      f"(tol={tol_here:.1e}, soft_tol={soft_tol:.1e})",
                      flush=True)
            if success_count > 4 and dg < DG_MAX:
                dg = min(dg * 1.3, DG_MAX)
                success_count = 0

            ref_val = ref_dict.get(round(g, 2))
            if ref_val and abs(g - round(g, 2)) < 0.003:
                digits = -math.log10(max(abs(D - ref_val) / abs(ref_val), 1e-16))
                dt = time.time() - t_start
                print(f"g={round(g, 2):.2f}: D={D:.10f} ref={ref_val:.10f} "
                      f"dig={digits:.1f} ||E||={norm:.1e} tol={tol_here:.1e} "
                      f"dg={dg:.4f} cutP={scanner.cutP} "
                      f"[{mode} {len(solved_g)}pts {dt:.0f}s]",
                      flush=True)
            elif len(solved_g) % 5 == 0:
                dt = time.time() - t_start
                print(f"g={g:.4f}: D={D:.8f} ||E||={norm:.1e} "
                      f"tol={tol_here:.1e} dg={dg:.4f} cutP={scanner.cutP} "
                      f"[{mode} {len(solved_g)}pts {dt:.0f}s]",
                      flush=True)

            # Phase 15A: bidirectional reduction — shrink cutP when residual is
            # well below tol AND cutP is above paper's schedule. Anti-thrash:
            # require ≥5 successful points since the last cutP change.
            if (norm < 0.5 * tol_here
                    and scanner.cutP > target_cutP(g_new) + 2
                    and scanner.cutP > CUTP_INIT
                    and cutP_change_ago >= 5):
                new_cutP = scanner.cutP - 2
                dt = time.time() - t_start
                print(f"[{dt:.0f}s] cutP REDUCE: {scanner.cutP} -> {new_cutP} "
                      f"at g={g:.4f}, ||E||={norm:.1e} < "
                      f"{0.5*tol_here:.1e}", flush=True)
                N0_old = scanner.N0
                N0_new = new_cutP // 2
                solved_phys = [pad_phys(p, N0_old, N0_new) for p in solved_phys]
                scanner.set_cutP(new_cutP)
                J_inv_current = None
                broyden_age = 0
                cutP_change_ago = 0

            if len(solved_g) % 10 == 0:
                save_state(solved_g, solved_Delta, solved_phys,
                           scanner.cutP, cutP_hist=solved_cutP)
        else:
            # Reactive bump fallback. MAX_BUMPS=4 lets Newton climb to the
            # physics-required cutP within a single g_new. Bump by
            # CUTP_BUMP_STEP (4) so each bump buys head room for many
            # subsequent points (Phase 15B Broyden amortization).
            #
            # Cap bumps at target_cutP(g_new) + CUTP_CEIL_MARGIN — past this,
            # cutP injects truncation noise (Implementation-33) and further
            # bumps won't help.
            MAX_BUMPS = 4
            cutP_ceiling = min(target_cutP(g_new) + CUTP_CEIL_MARGIN, CUTP_MAX)
            if (norm < 1e-2 and scanner.cutP < cutP_ceiling
                    and consecutive_bumps < MAX_BUMPS):
                new_cutP = min(scanner.cutP + CUTP_BUMP_STEP, cutP_ceiling)
                dt = time.time() - t_start
                print(f"[{dt:.0f}s] cutP BUMP #{consecutive_bumps+1}: "
                      f"{scanner.cutP} -> {new_cutP} at g={g_new:.4f}, "
                      f"||E||={norm:.1e} tol={tol_here:.1e}", flush=True)
                N0_old = scanner.N0
                N0_new = new_cutP // 2
                solved_phys = [pad_phys(p, N0_old, N0_new) for p in solved_phys]
                params_new = np.array(V_to_params(V_new, scanner.gauge_indices,
                                                  scanner.N0), dtype=np.complex128)
                phys_new = i2p(params_new, g_new, N0_old)
                phys_padded = pad_phys(phys_new, N0_old, N0_new)
                params_padded = p2i(phys_padded, g_new, N0_new)
                scanner.set_cutP(new_cutP)
                V_cur = np.array(params_to_V(params_padded, scanner.gauge_indices,
                                              scanner.N0), dtype=np.complex128)
                consecutive_bumps += 1
                J_inv_current = None
                broyden_age = 0
                cutP_change_ago = 0
                continue

            dg /= 2
            success_count = 0
            consecutive_bumps = 0
            V_cur = None
            J_inv_current = None
            broyden_age = 0
            if dg < DG_MIN:
                print(f"STUCK g={g_new:.5f} ||E||={norm:.1e} "
                      f"tol={tol_here:.1e} dg<{DG_MIN:.0e} "
                      f"cutP={scanner.cutP}", flush=True)
                break

    save_state(solved_g, solved_Delta, solved_phys,
               scanner.cutP, cutP_hist=solved_cutP)
    dt = time.time() - t_start
    n_new = len(solved_g) - 110 if len(solved_g) > 110 else len(solved_g)
    ad_calls = getattr(scanner, "ad_calls", 0)
    print(f"\nDone: cutP={scanner.cutP}, {len(solved_g)} pts in {dt:.0f}s, "
          f"g=[{solved_g[0]:.3f}, {solved_g[-1]:.4f}]", flush=True)
    print(f"AD Jacobian calls: {ad_calls} "
          f"({100.0*ad_calls/max(n_new,1):.0f}% of new points)", flush=True)


if __name__ == "__main__":
    main()
