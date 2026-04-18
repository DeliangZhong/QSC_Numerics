"""QaiShift sweep diagnostic at g=0.2499 (Phase 14 converged state).

Loads the Phase 14 solution and evaluates forward-map residual at various
(QaiShift, cutP) combinations. The optimum (QS, cutP) pair gives the lowest
||E|| — this reveals the truncation-amplification balance at g~0.25.
"""
import os
import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from qsc.forward_map import V_to_params, params_to_V
from qsc.forward_map_flint import forward_map_flint
from qsc.quantum_numbers import KONISHI, compute_gauge_info, compute_Mtint

DPS = 50
CUTQAI = 24
Mt = np.array([2., 1., 0., -1.])
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


def main():
    # Load Phase 14 state at its last point (g≈0.2499, cutP=64 / N0=32).
    d = np.load("data/konishi_adaptive_scan_phase14.npz")
    g = float(d["g"][-1])
    D_ref_approx = float(d["Delta"][-1])
    phys = d["phys"][-1]
    N0_source = 32    # Phase 14 saved at cutP=64
    assert phys.shape[0] == 1 + 4 * N0_source, f"shape mismatch: {phys.shape}"

    print(f"Source state: g={g:.6f}, cutP_src={2*N0_source}, Δ={D_ref_approx:.8f}")
    print(f"Sweeping (QaiShift, cutP) pairs. Evaluating ||F(V_src_padded)||")
    print()

    QS_values = [4, 8, 12, 16, 20, 24]
    cutP_values = [64, 72, 80, 88]

    print(f"{'cutP':>5} | " + " ".join(f"QS={q:2d}" for q in QS_values))
    print("-" * 60)
    for cutP in cutP_values:
        N0 = cutP // 2
        phys_padded = pad_phys(phys, N0_source, N0)
        params = p2i(phys_padded, g, N0)
        gauge_info = compute_gauge_info(Mtint, N0)
        gauge_indices = gauge_info["gauge_indices"]

        row = f"{cutP:>5} | "
        norms = []
        for QS in QS_values:
            F = forward_map_flint(
                params, KONISHI, g,
                cutP=cutP, nPoints=cutP + 2,
                cutQai=CUTQAI, QaiShift=QS, dps=DPS,
            )
            norm = float(np.max(np.abs(F)))
            norms.append(norm)
            row += f"{norm:>7.1e} "
        print(row)

    print()
    print("(Smaller ||F|| = better truncation balance at source state.)")


if __name__ == "__main__":
    main()
