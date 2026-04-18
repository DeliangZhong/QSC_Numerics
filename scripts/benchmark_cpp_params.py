"""Benchmark forward_map_flint at C++ parameters vs our Phase 14 parameters.

C++ reference (run_konishi.py):
  cutP=16 (initial), cutQai=30, QaiShift=50, precGoal=-24 → dps≈186
Our Phase 14 (QS=8/dps=50 regime):
  cutP=16, cutQai=24, QaiShift=8, dps=50

Measures wall time per forward-map evaluation at several g values. Loads the
Phase 14 converged state as the input point (zero-padded to any needed cutP).
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from qsc.forward_map_flint import forward_map_flint
from qsc.quantum_numbers import KONISHI, compute_Mtint

Mt = np.array([2., 1., 0., -1.])
Mtint = compute_Mtint(KONISHI)


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


def bench_point(phys_src, g_src, cfg, g_test, label, n_trials=3):
    """Run forward_map with given config at g_test. Input state is the Phase 14
    state interpreted at g_test (same phys vector — residual will be large if
    g_test != g_src, which is fine; we just want timing)."""
    cutP = cfg["cutP"]
    N0 = cutP // 2
    phys_padded = pad_phys(phys_src, 32, N0)
    params = p2i(phys_padded, g_test, N0)

    # Warmup
    F0 = forward_map_flint(params, KONISHI, g_test,
                            cutP=cutP, nPoints=cutP + 2,
                            cutQai=cfg["cutQai"], QaiShift=cfg["QaiShift"],
                            dps=cfg["dps"])
    norm = float(np.max(np.abs(F0)))

    ts = []
    for _ in range(n_trials):
        t0 = time.time()
        _ = forward_map_flint(params, KONISHI, g_test,
                               cutP=cutP, nPoints=cutP + 2,
                               cutQai=cfg["cutQai"], QaiShift=cfg["QaiShift"],
                               dps=cfg["dps"])
        ts.append(time.time() - t0)
    t_med = sorted(ts)[len(ts) // 2]
    print(f"  {label:>30}: {t_med*1000:7.1f} ms/call  ||F||={norm:.2e}")


def main():
    d = np.load("data/konishi_adaptive_scan_phase14.npz")
    g_src = float(d["g"][-1])
    phys = d["phys"][-1]
    print(f"Source: g={g_src:.4f}, Phase 14 converged state (N0=32)")
    print()

    configs = {
        "OUR (QS=8, dps=50, cQ=24)":
            dict(cutP=16, cutQai=24, QaiShift=8, dps=50),
        "OUR (QS=8, dps=50, cutP=64)":
            dict(cutP=64, cutQai=24, QaiShift=8, dps=50),
        "C++ (QS=50, dps=186, cQ=30)":
            dict(cutP=16, cutQai=30, QaiShift=50, dps=186),
        "C++ (QS=50, dps=186, cutP=64)":
            dict(cutP=64, cutQai=30, QaiShift=50, dps=186),
    }

    for g_test in [0.10, 0.25, 0.50, 1.00]:
        print(f"--- g = {g_test:.2f} ---")
        for label, cfg in configs.items():
            try:
                bench_point(phys, g_src, cfg, g_test, label)
            except Exception as e:
                print(f"  {label:>30}: ERROR {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
