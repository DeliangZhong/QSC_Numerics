"""Diagnostic: sweep QaiShift at fixed g to find where accuracy degrades.

Tests both flint and mpmath forward maps at proportional dps to determine
whether the pulldown works correctly at high QaiShift.
"""

import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from qsc.quantum_numbers import KONISHI

# --- Load converged solution at g ≈ 0.1 ---
CUTP = 16
N0 = CUTP // 2
Mt = np.array([2., 1., 0., -1.])


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


def main():
    data = np.load("data/konishi_mp_scan.npz")
    g_arr = data["g"]
    phys_arr = data["phys"]

    # Use g closest to 0.1
    idx = np.argmin(np.abs(g_arr - 0.1))
    g = float(g_arr[idx])
    phys = phys_arr[idx]
    params = p2i(phys, g)
    print(f"g = {g:.6f}, Delta = {phys[0] + 2:.10f}")
    print(f"params[0] (Delta-2) = {params[0]:.10f}")
    print()

    # Import forward maps
    try:
        from qsc.forward_map_flint import forward_map_flint
        has_flint = True
    except ImportError:
        has_flint = False
        print("FLINT not available, using mpmath only")

    from qsc.forward_map_mp import forward_map_mp

    # Sweep QaiShift with proportional dps
    npoints = 18
    cutqai_values = [24, 30]  # also test C++ default cutQai=30
    qaishift_values = [4, 8, 12, 20, 30, 50]
    dps_base = 50  # minimum dps

    print(f"{'QaiShift':>8} {'cutQai':>6} {'dps':>5} {'||E||_flint':>14} "
          f"{'||E||_mp':>14} {'t_fl(ms)':>10} {'t_mp(ms)':>10}")
    print("-" * 85)

    for cutqai in cutqai_values:
        for qs in qaishift_values:
            # dps scales with QaiShift: ~3 digits lost per pulldown step
            dps = max(dps_base, 4 * qs + 20)

            # FLINT
            t_fl, norm_fl = 0.0, float('nan')
            if has_flint:
                t0 = time.perf_counter()
                try:
                    E_fl = forward_map_flint(params, KONISHI, g,
                                             cutP=CUTP, nPoints=npoints,
                                             cutQai=cutqai, QaiShift=qs,
                                             dps=dps)
                    norm_fl = float(np.max(np.abs(E_fl)))
                except Exception as e:
                    print(f"  FLINT error at QS={qs}: {e}")
                t_fl = (time.perf_counter() - t0) * 1000

            # mpmath
            t0 = time.perf_counter()
            try:
                E_mp = forward_map_mp(params, KONISHI, g,
                                      cutP=CUTP, nPoints=npoints,
                                      cutQai=cutqai, QaiShift=qs,
                                      dps=dps)
                norm_mp = float(np.max(np.abs(E_mp)))
            except Exception as e:
                norm_mp = float('nan')
                print(f"  mpmath error at QS={qs}: {e}")
            t_mp = (time.perf_counter() - t0) * 1000

            fl_str = f"{norm_fl:.4e}" if not np.isnan(norm_fl) else "N/A"
            mp_str = f"{norm_mp:.4e}" if not np.isnan(norm_mp) else "N/A"
            print(f"{qs:>8} {cutqai:>6} {dps:>5} {fl_str:>14} "
                  f"{mp_str:>14} {t_fl:>10.0f} {t_mp:>10.0f}")

        print()


if __name__ == "__main__":
    main()
