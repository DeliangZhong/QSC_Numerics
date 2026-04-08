"""Scan Konishi Delta(g) using predictor-corrector continuation."""

import json
import math
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qsc.continuation import scan_predictor_corrector
from qsc.forward_map import SolverConfig
from qsc.quantum_numbers import KONISHI


def main():
    with open("tests/fixtures/konishi_cpp_internal.json") as f:
        cpp = json.load(f)

    N0 = 8
    c_full = jnp.zeros((4, N0 + 1), dtype=complex)
    for a in range(4):
        raw = cpp[f"c_internal_{a}"]
        for n in range(N0 + 1):
            val = raw[n]
            if a == 0 or a == 2:
                val = 1j * val
            c_full = c_full.at[a, n].set(val)

    params0 = jnp.concatenate([
        jnp.array([cpp["anomalous_delta"] + 0j]),
        c_full[0, 1:], c_full[1, 1:], c_full[2, 1:], c_full[3, 1:],
    ])

    config = SolverConfig(cutP=16, nPoints=18, cutQai=24, QaiShift=4)

    print("Konishi Delta(g) scan with predictor-corrector continuation")
    print(f"Config: cutP={config.cutP}, cutQai={config.cutQai}, QaiShift={config.QaiShift}")
    print()

    t0 = time.time()
    results = scan_predictor_corrector(
        params0, KONISHI, g_start=0.1, g_end=1.0, config=config,
        dg_init=0.005, tol=1e-10, max_iter=8,
    )
    dt = time.time() - t0

    n_ok = sum(1 for r in results if r["converged"])
    g_max = max((r["g"] for r in results), default=0)
    print(f"\n{len(results)} points, {n_ok} converged, g_max={g_max:.3f}, {dt:.0f}s")

    # Compare with reference
    with open("tests/fixtures/reference_spectral_data.json") as f:
        ref = json.load(f)
    konishi_ref = ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]
    ref_dict = {round(r[0], 4): r[1] for r in konishi_ref}

    print(f"\n{'g':>6s}  {'Delta':>14s}  {'Ref':>14s}  {'diff':>10s}  {'digits':>6s}")
    print("-" * 58)
    for r in results:
        g_r = round(r["g"], 2)
        if abs(r["g"] - g_r) < 0.003:
            ref_val = ref_dict.get(g_r)
            if ref_val:
                diff = abs(r["Delta"] - ref_val)
                digits = -math.log10(max(diff / abs(ref_val), 1e-16))
                print(f"{g_r:6.2f}  {r['Delta']:14.10f}  {ref_val:14.10f}  "
                      f"{diff:10.2e}  {digits:6.1f}")


if __name__ == "__main__":
    main()
