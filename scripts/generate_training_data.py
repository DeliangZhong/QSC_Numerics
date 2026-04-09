"""Generate converged QSC solutions at weak coupling for ML training."""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from qsc.forward_map import SolverConfig, params_to_V, V_to_params
from qsc.newton import solve_newton
from qsc.perturbative import perturbative_params
from qsc.quantum_numbers import KONISHI, compute_gauge_info, compute_Mtint


def load_cpp_fixture(path: str) -> jnp.ndarray:
    """Load C++ internal params from fixture."""
    with open(path) as f:
        cpp = json.load(f)
    N0 = 8
    c = jnp.zeros((4, N0 + 1), dtype=complex)
    for a in range(4):
        for n in range(N0 + 1):
            v = cpp[f"c_internal_{a}"][n]
            if a in (0, 2): v = 1j * v
            c = c.at[a, n].set(v)
    return jnp.concatenate([jnp.array([cpp["anomalous_delta"] + 0j]),
                            c[0, 1:], c[1, 1:], c[2, 1:], c[3, 1:]])


def internal_to_physical(params, g, Mt, N0):
    """Convert internal params to physical convention (smooth in g)."""
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


def main():
    config = SolverConfig(cutP=16, nPoints=18, cutQai=24, QaiShift=4,
                          use_mpmath=False)
    N0 = config.N0
    Mt = np.array([2., 1., 0., -1.])
    Mtint = compute_Mtint(KONISHI)
    gauge_info = compute_gauge_info(Mtint, N0)

    # Collect solutions
    solutions = []
    t_start = time.time()

    # From perturbative initial guesses (g = 0.02 to 0.10)
    print("Generating from perturbative initial guesses:", flush=True)
    for g_int in range(2, 11):
        g = g_int / 100.0
        params0 = perturbative_params(g, N0)
        result = solve_newton(params0, KONISHI, g, config, tol=1e-10,
                             max_iter=15, damped=True)
        if result["converged"] or result["residual_norm"] < 1e-5:
            phys = internal_to_physical(result["params"], g, Mt, N0)
            Delta = float(jnp.real(result["params"][0])) + 2
            solutions.append({"g": g, "Delta": Delta, "phys": phys,
                            "residual": result["residual_norm"]})
            print(f"  g={g:.2f}: Delta={Delta:.10f}, ||E||={result['residual_norm']:.1e}, "
                  f"iter={result['iterations']}", flush=True)
        else:
            print(f"  g={g:.2f}: FAILED ||E||={result['residual_norm']:.1e}", flush=True)

    # From C++ fixtures
    print("\nAdding C++ fixtures:", flush=True)
    for path, g in [("tests/fixtures/konishi_cpp_internal.json", 0.1),
                    ("tests/fixtures/konishi_cpp_g02.json", 0.2)]:
        params = load_cpp_fixture(path)
        phys = internal_to_physical(params, g, Mt, N0)
        Delta = float(jnp.real(params[0])) + 2
        solutions.append({"g": g, "Delta": Delta, "phys": phys, "residual": 0.0})
        print(f"  g={g:.2f}: Delta={Delta:.10f} (C++ fixture)", flush=True)

    # From reference spectral data (Delta only, no c-coefficients)
    with open("tests/fixtures/reference_spectral_data.json") as f:
        ref = json.load(f)
    konishi_ref = ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]

    dt = time.time() - t_start
    print(f"\n{len(solutions)} full solutions in {dt:.0f}s", flush=True)
    print(f"{len(konishi_ref)} reference (g, Delta) pairs available", flush=True)

    # Save
    g_arr = np.array([s["g"] for s in solutions])
    Delta_arr = np.array([s["Delta"] for s in solutions])
    phys_arr = np.array([s["phys"] for s in solutions])
    residual_arr = np.array([s["residual"] for s in solutions])

    ref_g = np.array([r[0] for r in konishi_ref])
    ref_Delta = np.array([r[1] for r in konishi_ref])

    np.savez("data/konishi_solutions.npz",
             g=g_arr, Delta=Delta_arr, phys=phys_arr, residual=residual_arr,
             ref_g=ref_g, ref_Delta=ref_Delta)
    print(f"\nSaved to data/konishi_solutions.npz", flush=True)


if __name__ == "__main__":
    main()
