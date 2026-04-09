"""Bootstrap scan: solve Konishi at increasing g, adding each solution to ML training."""

import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from qsc.forward_map import SolverConfig
from qsc.ml_predictor import DeltaPredictor, FullParamPredictor
from qsc.newton import solve_newton
from qsc.quantum_numbers import KONISHI


def internal_to_physical(params, g, Mt, N0):
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

    # Load existing solutions
    data = np.load("data/konishi_solutions.npz")

    # Load reference for validation
    with open("tests/fixtures/reference_spectral_data.json") as f:
        ref = json.load(f)
    ref_data = ref["Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1"]["data"]
    ref_dict = {round(r[0], 4): r[1] for r in ref_data}

    # Initialize predictor with existing solutions
    pred = DeltaPredictor(hidden_sizes=(128, 128))
    full_pred = FullParamPredictor(pred)

    for i in range(len(data['g'])):
        full_pred.add_solution(data['g'][i], data['phys'][i])

    # Train initial Delta predictor on existing + reference data
    g_train = np.concatenate([data['g'], data['ref_g']])
    D_train = np.concatenate([data['Delta'], data['ref_Delta']])
    print(f"Initial training: {len(g_train)} points", flush=True)
    pred.train(g_train, D_train, lr=3e-3, epochs=10000, verbose=False)

    # Bootstrap: solve at increasing g values
    g_targets = np.arange(0.02, 1.01, 0.01)  # g = 0.02, 0.03, ..., 1.00
    solved = {round(g, 3) for g in data['g']}  # already solved

    results = []
    t_start = time.time()
    retrain_counter = 0

    for g in g_targets:
        g_round = round(g, 3)
        if g_round in solved:
            continue

        # Predict initial guess
        params_pred = full_pred.predict_internal(g, N0)

        # Newton solve
        result = solve_newton(params_pred, KONISHI, g, config, tol=1e-10,
                             max_iter=15, damped=True)

        if result['converged'] or result['residual_norm'] < 1e-5:
            D = float(jnp.real(result['params'][0])) + 2
            phys = internal_to_physical(result['params'], g, Mt, N0)
            full_pred.add_solution(g, phys)
            solved.add(g_round)
            retrain_counter += 1

            # Validate against reference
            ref_val = ref_dict.get(round(g, 2))
            ref_str = ""
            if ref_val:
                diff = abs(D - ref_val)
                digits = -math.log10(max(diff / abs(ref_val), 1e-16))
                ref_str = f"  ref={ref_val:.6f} digits={digits:.1f}"

            if len(results) % 5 == 0 or ref_val:
                print(f"  g={g:.3f}: D={D:.8f} iter={result['iterations']} "
                      f"||E||={result['residual_norm']:.1e}{ref_str}", flush=True)

            results.append({"g": g, "Delta": D, "iter": result['iterations'],
                          "residual": result['residual_norm']})

            # Retrain Delta predictor every 10 new solutions
            if retrain_counter >= 10:
                g_new = np.array([r['g'] for r in results])
                D_new = np.array([r['Delta'] for r in results])
                g_all = np.concatenate([g_train, g_new])
                D_all = np.concatenate([D_train, D_new])
                pred.train(g_all, D_all, lr=1e-3, epochs=5000, verbose=False)
                retrain_counter = 0
                print(f"  [retrained on {len(g_all)} points]", flush=True)
        else:
            print(f"  g={g:.3f}: FAILED ||E||={result['residual_norm']:.1e} "
                  f"(ML Delta={float(pred.predict(np.array([g]))[0]):.6f})", flush=True)

    dt = time.time() - t_start
    n_solved = len(results)
    g_max = max(r['g'] for r in results) if results else 0
    print(f"\n{n_solved} new solutions in {dt:.0f}s, max g={g_max:.3f}", flush=True)

    # Save all solutions
    all_g = np.concatenate([data['g'], np.array([r['g'] for r in results])])
    all_D = np.concatenate([data['Delta'], np.array([r['Delta'] for r in results])])

    # Summary comparison
    print(f"\n{'g':>6s}  {'Delta':>12s}  {'Ref':>12s}  {'diff':>10s}  {'digits':>6s}")
    seen = set()
    for r in results:
        g_r = round(r['g'] * 20) / 20  # nearest 0.05
        if abs(r['g'] - g_r) < 0.01 and g_r not in seen:
            ref_val = ref_dict.get(round(g_r, 4))
            if ref_val:
                diff = abs(r['Delta'] - ref_val)
                digits = -math.log10(max(diff / abs(ref_val), 1e-16))
                print(f"{g_r:6.2f}  {r['Delta']:12.8f}  {ref_val:12.8f}  "
                      f"{diff:10.2e}  {digits:6.1f}")
                seen.add(g_r)


if __name__ == "__main__":
    main()
