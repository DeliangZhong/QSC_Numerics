"""Diagnostic: measure Newton basin of attraction at g=0.1 and g=0.2."""

import json
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qsc.forward_map import SolverConfig, forward_map_typeI, params_to_V, V_to_params
from qsc.quantum_numbers import KONISHI, compute_gauge_info, compute_Mtint


def load_cpp_params(fixture_path: str) -> jnp.ndarray:
    """Load C++ internal params from fixture JSON."""
    with open(fixture_path) as f:
        cpp = json.load(f)
    N0 = 8
    c_full = jnp.zeros((4, N0 + 1), dtype=complex)
    for a in range(4):
        raw = cpp[f"c_internal_{a}"]
        for n in range(N0 + 1):
            val = raw[n]
            if a in (0, 2):
                val = 1j * val
            c_full = c_full.at[a, n].set(val)
    return jnp.concatenate([
        jnp.array([cpp["anomalous_delta"] + 0j]),
        c_full[0, 1:], c_full[1, 1:], c_full[2, 1:], c_full[3, 1:],
    ])


def test_basin(g: float, fixture_path: str):
    """Test Newton convergence from perturbed solutions."""
    params_exact = load_cpp_params(fixture_path)
    config = SolverConfig(cutP=16, nPoints=18, cutQai=24, QaiShift=4, use_mpmath=False)
    N0 = config.N0
    Mtint = compute_Mtint(KONISHI)
    gauge_info = compute_gauge_info(Mtint, N0)
    gauge_indices = gauge_info["gauge_indices"]

    V_exact = params_to_V(params_exact, gauge_indices, N0)

    def F_V(V):
        p = V_to_params(V, gauge_indices, N0)
        return forward_map_typeI(p, KONISHI, g, config)

    # Baseline residual
    E0 = F_V(V_exact)
    print(f"\n=== g = {g} ===")
    print(f"Baseline ||E|| = {float(jnp.max(jnp.abs(E0))):.2e}")

    # Test perturbations
    print(f"\n{'pert':>6s}  {'||E0||':>10s}  {'undamped':>10s}  {'alpha=0.5':>10s}  "
          f"{'alpha=0.1':>10s}  {'best_alpha':>10s}  {'best_||E||':>10s}")
    print("-" * 80)

    for pert in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        # Perturb: multiply non-zero entries by (1 + pert)
        V_pert = V_exact * (1 + pert)

        E_pert = F_V(V_pert)
        norm_pert = float(jnp.max(jnp.abs(E_pert)))

        # Newton direction
        J = jax.jacfwd(F_V, holomorphic=True)(V_pert)
        delta, _, _, _ = jnp.linalg.lstsq(J, -E_pert, rcond=1e-12)

        # Test step sizes
        best_alpha = 0.0
        best_norm = norm_pert
        results = {}
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            V_trial = V_pert + alpha * delta
            E_trial = F_V(V_trial)
            n_trial = float(jnp.max(jnp.abs(E_trial)))
            results[alpha] = n_trial
            if n_trial < best_norm:
                best_norm = n_trial
                best_alpha = alpha

        print(f"{pert:6.3f}  {norm_pert:10.2e}  {results[1.0]:10.2e}  {results[0.5]:10.2e}  "
              f"{results[0.1]:10.2e}  {best_alpha:10.3f}  {best_norm:10.2e}")

    # Full Newton iterations (undamped) from 1% perturbation
    print(f"\nFull Newton (undamped) from 1% perturbation at g={g}:")
    V = V_exact * 1.01
    for it in range(15):
        E = F_V(V)
        norm = float(jnp.max(jnp.abs(E)))
        Delta = float(jnp.real(V[0])) + 2
        print(f"  iter {it:2d}: ||E||={norm:.4e}, Delta={Delta:.12f}")
        if norm < 1e-10:
            print("  CONVERGED!")
            break
        if norm > 1e6:
            print("  DIVERGED!")
            break
        J = jax.jacfwd(F_V, holomorphic=True)(V)
        delta, _, _, _ = jnp.linalg.lstsq(J, -E, rcond=1e-12)
        V = V + delta

    # AD vs FD Jacobian comparison at perturbed point
    print(f"\nAD vs FD Jacobian at 3% perturbation, g={g}:")
    V_pert = V_exact * 1.03
    J_ad = jax.jacfwd(F_V, holomorphic=True)(V_pert)
    h = 1e-7
    dimV = len(V_pert)
    J_fd = jnp.zeros((dimV, dimV), dtype=complex)
    E_base = F_V(V_pert)
    for j in range(dimV):
        scale = max(float(jnp.abs(V_pert[j])), 1.0)
        V_p = V_pert.at[j].add(h * scale)
        E_p = F_V(V_p)
        J_fd = J_fd.at[:, j].set((E_p - E_base) / (h * scale))
    rel_err = float(jnp.max(jnp.abs(J_ad - J_fd)) / jnp.max(jnp.abs(J_ad)))
    print(f"  Max |J_ad - J_fd| / max|J_ad| = {rel_err:.2e}")
    print(f"  cond(J_ad) = {float(jnp.linalg.cond(J_ad)):.2e}")
    print(f"  cond(J_fd) = {float(jnp.linalg.cond(J_fd)):.2e}")


if __name__ == "__main__":
    test_basin(0.1, "tests/fixtures/konishi_cpp_internal.json")
    test_basin(0.2, "tests/fixtures/konishi_cpp_g02.json")
