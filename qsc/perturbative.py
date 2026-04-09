"""Perturbative initial guess from Marboe-Volin weak-coupling expansion."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def load_konishi_perturbative() -> dict:
    """Load Konishi perturbative coefficients."""
    fixture = Path(__file__).parent.parent / "tests" / "fixtures" / "konishi_perturbative.json"
    with open(fixture) as f:
        return json.load(f)


def perturbative_params(g: float, N0: int = 8) -> jnp.ndarray:
    """Evaluate perturbative expansion for Konishi at coupling g.

    Returns internal-format params: [anomalous_Delta, c[0][1],...,c[3][N0]]
    in the C++ denormalized convention.

    The perturbative expansion is:
      Delta = Delta0 + Σ_k gamma_k * g^k  (k=2,4,...,12)
      c_phys[a, n] = Σ_k cg[a, n, k] * g^k  (various k)

    Then denormalize: c_internal = c_phys / g^Mt[a]
    and apply imaginary factor for a=0,2.
    """
    data = load_konishi_perturbative()
    Mt = np.array([2.0, 1.0, 0.0, -1.0])

    # Anomalous dimension
    anomalous = 0.0
    for power, val in zip(data["delta_powers"], data["delta_values"]):
        anomalous += val * g ** power

    # c-coefficients in physical convention
    c_phys = np.zeros((4, N0))  # c_phys[a-1, n/2-1] for a=1..4, n=2,4,...,cutP
    for a_mma, n, k, val in zip(data["c_a"], data["c_n"], data["c_k"], data["c_val"]):
        a = a_mma - 1  # 0-indexed
        n_idx = n // 2 - 1  # n=2→0, n=4→1, ...
        if 0 <= n_idx < N0:
            c_phys[a, n_idx] += val * g ** k

    # Convert to internal convention
    params = np.zeros(1 + 4 * N0, dtype=complex)
    params[0] = anomalous
    for a in range(4):
        start = 1 + a * N0
        denorm = c_phys[a] / g ** Mt[a]
        if a in (0, 2):
            params[start:start + N0] = 1j * denorm
        else:
            params[start:start + N0] = denorm

    return jnp.array(params)
