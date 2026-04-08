"""I/O utilities for converting between Mathematica and internal formats."""

import json

import jax.numpy as jnp

from qsc.quantum_numbers import QuantumNumbers, compute_kettoLAMBDA, compute_Mt, compute_Mtint


def load_konishi_fixture(path: str) -> dict:
    """Load the Konishi converged fixture JSON."""
    with open(path) as f:
        return json.load(f)


def mathematica_to_internal_params(mma_params: list, qn: QuantumNumbers,
                                   cutP: int) -> jnp.ndarray:
    """Convert Mathematica-format parameter vector to internal format.

    Mathematica TypeI format:
      [g, Delta, c[1,2], c[1,4], ..., c[1,cutP],
                 c[2,2], c[2,4], ..., c[2,cutP],
                 c[3,2], c[3,4], ..., c[3,cutP],
                 c[4,2], c[4,4], ..., c[4,cutP]]

    Internal format (C++ convention):
      [Delta, c[0,1], c[0,2], ..., c[0,N0],
              c[1,1], c[1,2], ..., c[1,N0],
              c[2,1], c[2,2], ..., c[2,N0],
              c[3,1], c[3,2], ..., c[3,N0]]

    Key differences:
    1. Mathematica uses 1-indexed a (1..4), C++ uses 0-indexed (0..3)
    2. Mathematica stores only even powers c[a,2], c[a,4], ...
       which map to internal c[a][1], c[a][2], ... (n = even_power/2)
    3. Mathematica normalizes by g^Mt[a]; C++ denormalizes
    4. c[0] and c[2] are purely imaginary in C++ convention,
       c[1] and c[3] are real

    For TypeI: N0 = cutP/2, and Mathematica has cutP/2 entries per a.
    """
    g = mma_params[0]
    Delta = mma_params[1]
    N0 = cutP // 2

    Mtint = compute_Mtint(qn)
    kettoLAMBDA = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kettoLAMBDA)

    # Extract c[a] blocks from Mathematica format
    # Mathematica c[a, 2k] for k=1..N0 → internal c[a-1][k]
    # But Mathematica stores them normalized: c_mma = c_phys
    # C++ denormalizes: c_cpp = c_mma / g^Mt[a]

    c_internal = jnp.zeros(4 * N0, dtype=complex)
    for a_mma in range(1, 5):
        a_cpp = a_mma - 1
        start_mma = 2 + (a_mma - 1) * N0
        c_block = jnp.array(mma_params[start_mma:start_mma + N0])

        # Denormalize: divide by g^Mt[a]
        c_denorm = c_block / g ** float(Mt[a_cpp])

        # Apply imaginary convention: a=0,2 are imaginary, a=1,3 are real
        if a_cpp == 0 or a_cpp == 2:
            c_denorm = 1j * c_denorm
        # else: c_denorm stays real

        start_internal = a_cpp * N0
        c_internal = c_internal.at[start_internal:start_internal + N0].set(c_denorm)

    # Anomalous dimension (Delta - Delta0 is what C++ calls "Delta")
    # Actually, looking at C++ code: Delta0 in C++ is the ANOMALOUS dimension,
    # i.e., what Mathematica calls Gamma = Delta - Delta0_bare.
    # But in forward_map, we use the full Delta for computing Mhat.
    # Let's store the anomalous dimension as C++ does.
    Delta0_bare = qn.Delta0
    anomalous = Delta - Delta0_bare

    return jnp.concatenate([jnp.array([anomalous]), c_internal])
