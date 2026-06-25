"""Module 3: assemble the engine `params` seed vector from LiftLO.

Pure formatting, no new physics. Produces
    [anomalous_Delta, c[0][1..N0], c[1][1..N0], c[2][1..N0], c[3][1..N0]]
matching forward_map_mp / solve_newton_mp. Generalizes (and replaces) the
Konishi-only qsc/perturbative.py.
"""

import numpy as np

from qsc.quantum_numbers import (
    QuantumNumbers, compute_Mt, compute_Mtint, compute_kettoLAMBDA,
    compute_gauge_info,
)
from qsc.seed.lift_lo import LiftLO


def assemble_seed(lift_lo: LiftLO, qn: QuantumNumbers, g: float,
                  cutP: int = 16) -> np.ndarray:
    """Pack LiftLO into the engine `params` vector at coupling ``g``."""
    N0 = cutP // 2
    Mtint = compute_Mtint(qn)
    kL = compute_kettoLAMBDA(Mtint)
    Mt = np.array(compute_Mt(Mtint, kL), dtype=np.float64)
    gauge_indices = compute_gauge_info(Mtint, N0)["gauge_indices"]

    params = np.zeros(1 + 4 * N0, dtype=np.complex128)
    # Anomalous dimension at LO: Δ - Δ0 = γ₂ g² (real).
    params[0] = lift_lo.anomalous_lo_coeff * g ** 2

    for a in range(4):
        denorm = lift_lo.c_lo[a] / g ** Mt[a]          # physical → internal
        if a in (0, 2):                                 # imaginary channels (MMA a=1,3)
            denorm = 1j * denorm
        params[1 + a * N0: 1 + (a + 1) * N0] = denorm

    # Enforce gauge zeros.
    for (a, n_idx) in gauge_indices:
        params[1 + a * N0 + n_idx] = 0.0

    return params
