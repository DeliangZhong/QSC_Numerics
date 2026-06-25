"""Module 2: leading-order lift dictionary.

Maps the one-loop QQ-system to the leading QSC P-function data:
A_a, B_i (fixed by quantum numbers alone) and c[a,n]^LO (physical convention).

MVP: c_lo = 0. The spec treats c_lo as verification-gated — if the asymptotics-
only seed is not in the Newton basin at a usable g_start (Task 5), derive c_lo
from the O(g^0) QQ-relation here. order != "LO" is reserved for engine B.
"""

from dataclasses import dataclass

import numpy as np

from qsc.quantum_numbers import (
    compute_A, compute_B, compute_Mhat, compute_Mhat0, compute_Mt,
    compute_Mtint, compute_kettoLAMBDA,
)
from qsc.seed.oneloop_qq import OneLoopQQ


@dataclass(frozen=True)
class LiftLO:
    A: np.ndarray              # (4,) complex — MV asymptotic coefficients A_a
    B: np.ndarray              # (4,) complex — MV asymptotic coefficients B_i
    c_lo: np.ndarray           # (4, N0) complex — leading c[a,n], physical convention
    anomalous_lo_coeff: float  # γ₂ in Δ = Δ0 + γ₂ g² + O(g⁴)


def lift(state: OneLoopQQ, N0: int, order: str = "LO") -> LiftLO:
    if order != "LO":
        raise NotImplementedError(
            f"order={order!r} requires the algebraic Marboe-Volin engine (B); "
            "MVP implements LO only"
        )
    qn = state.qn
    Mtint = compute_Mtint(qn)
    kL = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kL)
    Mhat0 = compute_Mhat0(qn, kL)
    Mhat = compute_Mhat(Mhat0, 0.0)  # LO: anomalous = 0

    A_arr, _, _ = compute_A(Mt, Mhat)
    B_arr = compute_B(Mt, Mhat)

    A = np.array(A_arr, dtype=np.complex128)
    B = np.array(B_arr, dtype=np.complex128)
    c_lo = np.zeros((4, N0), dtype=np.complex128)
    return LiftLO(A=A, B=B, c_lo=c_lo, anomalous_lo_coeff=state.one_loop_anomalous_coeff)
