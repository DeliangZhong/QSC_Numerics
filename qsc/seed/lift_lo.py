"""Module 2: leading-order lift dictionary.

Maps the one-loop QQ-system to the leading QSC P-function data:
c[a,n]^LO (physical convention) and the one-loop anomalous coefficient.

A_a, B_i (MV asymptotic coefficients) are intentionally NOT carried here.
They are singular at the bare point (anomalous Δ = 0) — for Konishi the
relevant entry is 0/0, yielding NaN — and the finite-coupling engine
recomputes them from Δ at the real coupling g.  The MVP seed assembler
never consumes A/B from this module.

c_lo = 0 is the verification-gated MVP stub.  If the asymptotics-only seed
is not in the Newton basin at a usable g_start (Task 5), derive c_lo from
the O(g^0) QQ-relation here.  order != "LO" is reserved for the algebraic
Marboe-Volin engine B.
"""

from dataclasses import dataclass

import numpy as np

from qsc.seed.oneloop_qq import OneLoopQQ


@dataclass(frozen=True)
class LiftLO:
    c_lo: np.ndarray           # (4, N0) complex128 — leading c[a,n], physical convention
    anomalous_lo_coeff: float  # γ₂ in Δ = Δ0 + γ₂ g² + O(g⁴)


def lift(state: OneLoopQQ, N0: int, order: str = "LO") -> LiftLO:
    if order != "LO":
        raise NotImplementedError(
            f"order={order!r} requires the algebraic Marboe-Volin engine (B); "
            "MVP implements LO only"
        )
    c_lo = np.zeros((4, N0), dtype=np.complex128)
    return LiftLO(c_lo=c_lo, anomalous_lo_coeff=state.one_loop_anomalous_coeff)
