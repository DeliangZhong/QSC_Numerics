"""Module 2: weak-coupling lift dictionary.

Maps the one-loop QQ-system to QSC P-function seed data: the leading
c[a,n] coefficients (physical convention) and the one-loop anomalous
coefficient γ₂.

A_a, B_i (MV asymptotic coefficients) are intentionally NOT carried here.
They are singular at the bare point (anomalous Δ = 0) — for Konishi the
relevant entry is 0/0, yielding NaN — and the finite-coupling engine
recomputes them from Δ at the real coupling g.  The MVP seed assembler
never consumes A/B from this module.

Supported ``order`` values:

``"asymptotic"`` (default)
    The honest MVP stub: ``c_lo = 0``.  The seed carries only the one-loop
    anomalous Δ and the (engine-supplied) asymptotics — NOT the true
    Marboe-Volin leading-order P-coefficients.  Diagnostics show this seed
    is marginal (reaches Δ to ~1e-4 but stalls ‖F‖ above the floor), so it
    is explicitly labelled a stub, not a complete leading-order lift.

``"LO"``
    The true Marboe-Volin leading-order ``c_lo`` (e.g. for Konishi the
    nonzero integers cg[2,2]=3, cg[3,4]=-6, cg[4,2]=6, ...).  Not yet
    implemented — the order-by-order MV solver is being ported separately;
    until then this raises ``NotImplementedError`` rather than silently
    returning zeros that masquerade as the real leading order.
"""

from dataclasses import dataclass

import numpy as np

from qsc.seed.oneloop_qq import OneLoopQQ


@dataclass(frozen=True)
class LiftLO:
    c_lo: np.ndarray           # (4, N0) complex128 — leading c[a,n], physical convention
    anomalous_lo_coeff: float  # γ₂ in Δ = Δ0 + γ₂ g² + O(g⁴)


def lift(state: OneLoopQQ, N0: int, order: str = "asymptotic") -> LiftLO:
    """Build the weak-coupling seed lift; see module docstring for ``order``."""
    if order == "LO":
        raise NotImplementedError(
            "the true Marboe-Volin leading-order c_lo is not yet implemented "
            "(order-by-order MV solver is being ported); use order='asymptotic' "
            "for the explicit zero-c_lo stub seed"
        )
    if order != "asymptotic":
        raise NotImplementedError(
            f"order={order!r} is not supported; use 'asymptotic' (stub) "
            "or 'LO' (true leading order, not yet implemented)"
        )
    c_lo = np.zeros((4, N0), dtype=np.complex128)
    return LiftLO(c_lo=c_lo, anomalous_lo_coeff=state.one_loop_anomalous_coeff)
