"""Module 1: one-loop (rational, g=0) QQ-system data for a state.

Self-contained Python. For Konishi (sl(2) twist-2, spin S=2) the one-loop
solution is analytic. The dataclass and signature are shaped so a QQ_Galois
adapter can later supply OneLoopQQ for general/nested/super states.
"""

from dataclasses import dataclass

from qsc.quantum_numbers import QuantumNumbers


@dataclass(frozen=True)
class OneLoopQQ:
    """One-loop QQ-system solution for a single state/branch."""

    qn: QuantumNumbers
    bethe_roots: tuple[complex, ...]      # momentum-carrying Bethe roots
    one_loop_anomalous_coeff: float        # γ₂ in Δ = Δ0 + γ₂ g² + O(g⁴)
    sol: int                               # solution-branch label


def _konishi_oneloop() -> OneLoopQQ:
    """Konishi: sl(2) twist L=2, spin S=2. One-loop γ₂ = 12.

    One-loop Bethe roots for the two-magnon sl(2) state of length 2:
    the symmetric pair u = ±1/(2*sqrt(3)) (Baxter/Bethe solution).
    """
    from qsc.quantum_numbers import KONISHI

    u = 1.0 / (2.0 * 3.0 ** 0.5)
    return OneLoopQQ(
        qn=KONISHI,
        bethe_roots=(-u, u),
        one_loop_anomalous_coeff=12.0,
        sol=1,
    )


def solve_oneloop_qq(qn: QuantumNumbers) -> OneLoopQQ:
    """Return the one-loop QQ-system data for ``qn``.

    MVP supports Konishi only; other states raise NotImplementedError until
    the QQ_Galois adapter or a general nested solver is wired in.
    """
    if qn.nb == (0, 0) and qn.nf == (1, 1, 1, 1) and qn.na == (0, 0):
        return _konishi_oneloop()
    raise NotImplementedError(
        f"one-loop QQ-system not implemented for {qn}; "
        "wire in QQ_Galois adapter for general states"
    )
