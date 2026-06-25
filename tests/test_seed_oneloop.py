import numpy as np
from qsc.quantum_numbers import KONISHI
from qsc.seed.oneloop_qq import OneLoopQQ, solve_oneloop_qq


def test_konishi_oneloop_anomalous_coeff_matches_csv_slope():
    """Δ ≈ Δ0 + γ₂ g² at weak coupling; konishi_gDelta.csv gives γ₂ = 12.

    Check against the smallest-g rows: Δ(0.001)=2.000011999952 ⇒
    (Δ-2)/g² = 11.999952 ≈ 12.
    """
    qq = solve_oneloop_qq(KONISHI)
    assert isinstance(qq, OneLoopQQ)
    assert qq.sol == 1
    assert abs(qq.one_loop_anomalous_coeff - 12.0) < 1e-3


def test_konishi_oneloop_has_two_bethe_roots():
    """Konishi is the sl(2) twist-2 spin-2 state: 2 magnon roots, symmetric ±u."""
    qq = solve_oneloop_qq(KONISHI)
    assert len(qq.bethe_roots) == 2
    roots = np.array(qq.bethe_roots)
    # roots come in a ± symmetric pair summing to ~0
    assert abs(roots.sum()) < 1e-9
