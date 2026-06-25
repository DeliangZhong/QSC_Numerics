import numpy as np
import pytest
from qsc.quantum_numbers import (
    KONISHI, compute_Mtint, compute_kettoLAMBDA, compute_Mt,
    compute_Mhat0, compute_Mhat, compute_A, compute_B,
)
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.lift_lo import LiftLO, lift


def test_lift_lo_asymptotics_match_quantum_numbers():
    """A_a, B_i at LO use anomalous=0 (Δ=Δ0); must equal compute_A/compute_B
    evaluated at Mhat(Δ0)."""
    N0 = 8
    qq = solve_oneloop_qq(KONISHI)
    res = lift(qq, N0=N0, order="LO")
    assert isinstance(res, LiftLO)

    Mtint = compute_Mtint(KONISHI)
    kL = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kL)
    Mhat0 = compute_Mhat0(KONISHI, kL)
    Mhat = compute_Mhat(Mhat0, 0.0)  # anomalous = 0 at LO
    A_ref, _, _ = compute_A(Mt, Mhat)
    B_ref = compute_B(Mt, Mhat)

    assert np.allclose(np.array(res.A), np.array(A_ref), equal_nan=True)
    assert np.allclose(np.array(res.B), np.array(B_ref), equal_nan=True)


def test_lift_lo_shapes_and_anomalous():
    N0 = 8
    qq = solve_oneloop_qq(KONISHI)
    res = lift(qq, N0=N0, order="LO")
    assert res.A.shape == (4,)
    assert res.B.shape == (4,)
    assert res.c_lo.shape == (4, N0)
    assert res.anomalous_lo_coeff == 12.0


def test_lift_higher_order_not_implemented():
    qq = solve_oneloop_qq(KONISHI)
    with pytest.raises(NotImplementedError):
        lift(qq, N0=8, order="NLO")
