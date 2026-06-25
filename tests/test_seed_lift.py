import pytest
from qsc.quantum_numbers import KONISHI
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.lift_lo import LiftLO, lift


def test_lift_lo_shapes_and_anomalous():
    N0 = 8
    qq = solve_oneloop_qq(KONISHI)
    res = lift(qq, N0=N0, order="LO")
    assert isinstance(res, LiftLO)
    assert res.c_lo.shape == (4, N0)
    assert res.anomalous_lo_coeff == 12.0


def test_lift_higher_order_not_implemented():
    qq = solve_oneloop_qq(KONISHI)
    with pytest.raises(NotImplementedError):
        lift(qq, N0=8, order="NLO")
