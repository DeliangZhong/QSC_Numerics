import pytest
from qsc.quantum_numbers import KONISHI
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.lift_lo import LiftLO, lift


def test_lift_asymptotic_shapes_and_anomalous():
    N0 = 8
    qq = solve_oneloop_qq(KONISHI)
    res = lift(qq, N0=N0, order="asymptotic")
    assert isinstance(res, LiftLO)
    assert res.c_lo.shape == (4, N0)
    # asymptotic stub carries zero leading P-coefficients
    assert (res.c_lo == 0).all()
    assert res.anomalous_lo_coeff == 12.0


def test_lift_default_order_is_asymptotic_stub():
    qq = solve_oneloop_qq(KONISHI)
    res = lift(qq, N0=8)  # default order
    assert (res.c_lo == 0).all()


def test_lift_true_LO_not_implemented():
    """order='LO' must NOT silently return zeros — the real MV leading order
    is not yet ported, so it must raise rather than masquerade as complete."""
    qq = solve_oneloop_qq(KONISHI)
    with pytest.raises(NotImplementedError):
        lift(qq, N0=8, order="LO")


def test_lift_unknown_order_not_implemented():
    qq = solve_oneloop_qq(KONISHI)
    with pytest.raises(NotImplementedError):
        lift(qq, N0=8, order="NLO")
