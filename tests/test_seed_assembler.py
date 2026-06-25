import numpy as np
from qsc.quantum_numbers import KONISHI, compute_Mtint, compute_gauge_info
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.lift_lo import lift
from qsc.seed.seed_assembler import assemble_seed


def test_seed_layout_length_and_anomalous_slot():
    cutP = 16
    g = 0.1
    qq = solve_oneloop_qq(KONISHI)
    lo = lift(qq, N0=cutP // 2, order="LO")
    seed = assemble_seed(lo, KONISHI, g, cutP=cutP)

    assert seed.dtype == np.complex128
    assert seed.shape == (1 + 4 * (cutP // 2),)
    # params[0] is anomalous Δ = γ₂ g² = 12 * 0.01 = 0.12 (LO estimate)
    assert abs(seed[0].real - 12.0 * g**2) < 1e-12
    assert abs(seed[0].imag) < 1e-15


def test_seed_gauge_zeros_present():
    """c_lo = 0 in MVP, so all c-entries are 0; gauge positions must be 0 too."""
    cutP = 16
    N0 = cutP // 2
    qq = solve_oneloop_qq(KONISHI)
    lo = lift(qq, N0=N0, order="LO")
    seed = assemble_seed(lo, KONISHI, 0.1, cutP=cutP)

    gauge = compute_gauge_info(compute_Mtint(KONISHI), N0)["gauge_indices"]
    for (a, n_idx) in gauge:
        flat = 1 + a * N0 + n_idx
        assert seed[flat] == 0
    # all c-entries zero in the MVP stub
    assert np.allclose(seed[1:], 0)


def test_seed_denorm_and_ifactor_with_nonzero_clo():
    """With a synthetic nonzero c_lo, check denorm c/g^Mt and i-factor for a∈{0,2}."""
    cutP = 16
    N0 = cutP // 2
    g = 0.1
    Mt = [2.0, 1.0, 0.0, -1.0]
    qq = solve_oneloop_qq(KONISHI)
    lo = lift(qq, N0=N0, order="LO")
    c_lo = np.zeros((4, N0), dtype=np.complex128)
    c_lo[1, 0] = 0.5   # a=1 (real channel), n_idx=0
    c_lo[0, 0] = 0.3   # a=0 (imag channel), n_idx=0
    lo2 = type(lo)(c_lo=c_lo, anomalous_lo_coeff=lo.anomalous_lo_coeff)
    seed = assemble_seed(lo2, KONISHI, g, cutP=cutP)

    # a=1: real, c_internal = 0.5 / g^Mt[1] = 0.5 / 0.1^1 = 5.0
    assert abs(seed[1 + 1 * N0 + 0] - 5.0) < 1e-12
    # a=0: imag channel, c_internal = 1j * 0.3 / g^Mt[0] = 1j * 0.3 / 0.1^2 = 30j
    assert abs(seed[1 + 0 * N0 + 0] - 30.0j) < 1e-12
