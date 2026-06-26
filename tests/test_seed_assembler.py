import numpy as np
from qsc.quantum_numbers import KONISHI, compute_Mtint, compute_gauge_info
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.lift_lo import LiftLO, lift
from qsc.seed.seed_assembler import assemble_seed


def test_seed_layout_length_and_anomalous_slot():
    cutP = 16
    g = 0.1
    qq = solve_oneloop_qq(KONISHI)
    lo = lift(qq, N0=cutP // 2, order="asymptotic")
    seed = assemble_seed(lo, KONISHI, g, cutP=cutP)

    assert seed.dtype == np.complex128
    assert seed.shape == (1 + 4 * (cutP // 2),)
    # params[0] is anomalous Δ = γ₂ g² = 12 * 0.01 = 0.12 (LO estimate)
    assert abs(seed[0].real - 12.0 * g**2) < 1e-12
    assert abs(seed[0].imag) < 1e-15


def test_seed_gauge_zeros_present_stub():
    """asymptotic stub has c_lo = 0, so every c-entry (and gauge slot) is 0."""
    cutP = 16
    N0 = cutP // 2
    qq = solve_oneloop_qq(KONISHI)
    lo = lift(qq, N0=N0, order="asymptotic")
    seed = assemble_seed(lo, KONISHI, 0.1, cutP=cutP)
    assert np.allclose(seed[1:], 0)


def test_gauge_zero_targets_correct_packed_slot():
    """Regression for the gauge off-by-one.

    compute_gauge_info indexes the FULL c[a][0..N0] array (index 0 = the A_a
    term, not in params); params hold c[a][1..N0], so full index n_idx maps to
    packed offset n_idx-1.  With a synthetic nonzero c_lo we assert the gauge
    coefficient is zeroed and the ADJACENT free coefficient is preserved — a
    `+ n_idx` (off-by-one) assembler zeros the free slot and leaves the gauge
    slot polluted, failing this test.
    """
    cutP = 16
    N0 = cutP // 2
    g = 0.1
    gauge = compute_gauge_info(compute_Mtint(KONISHI), N0)["gauge_indices"]
    # Konishi has exactly one gauge index, (a=2, n_idx=1).
    assert gauge == [(2, 1)]
    a, n_idx = gauge[0]
    assert n_idx >= 1  # the n_idx == 0 (A-term) edge case is not exercised here

    c_lo = np.zeros((4, N0), dtype=np.complex128)
    c_lo[a, n_idx - 1] = 0.7   # the gauge coefficient (packed offset n_idx-1)
    c_lo[a, n_idx] = 0.9       # the adjacent free coefficient (packed offset n_idx)
    lo = LiftLO(c_lo=c_lo, anomalous_lo_coeff=12.0)
    seed = assemble_seed(lo, KONISHI, g, cutP=cutP)

    gauge_slot = 1 + a * N0 + (n_idx - 1)
    free_slot = 1 + a * N0 + n_idx
    # a=2 is an imaginary channel (i-factor), Mt[2]=0 ⇒ c_internal = 1j*c_phys.
    assert seed[gauge_slot] == 0                       # gauge coefficient zeroed
    assert abs(seed[free_slot] - 1j * 0.9) < 1e-12     # free coefficient preserved


def test_seed_denorm_and_ifactor_with_nonzero_clo():
    """With a synthetic nonzero c_lo, check denorm c/g^Mt and i-factor for a∈{0,2}."""
    cutP = 16
    N0 = cutP // 2
    g = 0.1
    c_lo = np.zeros((4, N0), dtype=np.complex128)
    c_lo[1, 0] = 0.5   # a=1 (real channel), n_idx=0
    c_lo[0, 0] = 0.3   # a=0 (imag channel), n_idx=0
    lo = LiftLO(c_lo=c_lo, anomalous_lo_coeff=12.0)
    seed = assemble_seed(lo, KONISHI, g, cutP=cutP)

    # a=1: real, c_internal = 0.5 / g^Mt[1] = 0.5 / 0.1^1 = 5.0
    assert abs(seed[1 + 1 * N0 + 0] - 5.0) < 1e-12
    # a=0: imag channel, c_internal = 1j * 0.3 / g^Mt[0] = 1j * 0.3 / 0.1^2 = 30j
    assert abs(seed[1 + 0 * N0 + 0] - 30.0j) < 1e-12
