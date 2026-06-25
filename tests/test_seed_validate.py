"""Tests for qsc.seed.validate_seed (module 4).

The mpmath solve is ~5 min, so we do NOT call seed_and_solve/assess_seed for
real.  All slow paths are monkeypatched.
"""

import numpy as np
import pytest

from qsc.quantum_numbers import KONISHI
from qsc.seed.validate_seed import assess_seed, delta_reference, seed_and_solve


# ---------------------------------------------------------------------------
# test 1 — CSV reader
# ---------------------------------------------------------------------------


def test_delta_reference_reads_csv():
    # g=0.1 row is "0.1","2.1155063779452210568"
    val = delta_reference(0.1)
    assert val is not None
    assert abs(val - 2.1155) < 1e-3

    # No g=0.0731 row in the reference CSV
    assert delta_reference(0.0731) is None


# ---------------------------------------------------------------------------
# test 2 — seed_and_solve wiring (monkeypatched solver)
# ---------------------------------------------------------------------------


def test_seed_and_solve_wiring(monkeypatch):
    fake_params = np.array([0.0155 + 0j] + [0] * 32, dtype=np.complex128)
    fake_result = {
        "params": fake_params,
        "residual_norm": 0.3,
        "iterations": 2,
        "converged": False,
    }

    import qsc.seed.validate_seed as vsmod

    monkeypatch.setattr(vsmod, "solve_newton_mp", lambda *a, **kw: fake_result)

    result = seed_and_solve(KONISHI, 0.1)

    # Delta = Delta0 + real(params[0]) = 2 + 0.0155 = 2.0155
    assert abs(result["Delta"] - 2.0155) < 1e-10

    # seed must be a 1-D complex array of length 33 (1 + 4*8 for cutP=16)
    seed = result["seed"]
    assert isinstance(seed, np.ndarray)
    assert seed.dtype == np.complex128
    assert seed.shape == (33,)


# ---------------------------------------------------------------------------
# test 3 — assess_seed in_basin flag
# ---------------------------------------------------------------------------


def test_assess_seed_in_basin_flag(monkeypatch):
    # Produce params[0] = Delta_ref - Delta0 so that Delta == Delta_ref(0.1)
    delta_ref_01 = delta_reference(0.1)  # ~2.1155...
    anomalous = delta_ref_01 - KONISHI.Delta0  # = delta_ref_01 - 2

    fake_params = np.array([anomalous + 0j] + [0] * 32, dtype=np.complex128)
    fake_result = {
        "params": fake_params,
        "residual_norm": 0.3,
        "iterations": 3,
        "converged": False,
    }

    import qsc.seed.validate_seed as vsmod

    monkeypatch.setattr(vsmod, "solve_newton_mp", lambda *a, **kw: fake_result)

    out = assess_seed(KONISHI, 0.1)

    assert out["in_basin"] is True
    assert out["delta_err"] is not None
    assert out["delta_err"] < 1e-3
