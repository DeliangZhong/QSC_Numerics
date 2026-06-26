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


def _patch_solver(monkeypatch, anomalous, residual_norm, converged, iterations=3):
    fake_result = {
        "params": np.array([anomalous + 0j] + [0] * 32, dtype=np.complex128),
        "residual_norm": residual_norm,
        "iterations": iterations,
        "converged": converged,
    }
    import qsc.seed.validate_seed as vsmod

    monkeypatch.setattr(vsmod, "solve_newton_mp", lambda *a, **kw: fake_result)


def test_assess_seed_in_basin_when_delta_close_and_residual_at_floor(monkeypatch):
    # Δ == Δ_ref(0.1) and ‖F‖ at the floor (0.3 < residual_tol=1.0).
    anomalous = delta_reference(0.1) - KONISHI.Delta0
    _patch_solver(monkeypatch, anomalous, residual_norm=0.3, converged=False)

    out = assess_seed(KONISHI, 0.1)

    assert out["in_basin"] is True
    assert out["delta_err"] is not None and out["delta_err"] < 1e-3
    # transparency fields surfaced even though Newton's own flag is False
    assert out["converged"] is False
    assert out["iterations"] == 3
    assert out["residual_norm"] == 0.3


def test_assess_seed_not_in_basin_when_delta_close_but_residual_stalled(monkeypatch):
    """The decisive negative case: Δ lands within delta_tol of the reference,
    but ‖F‖ stalled far above the floor (a non-converged / degenerate solve).
    The gate MUST reject it — the Δ-only gate would have falsely certified it."""
    # Δ within 1e-3 of the reference but ‖F‖ = 56 (the observed c=0 stall).
    anomalous = (delta_reference(0.1) - KONISHI.Delta0) + 5e-4
    _patch_solver(monkeypatch, anomalous, residual_norm=56.0, converged=False)

    out = assess_seed(KONISHI, 0.1)

    assert out["delta_err"] < 1e-3          # Δ alone looks fine ...
    assert out["residual_norm"] == 56.0     # ... but the solve stalled
    assert out["in_basin"] is False         # so the gate rejects it
