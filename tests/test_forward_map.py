"""Test the forward map against the converged Konishi reference solution."""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from qsc.forward_map import SolverConfig, forward_map_typeI
from qsc.io_utils import mathematica_to_internal_params
from qsc.quantum_numbers import KONISHI


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def konishi_data():
    """Load Konishi converged reference data."""
    with open(FIXTURE_DIR / "konishi_converged_g01.json") as f:
        return json.load(f)


def test_forward_map_residual_near_zero(konishi_data):
    """The forward map evaluated at a converged solution should give ~0 residual."""
    g = konishi_data["g"]
    cutP = konishi_data["cutP"]
    config = SolverConfig(
        cutP=cutP,
        nPoints=konishi_data["nPoints"],
        cutQai=konishi_data["cutQai"],
        QaiShift=konishi_data["QaiShift"],
    )

    # Convert Mathematica params to internal format
    params = mathematica_to_internal_params(
        konishi_data["converged_params"], KONISHI, cutP
    )

    # Evaluate forward map
    residual = forward_map_typeI(params, KONISHI, g, config)

    # The residual should be very small (converged solution)
    norm = float(jnp.max(jnp.abs(residual)))
    print(f"Residual norm: {norm:.2e}")
    assert norm < 1e-8, f"Forward map residual too large: {norm:.2e}"


def test_quantum_numbers_konishi():
    """Verify Konishi quantum number derived quantities."""
    assert KONISHI.L == 2
    assert KONISHI.Delta0 == 2
