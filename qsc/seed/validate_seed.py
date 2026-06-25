"""Module 4: Δ-gated reliability harness for the QSC seed generator.

Success metric: anomalous dimension Δ, NOT residual norm.
The mpmath forward map has a structural ‖F‖ floor (~0.27 at cutP=16)
even at the exact solution, so ‖F‖ < tol is unreachable.  Instead we
compare Newton's converged Δ against the reference CSV.

Public API
----------
delta_reference(g)        -> float | None
seed_and_solve(qn, g, ...) -> dict
assess_seed(qn, g, ...)   -> dict
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qsc.newton_mp import solve_newton_mp
from qsc.quantum_numbers import QuantumNumbers
from qsc.seed.lift_lo import lift
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.seed_assembler import assemble_seed

# ---------------------------------------------------------------------------
# Path to the reference CSV (relative to this file's package root).
# Layout: data/konishi_gDelta.csv with header "g","Delta"
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_KONISHI_CSV = _DATA_DIR / "konishi_gDelta.csv"


def delta_reference(g: float, csv_path: Path = _KONISHI_CSV) -> float | None:
    """Return the reference Δ for coupling *g* from the Konishi CSV, or None.

    Matches rows whose ``g`` value agrees with the requested ``g`` to within
    an absolute tolerance of 1e-12.  Returns ``None`` when no such row exists.

    Parameters
    ----------
    g:
        The coupling constant to look up.
    csv_path:
        Path to the reference CSV (default: ``data/konishi_gDelta.csv``).
    """
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            g_row = float(row["g"])
            if abs(g_row - g) < 1e-12:
                return float(row["Delta"])
    return None


def seed_and_solve(
    qn: QuantumNumbers,
    g: float,
    cutP: int = 16,
    nPoints: int = 18,
    cutQai: int = 24,
    QaiShift: int = 60,
    dps: int = 50,
    **newton_kw,
) -> dict:
    """Assemble the LO seed, run Newton, and return enriched result dict.

    Pipeline:
        solve_oneloop_qq(qn)
        → lift(state, N0, order="LO")
        → assemble_seed(lift_lo, qn, g, cutP)
        → solve_newton_mp(seed, qn, g, ...)

    Returns
    -------
    dict with all keys from ``solve_newton_mp`` plus:

    ``"seed"``
        The assembled seed vector (complex128 ndarray of length 1+4*N0).
    ``"Delta"``
        ``qn.Delta0 + float(np.real(result["params"][0]))`` — the converged
        anomalous dimension estimate.
    """
    N0 = cutP // 2

    # Build seed from one-loop data.
    state = solve_oneloop_qq(qn)
    lo = lift(state, N0, order="LO")
    seed = assemble_seed(lo, qn, g, cutP=cutP)

    # Call Newton solver (may be monkeypatched in tests).
    result = solve_newton_mp(
        seed,
        qn,
        g,
        cutP=cutP,
        nPoints=nPoints,
        cutQai=cutQai,
        QaiShift=QaiShift,
        dps=dps,
        **newton_kw,
    )

    # Enrich result.
    result = dict(result)  # shallow copy so we don't mutate caller's dict
    result["seed"] = seed
    result["Delta"] = qn.Delta0 + float(np.real(result["params"][0]))

    return result


def assess_seed(
    qn: QuantumNumbers,
    g: float,
    delta_tol: float = 1e-3,
    **kw,
) -> dict:
    """Run the reliability assessment for a single (qn, g) point.

    Calls :func:`seed_and_solve`, looks up the reference Δ via
    :func:`delta_reference`, and returns a diagnostic dict.

    Parameters
    ----------
    qn:
        Quantum numbers of the operator.
    g:
        Coupling constant.
    delta_tol:
        Tolerance on |Δ - Δ_ref| for the ``in_basin`` flag.
    **kw:
        Forwarded to :func:`seed_and_solve`.

    Returns
    -------
    dict with keys:

    ``"g"``            – the coupling constant.
    ``"Delta"``        – Newton's converged Δ estimate.
    ``"Delta_ref"``    – reference Δ from CSV (float or ``None``).
    ``"delta_err"``    – ``|Delta - Delta_ref|`` (float or ``None`` if no ref).
    ``"residual_norm"``– ``‖F‖`` from Newton.
    ``"in_basin"``     – ``True`` iff ``delta_err < delta_tol``.
    """
    result = seed_and_solve(qn, g, **kw)
    Delta = result["Delta"]
    Delta_ref = delta_reference(g)

    if Delta_ref is not None:
        delta_err: float | None = abs(Delta - Delta_ref)
        in_basin = delta_err < delta_tol
    else:
        delta_err = None
        in_basin = False

    return {
        "g": g,
        "Delta": Delta,
        "Delta_ref": Delta_ref,
        "delta_err": delta_err,
        "residual_norm": result["residual_norm"],
        "in_basin": in_basin,
    }
