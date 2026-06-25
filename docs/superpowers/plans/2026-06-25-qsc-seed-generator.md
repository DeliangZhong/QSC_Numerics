# QSC Weak-Coupling Seed Generator (Type I / Konishi) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, fast weak-coupling seed generator that reliably bootstraps the existing QSC finite-coupling engine for the Konishi operator (Type I).

**Architecture:** Four small modules in `qsc/seed/` — (1) one-loop QQ-system, (2) leading-order lift dictionary, (3) seed assembler into the engine's `params` layout, (4) reliability validation. Approach "C now, B-ready": leading-order dictionary feeds the existing `newton_mp`/continuation, which acts as the order-by-order solver. Module boundaries are plain dataclasses so `QQ_Galois` (module 1) and an algebraic higher-order engine (module 2) can slot in later.

**Tech Stack:** Python 3.13, numpy, mpmath; existing `qsc/` package (`forward_map_mp`, `newton_mp`, `quantum_numbers`); pytest; uv-managed venv (`uv run pytest`).

## Global Constraints

- Seed does NOT affect the QSC result — only convergence reliability. All seed choices are reliability/speed decisions, never physics-correctness, except branch selection for degenerate Δ0 (out of scope for Konishi MVP).
- `params` layout consumed by `forward_map_mp`/`solve_newton_mp`: `[anomalous_Delta, c[0][1..N0], c[1][1..N0], c[2][1..N0], c[3][1..N0]]`, `dtype=np.complex128`, length `1 + 4*N0` where `N0 = cutP//2`. `params[0]` is the **anomalous** dimension (Δ−Δ0). `c[a][0]` is NOT in params (engine sets it from `A_a`).
- Internal normalization: `c_internal[a][n] = c_phys[a][n] / g**Mt[a]`, with a multiplicative `1j` factor for 0-indexed `a in {0, 2}` (Mathematica a=1,3). Konishi `Mt = [2, 1, 0, -1]`.
- Gauge-fixed zeros come from `qsc.quantum_numbers.compute_gauge_info(Mtint, N0)["gauge_indices"]` — list of `(a, n_idx)` set to 0.
- Konishi quantum numbers: `QuantumNumbers(nb=(0,0), nf=(1,1,1,1), na=(0,0), sol=1)`; `Δ0=2`, `L=2`.
- Default solver params (match `tests/fixtures/konishi_converged_g01.json`): `cutP=16, nPoints=18, cutQai=24, QaiShift=60, dps=50`.
- Run all commands via `uv run` from repo root `/Users/deliangzhong/Documents/Working/QSC_Numerics`.
- Validation anchors (real, in-repo): `konishi_gDelta.csv` leading slope ⇒ `γ₂=12` (Δ≈2+12g²); `tests/fixtures/konishi_internal_params.json[0]=0.11550637794522…` is the converged anomalous Δ at g=0.1.

---

## File Structure

- Create `qsc/seed/__init__.py` — package marker, re-exports `solve_oneloop_qq`, `lift`, `assemble_seed`.
- Create `qsc/seed/oneloop_qq.py` — module 1: `OneLoopQQ` dataclass + `solve_oneloop_qq(qn)`.
- Create `qsc/seed/lift_lo.py` — module 2: `LiftLO` dataclass + `lift(state, N0, order="LO")`.
- Create `qsc/seed/seed_assembler.py` — module 3: `assemble_seed(lift_lo, qn, g, cutP)`.
- Create `qsc/seed/validate_seed.py` — module 4: `find_basin_g(...)`, `seed_and_solve(...)`.
- Create `tests/test_seed_oneloop.py`, `tests/test_seed_lift.py`, `tests/test_seed_assembler.py`, `tests/test_seed_endtoend.py`.

---

## Task 1: Package scaffold + one-loop QQ-system (module 1)

**Files:**
- Create: `qsc/seed/__init__.py`
- Create: `qsc/seed/oneloop_qq.py`
- Test: `tests/test_seed_oneloop.py`

**Interfaces:**
- Consumes: `qsc.quantum_numbers.QuantumNumbers`, `KONISHI`.
- Produces:
  - `@dataclass(frozen=True) class OneLoopQQ: qn: QuantumNumbers; bethe_roots: tuple[complex, ...]; one_loop_anomalous_coeff: float; sol: int`
  - `solve_oneloop_qq(qn: QuantumNumbers) -> OneLoopQQ`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed_oneloop.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_oneloop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qsc.seed'`.

- [ ] **Step 3: Write minimal implementation**

```python
# qsc/seed/__init__.py
"""Weak-coupling seed generator for the QSC engine."""

from qsc.seed.oneloop_qq import OneLoopQQ, solve_oneloop_qq

__all__ = ["OneLoopQQ", "solve_oneloop_qq"]
```

```python
# qsc/seed/oneloop_qq.py
"""Module 1: one-loop (rational, g=0) QQ-system data for a state.

Self-contained Python. For Konishi (sl(2) twist-2, spin S=2) the one-loop
solution is analytic. The dataclass and signature are shaped so a QQ_Galois
adapter can later supply OneLoopQQ for general/nested/super states.
"""

from dataclasses import dataclass

from qsc.quantum_numbers import QuantumNumbers


@dataclass(frozen=True)
class OneLoopQQ:
    """One-loop QQ-system solution for a single state/branch."""

    qn: QuantumNumbers
    bethe_roots: tuple[complex, ...]      # momentum-carrying Bethe roots
    one_loop_anomalous_coeff: float        # γ₂ in Δ = Δ0 + γ₂ g² + O(g⁴)
    sol: int                               # solution-branch label


def _konishi_oneloop() -> OneLoopQQ:
    """Konishi: sl(2) twist L=2, spin S=2. One-loop γ₂ = 12.

    One-loop Bethe roots for the two-magnon sl(2) state of length 2:
    the symmetric pair u = ±1/(2*sqrt(3)) (Baxter/Bethe solution).
    """
    from qsc.quantum_numbers import KONISHI

    u = 1.0 / (2.0 * 3.0 ** 0.5)
    return OneLoopQQ(
        qn=KONISHI,
        bethe_roots=(-u, u),
        one_loop_anomalous_coeff=12.0,
        sol=1,
    )


def solve_oneloop_qq(qn: QuantumNumbers) -> OneLoopQQ:
    """Return the one-loop QQ-system data for ``qn``.

    MVP supports Konishi only; other states raise NotImplementedError until
    the QQ_Galois adapter or a general nested solver is wired in.
    """
    if qn.nb == (0, 0) and qn.nf == (1, 1, 1, 1) and qn.na == (0, 0):
        return _konishi_oneloop()
    raise NotImplementedError(
        f"one-loop QQ-system not implemented for {qn}; "
        "wire in QQ_Galois adapter for general states"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_oneloop.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add qsc/seed/__init__.py qsc/seed/oneloop_qq.py tests/test_seed_oneloop.py
git commit -m "feat(seed): module 1 — one-loop QQ-system for Konishi"
```

---

## Task 2: Leading-order lift dictionary (module 2)

**Files:**
- Create: `qsc/seed/lift_lo.py`
- Modify: `qsc/seed/__init__.py` (add re-export)
- Test: `tests/test_seed_lift.py`

**Interfaces:**
- Consumes: `OneLoopQQ` (Task 1); `qsc.quantum_numbers` functions `compute_Mtint, compute_kettoLAMBDA, compute_Mt, compute_Mhat0, compute_Mhat, compute_A, compute_B`.
- Produces:
  - `@dataclass(frozen=True) class LiftLO: A: np.ndarray; B: np.ndarray; c_lo: np.ndarray; anomalous_lo_coeff: float` where `A,B` are shape `(4,)` complex, `c_lo` is shape `(4, N0)` complex in **physical** convention.
  - `lift(state: OneLoopQQ, N0: int, order: str = "LO") -> LiftLO`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed_lift.py
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

    assert np.allclose(np.array(res.A), np.array(A_ref))
    assert np.allclose(np.array(res.B), np.array(B_ref))


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_lift.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qsc.seed.lift_lo'`.

- [ ] **Step 3: Write minimal implementation**

```python
# qsc/seed/lift_lo.py
"""Module 2: leading-order lift dictionary.

Maps the one-loop QQ-system to the leading QSC P-function data:
A_a, B_i (fixed by quantum numbers alone) and c[a,n]^LO (physical convention).

MVP: c_lo = 0. The spec treats c_lo as verification-gated — if the asymptotics-
only seed is not in the Newton basin at a usable g_start (Task 5), derive c_lo
from the O(g^0) QQ-relation here. order != "LO" is reserved for engine B.
"""

from dataclasses import dataclass

import numpy as np

from qsc.quantum_numbers import (
    compute_A, compute_B, compute_Mhat, compute_Mhat0, compute_Mt,
    compute_Mtint, compute_kettoLAMBDA,
)
from qsc.seed.oneloop_qq import OneLoopQQ


@dataclass(frozen=True)
class LiftLO:
    A: np.ndarray              # (4,) complex — MV asymptotic coefficients A_a
    B: np.ndarray              # (4,) complex — MV asymptotic coefficients B_i
    c_lo: np.ndarray           # (4, N0) complex — leading c[a,n], physical convention
    anomalous_lo_coeff: float  # γ₂ in Δ = Δ0 + γ₂ g² + O(g⁴)


def lift(state: OneLoopQQ, N0: int, order: str = "LO") -> LiftLO:
    if order != "LO":
        raise NotImplementedError(
            f"order={order!r} requires the algebraic Marboe-Volin engine (B); "
            "MVP implements LO only"
        )
    qn = state.qn
    Mtint = compute_Mtint(qn)
    kL = compute_kettoLAMBDA(Mtint)
    Mt = compute_Mt(Mtint, kL)
    Mhat0 = compute_Mhat0(qn, kL)
    Mhat = compute_Mhat(Mhat0, 0.0)  # LO: anomalous = 0

    A_arr, _, _ = compute_A(Mt, Mhat)
    B_arr = compute_B(Mt, Mhat)

    A = np.array(A_arr, dtype=np.complex128)
    B = np.array(B_arr, dtype=np.complex128)
    c_lo = np.zeros((4, N0), dtype=np.complex128)
    return LiftLO(A=A, B=B, c_lo=c_lo, anomalous_lo_coeff=state.one_loop_anomalous_coeff)
```

Also add to `qsc/seed/__init__.py`:

```python
from qsc.seed.lift_lo import LiftLO, lift

__all__ = ["OneLoopQQ", "solve_oneloop_qq", "LiftLO", "lift"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_lift.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add qsc/seed/lift_lo.py qsc/seed/__init__.py tests/test_seed_lift.py
git commit -m "feat(seed): module 2 — leading-order lift dictionary (asymptotics + c_lo stub)"
```

---

## Task 3: Seed assembler into engine params layout (module 3)

**Files:**
- Create: `qsc/seed/seed_assembler.py`
- Modify: `qsc/seed/__init__.py` (add re-export)
- Test: `tests/test_seed_assembler.py`

**Interfaces:**
- Consumes: `LiftLO` (Task 2); `qsc.quantum_numbers` `compute_Mtint, compute_kettoLAMBDA, compute_Mt, compute_gauge_info`; `QuantumNumbers`.
- Produces: `assemble_seed(lift_lo: LiftLO, qn: QuantumNumbers, g: float, cutP: int = 16) -> np.ndarray` returning the `params` vector (`complex128`, length `1 + 4*(cutP//2)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed_assembler.py
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
    lo2 = type(lo)(A=lo.A, B=lo.B, c_lo=c_lo, anomalous_lo_coeff=lo.anomalous_lo_coeff)
    seed = assemble_seed(lo2, KONISHI, g, cutP=cutP)

    # a=1: real, c_internal = 0.5 / g^Mt[1] = 0.5 / 0.1^1 = 5.0
    assert abs(seed[1 + 1 * N0 + 0] - 5.0) < 1e-12
    # a=0: imag channel, c_internal = 1j * 0.3 / g^Mt[0] = 1j * 0.3 / 0.1^2 = 30j
    assert abs(seed[1 + 0 * N0 + 0] - 30.0j) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_assembler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qsc.seed.seed_assembler'`.

- [ ] **Step 3: Write minimal implementation**

```python
# qsc/seed/seed_assembler.py
"""Module 3: assemble the engine `params` seed vector from LiftLO.

Pure formatting, no new physics. Produces
    [anomalous_Delta, c[0][1..N0], c[1][1..N0], c[2][1..N0], c[3][1..N0]]
matching forward_map_mp / solve_newton_mp. Generalizes (and replaces) the
Konishi-only qsc/perturbative.py.
"""

import numpy as np

from qsc.quantum_numbers import (
    QuantumNumbers, compute_Mt, compute_Mtint, compute_kettoLAMBDA,
    compute_gauge_info,
)
from qsc.seed.lift_lo import LiftLO


def assemble_seed(lift_lo: LiftLO, qn: QuantumNumbers, g: float,
                  cutP: int = 16) -> np.ndarray:
    """Pack LiftLO into the engine `params` vector at coupling ``g``."""
    N0 = cutP // 2
    Mtint = compute_Mtint(qn)
    kL = compute_kettoLAMBDA(Mtint)
    Mt = np.array(compute_Mt(Mtint, kL), dtype=np.float64)
    gauge_indices = compute_gauge_info(Mtint, N0)["gauge_indices"]

    params = np.zeros(1 + 4 * N0, dtype=np.complex128)
    # Anomalous dimension at LO: Δ - Δ0 = γ₂ g² (real).
    params[0] = lift_lo.anomalous_lo_coeff * g ** 2

    for a in range(4):
        denorm = lift_lo.c_lo[a] / g ** Mt[a]          # physical → internal
        if a in (0, 2):                                 # imaginary channels (MMA a=1,3)
            denorm = 1j * denorm
        params[1 + a * N0: 1 + (a + 1) * N0] = denorm

    # Enforce gauge zeros.
    for (a, n_idx) in gauge_indices:
        params[1 + a * N0 + n_idx] = 0.0

    return params
```

Also add to `qsc/seed/__init__.py`:

```python
from qsc.seed.seed_assembler import assemble_seed

__all__ = ["OneLoopQQ", "solve_oneloop_qq", "LiftLO", "lift", "assemble_seed"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_assembler.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add qsc/seed/seed_assembler.py qsc/seed/__init__.py tests/test_seed_assembler.py
git commit -m "feat(seed): module 3 — assemble seed into engine params layout"
```

---

## Task 4: Reliability harness — basin & convergence (module 4)

This is the empirical gate that answers "how much dictionary does the engine need?" It runs the real `solve_newton_mp` (slow, mpmath) so tests are marked `slow`.

**Files:**
- Create: `qsc/seed/validate_seed.py`
- Modify: `qsc/seed/__init__.py` (add re-exports)
- Test: `tests/test_seed_endtoend.py`

**Interfaces:**
- Consumes: `assemble_seed` (Task 3), `solve_oneloop_qq` (Task 1), `lift` (Task 2); `qsc.newton_mp.solve_newton_mp`; `qsc.quantum_numbers.QuantumNumbers`.
- Produces:
  - `seed_and_solve(qn, g, cutP=16, nPoints=18, cutQai=24, QaiShift=60, dps=50, **newton_kw) -> dict` — assembles the seed and runs `solve_newton_mp`, returning its result dict plus `"seed"` and `"Delta"` keys (`Delta = Δ0 + params[0].real`).
  - `find_basin_g(qn, g_candidates=(0.005,0.01,0.02,0.05,0.1), **kw) -> float | None` — largest candidate g that converges; `None` if none.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed_endtoend.py
import numpy as np
import pytest
from qsc.quantum_numbers import KONISHI
from qsc.seed.validate_seed import seed_and_solve, find_basin_g


@pytest.mark.slow
def test_seed_converges_to_konishi_at_small_g():
    """The LO seed must drive newton_mp to the Konishi solution at a small g.

    Δ(g) ≈ 2 + 12 g². At g=0.02 the true Δ ≈ 2.0048. Assert convergence and
    that the converged anomalous Δ matches the perturbative value to ~1e-3
    (confirming the correct branch, not a neighbor)."""
    g = 0.02
    res = seed_and_solve(KONISHI, g)
    assert res["converged"], f"did not converge: ||F||={res['residual_norm']:.2e}"
    # leading anomalous ≈ 12 g²; tolerate higher-order corrections
    assert abs(res["Delta"] - (2.0 + 12.0 * g**2)) < 5e-3


@pytest.mark.slow
def test_find_basin_g_returns_a_usable_start():
    g_start = find_basin_g(KONISHI)
    assert g_start is not None and g_start >= 0.005
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_endtoend.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'qsc.seed.validate_seed'`.

- [ ] **Step 3: Write minimal implementation**

```python
# qsc/seed/validate_seed.py
"""Module 4: reliability harness — does the seed bootstrap the QSC engine?"""

import numpy as np

from qsc.newton_mp import solve_newton_mp
from qsc.quantum_numbers import QuantumNumbers
from qsc.seed.lift_lo import lift
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.seed_assembler import assemble_seed


def seed_and_solve(qn: QuantumNumbers, g: float, cutP: int = 16,
                   nPoints: int = 18, cutQai: int = 24, QaiShift: int = 60,
                   dps: int = 50, **newton_kw) -> dict:
    """Assemble the LO seed at ``g`` and run the mpmath Newton solver."""
    qq = solve_oneloop_qq(qn)
    lo = lift(qq, N0=cutP // 2, order="LO")
    seed = assemble_seed(lo, qn, g, cutP=cutP)

    result = solve_newton_mp(seed, qn, g, cutP=cutP, nPoints=nPoints,
                             cutQai=cutQai, QaiShift=QaiShift, dps=dps,
                             **newton_kw)
    Delta0 = qn.Delta0
    result["seed"] = seed
    result["Delta"] = Delta0 + float(np.real(result["params"][0]))
    return result


def find_basin_g(qn: QuantumNumbers,
                 g_candidates=(0.005, 0.01, 0.02, 0.05, 0.1),
                 **kw) -> float | None:
    """Largest candidate g whose LO seed converges; None if none converge."""
    best = None
    for g in sorted(g_candidates):
        try:
            res = seed_and_solve(qn, g, **kw)
        except Exception:
            continue
        if res["converged"]:
            best = g
    return best
```

Also extend `qsc/seed/__init__.py`:

```python
from qsc.seed.validate_seed import seed_and_solve, find_basin_g

__all__ = [
    "OneLoopQQ", "solve_oneloop_qq", "LiftLO", "lift", "assemble_seed",
    "seed_and_solve", "find_basin_g",
]
```

Register the `slow` marker in `pyproject.toml` under `[tool.pytest.ini_options]` (add the section if absent):

```toml
[tool.pytest.ini_options]
markers = ["slow: end-to-end mpmath solves (deselect with -m 'not slow')"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_endtoend.py -v -m slow`
Expected: PASS. **If `test_seed_converges_to_konishi_at_small_g` FAILS** (LO `c_lo=0` seed not in basin even at g=0.005), this is the documented gate: go to Task 5 (derive `c_lo`). Record the failure in the commit message and proceed to Task 5.

- [ ] **Step 5: Commit**

```bash
git add qsc/seed/validate_seed.py qsc/seed/__init__.py tests/test_seed_endtoend.py pyproject.toml
git commit -m "feat(seed): module 4 — basin/convergence reliability harness"
```

---

## Task 5 (conditional): Derive c_lo if the asymptotics-only seed is out of basin

Run this task ONLY if Task 4's convergence test failed at all candidate g. If Task 4 passed, skip to Task 6.

**Files:**
- Modify: `qsc/seed/lift_lo.py` (`lift` fills `c_lo` from the O(g⁰) QQ-relation)
- Test: `tests/test_seed_lift.py` (add `c_lo` value test), `tests/test_seed_endtoend.py` (re-run)

**Interfaces:**
- Unchanged signatures. `lift(...).c_lo` becomes nonzero, derived from `state.bethe_roots` / the O(g⁰) QQ-relation matching `P_a = A_a + Σ c[a,n] x^{-n}` to the one-loop Q-functions.

- [ ] **Step 1: Write the failing test** — assert the assembled seed at g=0.1 reproduces `tests/fixtures/konishi_internal_params.json` to within seed tolerance (leading c-entries within 10% — a seed, not the converged solution), and that `seed_and_solve` now converges at g=0.1.

```python
# add to tests/test_seed_endtoend.py
import json
from pathlib import Path
import numpy as np
import pytest
from qsc.quantum_numbers import KONISHI
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.lift_lo import lift
from qsc.seed.seed_assembler import assemble_seed


def test_clo_seed_in_neighborhood_of_converged_g01():
    """Derived c_lo seed at g=0.1 must be in the neighborhood of the converged
    internal params (within 25% on the dominant c[1][1] entry)."""
    conv = json.loads(
        Path("tests/fixtures/konishi_internal_params.json").read_text())
    conv = np.array(conv, dtype=float)
    qq = solve_oneloop_qq(KONISHI)
    lo = lift(qq, N0=8, order="LO")
    seed = assemble_seed(lo, KONISHI, 0.1, cutP=16)
    # c[1][1] lives at flat index 1 + 1*8 + 0 = 9; converged ≈ 0.31936
    assert abs(seed[9].real - conv[9]) / abs(conv[9]) < 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_endtoend.py::test_clo_seed_in_neighborhood_of_converged_g01 -v`
Expected: FAIL (with `c_lo=0`, `seed[9]=0`, relative error = 1.0).

- [ ] **Step 3: Write the derivation** in `lift`. Replace the `c_lo = np.zeros(...)` line with a derivation from the O(g⁰) QQ-relation for `P_a` against the one-loop Q-functions (`state.bethe_roots`). Pin the exact formula against `reference/.../auxiliary/TypeI_package.wl` and `prototype/Konishi_prototype.nb` (cells `In[37]`–`In[51]`): build the large-u `B[a,i,0] = -i A_a B_i / (powP[a]+powQ[i]+1)` relation and solve the resulting linear system for the leading `c[a,n]`. Implement the closed form for Konishi; raise `NotImplementedError` for other states. (Exact coefficients to be filled from the pinned reference cells during execution — this step's code is derived, not guessed, and is gated by Step 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_endtoend.py -v -m slow`
Expected: PASS — neighborhood test passes AND `seed_and_solve` converges at g=0.1.

- [ ] **Step 5: Commit**

```bash
git add qsc/seed/lift_lo.py tests/test_seed_endtoend.py tests/test_seed_lift.py
git commit -m "feat(seed): derive leading c_lo from O(g^0) QQ-relation (Konishi)"
```

---

## Task 6: Full-sweep reliability + retire dead perturbative path

**Files:**
- Test: `tests/test_seed_endtoend.py` (add full-sweep test)
- Modify: `qsc/perturbative.py` (deprecate the broken fixture path; delegate to the new seed package)

**Interfaces:**
- `qsc.perturbative.perturbative_params(g, N0=8)` keeps its signature but now delegates to `assemble_seed(lift(solve_oneloop_qq(KONISHI), N0), KONISHI, g)` instead of reading the missing `tests/fixtures/konishi_perturbative.json`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_seed_endtoend.py
import csv
from pathlib import Path
import pytest
from qsc.quantum_numbers import KONISHI
from qsc.seed.validate_seed import seed_and_solve


@pytest.mark.slow
def test_full_sweep_from_basin_start_matches_reference():
    """Seed at g_start, then re-seed-and-solve at each csv g up to 0.1,
    confirming the converged Δ tracks the reference curve (reliability of the
    seed across the sweep)."""
    rows = list(csv.DictReader(Path("data/konishi_gDelta.csv").open()))
    targets = [(float(r["g"]), float(r["Delta"])) for r in rows
               if 0.01 <= float(r["g"]) <= 0.1]
    assert targets, "expected csv rows in [0.01, 0.1]"
    for g, delta_ref in targets:
        res = seed_and_solve(KONISHI, g)
        assert res["converged"], f"g={g} did not converge"
        assert abs(res["Delta"] - delta_ref) < 1e-4, f"g={g}: {res['Delta']} vs {delta_ref}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_endtoend.py::test_full_sweep_from_basin_start_matches_reference -v -m slow`
Expected: FAIL if any g is out of basin or mismatches; otherwise this is the green reliability gate. (If it fails only at the largest g, lower the upper bound and note it — that g exceeds the LO seed's reach, the spec's "extend orders" trigger.)

- [ ] **Step 3: Retire the dead path** in `qsc/perturbative.py`:

```python
# qsc/perturbative.py  (replace the body of perturbative_params)
"""Konishi weak-coupling initial guess.

Delegates to the qsc.seed package (one-loop QQ → LO lift → assembled seed).
The previous fixture-based path (tests/fixtures/konishi_perturbative.json)
never shipped and is removed.
"""

import jax.numpy as jnp

from qsc.quantum_numbers import KONISHI
from qsc.seed.lift_lo import lift
from qsc.seed.oneloop_qq import solve_oneloop_qq
from qsc.seed.seed_assembler import assemble_seed


def perturbative_params(g: float, N0: int = 8) -> jnp.ndarray:
    """Internal-format seed params for Konishi at coupling ``g``."""
    cutP = 2 * N0
    lo = lift(solve_oneloop_qq(KONISHI), N0=N0, order="LO")
    seed = assemble_seed(lo, KONISHI, g, cutP=cutP)
    return jnp.array(seed)
```

Delete the now-unused `load_konishi_perturbative` function and its `json`/`Path` imports.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -m "not slow" -q && uv run pytest tests/test_seed_endtoend.py -v -m slow`
Expected: fast suite PASS; slow end-to-end PASS (note any documented upper-g limitation).

- [ ] **Step 5: Commit**

```bash
git add qsc/perturbative.py tests/test_seed_endtoend.py
git commit -m "feat(seed): full-sweep reliability test; retire broken perturbative fixture path"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** module 1 (Task 1), module 2 (Tasks 2, 5), module 3 (Task 3), module 4 (Tasks 4, 6); LO dictionary + bootstrap = Tasks 3–4; verification gates §7 = Tasks 4–6; B-readiness = `lift(order=...)` NotImplementedError (Task 2); fallback decision gate = Task 4 Step 4 → Task 5. Branch selection (degenerate Δ0) is out of scope per spec §9 — `solve_oneloop_qq` raises for non-Konishi, reserving the interface.
- **Convention anchors are real:** `params[0]` anomalous, `Mt=[2,1,0,-1]`, `γ₂=12` (csv slope), `konishi_internal_params.json[0]=0.1155`, i-factor for a∈{0,2}, gauge via `compute_gauge_info` — all verified against existing code/fixtures, not assumed.
- **A-fallback:** if Tasks 4–5 cannot get the engine into basin and the algebraic higher-order engine (B) is also needed but intractable, revert to spec approach A (numerical cut-matching); this plan does not implement A (deferred per spec §9).
