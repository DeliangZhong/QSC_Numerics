# QSC Weak-Coupling Seed Generator — Design

**Date:** 2026-06-25
**Status:** Approved design; pre-implementation
**Scope (first target):** Type I / Konishi operator
**Author:** Deliang Zhong (with Claude Code)

---

## 1. Purpose & context

The QSC numerical workflow has two stages:

1. **Seed generation** — produce an initial `(Δ, {c[a,n]})` at weak coupling.
2. **Finite-coupling iteration** — single-point Newton solver that, given a seed at coupling
   `g`, converges to the physical solution; continuation drives `g` upward.

Stage 2 already exists in this repo (`qsc/forward_map_mp.py`, `qsc/newton_mp.py`,
continuation scripts; reaches `g ≈ 0.25`). Stage 1 does **not** exist in general form:
`qsc/perturbative.py` handles only Konishi and references a missing fixture
(`tests/fixtures/konishi_perturbative.json`).

The reference implementation (`reference/qsc/local operators N4 SYM/`) builds the seed in
Mathematica (`auxiliary/TypeI_package.wl`, `prototype/Konishi_prototype.nb`) via the
**Marboe–Volin construction** (arXiv:1812.09238): start from the one-loop (rational)
QQ-system, then lift order-by-order in `g` to the weak-coupling expansion of the QSC
P-functions, producing `{Δg[i], cg[a,n,m]}` (stored as the `sbWeak` rule inside the 219
precomputed `data/perturbative/perturbative_data_*.mx` files). `FromPert[g]` just evaluates
that series numerically.

**Established physics fact governing the whole design:** the seed does **not** affect the
QSC *result*. Newton/QSC converges to the unique physical solution `(Δ(g), c(g))`
determined by the QSC equations, independent of the seed, **provided the seed is in the
basin of attraction**. Therefore every internal seed choice (closed-form `c^LO` vs.
`c^LO=0`, `g_start`, truncation order) is a **reliability/speed** decision, never a
physics-correctness decision — *with one caveat* (§6): for states whose bare dimension
`Δ0` is degenerate (multiple `sol` branches), the seed must land in the basin of the
*correct* branch.

**Goal of this work:** a fast, self-contained seed generator that **reliably bootstraps
the QSC numerics** for the target state — judged operationally (Newton converges, finds the
correct solution, continuation reaches the frontier), not by matching the reference `.mx`
bit-for-bit.

## 2. Decisions taken (from brainstorming)

| Decision | Choice |
|---|---|
| What "seed generator" means | The Marboe–Volin **lift**: QQ-system → weak-coupling expansion of QSC P/Q. The QQ part is assumed available (one-loop), the lift is the missing piece. |
| Lift depth | Only as deep as the QSC engine needs to converge reliably. Start at **leading order (LO)**; extend orders only if a larger `g_start` is required. |
| QQ-system input | **Self-contained Python** one-loop solver (Konishi is trivial), behind a clean interface so the sibling `QQ_Galois` Julia project can plug in later for general/nested states. |
| Lift algorithm | **C now, B-ready**: LO dictionary feeding the existing Stage-2 engine, with module boundaries so the pure-algebraic Marboe–Volin higher-order engine (B) slots in later. |
| Fallback | If C fails and B is intractable, revert to **A** (mirror the reference's numerical Zhukovsky/Fourier cut-matching). |

Approaches considered (for the order-by-order lift):
- **A — faithful cut-matching:** rebuild the reference's numerical P-ansatz + QQ/gluing on a
  Zhukovsky/Fourier grid. Bit-for-bit `.mx` validation, but heaviest and overlaps the
  existing finite-coupling forward map. *Documented fallback.*
- **B — pure algebraic Marboe–Volin:** solve the QQ-system order-by-order algebraically
  (exact rational `cg[a,n,m]`). The definitive fast analytic engine; most upfront derivation.
- **C — LO dictionary + bootstrap (chosen MVP):** compute LO exactly, let the existing
  Stage-2 Newton act as the order-by-order solver. Least new physics; fastest to a working seed.

## 3. Architecture

New package `qsc/seed/`. Data flows left→right; each boundary is a plain dataclass so a
module can be swapped (e.g. `QQ_Galois` for module 1, engine B for module 2) without
touching neighbors.

```
state (QuantumNumbers)
      │
      ▼
┌─────────────────────┐   one-loop Q-system        ┌──────────────────────┐
│ 1. oneloop_qq.py    │ ──(Q-polynomials/roots)──▶ │ 2. lift_lo.py        │
│  self-contained;    │   OneLoopQQ dataclass       │  LO dictionary:      │
│  QQ_Galois-ready    │                             │  Q-system → A_a, c^LO│
└─────────────────────┘                             └──────────┬───────────┘
                                                                │ {A_a, B_i, c[a,n]^LO, Δ0}
                                                                ▼
                                              ┌──────────────────────────────┐
                                              │ 3. seed_assembler.py         │
                                              │  → internal seed vector at   │
                                              │    g_start (Mt-denorm, gauge,│
                                              │    i-factors) for Stage-2    │
                                              └──────────────┬───────────────┘
                                                             ▼
                                       existing engine: forward_map_mp + newton_mp
                                                             │
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ 4. validate_seed.py          │
                                              │  Newton converges? correct   │
                                              │  branch? full-sweep reliable?│
                                              └──────────────────────────────┘
```

**Reuse, not rebuild.** Modules 2–3 lean on existing `qsc/quantum_numbers.py`:
`compute_A`/`compute_B` (the MV `A_a`, `B_i` asymptotics), `compute_Mt`,
`compute_gauge_info`, `compute_PhiV`. The genuinely new physics is **module 1** (one-loop
Q-system) and the **`c[a,n]^LO` part of module 2**. Module 3 generalizes
`perturbative.py` and removes the dead fixture dependency.

**B-readiness.** Module 2 exposes `lift(state, order) → {Δg[i], cg[a,n,m]}`; the MVP
implements `order=LO` only and raises `NotImplementedError` for higher orders, which engine
B later fills behind the identical signature.

## 4. Conventions (pinned)

Pinned from `reference/.../auxiliary/TypeI_package.wl:14–34`, cross-checked against
`qsc/quantum_numbers.py`. (Note: the exploratory notebook-extraction produced some
incorrect leading-order numbers, e.g. half-integer `λ`; those are **rejected**. The values
below are authoritative.)

- `L = ½(Σnf + Σna − Σnb)`,  `Δ0 = ½Σnf + Σna`
- `λ0[a] = nf[a] + {2,1,0,−1}[a]`  (a = 1..4)
- `Λ = (1 − λ0[1] − λ0[4]) / 2`
- `Mt[a] = λ[a] = λ0[a] + Λ`,  `powP = −λ`
- `ν0[i] = {−L−nb1−1, −L−nb2−2, na1+1, na2}[i]`
- `ν[i] = ν0[i] + (Δ−Δ0)/2 · {−1,−1,1,1}[i] − Λ`,  `powQ = −ν − 1`
- P-function: `P_a(u) = A_a + Σ_{n≥2, even} c[a,n] · x(u)^{−n}`, with `x` the Zhukovsky
  variable, `x ~ u/g` at large `u`. At LO, set `Δ = Δ0`.
- Internal normalization for the Stage-2 engine: `c_internal = c_phys / g^Mt[a]`, with an
  `i`-factor for `a ∈ {1,3}` (matches `qsc/perturbative.py` and the C++ format).
- Gauge fixing: specific `c[a,n] → 0` per `compute_gauge_info` (powP-difference gauge).

**Konishi sanity check** (`nb=(0,0)`, `nf=(1,1,1,1)`, `na=(0,0)`):
`λ0={3,2,1,0}`, `Λ=−1`, `Mt={2,1,0,−1}`, `powP={−2,−1,0,1}`, `L=2`, `Δ0=2`,
`ν0={−3,−4,1,0}`; gauge-fix at `(a=2, n=4)`. ✓

## 5. Module specifications

### Module 1 — `oneloop_qq.py`
- **Responsibility:** produce the one-loop (rational, `g=0`) QQ-system data for a state.
- **Output:** `OneLoopQQ` dataclass — Q-polynomial coefficients / Bethe roots and any
  nesting data needed by module 2; carries the `sol` branch label.
- **Konishi MVP:** the one-loop solution is analytically trivial (unique state at `Δ0=2`);
  implement it directly.
- **Interface:** signature and dataclass shaped so a `QQ_Galois` adapter (Julia bridge or
  reading its serialized nested Q-system) can later supply the same `OneLoopQQ` for
  general/nested/super states.

### Module 2 — `lift_lo.py`
- **Responsibility:** the LO dictionary `one-loop Q-system → (Δ_1-loop, {A_a, B_i, c[a,n]^LO})`.
- `A_a`, `B_i` come from `compute_A`/`compute_B` (fixed by quantum numbers; no QQ-solve).
- **New physics:** `c[a,n]^LO` from the `O(g⁰)` QQ-relation. **Verification-gated** (§7):
  no closed form is assumed up front. If a clean closed form proves elusive, the MVP falls
  back to `c[a,n]^LO = 0` (asymptotics-+`Δ0`-only seed) and lets Newton discover them at a
  small enough `g_start`; the basin test (§7) measures how much dictionary the engine
  actually needs.
- **API:** `lift(state, order='LO') → {Δg, cg}` structure; higher `order` → `NotImplementedError`.

### Module 3 — `seed_assembler.py`
- **Responsibility:** pure formatting, no new physics. Given `{A_a, B_i, c[a,n]^LO, Δ0}` and
  a `g_start`, emit the internal seed vector in the exact `params` layout consumed by
  `forward_map_mp`/`newton_mp`: apply `c_internal = c_phys/g^Mt[a]`, the `i`-factor for
  `a ∈ {1,3}`, the gauge zeros, and pack `[Δ−Δ0, c[0][·], …, c[3][·]]`.
- Generalizes `perturbative.py` to be state-agnostic and removes the dead fixture path.

### Module 4 — `validate_seed.py` (reliability harness)
- **Convergence:** `newton_mp` converges from the LO seed at `g_start` (auto-picked as the
  largest `g` in a sweep — e.g. `{0.005, 0.01, 0.02, 0.05, 0.1}` — that converges).
- **Correct branch & physics:** converged `Δ(g_start)` matches the existing converged data /
  C++ reference (`tests/fixtures/konishi_converged_g01.json`, `data/konishi_gDelta.csv`),
  confirming it found *Konishi*, not a neighbor.
- **Full-sweep reliability:** continuation runs from `g_start` up through the current
  frontier (~`g=0.25`) without re-seeding, reproducing `Δ(g)`.

## 6. The branch-selection caveat (why module 1 matters)

For states with **degenerate `Δ0`** (multiple operators / `sol=1,2,3,…`; e.g. the `Δ0=6`
sector with dozens of operators), the seed must land in the basin of the *correct* branch.
The one-loop QQ-system resolves this — it picks the right state and supplies its one-loop
Q-functions → the correct `c[a,n]^LO` branch. Konishi (unique at `Δ0=2`) does not exercise
this, which is why it is the right *first* target but branch-selection is the feature that
makes module 1 worth building properly for "full QSC."

## 7. Verification gates

1. **`c[a,n]^LO` correctness:** the assembled LO seed reproduces the leading-`g` behavior of
   the existing converged data to the expected `O(g²)` accuracy.
2. **Basin test:** LO seed lies inside the Newton basin at `g_start` (auto-tuned).
3. **Branch test:** converged solution matches Konishi reference `Δ(g)`, not a neighbor.
4. **Full-sweep test:** continuation from `g_start` reproduces `Δ(g)` to the frontier.

All gates are numerical/operational — consistent with the goal of *reliably bootstrapping
QSC*, not bit-matching the `.mx`.

## 8. Risks & fallback (decision gates)

1. **LO seed not in basin at any useful `g_start`** → try `c^LO=0` first; if still failing,
   derive `c^LO` from the one-loop QQ-system (full module 2).
2. **Branch ambiguity for degenerate states** → requires module 1; Konishi MVP defers it but
   module 1's interface is built for it.
3. **C insufficient for larger `g_start` / nested states** → promote to **B** (algebraic
   order-by-order MV) behind module 2's `lift(state, order)` signature.
4. **B intractable** → fall back to **A** (mirror the reference numerical cut-matching).

## 9. Out of scope (for the MVP)

- Types II / III / IV (general parity / general states).
- The full `.mx`-matching high-order expansion (engine B) — deferred, interface reserved.
- The `QQ_Galois` Julia bridge — deferred, interface reserved.
- Degenerate-`Δ0` branch selection beyond reserving the interface.

## 10. References

- Gromov, Kazakov, Leurent, Volin — arXiv:1305.1939, arXiv:1405.4857 (QSC).
- Marboe, Volin — arXiv:1812.09238 (fast analytic / weak-coupling solver; the lift).
- Hegedűs, Konczer — arXiv:1604.02346 (C++ numerical implementation).
- Gromov, Hegedűs, Julius, Sokolova — arXiv:2306.12379 (the reference solver/database).
- In-repo: `reference/.../auxiliary/TypeI_package.wl`, `prototype/Konishi_prototype.nb`,
  `core/TypeI_core.cpp`; `qsc/quantum_numbers.py`, `qsc/perturbative.py`,
  `qsc/forward_map_mp.py`, `qsc/newton_mp.py`.
