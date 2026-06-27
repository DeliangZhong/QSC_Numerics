# Fast 1-Loop Seed Generator — Handoff / Continuation

**Status:** task scoped and ready to implement. Read this first after any /compact.

## THE TASK (definite, small)
Build a **fast, pure-Python 1-loop seed generator**. It is confirmed (see §why) that the QSC
numerics (part 2) needs only the **1-loop** seed. So part 1 = produce, for a state:
- `γ₂` = 1-loop anomalous dimension (Konishi: 12), and
- the **leading `c[a,n]`** coefficients (the lowest-g-power term per channel).
No η-functions, no MZVs, no μ-loops, no Mathematica required.

## Why 1-loop is the answer (settled this session)
- L=0 is singular (anomalous Δ=0 ⇒ forward map `AA`→0, ZeroDivisionError). 1-loop is the minimum well-defined seed.
- For a proper quadratically-convergent QSC Newton (part 2, **independent**) at small `g_start`, the
  1-loop seed (error O(g⁴)) is inside the basin; continuation walks g up. Standard method: GLMS [1504.06640].
- The ‖F‖ floor / non-refinement seen with the local cutP=16 mpmath engine is a **part-2** issue, NOT part 1.

## The 1-loop seed, concretely
`params = [γ₂·g_start², c_internal[0][1..N0], c_internal[1][..], c_internal[2][..], c_internal[3][..]]`
(length 1+4·N0, N0=cutP//2=8, complex128) — the layout `forward_map_mp`/`solve_newton_mp` consume.
Build it from leading **physical** `c_phys[a,n]` via the EXISTING `qsc/seed/seed_assembler.assemble_seed`
(already does: `c_internal = c_phys/g^Mt[a]`, ×`1j` for Python a∈{0,2}, gauge zeros). So part 1 only needs to
produce `c_phys` leading + `γ₂`, then call `assemble_seed`.

## Konishi validation targets (physical leading c, from data/konishi_sbweak_reference.json)
MMA a=2,k=2 → **3**;  a=3,k=4 → **−6**;  a=4,k=2 → **6**;  a=4,k=4 → **24**;  a=4,k=6 → **6**;
a=1 has none at leading (its terms start at g⁶). γ₂=12. (MMA channel a ↔ Python a−1.)
Note the leading g-power differs per channel (a=4 starts at g⁻², a=2 at g²) — these are c_phys leading terms.

## Data source for the 1-loop coefficients
- `DQgl4[state]` (QSCsolver leading rational Q-system) runs in **0.0s** headless and gives the one-loop
  Q-system (Konishi: Baxter `u²−1/12`). The leading `c[a,n]` come from the 1-loop P-construction off this.
- The one-loop QQ-system is exactly what `QQ_Galois` (sibling Julia project) solves, and what
  `qsc/seed/oneloop_qq.py` stubs. Port the leading P→c map (algebraic, no MZVs) to Python.
- Cross-check / ground truth: `data/konishi_sbweak_reference.json` (full reference series, extracted via
  wolframscript from the .mx) and `tests/fixtures/konishi_internal_params.json` (converged seed at g=0.1).

## Conventions (pinned, validated)
- `Mt = λ0 + Λ`, `λ0[a]=nf[a]+{2,1,0,−1}`, `Λ=(1−λ0[1]−λ0[4])/2`. Konishi `Mt={2,1,0,−1}`.
- Channel map: MMA a=1,2,3,4 ↔ Python a=0,1,2,3. Dominant real channels = Python a=1,3 (MMA 2,4).
- i-factor for Python a∈{0,2}; gauge zeros via `compute_gauge_info` mapped `n_idx→offset n_idx−1` (fixed).

## What's already built (on main, tested)
`qsc/seed/`: `oneloop_qq.py` (γ₂=12 stub), `lift_lo.py` (order="asymptotic"=zero stub works;
order="LO" RAISES — this is what the 1-loop generator fills), `seed_assembler.py` (assemble_seed; gauge
off-by-one fixed), `validate_seed.py` (Δ-gated harness). All fast tests green (14). Adversarial review addressed.
Reference: `reference/MV_1812.09238/QSCsolver.nb` (+ QSCdata.nb, gitignored). Capstone proved a full
sbWeak seed → engine gives Δ to 7 digits.

## NEXT STEP
Implement the 1-loop `c_phys` computation in Python (from the one-loop QQ-system / DQgl4-equivalent),
fill `lift(state, N0, order="LO")` to return it, add a test vs the Konishi targets above, then generalize
to other states. Keep it pure-Python and fast (no wolframscript in the hot path).

## Dead ends (do NOT repeat)
- Running QSCsolver.nb headless for findγ/full series: loading errors (fix/findγ undefined out of order),
  slow (~min/state), license-style runaways. Abandoned. (DQgl4 alone IS fine/fast.)
- The ‖F‖≈0.27 floor at cutP=16 is structural forward-map fidelity — part 2's concern, not part 1's.
- Empirical basin sweeps: settled — 1 loop is the answer; don't re-run.
