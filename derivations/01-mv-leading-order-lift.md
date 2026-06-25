# Marboe–Volin Leading-Order Lift — Derivation Working Doc (Konishi / Type I)

**Status:** in progress. Goal: derive the closed-form map `one-loop QQ-system → leading c[a,n]` so `qsc/seed/lift_lo.py` produces a real (non-zero) `c_lo`, replacing the `.mx` data. All numerical relationships noted below are **observed, not yet proven** — they must be derived from the Marboe–Volin construction (arXiv:1812.09238) before being committed to code. Per project rules: no guessed formulas in the implementation; every coefficient formula needs a derivation or a cited MV equation, validated against the ground-truth table.

## Conventions (pinned; see also the spec)
- `λ0[a]=nf[a]+{2,1,0,−1}`, `Λ=(1−λ0[1]−λ0[4])/2`, `Mt[a]=λ[a]=λ0[a]+Λ`, `powP=−λ`.
- Konishi: `Mt={2,1,0,−1}`, `powP={−2,−1,0,1}`, `Δ0=2`, `L=2`.
- `P_a = A_a + Σ_{n≥2,even} c[a,n] x^{−n}` (Zhukovsky x, x~u/g at large u).
- Internal/engine convention: `c_internal[a][n] = c_phys[a][n]/g^Mt[a]`, ×`1j` for MMA a∈{1,3} (= Python a∈{0,2}).
- Channel map: **MMA a=1,2,3,4 ↔ Python a=0,1,2,3.** Engine gauge sets Python a=0,2 (MMA 1,3) to 0; dominant real channels are Python a=1,3 (MMA 2,4).

## Scaffold (verbatim from prototype Konishi_prototype.nb In[37]–In[43])
- `AMV[a] = -(λ[a]+ν[1])(λ[a]+ν[2]) / Π_{b>a} i(λ[a]-λ[b])`
- `BMV[a<3] = 1 / Π_{b>a} i(ν[a]-ν[b])`;  `BMV[a>2] = Π_b (λ[b]+ν[a]) / Π_{b>a} i(ν[a]-ν[b])`
- `sbA = {A1→AMV1, A2→AMV2, A3→−AMV3, A4→−AMV4}`, `sbB = {Bi→BMVi}`
- Leading Q-expansion coefficient: `B[a,i,0] = -i·A[a]·B[i] / (powP[a]+powQ[i]+1)`
- `powQ = -ν - 1`. **Pinned by derivation** from TypeI_package.wl:28–34: ν0={−3,−4,1,0}, Λ=−1, at Δ=Δ0 ⇒ ν=ν0−Λ={−2,−3,2,1} ⇒ **`powQ={1,2,−3,−2}`**. (The extraction agent's `{3,4,−1,0}` was wrong.)

## One-loop quantities (computed, small-anom limit Δ→Δ0⁺)
`A ≈ [0, 1, −6i, 12]`,  `B ≈ [i/12, −1/20, −24i, 0]`.
NOTE: `A_a` has a 0/0 at exactly Δ=Δ0 (A[0]=0, A[3]=NaN); evaluate as the Δ→Δ0 limit or via the AMV closed form.

## Ground-truth leading coefficients (from data/konishi_sbweak_reference.json)
Lowest-power-of-g coefficient `cg[a,k,m_min]` per (MMA a, k=n):

| MMA a | k(=n) | m_min | cg (leading) |
|------:|------:|------:|-------------:|
| 1 | 2 | 6 | −14.4246828 (irrational; NOT one-loop) |
| 1 | 4 | 8 | 649.003125 (irrational; NOT one-loop) |
| 2 | 2 | 2 | **3** |
| 3 | 4 | 2 | **−6** |
| 4 | 2 | −2 | **6** |
| 4 | 4 | 0 | **24** |
| 4 | 6 | 2 | **6** |

The bolded integers are the genuine one-loop data to reproduce. Channel a=1's leading terms appear only at g⁶⁺ (subleading), consistent with its role.

## Observed relations (CONJECTURES — to be derived, not coded as-is)
- `cg[4,4,0] = 24 = 2·A[4]` (A[4]=12)
- `cg[4,2,-2] = 6 = A[4]/2`
- `cg[3,4,2] = −6`, and `A[3] = −6i` ⇒ `−6 = A[3]/i`
- `cg[2,2,2] = 3`, `A[2] = 1`
These suggest the leading c[a,n] are simple functions of A_a (and powP/powQ, B_i) via the MV QQ-relations. DERIVE the exact relation from arXiv:1812.09238 (P↔Q↔c closure) and confirm it reproduces ALL bolded entries before implementing.

## CONFIRMED SCOPE (closure extraction, In[43]–In[66])
The prototype does **not** solve the weak-coupling expansion order-by-order. It builds the full
residual system `GE = ℱfur[Plist]` (P-ansatz → DoQailarge recursion for B[a,i,n] → SolveQPP
pulldown → Fourier/Chebyshev gluing `coefpGluing[a] == coefpDef[a]` → excluded/gauge set) and
**Newton-solves it numerically, seeded by the `.mx` perturbative data** (`Get[fnameA]`). The
order-by-order algebraic MV *generator* that produced the `.mx` is **NOT in this repository** — it
is the separate code of arXiv:1812.09238. Therefore a self-contained `c_lo` requires **implementing
the leading order of the MV algebraic weak-coupling solver** (not extractable from this repo).

Scaffold that IS available (verbatim, usable):
- P-ansatz In[41]; `slB0` leading-Q relation In[43]; `DoQailarge` B-recursion In[51];
  gluing `equations` / `coefpGluing`/`coefpDef` In[57]; `CoefsX` Fourier In[56]; Newton In[65–66].

## Plan to complete (Task 5) — focused research-implementation
1. ~~Pin ν, powQ~~ DONE: `powQ={1,2,−3,−2}`.
2. Get the MV leading-order construction from arXiv:1812.09238 (weak-coupling P parametrization +
   the one-loop algebraic solution). Derive the closed-form leading `c[a,n]` for Type I.
3. Implement `c_lo` in `qsc/seed/lift_lo.py` from the one-loop QQ data + A_a/B_i (DERIVED, not
   pattern-matched against the integers below).
4. Validate against: the integer targets in this doc; the dominant-channel converged fixture
   (c[1][1]≈0.319, c[3][1]≈69.89); and `seed_and_solve` reaching Δ to engine accuracy.

RECOMMENDATION: do this as a fresh focused pass with the paper in hand — it is a real research task,
not a transcription, and the project's rigor rules forbid coding the observed integer relations
without a derivation.

## Validation harness (available now)
`data/konishi_sbweak_reference.json` (exact reference series) + `tests/fixtures/konishi_internal_params.json` (engine-gauge converged seed at g=0.1) + `qsc/seed/validate_seed.py` (Δ-gated). Capstone proved a reference-quality seed → Δ to 7 digits at iter 0.
