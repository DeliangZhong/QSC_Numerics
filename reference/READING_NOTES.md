# Reference QSC Solver — Reading Notes

These notes document the existing QSC numerical solver at `qsc/local operators N4 SYM/`.
The goal is to understand the full pipeline before reimplementing it in JAX/Python.

**Key references:**
- QSC formulation: Gromov, Kazakov, Leurent, Volin — arXiv:1305.1939, arXiv:1405.4857
- Original numerical algorithm: Gromov, Levkovich-Maslyuk, Sizov — arXiv:1504.06640
- Fast QSC Solver (this code): Gromov, Hegedűs, Julius, Sokolova — arXiv:2306.12379
- Perturbative QSC data: Marboe, Volin — arXiv:1812.09238

---

## 1. Architecture Overview

The solver is a **3-tier pipeline**:

```
Python Jupyter (orchestration & continuation in g)
  → Mathematica wolframscript (parameter management, perturbative init)
    → C++ executable (core Newton solver, CLN arbitrary precision)
      → Mathematica (output parsing, data persistence)
        → Python (adaptive step-size decisions)
```

### Data flow for a single (state, g) point

1. **Python** picks next coupling `g`, sets hyperparameters
2. **Python** calls `wolframscript TypeX_script.wls` via subprocess
3. **Mathematica script** loads `TypeX_package.wl`, computes initial guess (perturbative or interpolated)
4. **Mathematica script** denormalizes coefficients, builds C++ command string
5. **C++ executable** runs Newton iteration to self-consistency
6. **C++ → Mathematica**: output parsed, coefficients renormalized, data saved to `.mx`
7. **Mathematica → Python**: diagnostics emitted as `key||value||` on stdout
8. **Python** parses diagnostics, decides step-size for next `g`

### File organization

```
local operators N4 SYM/
├── core/           C++ solvers (TypeI-IV_core.cpp) + Mathematica examples (.nb)
├── auxiliary/      Mathematica packages (.wl), scripts (.wls), Python modules (.ipynb)
├── run/            Python Jupyter notebooks for automated continuation
├── data/
│   ├── numerical/      219 precomputed spectral data files (.mx)
│   ├── perturbative/   219 weak-coupling initial data files (.mx)
│   └── output/         Default output directory (empty)
└── prototype/      Konishi_prototype.nb — pure Mathematica reference implementation
```

---

## 2. C++ Core Algorithm

Each `TypeX_core.cpp` (3200–6244 lines) implements the full Newton solver using the CLN library for arbitrary-precision arithmetic.

### 2.1 Input format

The executable takes **21 + 4×N0** command-line arguments:

| Arg | Name | Meaning |
|-----|------|---------|
| 1 | WP | Working precision (decimal digits, e.g. 186) |
| 2 | N0 | Truncation order for c_{a,n} coefficients |
| 3 | NCb | Cutoff for Q large-u expansion |
| 4 | NIcutoff | Imaginary pull-down cutoff |
| 5 | lc | Number of Chebyshev sampling points on the cut |
| 6 | DH | Derivative step: h = 10^{-DH} for finite differences |
| 7 | precssf | Target: ‖E‖ < 10^{-precssf} |
| 8 | precDelta | Target: |δΔ| < 10^{-precDelta} |
| 9 | maxiter | Maximum Newton iterations |
| 10–17 | nb, nf, na | Quantum numbers (8 oscillator numbers) |
| 18 | sol | Multiplicity label |
| 19 | g | Coupling constant (high-precision string) |
| 20 | Δ | Initial anomalous dimension |
| 21+ | c_{a,n} | Initial expansion coefficients (4×(N0+1) values) |

### 2.2 Initialization

1. Compute dimensional exponents: `Mt[a] = Mtint[a] + Λ/2` where Λ encodes quantum numbers
2. Compute hatted dimensions: `M̂[a] = M̂₀[a] ± Δ/2`
3. Build `A_a`, `B_i` factors (rational functions of quantum numbers and Δ)
4. Construct `AA[a][b] = A_a · A^b` and `BB[a][i]` matrices (4×4)
5. Compute `α[a][i] = M̂[i] - Mt[a]` (shift exponents for Q-propagation)

### 2.3 Symmetry reduction and gauge fixing

The Newton system dimension depends on the operator type:

| Type | Symmetry | dimV (approx) | Relation |
|------|----------|---------------|----------|
| I | LR + parity | 1 + N0 + Σ Nch[a] | Smallest — c_{a,n} = c̃_{a,n}, even powers only |
| II | LR only | ~2× Type I | Zero-mode complications (ambiguous Q_{a|i,n}) |
| III | Parity only | ~2× Type I | No LR reduction |
| IV | General | ~4× Type I | Full parameter space |

**CtoV2LR / VtoC2LR** functions map between the constrained coefficient space {Δ, c_{a,n}} and the unconstrained Newton variable vector V. Gauge constraints remove coefficients where `2n = Mtint[0] - Mtint[a]`.

### 2.4 Chebyshev grid

- **lc** points on the cut `[-2|g|, 2|g|]` via Chebyshev-Gauss quadrature:
  ```
  u_k = -2·Re(g)·cos(π(2k+1)/(2·lc)),  k = 0, ..., lc-1
  ```
- Precompute Chebyshev matrices **CT** (cosine transform) and **CU** (Chebyshev-U) of size lc × lc
- Compute `√(4g² - u_k²)` weights for Fourier inversion

### 2.5 P-function construction

P-functions are expanded in inverse powers of the Zhukovsky variable:

```
P_a(u) = X(u)^{Mt[a]} × Σ_{n≥0} c_{a,n} · X(u)^{-2n}
```

where `X(u) = u/(2g) + √((u/(2g))² - 1)`.

The code precomputes:
- **σ-coefficients** (`sigmasubfunc2`): encode the 1/u expansion of X(u)^n via kappa/kappabar binomial recursions
- **TXm, TXmi**: Tables of X(u_k)^{2n} for n = 0,...,N0
- **PaT, PtaT**: P_a(u_k) evaluated on the Chebyshev grid
- **TxAnm**: Powers of X(u_k + im) for imaginary shifts (used in Q-propagation)

### 2.6 Q-propagation (central computation)

**Function: QconstructorUJ2LRi** (~250 lines)

This is the most expensive part. Given P_a(u), it constructs Q_{a|i}(u) via:

**Step A — Build b_{a|i,n} coefficients:**
For each n = 1,...,NQ_i, solve a 4×4 linear system:
```
scT[i][m] · b[i][m] = f1[m] - f2[m]
```
where:
- scT encodes the shift structure: `α_{a|i}·(2m-1) - i·β_{a|i}`
- f1 comes from P-function source terms (T1, T2, T3, T41, T5 arrays — binomial expansions)
- f2 comes from convolution of previously computed b-coefficients (S1n, S32 arrays)
- Solved via `linsolvepE`: LU decomposition with partial pivoting

**Step B — Evaluate Q at sampling points:**
Starting from large-u asymptotics `Q_{a|i}(u) ~ u^{M̂_i}`, propagate down to the cut using the imaginary "pull-down" process:
```
Q_{a|i}(u) = BB[a][i] · u^{-M̂_i-NI} · Σ_n b_{a|i,n} · u^{-n}
```
then apply NI steps of the shift recursion to reach u_k on the Chebyshev grid.

**Step C — Compute gluing constant α_Q:**
```
α_Q = average of Q_0/conj(Q_2) and -Q_1/conj(Q_3) ratios across lc points
```
(Real part only; imaginary part discarded as an error diagnostic.)

**Step D — Recover P_a via gluing:**
The gluing equations express:
```
ΔP_a(u_k) = Q_{a|0}(u_k)·[Q_0(u_k) + conj(Q_1(u_k))/α_Q] - ...
```
The residual `δP[n·4+a] = P_a(u_k) - P_{glued,a}(u_k)` forms the equation system.

**Step E — Error estimation:**
Two metrics: L2 norm of Q-shift deviations (NI vs NI-1) and α_Q ratio deviations across the grid.

### 2.7 Fourier inversion (equations from Q-values)

**QtoEtypeIInewton**: transforms from sampling-point residuals back to Fourier-coefficient residuals.
- Splits into symmetric (cS) and antisymmetric (cA) modes
- Uses the precomputed CT and CU Chebyshev matrices
- Produces the equation vector E of dimension dimV

### 2.8 Newton iteration (main loop)

For each Newton step:

1. **Jacobian computation** (the bottleneck):
   - For each variable j = 0,...,dimV-1:
     - Perturb V[j] by h = 10^{-DH}
     - Recompute ALL Δ-dependent quantities (M̂, A, B, BB, α, scT, etc.)
     - Call `QconstructorUJ2LRi` → get shifted residual
     - Apply `QtoEtypeIInewton` → get shifted equations
     - Finite difference: `DE[j] = (E_shifted - E_0) / h`
   - **Cost: O(dimV) full forward evaluations**

2. **Linear solve**: `DE · δc = -E_0` via LU with pivoting

3. **Update**: `V ← V - δc`

4. **Convergence check**: both `‖E‖ < 10^{-precssf}` AND `|δΔ| < 10^{-precDelta}`

### 2.9 Output format

Structured output to stdout containing:
- Error code (0 = success, 1 = maxiter reached)
- Final Δ and all c_{a,n} coefficients
- Iteration count, final norms, running cutoffs
- Q-shift deviations per iteration (for diagnostics)

---

## 3. Orchestration Pipeline

### 3.1 Python run notebooks (`run/TypeX_run.ipynb`)

Implement adaptive continuation in the coupling g:

```python
gloop = Fraction(1, 10000)   # start at weak coupling
dgloop = Fraction(1, 1280)   # initial step size

while gloop < gmax:
    listOut = runit(stateID, gloop, listParams, debugmode)

    if flagSaved and succescount > 3:
        dgloop *= 2            # double step when converging well
        succescount = 0
    elif not flagSaved:
        gloop -= dgloop / 2    # step back
        dgloop /= 2            # halve step when struggling
        succescount = 0
    
    gloop += dgloop
```

Also attempts to land on "round" g values (0.1, 0.2, ...) for clean output.

**Typical hyperparameters:**
- cutP: 16–24 (Fourier mode cutoff)
- QaiShift: 50–100 (large-u shift)
- cutQai: 24–30 (Q expansion order)
- errorTolerance: -24 (i.e. 10^{-24})
- maxIterations: 5

### 3.2 Python module notebooks (`auxiliary/TypeX_module.ipynb`)

**runit(stateID, gloop, listParams, debugmode):**
- Calls `subprocess.Popen(["wolframscript", "-script", scriptPath, ...])` with 20 arguments
- Parses stdout line-by-line for `key||value||` format
- Returns: `[iniS, resS, resDelta, dDelta, itr, shifterror, atime, newtonGoal, flagError, flagSaved]`

**BoostShift(stateID, gloop, listParams, debugmode):**
- Adaptive hyperparameter tuning: compares error-reduction rate when increasing QaiShift vs cutQai
- Runs 3 configurations, picks the one with best `(error reduction) / (time cost)` ratio

**IniFromExisting(stateID, gloop, listParams, debugmode):**
- Iterative parameter tuning loop (up to 200 rounds)
- Increases cutP when Newton residual is too large
- Calls BoostShift when shift precision is insufficient
- Exits when both shift and Newton precision meet the goal

### 3.3 Mathematica packages (`auxiliary/TypeX_package.wl`)

**Key functions:**

- **Oscillator algebra setup**: Computes L, Δ₀, Λ[a], ν[i] from quantum numbers
- **FromPert(stateID, gloop, listParams)**: Loads perturbative data from `.mx` file, evaluates weak-coupling expansion at given g
- **InterpolateWeak(g0, eOrder, prec)**: Blends perturbative + saved numerical data using `FindFit`
- **InterpolateIn(g0, eOrder, prec)**: Pure polynomial extrapolation from saved numerical points
- **IncrBs(BS, cutPold, cutPnew)**: Pads coefficient vector with zeros when increasing cutP
- **GoodB(Plist)**: Boosts precision to `WP × 3/2`

**Parameter vector structure** differs by type:
- TypeI: `{g, Δ, c[1,2], c[1,4], ..., c[4,cutP]}` — even powers only, ~2·cutP + 2 elements
- TypeII: `{g, Δ, c[1,1], c[1,2], ..., c[4,cutP]}` — all powers, ~4·cutP + 2 elements
- TypeIII: `{g, Δ, c[1,2], ..., c[4,cutP], c^{1,2}, ..., c^{4,cutP}}` — both P_a and P^a, ~4·cutP + 2
- TypeIV: full ~8·cutP + 2 elements

### 3.4 Mathematica scripts (`auxiliary/TypeX_script.wls`)

The script bridges package → C++ → output:

1. **Load initial guess** via `InterpolateWeak` or `InterpolateIn`
2. **Denormalize coefficients**: undo the `g^{Mt[a]}` scaling used in the package representation
3. **Build C++ command string**: 21 + 4×N0 arguments with high-precision `CForm` formatting
4. **Execute**: `Import["!./TypeX_exec.out " <> args, "String"]`
5. **Parse output**: extract error code, Δ, coefficients, convergence metrics
6. **Renormalize**: restore the `g^{Mt[a]}` scaling
7. **Save**: append to `.mx` file with `{saved, init, params, error, values, digits, command}` functions
8. **Emit diagnostics**: `Print["key||", value, "||"]` for Python parsing

### 3.5 Konishi prototype (`prototype/Konishi_prototype.nb`)

Pure Mathematica implementation of the TypeI algorithm. No C++ dependency.
Useful as a correctness reference — transparent but much slower.
Implements the same P→Q→gluing→Fourier cycle but using Mathematica's native numerics.

---

## 4. Data Inventory

### 4.1 Numerical spectral data (`data/numerical/`)

**219 files** in Mathematica binary `.mx` format.

Each file contains a function `SpectralData[{Δ₀, nb1, nb2, nf1, nf2, nf3, nf4, na1, na2, sol}]` that returns a list of `{g, Δ}` pairs with ≥12 digit precision (often 20+).

**Distribution by bare dimension Δ₀:**

| Δ₀ | Count | Fraction |
|----|-------|----------|
| 2 | 1 | 0.5% |
| 3 | 1 | 0.5% |
| 4 | 10 | 4.6% |
| 11/2 | 36 | 16.4% |
| 5 | 27 | 12.3% |
| 6 | 144 | 65.8% |

73 distinct base states (unique quantum number sets), with 1–16 solutions per state.

### 4.2 Perturbative data (`data/perturbative/`)

**219 files** — exact 1-to-1 correspondence with numerical data.

Each contains substitution rule `sbWeak` with weak-coupling expansions of Δ and all c_{a,n} from the Marboe-Volin perturbative QSC solver (arXiv:1812.09238).

### 4.3 File naming convention

```
{type}_data_Delta0{Δ₀}_b1{nb1}_b2{nb2}_f1{nf1}_f2{nf2}_f3{nf3}_f4{nf4}_a1{na1}_a2{na2}_sol{s}.mx
```

Half-integer Δ₀ encoded as e.g. `Delta011by2` for Δ₀ = 11/2.

Example — Konishi operator: `numerical_spectral_data_Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1.mx`

---

## 5. Hyperparameter Reference

| Parameter | Physical meaning | Typical values | Role in algorithm |
|-----------|-----------------|----------------|-------------------|
| cutP | Fourier mode cutoff for P_a(x) | 12–24 | Truncation order N_trunc |
| nPoints | Chebyshev grid size on cut | cutP + 2 | Number of probe equations |
| cutQai | Large-u expansion order for Q | 24–30 | Asymptotic accuracy |
| QaiShift | Shift parameter in Q evaluation | 50–100 | Numerical stability of pull-down |
| epsilon | Newton residual tolerance | 10^{-30} | Inner-loop convergence goal |
| WP | Working precision (digits) | 186–500 | CLN decimal digit count |
| errorTolerance | Required precision for saving | 20–36 | Data quality threshold |
| maxIterations | Newton iteration cap | 5 | Safety limit per g-point |
| DH | Finite-difference step exponent | 30 | h = 10^{-DH} for Jacobian |

---

## 6. Key Bottlenecks for Optimization

### 6.1 Jacobian by finite differences — O(dimV) forward passes

The #1 bottleneck. Each Newton step requires perturbing every variable independently and re-evaluating the full forward map. For TypeI with cutP=16, dimV ≈ 33. For TypeIV with cutP=24, dimV ≈ 193.

**Optimization target**: Replace with automatic differentiation (JAX `jacfwd`/`jacrev`) → 1–5 forward-pass equivalents.

### 6.2 Full arbitrary precision throughout

The C++ code uses CLN at 186+ decimal digits for ALL arithmetic, even during early Newton iterations where float64 (~15 digits) would suffice. Mixed-precision refinement (float64 for early iterations, multiprecision for final polish) could save 50× per iteration.

### 6.3 Serial state processing

All 219 states are independent problems processed one at a time. With JAX `vmap`, these can be batched across GPU cores for near-linear speedup.

### 6.4 Subprocess chain latency

Each (state, g) evaluation requires Python → wolframscript → C++ → wolframscript → Python with shell overhead. A single JAX function call eliminates this entirely.

### 6.5 No quasi-Newton methods

The code always recomputes the full Jacobian. Broyden's method (Sherman-Morrison rank-1 update of J⁻¹) would reduce per-iteration cost from O(dimV) to O(1) forward passes after the first step.

### 6.6 No adaptive truncation

The code uses a fixed cutP throughout. A multigrid strategy (start at cutP=4, solve coarsely, double cutP, use coarse solution as initial guess) would dramatically reduce the number of expensive large-N iterations.

### 6.7 No predictor-corrector continuation

The adaptive step-size in g is purely reactive (double/halve based on success). Tangent-line prediction using the implicit derivative `dc/dg = -J⁻¹ ∂F/∂g` would provide much better initial guesses, reducing corrector iterations.

---

## 7. Notation Cross-Reference

| This code | Implementation guide | Physics meaning |
|-----------|---------------------|-----------------|
| c[a,n] | c_{a,n} | P-function expansion coefficients |
| Mt[a] | M_a | Leading power of P_a in Zhukovsky variable |
| Mhat[i] | M̂_i | Large-u exponent of Q_i (encodes Δ) |
| cutP | N_trunc | Truncation order |
| lc / nPoints | — | Chebyshev grid size |
| QaiShift / NIcutoff | — | Imaginary pull-down parameters |
| WP | — | Working precision in decimal digits |
| precssf | — | Equation norm convergence target |
| DH | — | Finite-difference step: h = 10^{-DH} |
| nb, nf, na | [q₁, p, q₂], S | Oscillator quantum numbers |
| gloop | g | 't Hooft coupling √λ/(4π) |
