# AI Development Discussion Log

<!-- INSTRUCTIONS FOR AI:
  - Reverse chronological order: newest entries on top.
  - Only read the TOP entry — everything below has already been discussed.
  - Naming convention:
      "Discussion-" = AI/human open-ended discussion and brainstorming
      "Implementation-" = AI's implementation results and findings
      "Feedback-"       = Human/AI feedback on the preceding implementation
  - When adding a new entry, prepend it above the previous top entry.
-->

## Implementation-3: Forward Map + Newton Status (Apr 8, 2026)

### What Works
- Forward map runs end-to-end for TypeI Konishi at g=0.1
- Initial residual ||E|| = 0.227 (float64, QaiShift=2)
- FD Newton with alpha=0.1 damping: residual decreases 0.227 → 0.207 → 0.185 over 2 iterations
- AD Jacobian compiles but Python for-loops make it impractically slow (~10 min/iteration)

### Key Findings
1. **float64 precision limit**: QaiShift > 2 causes growing errors in pulldown (exponential accumulation). C++ uses QaiShift=60 with 186-digit precision. At float64, QaiShift=2 is optimal.
2. **Jacobian effective rank = 31** (out of 32): one gauge direction is null. Using lstsq with rcond=1e-10 handles this.
3. **Convergence**: residual decreasing but Delta drifts from reference value — expected since float64 with QaiShift=2 solves a truncated problem.

### Performance Blockers
- Forward map uses Python for-loops (b-coefficient recursion, pulldown, gluing) → ~30s per evaluation
- Need vectorization with `jax.lax.scan` for both speed and AD traceability
- FD Jacobian: 33 evaluations × 30s = 10 min per Newton step

### Next Steps
1. Vectorize forward map internals (especially b-coefficient recursion and pulldown) using jax.lax.scan/fori_loop
2. Enable jax.jacfwd for exact AD Jacobian (~100× faster than FD)
3. Run Newton to convergence and verify Delta matches reference to float64 precision
4. Add Broyden solver (rank-1 Jacobian updates)
5. Continuation in g to reproduce full Konishi curve

---

## Discussion-2: Implementation Plan — TypeI Forward Map First (Apr 8, 2026)

### Strategy

After studying the C++ core (`TypeI_core.cpp`, 3200 lines) and the Mathematica orchestration pipeline in detail, the critical insight is: **the forward map is the hard part**. Newton/Broyden/continuation are standard numerical methods; the physics lives entirely in the forward map `(c, Δ, g) → F`. Once that's correct and JAX-traceable, everything else follows.

**Approach: TypeI Konishi first, validate obsessively, then generalize.**

TypeI (LR + parity symmetric) is the simplest case — only even powers in the P-expansion, `c_{a,n} = c̃_{a,n}`, and dimV ≈ 1 + 4×(cutP/2) after gauge fixing. The Konishi operator (Δ₀=2, `[0,0,1,1,1,1,0,0]`) is the canonical test case with abundant published data.

### Milestone 0: Validation Data Extraction

**Goal:** Extract intermediate values from the Konishi Mathematica prototype at a specific g (e.g., g = 0.1) so every module can be unit-tested against ground truth — not just final (g, Δ) pairs.

Run `prototype/Konishi_prototype.nb` via wolframscript and export at each algorithm stage:
- Quantum numbers and derived quantities: L, Λ[a], ν[i], Mt[a], M̂[i], A_a, B_i, AA, BB, α[a][i]
- Chebyshev grid: u_k points, CT/CU matrices, suA weights
- σ-coefficients (kappa/kappabar tables)
- P_a(u_k) on the grid (both sheets)
- ksub[a][n] (1/u expansion of P_a)
- q-array (convolution products)
- b_{a|i,n} coefficients for each i (the sequential 4×4 solves)
- Q_{a|i}(u_k) before and after pull-down
- Q_lower, Q̃_lower at the cut
- α_Q gluing constant
- δP residual at the cut
- E (equation vector after Fourier inversion)

Save as JSON/npz for pytest fixtures. This is the single most important step — without intermediate ground truth, debugging the forward map is guesswork.

### Milestone 1: Core Mathematics Modules (TypeI only)

All modules written in JAX from the start (pure functional, no mutation). Each module has pytest tests against Milestone 0 data.

**1.1 `qsc/quantum_numbers.py`**

Dataclass `QuantumNumbers` holding nb, nf, na, sol. Derive:
```
L = (Σnf + Σna - Σnb) / 2
Δ₀ = Σnf/2 + Σna
Λ = (1 - Λ₀[1] - Λ₀[4]) / 2
Λ[a] = nf[a] + {2,1,0,-1}[a] + Λ
ν[i] = {-L-nb₁-1, -L-nb₂-2, na₁+1, na₂}[i] + (Δ-Δ₀)/2·{-1,-1,1,1}[i] - Λ
Mt[a] = -Λ[a]   (powP in Mathematica)
M̂[i] = -ν[i] - 1   (powQ in Mathematica)
```
Plus A_a, B_i, AA[a][b], BB[a][i] matrices, and α[a][i] = M̂[i] - Mt[a].

Determine gauge-fixed indices (where `2n = Mtint[0] - Mtint[a]`), and CtoV/VtoC mappings.

**1.2 `qsc/zhukovsky.py`**

Core functions:
- `x_of_u(u, g)`: Zhukovsky variable. Use the **long-cut** convention from C++: `x = u/2 - i/2 · √(4-u²)` (note: C++ uses `u` rescaled by `1/(2g)` in places — must be careful with conventions).
- `x_of_u_short(u, g)`: short-cut version for `|u| > 2g`.
- `sigma_coefficients(twiceMt, N_trunc, NQ, g)`: the kappa/kappabar recursion. This encodes the 1/u expansion of `X(u)^{Mt[a]}` via:
  ```
  σ(twiceMt, n, r, g) = Σ_{s=0}^{k-r} kappabar(twiceMt, s) · kappa(2r+q₀, k-r-s)  ×  (√g)^{twiceMt+2n}
  ```
  where `k = n÷2`, `q₀ = n mod 2`.

**1.3 `qsc/chebyshev.py`**

- `chebyshev_grid(g, lc)`: Chebyshev-Gauss points on `[-2|g|, 2|g|]`.
- `chebyshev_matrices(lc)`: CT (cosine) and CU (Chebyshev-U) transform matrices.
- `sqrt_weight(g, u_k)`: `√(4g² - u_k²)` weights.

**1.4 `qsc/p_functions.py`**

- `evaluate_P(c, Mt, u_grid, g, sigma)`: P_a(u_k) from coefficients + sigma tables.
  Also computes P̃_a (tilde = second sheet, x→1/x).
- `ksub_coefficients(c, sigma, NQ)`: 1/u expansion coefficients of P_a.
- `evaluate_P_shifted(c, Mt, u_grid, g, n_shifts, sigma)`: P_a(u_k + i·n) for n = 0,...,NI-1 (needed for pull-down).

**1.5 `qsc/qq_relations.py`** — THE HARD MODULE

This translates `QconstructorUJ2LRi` from C++. Three stages:

*Stage A: q-array (convolution products)*
```
q[(n,a,b)] = Σ_{m=0}^{n} ksub[a][m] · (-1)^{b+1} · ksub[3-b][n-m]  /  AA[a][b]
```

*Stage B: b-coefficients via sequential 4×4 linear solves*
For each i=0,...,3 and m=1,...,NQ[i]:
```
scT[m] · b[i][m] = F1(m) - F2(m)
```
where:
- `scT[m][a][b] = AA[a][b]·B[b][i] - i·B[a][i]·(2m - α[a][i])·δ_{ab}` (from `totalscTmaker2LRi`)
- F1 depends on: BB, α[a][i], previous b's, T1/T2/T3/T41/T5 tables (binomial expansions of α)
- F2 depends on: AA, BB, previous b's, q-array, S1n/S1/S31/S32 tables

This is sequential (each m depends on all previous m's) → implement as `jax.lax.scan` over m.

*Stage C: Q evaluation + pull-down*
1. Evaluate Q_{a|i} at large u: `Q[a][i][k] = BB[a][i] · u_k^{-M̂_i-NI} · Σ_n b[i][n,a] · u_k^{-n}`
2. Pull down through NI imaginary steps:
   ```
   for n = NI-1 down to 0:
     Q_new[a][i][k] = Σ_{b} (-1)^{b+1} · P[3-b](u_k+in) · Q_old[b][i][k] · P[a](u_k+in) + Q_old[a][i][k]
   ```
   This is also sequential → `jax.lax.scan` (or `fori_loop`) over n.

**1.6 `qsc/gluing.py`**

From Q_{a|i}(u_k), compute:
1. `Q_lower[k,i] = -Σ_a (-1)^{a+1} · P[3-a](u_k) · Q[a][i][k]` (contract upper indices)
2. Similarly `Q̃_lower` using P̃ instead of P
3. `α_Q = Re(mean(Q₀/Q₂* + Q̃₀/Q̃₂* - Q₁/Q₃* - Q̃₁/Q̃₃*)) / 4`
4. Residual: `δP[k,a] = Q[a,0,k]·(Q₃+Q₁*/αQ) - Q[a,1,k]·(Q₂-Q₀*/αQ) + Q[a,2,k]·(Q₁+Q₃*·αQ) - Q[a,3,k]·(Q₀-Q₂*·αQ)`

**1.7 `qsc/fourier.py`**

Transform δP(u_k) back to coefficient residuals:
- `QtoE_typeI(deltaP, deltaPt, CT, CU, suA, ...)`: splits into symmetric/antisymmetric modes, applies Chebyshev inversion, produces residual vector E of dimension dimV.

**1.8 `qsc/forward_map.py`**

Chain: params → quantum_numbers → P-functions → Q-propagation → gluing → Fourier → residual.

```python
def forward_map(params: jnp.ndarray, qn: QuantumNumbers, g: float, config: SolverConfig) -> jnp.ndarray:
    """Pure functional: (Δ, c_{a,n}) → residual F. JAX-traceable."""
```

**Integration test:** `||forward_map(known_Konishi_solution, g=0.1)|| < 10⁻¹⁰`

### Milestone 2: Newton Solver with AD

**2.1 `qsc/newton.py`**

```python
def solve(params0, qn, g, config, tol=1e-12, max_iter=30):
    F = lambda p: forward_map(p, qn, g, config)
    J = jax.jacfwd(F)
    # standard Newton loop
```

**2.2 Validation:**
- Solve Konishi at g=0.1 starting from perturbative data → check Δ matches reference
- Verify AD Jacobian vs finite-difference Jacobian to ~10⁻⁷
- Solve Konishi at g=0.5 starting from g=0.1 solution → check Δ ≈ 3.713

### Milestone 3: Continuation + Optimization

**3.1 `qsc/broyden.py`** — Sherman-Morrison rank-1 update of J⁻¹. Benchmark iteration count vs Newton.

**3.2 `qsc/continuation.py`** — Predictor-corrector in g:
- Predictor: `c(g+δg) ≈ c(g) + δg · (-J⁻¹ ∂F/∂g)` (∂F/∂g via AD, essentially free)
- Corrector: Newton/Broyden from predicted guess
- Adaptive δg: double if ≤3 iterations, halve if >8

**3.3 `qsc/adaptive_truncation.py`** — Multigrid in N_trunc: solve at N=4, pad to N=8, re-solve, etc.

**Validation:** Reproduce full Konishi Δ(g) curve for g ∈ [0, 5] and compare against 254-point reference data.

### Milestone 4: Generalization to TypeII–IV

**4.1 `qsc/symmetry.py`** — Detect operator type from quantum numbers. Handle:
- TypeII: LR symmetric but general parity → different CtoV mapping, zero-mode complications
- TypeIII: Parity symmetric but no LR → separate c and c̃ parameters
- TypeIV: General → full 8·N_trunc + 1 parameter space

**4.2 GPU batching** — `jax.vmap(solve_single_state)` over the 219 states.

### Milestone 5: High-Precision Mode

`qsc/precision.py` — float64 for early Newton iterations, switch to mpmath for final 2–3 iterations when >15 digits needed.

### Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Convention mismatch (signs, normalizations, index ordering) | Fatal — wrong results | Milestone 0: extract ALL intermediate values from Mathematica |
| b-coefficient recursion has subtle dependencies | Hard to debug | Test each m-step against Mathematica b_{a\|i,m} values |
| JAX tracing breaks on sequential Q-propagation | Blocks AD | Use `jax.lax.scan`/`fori_loop`; fall back to `jax.checkpoint` if memory issues |
| Pull-down numerically unstable at strong coupling | Wrong Q at cut | Start with weak coupling (g < 1); match C++ NI strategy |
| Performance regression vs C++ at float64 | Defeats purpose | Profile after correctness; JIT compilation should help |

### Decisions (resolved Apr 8)

1. **Milestone 0 scope**: Full set — extract ALL intermediate quantities (quantum numbers, sigma tables, P values, ksub, q-array, b-coefficients, Q values before/after pulldown, gluing constant, residual, equation vector). Safest approach.
2. **Starting coupling**: g = 0.1 (weak coupling, perturbative data available, fast convergence).
3. **Module granularity**: `qq_relations.py` is one function (~250 lines of C++ → comparable in Python). Keep it monolithic to match the C++ structure.
4. **TypeII–IV timeline**: Defer until Konishi works end-to-end (Milestone 2 complete).

---

## Discussion-1: Accelerated QSC Solver — Implementation Guide (Apr 8, 2026)

### Goal

Reimplement and accelerate the QSC numerical solver for planar $\mathcal{N}=4$ SYM, targeting 50–100× speedup over the existing C++/Mathematica pipeline via modern numerical methods (automatic differentiation, quasi-Newton solvers, GPU batching, adaptive truncation).

### Key References

- Original QSC formulation: Gromov, Kazakov, Leurent, Volin — arXiv:1305.1939, arXiv:1405.4857
- Original numerical algorithm: Gromov, Levkovich-Maslyuk, Sizov — arXiv:1504.06640
- Fast QSC Solver (our baseline): Gromov, Hegedűs, Julius, Sokolova — arXiv:2306.12379
- Existing code: https://github.com/julius-julius/qsc (97% Mathematica, 2% C++)

### Mathematical Structure of the QSC

**Q-system and P-functions.** The QSC involves Q-functions $\mathbf{Q}_i(u)$, $\tilde{\mathbf{Q}}_i(u)$ and P-functions $\mathbf{P}_a(u)$, $\tilde{\mathbf{P}}_a(u)$, where $a, i = 1, \ldots, 4$ and $u$ is the spectral parameter. These are connected by:

1. **QQ-relations** — bilinear functional equations involving shifts $u \to u \pm i/2$:
$$
\mathbf{Q}_{i}^{+} A_{ij} = \mathbf{P}_a \, M_{ai} \, \mathbf{Q}_j^{-} + \ldots
$$
where $f^{\pm}(u) \equiv f(u \pm i/2)$. See equations (17)–(31) of arXiv:2306.12379.

2. **Analyticity constraints** — P-functions have a single branch cut on $[-2g, 2g]$; Q-functions have an infinite ladder of cuts.

3. **Asymptotics** — large-$u$ behaviour of Q-functions encodes quantum numbers: $\Delta$, spin $S$, R-charges $[q_1, p, q_2]$.

**Zhukovsky parametrisation.**
$$
x(u) = \frac{u}{2g} + \sqrt{\frac{u}{2g} - 1}\sqrt{\frac{u}{2g} + 1}, \qquad u = g\left(x + \frac{1}{x}\right).
$$

P-functions expanded as $\mathbf{P}_a(u) = x^{-M_a} \sum_{n=0}^{\infty} c_{a,n} \, x^{-n}$, with free parameters $\{\Delta, c_{a,n}, \tilde{c}_{a,n}\}$. Effective unknowns: $\sim 8 N_{\mathrm{trunc}} + 1$.

**Numerical algorithm (iterative scheme).** Solves $F(\vec{c}, \Delta) = 0$ via Newton's method:

- **Step A (P → Q):** Construct $\mathbf{P}_a(u)$, solve QQ-relations iteratively via matrix recurrence $\mathbf{Q}_i(u + i/2) = T(u) \cdot \mathbf{Q}_i(u - i/2) + S(u)$, starting from large-$u$ asymptotics.
- **Step B (sheet continuation):** Compute $\tilde{\mathbf{Q}}_i$, $\tilde{\mathbf{P}}_a$ on second sheet via $x \to 1/x$.
- **Step C (gluing):** Impose $\tilde{\mathbf{Q}}_i(u_k) = M_{ij}(u_k) \mathbf{Q}_j(u_k)$ at probe points.
- **Step D (Fourier):** Extract updated $c_{a,n}^{\text{new}}$ by discrete Fourier transform.
- **Step E (Newton):** Form residual $F_n = c_{a,n}^{\text{new}} - c_{a,n}^{\text{input}}$, assemble Jacobian, solve $\delta\vec{c} = -J^{-1}F$.

**Symmetry reductions.** LR symmetry and parity can impose $c_{a,n} = \tilde{c}_{a,n}$, halving unknowns. Four operator types: TypeI (LR+parity) through TypeIV (general).

### Acceleration Strategies

**3.1 AD for the Jacobian (HIGH).** Replace finite-difference Jacobian ($\mathcal{O}(N)$ forward passes) with JAX `jacfwd`/`jacrev` ($\sim 1$–$5$ passes). Forward map must be pure functional for JAX tracing; use `jax.lax.scan` for Q-propagation.

**3.2 Quasi-Newton / Broyden (HIGH).** Sherman–Morrison rank-1 update of $J^{-1}$: per-iteration cost drops from $\mathcal{O}(N)$ to $\mathcal{O}(1)$ forward passes. Fallback to full Jacobian if stalled.

**3.3 Adaptive Truncation (MEDIUM).** Multigrid in $N_{\rm trunc}$: start at $N=4$, solve coarsely, double $N$, pad with zeros, re-solve. Reduces expensive large-$N$ iterations.

**3.4 Predictor-Corrector Continuation (MEDIUM).** Tangent extrapolation $\vec{c}^{(0)}(g_{k+1}) = \vec{c}(g_k) + \delta g \cdot (-J^{-1} \partial F/\partial g)$ gives much better initial guesses for the next $g$-point. Adaptive step size.

**3.5 GPU Batching (MEDIUM).** 219 independent states → `jax.vmap(solve_single_state)` for near-linear GPU speedup.

**3.6 Mixed-Precision Refinement (LOW).** float64 for early iterations, `mpmath`/`arb` only for final 2–3 iterations to reach $10^{-30}$.

**3.7 Spectral Shift Operator (EXPLORATORY).** Shift $u \to u+i/2$ becomes multiplication by $e^{-\pi k}$ in Fourier space → banded/diagonal QQ-relations in coefficient space.

### Proposed Project Structure

```
qsc-fast/
├── qsc/
│   ├── zhukovsky.py              # x(u), u(x), branch-cut utilities
│   ├── quantum_numbers.py        # State specification
│   ├── p_functions.py            # P_a(u) from {c_{a,n}}
│   ├── qq_relations.py           # Q-propagation via shifted recurrence
│   ├── analytic_continuation.py  # x → 1/x sheet continuation
│   ├── gluing.py                 # Gluing conditions at probe points
│   ├── fourier.py                # Discrete transform: values → coefficients
│   ├── forward_map.py            # Full pipeline (c, Δ) → F  [JAX-traceable]
│   ├── newton.py                 # Newton solver with AD Jacobian
│   ├── broyden.py                # Broyden quasi-Newton solver
│   ├── continuation.py           # Predictor-corrector in g
│   ├── adaptive_truncation.py    # Multigrid-in-N strategy
│   ├── symmetry.py               # LR / parity symmetry reductions
│   └── precision.py              # Mixed-precision wrapper
├── scripts/                      # CLI tools
├── notebooks/                    # Exploration notebooks
├── tests/                        # pytest suite
└── data/                         # Reference spectra, perturbative starts
```

### Implementation Roadmap

**Phase 1 (Core Forward Map):** JAX-traceable `forward_map(params, quantum_numbers, g, config) -> residual`. Validate: Konishi at $g=1$ gives $\Delta \approx 4.189$, `forward_map(known_solution) ≈ 0`.

**Phase 2 (Newton + AD):** `jax.jacfwd(forward_map)` → exact Jacobian. Verify AD vs FD to machine precision.

**Phase 3 (Broyden):** Sherman–Morrison update. Benchmark iteration count vs Newton.

**Phase 4 (Continuation + Adaptive Truncation):** Predictor-corrector in $g$; multigrid $N_{\rm trunc} = [4, 8, 16, 32]$.

**Phase 5 (GPU Batching):** `jax.vmap` over states, `jax.pmap` for multi-GPU. Reproduce 219-state scan.

**Phase 6 (High Precision):** `mpmath`-based forward map for $>15$ digit results.

### Key Subtleties

- **Branch cuts:** Zhukovsky cut on $[-2g, 2g]$ — use `jnp.where` for sheet selection.
- **Asymptotic normalisation:** Q-functions $\sim u^{\hat{\Delta}_i}$ at large $u$.
- **Gauge freedom:** Residual $H$-symmetry — gauge-fixed parameters should be zero (precision diagnostic).
- **Weak-coupling init:** Marboe–Volin perturbative expansions (arXiv:1812.09238) as starting points.
- **Level crossings:** Track states by quantum numbers, not energy ordering.
- **Convergence at strong coupling:** $1/x$ expansion converges slowly for $g \gg 1$ — monitor $|c_{a,n}|$ decay.

### Validation Checkpoints

| Test | Expected | Source |
|------|----------|--------|
| Konishi $\Delta(g{=}1)$ | $\approx 4.189$ | arXiv:1504.06640 |
| Konishi $\Delta(g{=}0.1)$ | matches 8-loop perturbation theory | arXiv:1812.09238 |
| Konishi $\Delta(g{=}5)$ | $\approx 2\sqrt[4]{\lambda} - 2 + \ldots$ | arXiv:2306.12379 |
| 45 LR+parity states | match published data, $g \in [0,5]$ | GitHub repo |
| All 219 states at $g = 0.5$ | match published data | GitHub repo |
| AD vs FD Jacobian | agree to $\sim 10^{-7}$ (float64) | internal |
| Broyden iterations | $\leq 2\times$ Newton for same precision | benchmark |

### Performance Targets

| Metric | Current (C++/Mathematica) | Target (JAX) |
|--------|--------------------------|---------------|
| Single state, single $g$, to $10^{-12}$ | ~10 s | ~0.2 s |
| Konishi $g \in [0, 5]$, 100 points | ~15 min | ~1 min |
| All 219 states, $g \in [0, 1]$, 50 points | ~weeks (1 PC) | ~hours (1 GPU) |
| Jacobian evaluation | $N$ forward passes | 1 reverse-mode pass |
| Per-iteration cost (Broyden vs Newton) | $N$ forward passes | 1 forward pass |
