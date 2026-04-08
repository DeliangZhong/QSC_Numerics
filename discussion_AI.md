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
