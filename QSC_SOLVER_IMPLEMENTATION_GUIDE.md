# Accelerated Quantum Spectral Curve Solver — Implementation Guide

## For use with Claude Code (or any agentic coding assistant)

---

## 1. Physics Context

We are solving the **Quantum Spectral Curve (QSC)** equations for planar $\mathcal{N}=4$ Super-Yang–Mills theory. The QSC is a finite system of functional equations that encodes the exact (non-perturbative, all-loop) spectrum of anomalous dimensions $\Delta(g)$ of local operators, where $g$ is related to the 't Hooft coupling $\lambda = 16\pi^2 g^2$.

**Key references:**
- Original QSC formulation: Gromov, Kazakov, Leurent, Volin — arXiv:1305.1939, arXiv:1405.4857
- Original numerical algorithm: Gromov, Levkovich-Maslyuk, Sizov — arXiv:1504.06640
- Fast QSC Solver (our baseline): Gromov, Hegedűs, Julius, Sokolova — arXiv:2306.12379
- Existing code: https://github.com/julius-julius/qsc (97% Mathematica, 2% C++)

The goal of this project is to **reimplement and accelerate** the QSC numerical solver, targeting 50–100× speedup over the existing C++/Mathematica pipeline via modern numerical methods (automatic differentiation, quasi-Newton solvers, GPU batching, adaptive truncation).

---

## 2. Mathematical Structure of the QSC

### 2.1 The Q-system and P-functions

The QSC involves **Q-functions** $\mathbf{Q}_i(u)$, $\tilde{\mathbf{Q}}_i(u)$ and **P-functions** $\mathbf{P}_a(u)$, $\tilde{\mathbf{P}}_a(u)$, where $a, i = 1, \ldots, 4$ and $u$ is the spectral parameter. These are functions of a single complex variable, connected by:

1. **QQ-relations** — bilinear functional equations involving shifts $u \to u \pm i/2$:
$$
\mathbf{Q}_{i}^{+} A_{ij} = \mathbf{P}_a \, M_{ai} \, \mathbf{Q}_j^{-} + \ldots
$$
where $f^{\pm}(u) \equiv f(u \pm i/2)$. The precise form involves the $\mathrm{PSU}(2,2|4)$ Q-system; see equations (17)–(31) of arXiv:2306.12379.

2. **Analyticity constraints** — the P-functions have a single branch cut on $[-2g, 2g]$ in the $u$-plane, and the Q-functions have an infinite ladder of cuts (Zhukovsky cuts).

3. **Asymptotics** — the large-$u$ behaviour of the Q-functions encodes the quantum numbers of the state: $\Delta$, Lorentz spin $S$, R-charges $[q_1, p, q_2]$, etc.

### 2.2 Zhukovsky parametrisation

The **Zhukovsky variable** is defined by:
$$
x(u) = \frac{u}{2g} + \sqrt{\frac{u}{2g} - 1}\sqrt{\frac{u}{2g} + 1}, \qquad u = g\left(x + \frac{1}{x}\right).
$$

The P-functions are expanded in powers of $1/x$:
$$
\mathbf{P}_a(u) = x^{-M_a} \sum_{n=0}^{\infty} c_{a,n} \, x^{-n},
$$
where $M_a$ are known integers determined by the quantum numbers of the state. The **free parameters** of the problem are:
$$
\text{unknowns} = \{\Delta,\; c_{a,n},\; \tilde{c}_{a,n}\}, \quad a=1,\ldots,4, \quad n = 0, 1, \ldots, N_{\mathrm{trunc}}.
$$

Some of these are fixed by normalisation/gauge conditions; the effective number of unknowns is $\sim 8 N_{\mathrm{trunc}} + 1$.

### 2.3 The numerical algorithm (iterative scheme)

The algorithm solves a **nonlinear system** $F(\vec{c}, \Delta) = 0$ via Newton's method. Each evaluation of $F$ involves:

#### Step A: P → Q propagation
Given the current $\{c_{a,n}\}$ and $\Delta$, construct $\mathbf{P}_a(u)$ on the first sheet. Then solve the QQ-relations iteratively to obtain $\mathbf{Q}_i(u)$ at a set of **probe points** $\{u_k\}$ near the branch cut.

The QQ-relations are **functional equations with shifts**: they relate $Q(u+i/2)$ to $Q(u-i/2)$ and $P(u)$. In practice, one builds $Q$ column-by-column on a grid extending from $u \approx 2g$ out to $|u| \gg 1$, using the recursion:
$$
\mathbf{Q}_i(u + i/2) = [\text{known matrix}(u)] \cdot \mathbf{Q}_i(u - i/2) + [\text{source from } \mathbf{P}].
$$
This is iterated starting from the known large-$u$ asymptotics.

#### Step B: Analytic continuation (first sheet → second sheet)
Compute $\tilde{\mathbf{Q}}_i(u)$ and $\tilde{\mathbf{P}}_a(u)$ on the second sheet of the Zhukovsky cut. The continuation through the cut $[-2g, 2g]$ uses:
$$
\tilde{f}(u) = f(u)\big|_{x \to 1/x}.
$$
In practice, this means replacing $x^{-n} \to x^{n}$ in the P-function expansion, and similarly propagating the Q-functions on the second sheet.

#### Step C: Gluing conditions
At the probe points $\{u_k\}$, impose the **gluing conditions**:
$$
\tilde{\mathbf{Q}}_i(u_k) = M_{ij}(u_k) \, \mathbf{Q}_j(u_k),
$$
where $M_{ij}$ is a known (coupling-dependent, state-dependent) gluing matrix. These conditions ensure that the Q-functions have the correct monodromy around the branch cut.

#### Step D: Fourier transform back to coefficients
From the values of $\mathbf{P}_a(u_k)$ obtained via the gluing conditions, extract updated coefficients $\{c_{a,n}^{\text{new}}\}$ by a discrete Fourier-type transform (since $\mathbf{P}_a$ is expanded in $x^{-n}$ and $x = e^{i\theta}$ on the cut).

#### Step E: Newton update
Form the residual:
$$
F_n = c_{a,n}^{\text{new}} - c_{a,n}^{\text{input}}, \qquad \text{plus similar for } \tilde{c}_{a,n}.
$$
Assemble the Jacobian $J = \partial F / \partial (\vec{c}, \Delta)$ and solve:
$$
\begin{pmatrix} \delta \vec{c} \\ \delta \Delta \end{pmatrix} = -J^{-1} F.
$$
Update and iterate until $\|F\| < \epsilon$.

### 2.4 Symmetry reductions

Many states have discrete symmetries (left-right symmetry, parity) that relate $\mathbf{P}_a \leftrightarrow \tilde{\mathbf{P}}_a$ or impose $c_{a,n} = \tilde{c}_{a,n}$, halving the number of unknowns. The code should detect and exploit these automatically.

---

## 3. Acceleration Strategies to Implement

### 3.1 Automatic Differentiation for the Jacobian (HIGH PRIORITY)

**Current approach:** The Jacobian is computed by finite differences — perturbing each of the $\sim 8 N_{\rm trunc}$ parameters one at a time and re-evaluating the full forward map. This costs $\mathcal{O}(N)$ forward passes.

**New approach:** Implement the forward map $(\vec{c}, \Delta) \mapsto F$ in **JAX** (Python). Then use `jax.jacfwd` or `jax.jacrev` to get the exact Jacobian in $\sim 1$–$5$ forward-pass equivalents.

Key implementation notes:
- The forward map must be written in **pure functional style** (no in-place mutation) for JAX tracing to work.
- Use `jax.lax.scan` for the iterative Q-propagation (Step A), not Python for-loops.
- Complex arithmetic is natively supported in JAX.
- For the Fourier-transform step, use `jax.numpy.fft`.

### 3.2 Quasi-Newton (Broyden) Solver (HIGH PRIORITY)

Replace the full-Newton iteration with **Broyden's method**:
```
J_{k+1}^{-1} = J_k^{-1} + (δx - J_k^{-1} δF) ⊗ (δx^T J_k^{-1}) / (δx^T J_k^{-1} δF)
```
where $\delta x = x_{k+1} - x_k$ and $\delta F = F_{k+1} - F_k$. This is the Sherman–Morrison update of $J^{-1}$.

- Initialise $J_0$ with the true Jacobian (from AD) at the first step.
- Subsequent iterations only require **one** forward evaluation each.
- Convergence is superlinear (not quadratic), but the per-iteration cost drops from $\mathcal{O}(N)$ to $\mathcal{O}(1)$ forward passes.
- Fallback: if Broyden stalls (residual not decreasing), recompute the full Jacobian via AD and restart.

### 3.3 Adaptive Truncation Order (MEDIUM PRIORITY)

Implement a **multigrid-in-$N_{\rm trunc}$** strategy:
1. Start with $N_{\rm trunc} = 4$ (very cheap, $\sim 33$ unknowns).
2. Solve to modest precision ($\|F\| < 10^{-6}$).
3. Increase $N_{\rm trunc} \to 2 N_{\rm trunc}$. Pad the new coefficients with zeros.
4. Use the coarse solution as the initial guess for the refined problem.
5. Repeat until the desired $N_{\rm trunc}$ is reached.

This dramatically reduces the number of expensive (large-$N$) iterations.

### 3.4 Continuation in $g$ (Predictor-Corrector) (MEDIUM PRIORITY)

When scanning over a grid of couplings $g_1 < g_2 < \ldots < g_K$:

**Predictor step** (tangent extrapolation):
$$
\vec{c}^{(0)}(g_{k+1}) = \vec{c}(g_k) + (g_{k+1} - g_k) \frac{d\vec{c}}{dg}\bigg|_{g_k}.
$$
The derivative $d\vec{c}/dg$ is obtained from implicit differentiation of $F(\vec{c}, \Delta; g) = 0$:
$$
\frac{d\vec{c}}{dg} = -J^{-1} \frac{\partial F}{\partial g}.
$$
Since we already have $J^{-1}$ from the Newton solve at $g_k$, this is essentially free.

**Corrector step:** Run Newton/Broyden from the predicted initial guess.

**Adaptive step size:** If the corrector converges in ≤ 3 iterations, double $\delta g$; if it takes > 8 iterations (or fails), halve $\delta g$ and retry.

### 3.5 GPU Batching over States (MEDIUM PRIORITY)

The 219+ states are **independent** problems. With JAX:
```python
# vmap over state index
batched_solve = jax.vmap(solve_single_state)
all_results = batched_solve(all_initial_guesses, all_quantum_numbers)
```
This fills GPU cores and gives near-linear speedup up to hardware limits.

### 3.6 Mixed-Precision Iterative Refinement (LOW PRIORITY, HIGH IMPACT AT HIGH PRECISION)

1. Run Newton iterations in `float64` until $\|F\| < 10^{-12}$.
2. Switch to `mpfr` (via `mpmath` or `arb`) only for the final 2–3 iterations to reach $10^{-30}$ or beyond.

Since multi-precision arithmetic is $\sim 50\times$ slower than `float64`, this saves most of the runtime when high precision is needed.

### 3.7 Spectral Representation of the Shift Operator (EXPLORATORY)

The shift $u \to u + i/2$ in Fourier space is multiplication by $e^{-\pi k}$. If the Q-functions are expanded in a suitable spectral basis (Chebyshev polynomials on the cut, Laurent series outside), the QQ-relations become **banded or diagonal** in coefficient space, replacing the pointwise grid-based propagation with sparse linear algebra. This is a deeper refactoring but could yield asymptotic speedups for large $N_{\rm trunc}$.

---

## 4. Recommended Project Structure

```
qsc-fast/
├── README.md
├── pyproject.toml                # dependencies: jax, jaxlib, mpmath, numpy, scipy
│
├── qsc/
│   ├── __init__.py
│   ├── zhukovsky.py              # x(u), u(x), and branch-cut utilities
│   ├── quantum_numbers.py        # State specification: Δ₀, S, [q1,p,q2], symmetry flags
│   ├── p_functions.py            # P_a(u) construction from {c_{a,n}} in Zhukovsky expansion
│   ├── qq_relations.py           # Q-propagation: P → Q via shifted functional equations
│   ├── analytic_continuation.py  # First sheet → second sheet (x → 1/x)
│   ├── gluing.py                 # Gluing conditions: Q̃ = M · Q at probe points
│   ├── fourier.py                # Discrete transform: probe-point values → updated c_{a,n}
│   ├── forward_map.py            # Full pipeline: (c, Δ) → F residual  [JAX-traceable]
│   ├── newton.py                 # Newton solver with AD Jacobian
│   ├── broyden.py                # Broyden quasi-Newton solver
│   ├── continuation.py           # Predictor-corrector continuation in g
│   ├── adaptive_truncation.py    # Multigrid-in-N strategy
│   ├── symmetry.py               # Detect and impose LR / parity symmetry reductions
│   └── precision.py              # Mixed-precision wrapper (float64 ↔ mpfr)
│
├── scripts/
│   ├── solve_single_state.py     # CLI: solve one state at one coupling
│   ├── scan_coupling.py          # CLI: scan Δ(g) for one state over g-range
│   ├── scan_all_states.py        # CLI: batch-solve all 219 states (GPU)
│   └── validate_against_data.py  # Compare against published data from the GitHub repo
│
├── notebooks/
│   ├── 01_zhukovsky_basics.ipynb
│   ├── 02_forward_map_test.ipynb
│   ├── 03_ad_jacobian_demo.ipynb
│   ├── 04_broyden_vs_newton.ipynb
│   └── 05_continuation_demo.ipynb
│
├── mathematica/                  # Reference implementations for cross-checking
│   ├── QSCForwardMap.wl
│   └── KonishiTest.wl
│
├── tests/
│   ├── test_zhukovsky.py
│   ├── test_qq_relations.py
│   ├── test_forward_map.py       # Check F(exact_solution) ≈ 0 using known data
│   ├── test_jacobian_ad_vs_fd.py # Compare AD Jacobian against finite differences
│   └── test_konishi.py           # End-to-end: reproduce Konishi Δ(g) to known precision
│
└── data/
    ├── perturbative_starting_points/  # From Marboe-Volin weak-coupling QSC solver
    └── reference_spectra/             # Published Δ(g) data for validation
```

---

## 5. Implementation Roadmap

### Phase 1: Core Forward Map in JAX (Week 1–2)

**Goal:** A single function `forward_map(params, quantum_numbers, g, config) -> residual` that is fully JAX-traceable.

1. **`zhukovsky.py`**: Implement `x_of_u(u, g)` and `u_of_x(x, g)`. Handle the branch cut carefully: for $|u| < 2g$ the Zhukovsky variable is on the unit circle $x = e^{i\theta}$.

2. **`p_functions.py`**: Given coefficients `c_a = jnp.array([c_{a,0}, c_{a,1}, ..., c_{a,N}])` and the leading power $M_a$, compute:
   ```python
   def P_a(u, c_a, M_a, g):
       x = x_of_u(u, g)
       powers = x ** (-(M_a + jnp.arange(len(c_a))))
       return jnp.dot(c_a, powers)
   ```

3. **`qq_relations.py`**: This is the most delicate part. The Q-propagation is a **matrix recurrence** in the spectral parameter $u$:
   $$
   \vec{Q}(u + i/2) = T(u) \cdot \vec{Q}(u - i/2) + S(u)
   $$
   where $T(u)$ and $S(u)$ depend on $\mathbf{P}_a(u)$. Implement this as a `jax.lax.scan` over the $u$-grid:
   ```python
   def propagate_Q(P_values_on_grid, Q_asymptotic, u_grid):
       def step(Q_minus, u_and_P):
           u, P_vals = u_and_P
           T, S = build_transfer_matrix(u, P_vals)
           Q_plus = T @ Q_minus + S
           return Q_plus, Q_plus
       _, Q_all = jax.lax.scan(step, Q_asymptotic, (u_grid, P_values_on_grid))
       return Q_all
   ```

4. **`gluing.py`**: At probe points, compute the residual of $\tilde{Q}_i - M_{ij} Q_j = 0$.

5. **`forward_map.py`**: Chain everything together. The function signature should be:
   ```python
   def forward_map(params: jnp.ndarray, quantum_numbers: QuantumNumbers, 
                   g: float, config: SolverConfig) -> jnp.ndarray:
       """
       params: flat array [Δ, c_{1,0}, ..., c_{1,N}, c_{2,0}, ..., c̃_{4,N}]
       Returns: residual vector F of same length as params
       """
   ```

**Validation:** For the Konishi operator at $g = 1$, the known result is $\Delta \approx 4.189...$. Check that `forward_map(known_solution) ≈ 0`.

### Phase 2: Newton Solver with AD (Week 2–3)

1. **`newton.py`**:
   ```python
   def newton_solve(params0, quantum_numbers, g, config, tol=1e-12, max_iter=30):
       F = lambda p: forward_map(p, quantum_numbers, g, config)
       J_fn = jax.jacfwd(F)  # or jacrev depending on shape
       
       params = params0
       for i in range(max_iter):
           residual = F(params)
           if jnp.max(jnp.abs(residual)) < tol:
               break
           jacobian = J_fn(params)
           delta = jnp.linalg.solve(jacobian, -residual)
           params = params + delta
       return params
   ```

2. **`test_jacobian_ad_vs_fd.py`**: Verify the AD Jacobian against a finite-difference Jacobian to machine precision.

### Phase 3: Broyden Solver (Week 3)

1. **`broyden.py`**:
   ```python
   def broyden_solve(params0, quantum_numbers, g, config, tol=1e-12, max_iter=100):
       F = lambda p: forward_map(p, quantum_numbers, g, config)
       
       # Initialise with true Jacobian
       J_inv = jnp.linalg.inv(jax.jacfwd(F)(params0))
       
       x = params0
       f = F(x)
       for i in range(max_iter):
           dx = -J_inv @ f
           x_new = x + dx
           f_new = F(x_new)
           if jnp.max(jnp.abs(f_new)) < tol:
               return x_new
           df = f_new - f
           # Sherman-Morrison update of J^{-1}
           u = dx - J_inv @ df
           J_inv = J_inv + jnp.outer(u, dx @ J_inv) / (dx @ J_inv @ df)
           x, f = x_new, f_new
       return x
   ```

2. Benchmark Broyden vs Newton: count total forward-map evaluations to reach $10^{-12}$ for several test states.

### Phase 4: Continuation and Adaptive Truncation (Week 3–4)

1. **`continuation.py`**: Implement predictor-corrector with adaptive step size in $g$.

2. **`adaptive_truncation.py`**: Implement the multigrid-in-$N_{\rm trunc}$ scheme. Key function:
   ```python
   def solve_with_adaptive_truncation(quantum_numbers, g, N_levels=[4, 8, 16, 32]):
       params = initial_guess_from_perturbation_theory(quantum_numbers, g, N=N_levels[0])
       for N in N_levels:
           params = pad_to_truncation(params, N)
           config = SolverConfig(N_trunc=N)
           params = broyden_solve(params, quantum_numbers, g, config)
       return params
   ```

### Phase 5: GPU Batching and Production Runs (Week 4–5)

1. Use `jax.vmap` to batch over states.
2. Use `jax.pmap` for multi-GPU if available.
3. Reproduce the full 219-state scan from arXiv:2306.12379 and benchmark against their timings.

### Phase 6: High-Precision Mode (Optional, Week 5+)

1. Implement `mpmath`-based forward map for cases requiring > 15 digits.
2. Use float64 JAX for the first iterations, then switch to mpmath for refinement.

---

## 6. Key Subtleties and Pitfalls

### 6.1 Branch cuts and sheets
The Zhukovsky map has a branch cut on $[-2g, 2g]$. All arithmetic near this cut must be done with care. In JAX, use `jnp.where` to select the correct sheet rather than relying on the default branch of `jnp.sqrt`.

### 6.2 Asymptotic normalisation
The Q-functions have power-law asymptotics $Q_i(u) \sim u^{\hat{\Delta}_i}$ as $u \to \infty$, where $\hat{\Delta}_i$ depends on $\Delta$ and the quantum numbers. The propagation must be initialised from these asymptotics at a sufficiently large cutoff $u_{\max}$.

### 6.3 Gauge freedom
The QSC has a residual $H$-symmetry acting on the Q-functions. One must fix this gauge (e.g. by setting certain coefficients to specific values). The gauge-fixed parameters serve as a **precision diagnostic**: they should be zero, and their deviation from zero measures the accuracy of the solution.

### 6.4 Weak-coupling initialisation
At small $g$, the solution is close to the perturbative one. The Marboe–Volin perturbative QSC solver (arXiv:1812.09238) provides analytic weak-coupling expansions that serve as starting points. The code should be able to read these in (they are Mathematica expressions) and convert them to numerical initial guesses.

### 6.5 State identification and level crossings
Different states can cross as a function of $g$. The continuation code must track states by their quantum numbers, not by their energy ordering. If two states approach each other, decrease the step size in $g$.

### 6.6 Convergence of the $x$-expansion
At strong coupling ($g \gg 1$), the $1/x$ expansion of the P-functions converges slowly and large $N_{\rm trunc}$ is needed. The adaptive truncation scheme handles this, but one should also monitor the decay of $|c_{a,n}|$ and warn if the truncation is insufficient.

---

## 7. Validation Checkpoints

| Test | Expected Result | Source |
|------|----------------|--------|
| Konishi $\Delta(g{=}1)$ | $\approx 4.189$ | arXiv:1504.06640 |
| Konishi $\Delta(g{=}0.1)$ | matches 8-loop perturbation theory | arXiv:1812.09238 |
| Konishi $\Delta(g{=}5)$ | $\approx 2\sqrt[4]{\lambda} - 2 + \ldots$ (string theory) | arXiv:2306.12379 |
| 45 LR+parity-symmetric states | match published data in $g \in [0,5]$ | GitHub repo data/ |
| All 219 states at $g = 0.5$ | match published data | GitHub repo data/ |
| AD Jacobian vs FD Jacobian | agree to $\sim 10^{-7}$ (float64) | internal consistency |
| Broyden iterations ≤ 2× Newton iterations | for same final precision | benchmark |

---

## 8. Dependencies

```toml
[project]
name = "qsc-fast"
requires-python = ">=3.10"
dependencies = [
    "jax[cuda12]>=0.4.20",   # or jax[cpu] for CPU-only
    "jaxlib>=0.4.20",
    "numpy>=1.24",
    "scipy>=1.11",
    "mpmath>=1.3",
    "matplotlib>=3.7",       # for plotting Δ(g)
    "tqdm>=4.65",            # progress bars for scans
]

[project.optional-dependencies]
dev = ["pytest", "jupyter", "black", "ruff"]
```

---

## 9. Performance Targets

| Metric | Current (C++/Mathematica) | Target (JAX) |
|--------|--------------------------|---------------|
| Single state, single $g$, to $10^{-12}$ | ~10 s | ~0.2 s |
| Konishi $g \in [0, 5]$, 100 points | ~15 min | ~1 min |
| All 219 states, $g \in [0, 1]$, 50 points | ~weeks (1 PC) | ~hours (1 GPU) |
| Jacobian evaluation | $N$ forward passes | 1 reverse-mode pass |
| Per-iteration cost (Broyden vs Newton) | $N$ forward passes | 1 forward pass |

---

## 10. Getting Started (for the implementer)

1. **Clone the reference repo** and examine the Mathematica notebooks:
   ```bash
   git clone https://github.com/julius-julius/qsc
   ```
   Focus on the files in `local operators N4 SYM/`. The Mathematica code is the most complete reference for the algorithm.

2. **Start with the Konishi operator** — it is the simplest nontrivial state (quantum numbers: $\Delta_0 = 4$, $S = 0$, $[0,2,0]$ in the $\mathfrak{so}(6)$ notation, i.e. the $\mathbf{20'}$ representation). It has full LR and parity symmetry, so the parameter space is smallest.

3. **Implement the forward map first**, without any solver. Verify that plugging in the known Konishi solution gives a residual close to zero.

4. **Then add the Newton solver with AD**, and verify you can converge from a nearby initial guess.

5. **Then add Broyden**, and compare iteration counts.

6. **Then add continuation in $g$**, and reproduce the Konishi curve.

7. **Finally, batch over states** and reproduce the full scan.

Good luck, and remember: the physics is in the analyticity constraints and the QQ-relations. Everything else is numerical plumbing.
